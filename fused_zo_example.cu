/*
 * fused_zo_example.cu
 *
 *
 * MeZO-style zeroth-order (ZO) optimization on:
 *   out = layernorm(silu(inp @ W))
 * where
 *   inp    [B, M, d] — batch of M-row inputs
 *   W [d, I]    — shared weight, updated in-place
 *   out    [B, M, I] — batch outputs
 *
 * All batch samples share the same perturbation z ~ N(0,1) for W.
 * z is never materialized; each element is regenerated on-the-fly from a
 * seed via Philox counter-based PRNG.
 *
 * Two kernels:
 *   zo_fused_forward — perturbed forward pass + MSE loss accumulation
 *   zo_update        — ZO weight update, regenerates z from the same seed
 *
 * Thread layout (zo_fused_forward):
 *   t = m_local * ZO_TPR + n_quarter.  Thread t owns THREAD_N consecutive output
 *   columns n0 = n_start + THREAD_N*n_quarter for its own row m_local only.
 *   THREAD_N/PHILOX_WIDTH Philox calls per (weight-row k, n_quarter) yield THREAD_N
 *   N(0,1) samples total — all ZO_TILE_M rows use the same z for the same weight
 *   (shared perturbation). Register arrays v_cache_p/n[(I_DIM/ZO_TILE_N)*THREAD_N]
 *   accumulate the full GEMM output for both perturbations; SiLU is applied
 *   in-place after the loops.  Requires I % ZO_TILE_N == 0.
 *
 * Philox ILP:
 *   counter = k*(I/PHILOX_WIDTH) + n_start/PHILOX_WIDTH + n_quarter*(THREAD_N/PHILOX_WIDTH) + g
 *   uniquely addresses the g-th group-of-4 within thread n_quarter's THREAD_N columns.
 *
 * Compile: nvcc -arch=sm_89 -O2 fused_zo_example.cu -o fused_zo_example
 * Run:     ./fused_zo_example
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define checkCuda(ans) gpuAssert((ans), __FILE__, __LINE__)
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d — %s\n", file, line,
                cudaGetErrorString(code));
        exit(code);
    }
}

// ============================================================================
// Philox-4x32 counter-based PRNG
//
// Philox is counter-based: given a (seed, counter) pair it produces
// random-looking output via ~10 integer multiplications, with no sequential
// state to maintain. Any thread can independently evaluate any position in
// the stream, without storing intermediate state. Both kernels derive the
// counter from (k, n) alone, so z is regenerated on-the-fly without storage
// or coordination.
//
// philox_uniform_4 — (seed, counter) → 4 uniform floats in [0, 1)
// philox_normal_4  — Box-Muller pairs: (u0,u1)→(g0,g1), (u2,u3)→(g2,g3)
// ============================================================================
__host__ __device__ __forceinline__
float4 philox_uniform_4(unsigned long long seed, unsigned long long counter)
{
    const unsigned int P10A = 0x9E3779B9u, P10B = 0xBB67AE85u;
    const unsigned int PSA  = 0xD2511F53u, PSB  = 0xCD9E8D57u;
    uint2 key = make_uint2((unsigned int)(seed),
                           (unsigned int)(seed >> 32));
    uint4 ctr = make_uint4((unsigned int)(counter),
                           (unsigned int)(counter >> 32), 0u, 0u);
    #pragma unroll
    for (int r = 0; r < 7; ++r) {
        unsigned int h0 = (unsigned int)(((unsigned long long)PSA * ctr.x) >> 32);
        unsigned int h1 = (unsigned int)(((unsigned long long)PSB * ctr.z) >> 32);
        ctr = make_uint4(h1 ^ ctr.y ^ key.x,  PSA * ctr.x,
                         h0 ^ ctr.w ^ key.y,  PSB * ctr.z);
        key.x += P10A; key.y += P10B;
    }
    const float s = 2.3283064e-10f;  // 1/2^32
    return make_float4(ctr.x * s, ctr.y * s, ctr.z * s, ctr.w * s);
}

__host__ __device__ __forceinline__
float4 philox_normal_4(unsigned long long seed, unsigned long long counter)
{
    float4 u = philox_uniform_4(seed, counter);
    u.x = fmaxf(u.x, 1e-7f);  // avoid log(0) = inf
    u.z = fmaxf(u.z, 1e-7f);
    float r0 = sqrtf(-2.f * logf(u.x)), t0 = 6.28318530f * u.y;
    float r1 = sqrtf(-2.f * logf(u.z)), t1 = 6.28318530f * u.w;
    return make_float4(r0 * cosf(t0), r0 * sinf(t0),
                       r1 * cosf(t1), r1 * sinf(t1));
}

// ============================================================================
// zo_fused_forward_kernel
//
// MeZO estimates gradients without backpropagation.  It evaluates the loss at
// two slight weight perturbations — W+eps*z and W-eps*z — and estimates the
// gradient as (L(Φ(θ + εz, b)) - L(Φ(θ + εz, b))) / 2ε.
//   
// This kernel computes BOTH perturbed forward passes in a single launch,
// loading inp and w only once. It also generates and does not materialize
// the gaussian vector z, instead fusing ±εz into the weight loads.
//
// Computes both:
//   out_pos = layernorm(silu(inp @ (W + eps*z)))
//   out_neg = layernorm(silu(inp @ (W - eps*z)))
//
// Thread assignment
//   FWD_THREADS threads per block are arranged as a logical ZO_TILE_M × ZO_TPR grid:
//     m_local   = t / ZO_TPR  — which of the ZO_TILE_M output rows this thread owns
//     n_quarter = t % ZO_TPR  — which group of THREAD_N consecutive columns (0–ZO_TPR-1)
//   Thread t accumulates dot products for row m_local, columns n_quarter*THREAD_N
//   through n_quarter*THREAD_N+THREAD_N-1, and owns those same columns during LayerNorm.
//
// Loop order — K-outer, N-inner (same as the reference kernel):
//   inp_tile is loaded once per K-tile and reused across all N-tiles.
//   Loading it inside the N-loop would reload it I/ZO_TILE_N times per K-tile.
//   The Philox counter = k*(I/4) + n_start/4 + n_quarter depends on both k
//   and n; since both are available in the K-outer/N-inner body there is no
//   constraint on loop order from Philox.
//
// Grid:  dim3(ceil(M/ZO_TILE_M), B)  — one block per (row-tile, batch element)
// Block: FWD_THREADS = ZO_TILE_M * ZO_TPR threads (256 = 4 rows × 64 threads)
//
// Shared memory layout:
//   inp_tile_p [ZO_TILE_M × ZO_TILE_K floats] = 1 KB   + perturbation input
//   inp_tile_n [ZO_TILE_M × ZO_TILE_K floats] = 1 KB   - perturbation input
//   w_tile     [ZO_TILE_K × ZO_TILE_N halfs]  = 32 KB  weight strip (shared)
//   smem_red   [4 * FWD_THREADS floats]       = 4 KB   tree-reduce buffer (stats)
//   Total: ~38 KB
//
// w_tile is stored as half (not float).  The dot-product access pattern reads
// THREAD_N consecutive halves per thread, with a stride of THREAD_N between
// adjacent threads in a warp.  With half storage, that stride is THREAD_N*2
// bytes → THREAD_N/2 banks → (THREAD_N/2)-way conflicts.  THREAD_N=PHILOX_WIDTH=4
// gives 2-way conflicts, the minimum possible for any multiple-of-4 THREAD_N.
// Larger THREAD_N widens the stride and worsens conflicts.  Promoting to float
// at any THREAD_N doubles all strides and doubles the conflict count.
//
// Post-SiLU values live in v_cache_p/n register arrays and never touch shared
// memory, avoiding a large intermediate buffer.
//
// ---- Porting to Ampere / Hopper / Blackwell ---------------------------------
//
// Three changes are needed to make this a production kernel.  The concept is
// the same on all three architectures; the hardware primitives differ.
//
// (1) Replace the scalar FMA inner loop with a tensor-core MMA instruction.
//     Ampere:    mma.sync.aligned.m16n8k16.f32.f16.f16.f32
//                per-warp (32 threads); each thread holds a fixed slice of the
//                16×8 C-fragment, dictated by the PTX ISA.
//     Hopper:    wgmma.mma_async — warpgroup-level (128 threads); B is read
//                directly from shared memory; only C lives in registers.
//     Blackwell: tcgen05.mma — warpgroup-level; C accumulates in tensor memory
//                (tmem), a dedicated per-SM scratchpad separate from smem,
//                relieving register pressure for large accumulators.
//
//     Our simple (m_local, n_quarter*THREAD_N) decomposition does not survive
//     this change.  The thread→output mapping becomes hardware-dictated and
//     non-contiguous; THREAD_N is replaced by the MMA instruction's tile shape.
//
// (2) Replace blocking smem loads with asynchronous copies and pipeline stages.
//     Ampere:    cp.async (128-bit coalesced) + cp.async.commit_group /
//                cp.async.wait_group — allows compute and memory to overlap.
//     Hopper:    TMA (Tensor Memory Accelerator) — descriptor-based bulk copy;
//                a single thread issues the transfer; pairs with warp
//                specialization (dedicated producer warps do TMA, consumer
//                warps do wgmma) and cuda::barrier arrive/wait.
//     Blackwell: TMA for A/B tiles; tmem_load/store for the C accumulator.
//
//     Typically 2–4 pipeline stages in shared memory (ping-pong): while tile k
//     is being computed, tile k+1 is already being fetched.  This hides nearly
//     all DRAM latency and is the primary source of throughput in production.
//
// (3) Replace row-major w_tile with a swizzled shared memory layout.
//     All architectures: cute::Swizzle<B,M,S> composed into the tile layout
//     type, then ldmatrix (Ampere/Hopper) to load the MMA B-fragment.  The
//     swizzle parameters are chosen so that the 32 threads of a warp each land
//     on a different bank when executing ldmatrix — the right values depend on
//     the datatype and MMA shape, and differ from a fix for our scalar-load
//     access pattern.
//
// ZO-specific note:
//     The Philox perturbation and dual accumulators (v_cache_p/n) have no
//     CUTLASS analog; they are custom logic layered over the GEMM.  On all
//     three architectures, z generation remains per-thread (stateless,
//     counter-based), and the dual accumulator remains in registers.  Only
//     the surrounding GEMM infrastructure changes.
// ============================================================================
#define ZO_TILE_M     4
#define ZO_TILE_K     64
#define ZO_TILE_N     256
#define PHILOX_WIDTH  4     // output width of philox_normal_4; fixed by the API

#define THREAD_N      4     // outputs per thread per N-tile; must be multiple of
                            // PHILOX_WIDTH.  PHILOX_WIDTH=4 is also the
                            // bank-conflict-optimal choice: stride between adjacent
                            // threads in w_tile grows with THREAD_N, giving
                            // (THREAD_N/2)-way half conflicts; THREAD_N=4 minimizes.
#define I_DIM         4096  // hidden (intermediate) dimension
#define LAYERNORM_EPS 1e-7f

#define ZO_TPR        (ZO_TILE_N / THREAD_N)        // threads per row = 64
#define FWD_THREADS   (ZO_TILE_M * ZO_TPR)          // = 256 (derived, not set directly)


static_assert(THREAD_N % PHILOX_WIDTH == 0,
              "THREAD_N must be a multiple of PHILOX_WIDTH (philox_normal_4 output width)");
static_assert((ZO_TPR & (ZO_TPR - 1)) == 0,
              "ZO_TPR (= ZO_TILE_N / THREAD_N) must be a power of 2 (smem tree reduce)");

// ---------------------------------------------------------------------------
// Constraints
//
// ZO_TPR % 2 == 0      (Tree reduce)
//
// I % ZO_TILE_N == 0   (No partial tiles)
//
// ZO_TILE_K % 4 == 0   (Philox ILP)
//
// THREAD_N % PHILOX_WIDTH == 0   (Philox counter alignment)
//
// I_DIM == I at runtime
//   v_cache_p/n[(I_DIM/ZO_TILE_N)*THREAD_N] is compile-time sized.
// ---------------------------------------------------------------------------

__global__ void zo_fused_forward_kernel(
    half       * __restrict__ out_pos,   // [B, M, I] output for +eps
    half       * __restrict__ out_neg,   // [B, M, I] output for -eps
    const half * __restrict__ inp_pos,   // [B, M, d]
    const half * __restrict__ inp_neg,   // [B, M, d]
    const half * __restrict__ w,         // [d, I]
    const half * __restrict__ w_norm,    // [I]
    const half * __restrict__ b_norm,    // [I]
    int M, int d, int I,
    unsigned long long seed, float eps)
{
    // One shared memory allocation, manually partitioned:
    extern __shared__ char smem_raw[];
    float  *inp_tile_p = (float *)smem_raw;
    float  *inp_tile_n = inp_tile_p + ZO_TILE_M * ZO_TILE_K;
    half   *w_tile     = (half  *)(inp_tile_n + ZO_TILE_M * ZO_TILE_K);
    float  *smem_red   = (float *)((char *)w_tile + ZO_TILE_K * ZO_TILE_N * sizeof(half));
    // smem_red[4 * FWD_THREADS]: per-thread stats (sum_p, sumsq_p, sum_n, sumsq_n).

    const int tile_row = blockIdx.x;
    const int batch    = blockIdx.y;
    const int t        = threadIdx.x;
    const int IQ       = I / PHILOX_WIDTH;  // groups of PHILOX_WIDTH per weight row

    // Thread decomposition: t = m_local * ZO_TPR + n_quarter
    const int m_local   = t / ZO_TPR;
    const int n_quarter = t % ZO_TPR;
    const int row       = tile_row * ZO_TILE_M + m_local;

    const half *inp_pos_base = inp_pos + (size_t)batch * M * d;
    const half *inp_neg_base = inp_neg + (size_t)batch * M * d;

    // ----------------------------------------------------------------
    // Step 1: tiled GEMM — K-outer, N-inner.
    //
    // v_cache_p/n[64] accumulate the full dot products for both
    // perturbations across all K-tiles before SiLU is applied.
    //
    // K-outer means inp_tile is loaded once per K-tile and reused for all
    // N-tiles in the inner loop, rather than being reloaded once per
    // (K-tile, N-tile) pair.
    // ----------------------------------------------------------------
    float v_cache_p[(I_DIM / ZO_TILE_N) * THREAD_N] = {};
    float v_cache_n[(I_DIM / ZO_TILE_N) * THREAD_N] = {};

    for (int k_start = 0; k_start < d; k_start += ZO_TILE_K) {
        // Load both A tiles — once per K-tile, shared across all N-tiles.
        // In ZO, inp_pos and inp_neg are the same array; the L1 cache
        // handles the duplicate load transparently.
        for (int idx = t; idx < ZO_TILE_M * ZO_TILE_K; idx += FWD_THREADS) {
            int mi = idx / ZO_TILE_K, ki = idx % ZO_TILE_K;
            int r  = tile_row * ZO_TILE_M + mi;
            int k  = k_start + ki;
            bool valid = (r < M && k < d);
            inp_tile_p[idx] = valid ? __half2float(__ldg(&inp_pos_base[r * d + k])) : 0.f;
            inp_tile_n[idx] = valid ? __half2float(__ldg(&inp_neg_base[r * d + k])) : 0.f;
        }
        __syncthreads();

        for (int n_start = 0, ci = 0; n_start < I; n_start += ZO_TILE_N, ci += THREAD_N) {
            // Load B tile: [ZO_TILE_K × ZO_TILE_N] halfs, coalesced.
            for (int idx = t; idx < ZO_TILE_K * ZO_TILE_N; idx += FWD_THREADS) {
                int ki = idx / ZO_TILE_N, ni = idx % ZO_TILE_N;
                int k  = k_start + ki;
                w_tile[idx] = (k < d) ? __ldg(&w[(size_t)k * I + n_start + ni])
                                      : __float2half(0.f);
            }
            __syncthreads();

            // Dot product loop
            #pragma unroll 4
            for (int ki = 0; ki < ZO_TILE_K; ++ki) {
                int k = k_start + ki;
                float iv_p = inp_tile_p[m_local * ZO_TILE_K + ki];
                float iv_n = inp_tile_n[m_local * ZO_TILE_K + ki];
                const half *wt = w_tile + ki * ZO_TILE_N + n_quarter * THREAD_N;
                // Base counter for this thread's first PHILOX_WIDTH-group in this ki step.
                unsigned long long base_ctr = (unsigned long long)k * IQ
                                            + n_start / PHILOX_WIDTH
                                            + n_quarter * (THREAD_N / PHILOX_WIDTH);
                #pragma unroll
                for (int g = 0; g < THREAD_N / PHILOX_WIDTH; ++g) {
                    float4 z4 = philox_normal_4(seed, base_ctr + g);
                    float w0 = __half2float(wt[g * PHILOX_WIDTH + 0]), p0 = eps * z4.x;
                    float w1 = __half2float(wt[g * PHILOX_WIDTH + 1]), p1 = eps * z4.y;
                    float w2 = __half2float(wt[g * PHILOX_WIDTH + 2]), p2 = eps * z4.z;
                    float w3 = __half2float(wt[g * PHILOX_WIDTH + 3]), p3 = eps * z4.w;
                    v_cache_p[ci + g * PHILOX_WIDTH + 0] += iv_p * (w0 + p0);
                    v_cache_n[ci + g * PHILOX_WIDTH + 0] += iv_n * (w0 - p0);
                    v_cache_p[ci + g * PHILOX_WIDTH + 1] += iv_p * (w1 + p1);
                    v_cache_n[ci + g * PHILOX_WIDTH + 1] += iv_n * (w1 - p1);
                    v_cache_p[ci + g * PHILOX_WIDTH + 2] += iv_p * (w2 + p2);
                    v_cache_n[ci + g * PHILOX_WIDTH + 2] += iv_n * (w2 - p2);
                    v_cache_p[ci + g * PHILOX_WIDTH + 3] += iv_p * (w3 + p3);
                    v_cache_n[ci + g * PHILOX_WIDTH + 3] += iv_n * (w3 - p3);
                }
            }
            __syncthreads();
        }
    }

    // ----------------------------------------------------------------
    // Step 2: SiLU in-place on v_cache_p/n; collect per-row stats
    // (Σv, Σv²) for both perturbations; warp-shuffle reduction.
    // ----------------------------------------------------------------
    const int t_r = n_quarter;

    float2 local_p = {0.f, 0.f};
    float2 local_n = {0.f, 0.f};
    for (int ci = 0; ci < (I_DIM / ZO_TILE_N) * THREAD_N; ++ci) {
        float g = v_cache_p[ci];
        float v = g / (1.f + expf(-g));
        v_cache_p[ci] = v;
        local_p.x += v;
        local_p.y += v * v;
        g = v_cache_n[ci];
        v = g / (1.f + expf(-g));
        v_cache_n[ci] = v;
        local_n.x += v;
        local_n.y += v * v;
    }

    // Smem tree reduce across ZO_TPR threads per row, all 4 stats in parallel.
    smem_red[t]                    = local_p.x;
    smem_red[t +     FWD_THREADS]  = local_p.y;
    smem_red[t + 2 * FWD_THREADS]  = local_n.x;
    smem_red[t + 3 * FWD_THREADS]  = local_n.y;
    __syncthreads();
    for (int stride = ZO_TPR >> 1; stride > 0; stride >>= 1) {
        if (t_r < stride) {
            smem_red[t]                   += smem_red[t + stride];
            smem_red[t +     FWD_THREADS] += smem_red[t + stride +     FWD_THREADS];
            smem_red[t + 2 * FWD_THREADS] += smem_red[t + stride + 2 * FWD_THREADS];
            smem_red[t + 3 * FWD_THREADS] += smem_red[t + stride + 3 * FWD_THREADS];
        }
        __syncthreads();
    }
    const int row_base = m_local * ZO_TPR;
    float2 agg_p = { smem_red[row_base],                   smem_red[row_base +     FWD_THREADS] };
    float2 agg_n = { smem_red[row_base + 2 * FWD_THREADS], smem_red[row_base + 3 * FWD_THREADS] };
    float mean_p    = agg_p.x / I;
    float inv_std_p = rsqrtf(agg_p.y / I - mean_p * mean_p + LAYERNORM_EPS);
    float mean_n    = agg_n.x / I;
    float inv_std_n = rsqrtf(agg_n.y / I - mean_n * mean_n + LAYERNORM_EPS);

    // ----------------------------------------------------------------
    // Step 3: normalize both outputs from v_cache and write.
    // ----------------------------------------------------------------
    if (row < M) {
        half *out_row_p = out_pos + (size_t)(batch * M + row) * I;
        half *out_row_n = out_neg + (size_t)(batch * M + row) * I;
        for (int n_start = 0, ci = 0; n_start < I; n_start += ZO_TILE_N, ci += THREAD_N) {
            int n0 = n_start + n_quarter * THREAD_N;
            for (int j = 0; j < THREAD_N; ++j) {
                float gamma_j = __half2float(__ldg(&w_norm[n0 + j]));
                float beta_j  = __half2float(__ldg(&b_norm[n0 + j]));
                out_row_p[n0 + j] = __float2half(
                    (v_cache_p[ci + j] - mean_p) * inv_std_p * gamma_j + beta_j);
                out_row_n[n0 + j] = __float2half(
                    (v_cache_n[ci + j] - mean_n) * inv_std_n * gamma_j + beta_j);
            }
        }
    }
}

// ============================================================================
// zo_update_kernel
//
// Applies:  W[k, n] -= lr * grad_est * z[k, n]
//
// z is regenerated from the same (seed, counter) as zo_fused_forward:
//   counter = k*(I/PHILOX_WIDTH) + n/PHILOX_WIDTH = group
//   (linear index of the PHILOX_WIDTH-weight group)
//
// Since weights are stored row-major and I % PHILOX_WIDTH == 0, aligned groups
// of PHILOX_WIDTH consecutive weights always fall within the same weight row k
// — so the counter is equivalent to the forward kernel's base_ctr+g.
// All PHILOX_WIDTH Philox outputs are consumed per thread.
//
// Requires d*I % PHILOX_WIDTH == 0 (satisfied when I % PHILOX_WIDTH == 0).
// ============================================================================
#define UPD_THREADS 256

__global__ void zo_update_kernel(
    half * __restrict__ w,  // [d, I], updated in-place
    int d, int I,
    unsigned long long seed, float lr, float grad_est)
{
    // group = k*(I/PHILOX_WIDTH) + n/PHILOX_WIDTH — same counter as zo_fused_forward
    int group = blockIdx.x * UPD_THREADS + threadIdx.x;
    if ((size_t)group * PHILOX_WIDTH >= (size_t)d * I) return;

    float4 z4    = philox_normal_4(seed, (unsigned long long)group);
    float  scale = lr * grad_est;

    half *wp = w + (size_t)group * PHILOX_WIDTH;
    wp[0] = __float2half(__half2float(wp[0]) - scale * z4.x);
    wp[1] = __float2half(__half2float(wp[1]) - scale * z4.y);
    wp[2] = __float2half(__half2float(wp[2]) - scale * z4.z);
    wp[3] = __float2half(__half2float(wp[3]) - scale * z4.w);
}

// ============================================================================
// CPU reference — dual-perturbed forward pass, matching zo_fused_forward_kernel.
//
// Computes, for each (batch, row):
//   out_pos = layernorm(silu(inp_pos @ (W + eps*z)))
//   out_neg = layernorm(silu(inp_neg @ (W - eps*z)))
//
// z is regenerated from the same (seed, counter) formula as the GPU kernels:
//   counter = k * (I/PHILOX_WIDTH) + n/PHILOX_WIDTH
// so philox_normal_4(seed, counter)[j] gives z[k, n+j].
// ============================================================================
void cpu_reference(
    float      *out_pos,    // [B, M, I]
    float      *out_neg,    // [B, M, I]
    const half *inp_pos,    // [B, M, d]
    const half *inp_neg,    // [B, M, d]
    const half *w,          // [d, I]
    const half *w_norm,     // [I]
    const half *b_norm,     // [I]
    int B, int M, int d, int I,
    unsigned long long seed, float eps)
{
    const int IQ = I / PHILOX_WIDTH;
    float *tmp_p = (float *)malloc(I * sizeof(float));
    float *tmp_n = (float *)malloc(I * sizeof(float));
    for (int b = 0; b < B; ++b) {
        const half *inp_pos_b = inp_pos + (size_t)b * M * d;
        const half *inp_neg_b = inp_neg + (size_t)b * M * d;
        float      *out_pos_b = out_pos + (size_t)b * M * I;
        float      *out_neg_b = out_neg + (size_t)b * M * I;
        for (int m = 0; m < M; ++m) {
            for (int i = 0; i < I; ++i) tmp_p[i] = tmp_n[i] = 0.f;
            for (int k = 0; k < d; ++k) {
                float iv_p = __half2float(inp_pos_b[m*d + k]);
                float iv_n = __half2float(inp_neg_b[m*d + k]);
                for (int n = 0; n < I; n += PHILOX_WIDTH) {
                    unsigned long long ctr = (unsigned long long)k * IQ
                                          + (unsigned long long)(n / PHILOX_WIDTH);
                    float4 z4 = philox_normal_4(seed, ctr);
                    float zv[PHILOX_WIDTH] = {z4.x, z4.y, z4.z, z4.w};
                    for (int j = 0; j < PHILOX_WIDTH; ++j) {
                        float wval = __half2float(w[(size_t)k * I + n + j]);
                        float p    = eps * zv[j];
                        tmp_p[n + j] += iv_p * (wval + p);
                        tmp_n[n + j] += iv_n * (wval - p);
                    }
                }
            }
            float sum_p = 0.f, sumsq_p = 0.f;
            float sum_n = 0.f, sumsq_n = 0.f;
            for (int i = 0; i < I; ++i) {
                float g = tmp_p[i]; tmp_p[i] = g / (1.f + expf(-g));
                      g = tmp_n[i]; tmp_n[i] = g / (1.f + expf(-g));
                sum_p += tmp_p[i]; sumsq_p += tmp_p[i] * tmp_p[i];
                sum_n += tmp_n[i]; sumsq_n += tmp_n[i] * tmp_n[i];
            }
            float mean_p    = sum_p / I;
            float inv_std_p = 1.f / sqrtf(sumsq_p / I - mean_p * mean_p + LAYERNORM_EPS);
            float mean_n    = sum_n / I;
            float inv_std_n = 1.f / sqrtf(sumsq_n / I - mean_n * mean_n + LAYERNORM_EPS);
            float *out_row_p = out_pos_b + (size_t)m * I;
            float *out_row_n = out_neg_b + (size_t)m * I;
            for (int i = 0; i < I; ++i) {
                float gamma = __half2float(w_norm[i]);
                float beta  = __half2float(b_norm[i]);
                out_row_p[i] = (tmp_p[i] - mean_p) * inv_std_p * gamma + beta;
                out_row_n[i] = (tmp_n[i] - mean_n) * inv_std_n * gamma + beta;
            }
        }
    }
    free(tmp_p);
    free(tmp_n);
}

static float randf_11() { return (float)rand() / (float)RAND_MAX * 2.f - 1.f; }

int main() {
    const int B = 4, M = 128, d = 256, I = I_DIM;

    if (I % ZO_TILE_N != 0) {
        fprintf(stderr, "I (%d) must be divisible by ZO_TILE_N = %d\n", I, ZO_TILE_N);
        return 1;
    }

    const int n_inp  = B * M * d;
    const int n_w    = d * I;
    const int n_norm = I;
    const int n_out  = B * M * I;

    printf("\n");
    printf("=== zo_fused_forward + zo_update ===\n");
    printf("  B=%d  M=%d  d=%d  I=%d\n", B, M, d, I);
    printf("  ZO_TILE_M=%d  ZO_TILE_K=%d  ZO_TILE_N=%d  THREAD_N=%d  PHILOX_WIDTH=%d"
           "  FWD_THREADS=%d  UPD_THREADS=%d\n",
           ZO_TILE_M, ZO_TILE_K, ZO_TILE_N, THREAD_N, PHILOX_WIDTH, FWD_THREADS, UPD_THREADS);

    // Host arrays
    half  *h_inp         = (half *)malloc(n_inp  * sizeof(half));
    half  *h_w           = (half *)malloc(n_w    * sizeof(half));
    half  *h_w_norm      = (half *)malloc(n_norm * sizeof(half));
    half  *h_b_norm      = (half *)malloc(n_norm * sizeof(half));
    half  *h_target      = (half *)malloc(n_out  * sizeof(half));  // for host-side MSE in Test 2
    half  *h_out_gpu     = (half *)malloc(n_out  * sizeof(half));
    float *h_out_ref_pos = (float*)malloc(n_out  * sizeof(float));
    float *h_out_ref_neg = (float*)malloc(n_out  * sizeof(float));
    float *h_out_f32     = (float*)malloc(n_out  * sizeof(float));

    srand(42);
    for (int i = 0; i < n_inp;  ++i) h_inp[i]    = __float2half(randf_11());
    for (int i = 0; i < n_w;    ++i) h_w[i]       = __float2half(randf_11());
    for (int i = 0; i < n_norm; ++i) {
        h_w_norm[i] = __float2half(randf_11());
        h_b_norm[i] = __float2half((float)rand()/(float)RAND_MAX * 0.2f - 0.1f);
    }
    for (int i = 0; i < n_out; ++i) h_target[i] = __float2half(randf_11());

    // Device arrays
    half *d_inp, *d_w, *d_w_norm, *d_b_norm;
    half *d_out_pos, *d_out_neg;
    checkCuda(cudaMalloc(&d_inp,     n_inp  * sizeof(half)));
    checkCuda(cudaMalloc(&d_w,       n_w    * sizeof(half)));
    checkCuda(cudaMalloc(&d_w_norm,  n_norm * sizeof(half)));
    checkCuda(cudaMalloc(&d_b_norm,  n_norm * sizeof(half)));
    checkCuda(cudaMalloc(&d_out_pos, n_out  * sizeof(half)));
    checkCuda(cudaMalloc(&d_out_neg, n_out  * sizeof(half)));

    checkCuda(cudaMemcpy(d_inp,    h_inp,    n_inp  * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_w,      h_w,      n_w    * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_w_norm, h_w_norm, n_norm * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_b_norm, h_b_norm, n_norm * sizeof(half), cudaMemcpyHostToDevice));

    const int shm_fwd = 2 * ZO_TILE_M * ZO_TILE_K * sizeof(float)   // inp_tile_p + inp_tile_n
                      + ZO_TILE_K * ZO_TILE_N * sizeof(half)          // w_tile (half)
                      + 4 * FWD_THREADS * sizeof(float);              // smem_red (stats)
    dim3 grid((M + ZO_TILE_M - 1) / ZO_TILE_M, B);

    // -----------------------------------------------------------------------
    // Test 1: correctness — eps=0, perturbation vanishes, compare to CPU ref
    // -----------------------------------------------------------------------
    printf("\n[1] Correctness check (eps=0)\n");

    zo_fused_forward_kernel<<<grid, FWD_THREADS, shm_fwd>>>(
        d_out_pos, d_out_neg,
        d_inp, d_inp, d_w, d_w_norm, d_b_norm,
        M, d, I, /*seed=*/0ULL, /*eps=*/0.f);
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaMemcpy(h_out_gpu, d_out_pos, n_out * sizeof(half), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n_out; ++i) h_out_f32[i] = __half2float(h_out_gpu[i]);
    cpu_reference(h_out_ref_pos, h_out_ref_neg,
                  h_inp, h_inp, h_w, h_w_norm, h_b_norm,
                  B, M, d, I, /*seed=*/0ULL, /*eps=*/0.f);

    float abs_err = 0.f, max_ref = 0.f;
    for (int i = 0; i < n_out; ++i) {
        abs_err = fmaxf(abs_err, fabsf(h_out_ref_pos[i] - h_out_f32[i]));
        max_ref = fmaxf(max_ref, fabsf(h_out_ref_pos[i]));
    }
    float rel_err = abs_err / (max_ref + 1e-8f);
    printf("  Max abs error: %.4e  Max rel error: %.4e  %s\n",
           abs_err, rel_err, rel_err < 5e-3f ? "PASS" : "FAIL");

    printf("  First 4 outputs — %4s  %16s  %16s\n", "idx", "CPU (fp32)", "GPU (fp16→fp32)");
    for (int i = 0; i < 4; ++i)
        printf("    %-4d  %+15.8f  %+15.8f\n",
               i, h_out_ref_pos[i], h_out_f32[i]);

    // -----------------------------------------------------------------------
    // Test 2: ZO gradient estimate and weight update
    //
    // MeZO estimate:  grad_est = (f+ - f-) / (2*eps)
    // Weight update:  W  -= lr * grad_est * z   (z regenerated from seed)
    // -----------------------------------------------------------------------
    printf("\n[2] ZO gradient estimate and update\n");

    const unsigned long long seed = 0xDEADBEEF42ULL;
    const float eps    = 1e-2f;
    const float lr     = 1e-3f;
    const float n_elems = (float)(B * M * I);

    // Host-side MSE helper: reads h_out_gpu (half) vs h_target (half).
    auto host_mse = [&](const half *gpu_out) {
        float s = 0.f;
        for (int i = 0; i < n_out; ++i) {
            float d = __half2float(gpu_out[i]) - __half2float(h_target[i]);
            s += d * d;
        }
        return s;
    };

    // Unperturbed output before update → loss_before
    zo_fused_forward_kernel<<<grid, FWD_THREADS, shm_fwd>>>(
        d_out_pos, d_out_neg,
        d_inp, d_inp, d_w, d_w_norm, d_b_norm,
        M, d, I, seed, 0.f);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaMemcpy(h_out_gpu, d_out_pos, n_out * sizeof(half), cudaMemcpyDeviceToHost));
    float h_loss_before = host_mse(h_out_gpu);

    // f+ and f- in one dual-perturbation kernel call — loads inp and w once
    half *h_out_neg = (half *)malloc(n_out * sizeof(half));
    zo_fused_forward_kernel<<<grid, FWD_THREADS, shm_fwd>>>(
        d_out_pos, d_out_neg,
        d_inp, d_inp, d_w, d_w_norm, d_b_norm,
        M, d, I, seed, eps);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaMemcpy(h_out_gpu, d_out_pos, n_out * sizeof(half), cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(h_out_neg, d_out_neg, n_out * sizeof(half), cudaMemcpyDeviceToHost));
    float h_fpos = host_mse(h_out_gpu);
    float h_fneg = host_mse(h_out_neg);
    free(h_out_neg);

    float grad_est = (h_fpos - h_fneg) / (2.f * eps);
    printf("  loss (unperturbed, before):        %.6f\n", h_loss_before / n_elems);
    printf("  [Φ(θ+εz,b) - Φ(θ-εz,b)] / 2ε:  %.6f\n", grad_est);

    // Apply:  W -= lr * grad_est * z  (z regenerated from seed)
    int upd_groups = (d * I) / PHILOX_WIDTH;
    int upd_blocks = (upd_groups + UPD_THREADS - 1) / UPD_THREADS;
    zo_update_kernel<<<upd_blocks, UPD_THREADS>>>(
        d_w, d, I, seed, lr, grad_est);
    checkCuda(cudaDeviceSynchronize());

    // Unperturbed output after update → loss_after
    zo_fused_forward_kernel<<<grid, FWD_THREADS, shm_fwd>>>(
        d_out_pos, d_out_neg,
        d_inp, d_inp, d_w, d_w_norm, d_b_norm,
        M, d, I, seed, 0.f);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaMemcpy(h_out_gpu, d_out_pos, n_out * sizeof(half), cudaMemcpyDeviceToHost));
    float h_loss_after = host_mse(h_out_gpu);

    printf("  loss (unperturbed, after):         %.6f\n", h_loss_after / n_elems);
    printf("  delta: %+.6f  (%s)\n",
           (h_loss_after - h_loss_before) / n_elems,
           h_loss_after < h_loss_before ? "decreased" : "increased");

    // -----------------------------------------------------------------------
    // Timing
    // -----------------------------------------------------------------------
    printf("\n[3] Timing\n");
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    const int WARMUP = 5, RUNS = 50;
    for (int i = 0; i < WARMUP; ++i)
        zo_fused_forward_kernel<<<grid, FWD_THREADS, shm_fwd>>>(
            d_out_pos, d_out_neg,
            d_inp, d_inp, d_w, d_w_norm, d_b_norm,
            M, d, I, seed, eps);

    cudaEventRecord(t0);
    for (int i = 0; i < RUNS; ++i)
        zo_fused_forward_kernel<<<grid, FWD_THREADS, shm_fwd>>>(
            d_out_pos, d_out_neg,
            d_inp, d_inp, d_w, d_w_norm, d_b_norm,
            M, d, I, seed, eps);
    cudaEventRecord(t1);
    checkCuda(cudaDeviceSynchronize());
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("  zo_fused_forward: %.4f ms/call (%d calls)\n", ms / RUNS, RUNS);

    cudaEventRecord(t0);
    for (int i = 0; i < RUNS; ++i)
        zo_update_kernel<<<upd_blocks, UPD_THREADS>>>(
            d_w, d, I, seed, lr, grad_est);
    cudaEventRecord(t1);
    checkCuda(cudaDeviceSynchronize());
    cudaEventElapsedTime(&ms, t0, t1);
    printf("  zo_update:        %.4f ms/call (%d calls)\n", ms / RUNS, RUNS);

    // Cleanup
    free(h_inp); free(h_w); free(h_w_norm); free(h_b_norm);
    free(h_target); free(h_out_gpu);
    free(h_out_ref_pos); free(h_out_ref_neg); free(h_out_f32);
    cudaFree(d_inp); cudaFree(d_w); cudaFree(d_w_norm); cudaFree(d_b_norm);
    cudaFree(d_out_pos); cudaFree(d_out_neg);
    cudaEventDestroy(t0); cudaEventDestroy(t1);

    return rel_err < 5e-3f ? 0 : 1;
}
