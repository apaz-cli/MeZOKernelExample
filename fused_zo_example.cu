/*
 * fused_zo_example.cu
 *
 *
 * MeZO-style zeroth-order (ZO) optimization on:
 *   out = layernorm(silu(inp @ W_gate))
 * where
 *   inp    [B, M, d] — batch of M-row inputs
 *   W_gate [d, I]    — shared weight, updated in-place
 *   out    [B, M, I] — batch outputs
 *
 * All batch samples share the same perturbation z ~ N(0,1) for W_gate.
 * z is never materialized; each element is regenerated on-the-fly from a
 * seed via Philox counter-based PRNG (O(1) skip-ahead, no sequential state).
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
 *   (shared perturbation).  Register arrays v_cache_p/n[(I_DIM/ZO_TILE_N)*THREAD_N]
 *   accumulate the full GEMM output for both perturbations; SiLU is applied
 *   in-place after the loops.  Requires I % ZO_TILE_N == 0.
 *
 * Philox ILP:
 *   counter = k*(I/PHILOX_WIDTH) + n_start/PHILOX_WIDTH + n_quarter*(THREAD_N/PHILOX_WIDTH) + g
 *   uniquely addresses the g-th group-of-4 within thread n_quarter's THREAD_N columns.
 *   #pragma unroll 4 on the ki loop issues 4 independent counter chains (distinct k
 *   values → no data dependence); the GPU pipelines their 7-round multiply chains in
 *   parallel, hiding Philox latency behind itself.
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
// Philox-4x32 counter-based PRNG (7 rounds)
//
// Why not a standard sequential RNG?  ZO optimization needs the same noise
// vector z in both the forward kernel (to perturb each weight) and the update
// kernel (to scale the gradient step).  A sequential RNG would require either
// storing z (~d×I values) or replaying a stream from the beginning each time.
//
// Philox is counter-based: given a (seed, counter) pair it produces
// random-looking output via ~10 integer multiplications, with no sequential
// state to maintain.  Any thread can independently evaluate any position in
// the stream.  Both kernels derive the counter from (k, n) alone, so z is
// regenerated on-the-fly without storage or inter-kernel coordination.
//
// philox_uniform_4 — (seed, counter) → 4 uniform floats in [0, 1)
// philox_normal_4  — Box-Muller pairs: (u0,u1)→(g0,g1), (u2,u3)→(g2,g3)
// ============================================================================
__device__ __forceinline__
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
        unsigned int h0 = __umulhi(PSA, ctr.x);
        unsigned int h1 = __umulhi(PSB, ctr.z);
        ctr = make_uint4(h1 ^ ctr.y ^ key.x,  PSA * ctr.x,
                         h0 ^ ctr.w ^ key.y,  PSB * ctr.z);
        key.x += P10A; key.y += P10B;
    }
    const float s = 2.3283064e-10f;  // 1/2^32
    return make_float4(ctr.x * s, ctr.y * s, ctr.z * s, ctr.w * s);
}

__device__ __forceinline__
float4 philox_normal_4(unsigned long long seed, unsigned long long counter)
{
    float4 u = philox_uniform_4(seed, counter);
    u.x = fmaxf(u.x, 1e-7f);  // avoid log(0); negligible distribution error
    u.z = fmaxf(u.z, 1e-7f);
    float r0 = sqrtf(-2.f * logf(u.x)), t0 = 6.28318530f * u.y;
    float r1 = sqrtf(-2.f * logf(u.z)), t1 = 6.28318530f * u.w;
    return make_float4(r0 * __cosf(t0), r0 * __sinf(t0),
                       r1 * __cosf(t1), r1 * __sinf(t1));
}

// ============================================================================
// zo_fused_forward_kernel
//
// What is zeroth-order optimization?
//   ZO estimates gradients without backpropagation.  It evaluates the loss at
//   two slight weight perturbations — W+eps*z and W-eps*z — and estimates the
//   gradient as (f+ - f-) / (2*eps).  This kernel computes BOTH perturbed
//   forward passes in a single launch, loading inp and w_gate only once.
//   Crucially, z is never stored; Philox regenerates it on-the-fly for each
//   weight using the same counter formula as the update kernel.
//
// Computes both:
//   out_pos = layernorm(silu(inp @ (W_gate + eps*z)))
//   out_neg = layernorm(silu(inp @ (W_gate - eps*z)))
// and accumulates MSE vs target into *loss_pos and *loss_neg.
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
//   smem2_p    [ZO_TILE_M*2 float2]           = 64 B   warp-leader stats for +
//   smem2_n    [ZO_TILE_M*2 float2]           = 64 B   warp-leader stats for -
//   warp_mse_p [ZO_TILE_M*2 float]            = 32 B   warp-leader MSE for +
//   warp_mse_n [ZO_TILE_M*2 float]            = 32 B   warp-leader MSE for -
//   Total: ~34.2 KB
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

// ---------------------------------------------------------------------------
// Hyperparameter constraints
//
// ZO_TPR == ZO_TILE_N / THREAD_N   (derived, not set directly)
//   Thread decomposition t = m_local * ZO_TPR + n_quarter.  ZO_TILE_M rows,
//   ZO_TPR threads/row.  Each thread owns THREAD_N consecutive output columns
//   per N-tile, covering the full ZO_TILE_N column tile.
//
// FWD_THREADS == ZO_TILE_M * ZO_TPR   (derived)
//   Total threads per block.  ZO_TILE_M rows × ZO_TPR threads/row.
//   Changing TILE_N or THREAD_N automatically adjusts FWD_THREADS.
//
// ZO_TILE_K independent of ZO_TPR   (strided cooperative A-tile load)
//   The A-tile [ZO_TILE_M × ZO_TILE_K] is loaded by the strided loop
//     for (idx = t; idx < ZO_TILE_M*ZO_TILE_K; idx += FWD_THREADS)
//   Any combination of tile sizes works; the loop handles partial coverage.
//
// ZO_TPR == 64   (warp-shuffle epilogue hardcodes 2 warps/row)
//   Both stats and MSE reductions use smem2[m_local*2 + t_r/32] and
//   warp_mse[m_local*2 + t_r/32].  This assumes exactly 2 warps per row.
//   Changing ZO_TILE_N or THREAD_N such that ZO_TPR != 64 requires rewriting
//   both epilogue reductions.
//
// THREAD_N % PHILOX_WIDTH == 0   (enforced by static_assert above)
//   Each Philox call produces PHILOX_WIDTH=4 samples.  The inner g-loop runs
//   THREAD_N/PHILOX_WIDTH times, consuming exactly THREAD_N samples total.
//   Non-multiples would require a partial final call, wasting outputs and
//   breaking counter alignment with zo_update_kernel.
//   THREAD_N=PHILOX_WIDTH=4 is also bank-conflict-optimal (see w_tile note).
//
// I % ZO_TILE_N == 0   (checked in main)
//   Column-direction bounds guards are absent in the N-tile loop, so w_gate
//   and v_cache go out of bounds if I is not a multiple of ZO_TILE_N.  Also
//   ensures n_start / PHILOX_WIDTH is exact, keeping the Philox counter lossless.
//
// ZO_TILE_K % 4 == 0   (Philox ILP)
//   #pragma unroll 4 replicates the ki loop body 4 times, issuing 4
//   independent Philox chains (distinct counters k*IQ+...).  If ZO_TILE_K
//   is not divisible by 4, the compiler emits a remainder iteration whose
//   counter is not independent of the preceding group.
//
// I_DIM == I at runtime   (v_cache compile-time size)
//   float v_cache_p/n[(I_DIM/ZO_TILE_N)*THREAD_N] is compile-time sized.
//   I_DIM must match the I passed to the kernel.
//
// d % ZO_TILE_K: not required   (the check in main is overly conservative)
//   A-tile and B-tile loads are guarded by (k < d), so partial K-tiles are
//   correctly zero-padded.  The update kernel uses a flat linear group index
//   independent of ZO_TILE_K, so partial tiles cause no Philox aliasing.
//
// Philox counter uniqueness
//   counter = k*(I/PHILOX_WIDTH) + n_start/PHILOX_WIDTH
//             + n_quarter*(THREAD_N/PHILOX_WIDTH) + g
//   uniquely addresses each group of PHILOX_WIDTH weights.  The per-k offset
//   n_start/PHILOX_WIDTH + n_quarter*(THREAD_N/PHILOX_WIDTH) + g is always
//   < I/PHILOX_WIDTH, so no two weight rows share a counter.  Guaranteed once
//   I % ZO_TILE_N == 0.  zo_update_kernel uses the same counter formula with
//   group = k*(I/PHILOX_WIDTH) + n/PHILOX_WIDTH (linear group index).
//
// Hardware limits
//   FWD_THREADS <= 1024 (max threads/block).
//   shm_fwd ≈ 34 KB < 48 KB default; no cudaFuncSetAttribute needed.
// ---------------------------------------------------------------------------

__global__ void zo_fused_forward_kernel(
    half       * __restrict__ out_pos,   // [B*M, I] output for W + eps*z
    half       * __restrict__ out_neg,   // [B*M, I] output for W - eps*z
    float      * __restrict__ loss_pos,  // MSE accumulator for + perturbation
    float      * __restrict__ loss_neg,  // MSE accumulator for - perturbation
    const half * __restrict__ inp_pos,   // [B, M, d] — input for + perturbation (ZO: same as inp_neg)
    const half * __restrict__ inp_neg,   // [B, M, d] — input for - perturbation (ZO: same as inp_pos)
    const half * __restrict__ w_gate,    // [d, I]
    const half * __restrict__ target,    // [B*M, I]
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
    float2 *smem2_p    = (float2 *)((char *)w_tile + ZO_TILE_K * ZO_TILE_N * sizeof(half));
    float2 *smem2_n    = smem2_p   + ZO_TILE_M * 2;
    float  *warp_mse_p = (float  *)(smem2_n   + ZO_TILE_M * 2);
    float  *warp_mse_n = warp_mse_p + ZO_TILE_M * 2;

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
                w_tile[idx] = (k < d) ? __ldg(&w_gate[(size_t)k * I + n_start + ni])
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

    // Warp reduction for both perturbations simultaneously — one butterfly pass
    // each, no extra cost over reducing a single value.
    for (int s = 16; s > 0; s >>= 1) {
        local_p.x += __shfl_xor_sync(0xffffffff, local_p.x, s);
        local_p.y += __shfl_xor_sync(0xffffffff, local_p.y, s);
        local_n.x += __shfl_xor_sync(0xffffffff, local_n.x, s);
        local_n.y += __shfl_xor_sync(0xffffffff, local_n.y, s);
    }
    // Both warp leaders write in the same conditional; one __syncthreads() covers both.
    if (t_r % 32 == 0) {
        smem2_p[m_local * 2 + t_r / 32] = local_p;
        smem2_n[m_local * 2 + t_r / 32] = local_n;
    }
    __syncthreads();
    float2 agg_p = { smem2_p[m_local * 2].x + smem2_p[m_local * 2 + 1].x,
                     smem2_p[m_local * 2].y + smem2_p[m_local * 2 + 1].y };
    float2 agg_n = { smem2_n[m_local * 2].x + smem2_n[m_local * 2 + 1].x,
                     smem2_n[m_local * 2].y + smem2_n[m_local * 2 + 1].y };
    float mean_p    = agg_p.x / I;
    float inv_std_p = rsqrtf(agg_p.y / I - mean_p * mean_p + LAYERNORM_EPS);
    float mean_n    = agg_n.x / I;
    float inv_std_n = rsqrtf(agg_n.y / I - mean_n * mean_n + LAYERNORM_EPS);

    // ----------------------------------------------------------------
    // Step 3: normalize both outputs from v_cache; write both;
    // accumulate MSE for each vs the shared target.
    // ----------------------------------------------------------------
    float local_loss_p = 0.f, local_loss_n = 0.f;
    if (row < M) {
        half       *out_row_p = out_pos + (size_t)(batch * M + row) * I;
        half       *out_row_n = out_neg + (size_t)(batch * M + row) * I;
        const half *tgt_row   = target  + (size_t)(batch * M + row) * I;
        for (int n_start = 0, ci = 0; n_start < I; n_start += ZO_TILE_N, ci += THREAD_N) {
            int n0 = n_start + n_quarter * THREAD_N;
            for (int j = 0; j < THREAD_N; ++j) {
                float gamma_j = __half2float(__ldg(&w_norm[n0 + j]));
                float beta_j  = __half2float(__ldg(&b_norm[n0 + j]));
                float tgt_j   = __half2float(__ldg(&tgt_row[n0 + j]));
                float vp = (v_cache_p[ci + j] - mean_p) * inv_std_p * gamma_j + beta_j;
                float vn = (v_cache_n[ci + j] - mean_n) * inv_std_n * gamma_j + beta_j;
                out_row_p[n0 + j] = __float2half(vp);
                out_row_n[n0 + j] = __float2half(vn);
                float ep = vp - tgt_j, en = vn - tgt_j;
                local_loss_p += ep * ep;
                local_loss_n += en * en;
            }
        }
    }

    // ----------------------------------------------------------------
    // Step 4: reduce MSE partial sums for both perturbations simultaneously,
    // then atomic-add each to its global accumulator.
    // ----------------------------------------------------------------
    for (int s = 16; s > 0; s >>= 1) {
        local_loss_p += __shfl_xor_sync(0xffffffff, local_loss_p, s);
        local_loss_n += __shfl_xor_sync(0xffffffff, local_loss_n, s);
    }
    if (t_r % 32 == 0) {
        warp_mse_p[m_local * 2 + t_r / 32] = local_loss_p;
        warp_mse_n[m_local * 2 + t_r / 32] = local_loss_n;
    }
    __syncthreads();
    if (t_r == 0) {
        atomicAdd(loss_pos, warp_mse_p[m_local * 2] + warp_mse_p[m_local * 2 + 1]);
        atomicAdd(loss_neg, warp_mse_n[m_local * 2] + warp_mse_n[m_local * 2 + 1]);
    }
}

// ============================================================================
// zo_update_kernel
//
// Applies:  W_gate[k, n] -= lr * grad_est * z[k, n]
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
    half * __restrict__ w_gate,  // [d, I], updated in-place
    int d, int I,
    unsigned long long seed, float lr, float grad_est)
{
    // group = k*(I/PHILOX_WIDTH) + n/PHILOX_WIDTH — same counter as zo_fused_forward
    int group = blockIdx.x * UPD_THREADS + threadIdx.x;
    if ((size_t)group * PHILOX_WIDTH >= (size_t)d * I) return;

    float4 z4    = philox_normal_4(seed, (unsigned long long)group);
    float  scale = lr * grad_est;

    half *w = w_gate + (size_t)group * PHILOX_WIDTH;
    w[0] = __float2half(__half2float(w[0]) - scale * z4.x);
    w[1] = __float2half(__half2float(w[1]) - scale * z4.y);
    w[2] = __float2half(__half2float(w[2]) - scale * z4.z);
    w[3] = __float2half(__half2float(w[3]) - scale * z4.w);
}

// ============================================================================
// CPU reference — unperturbed forward pass for correctness verification
// ============================================================================
void cpu_reference(
    float      *out,
    const half *inp,
    const half *w_gate,
    const half *w_norm,
    const half *b_norm,
    int B, int M, int d, int I)
{
    float *tmp = (float *)malloc(I * sizeof(float));
    for (int b = 0; b < B; ++b) {
        const half *inp_b = inp + (size_t)b * M * d;
        float      *out_b = out + (size_t)b * M * I;
        for (int m = 0; m < M; ++m) {
            for (int i = 0; i < I; ++i) {
                float dot = 0.f;
                for (int k = 0; k < d; ++k)
                    dot += __half2float(inp_b[m*d + k]) *
                           __half2float(w_gate[k*I + i]);
                tmp[i] = dot / (1.f + expf(-dot));
            }
            float sum = 0.f, sumsq = 0.f;
            for (int i = 0; i < I; ++i) { sum += tmp[i]; sumsq += tmp[i]*tmp[i]; }
            float mean    = sum / I;
            float inv_std = 1.f / sqrtf(sumsq / I - mean*mean + LAYERNORM_EPS);
            for (int i = 0; i < I; ++i)
                out_b[m*I + i] = (tmp[i]-mean)*inv_std *
                                  __half2float(w_norm[i]) +
                                  __half2float(b_norm[i]);
        }
    }
    free(tmp);
}

// ============================================================================
// main
// ============================================================================
static float randf_11() { return (float)rand() / (float)RAND_MAX * 2.f - 1.f; }

int main() {
    const int B = 4, M = 128, d = 256, I = I_DIM;

    if (I % ZO_TILE_N != 0) {
        fprintf(stderr, "I (%d) must be divisible by ZO_TILE_N = %d\n", I, ZO_TILE_N);
        return 1;
    }
    if (d % ZO_TILE_K != 0) {
        fprintf(stderr, "d (%d) must be divisible by ZO_TILE_K = %d\n",
                d, ZO_TILE_K);
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
    half  *h_inp     = (half *)malloc(n_inp  * sizeof(half));
    half  *h_w_gate  = (half *)malloc(n_w    * sizeof(half));
    half  *h_w_norm  = (half *)malloc(n_norm * sizeof(half));
    half  *h_b_norm  = (half *)malloc(n_norm * sizeof(half));
    half  *h_target  = (half *)malloc(n_out  * sizeof(half));
    half  *h_out_gpu = (half *)malloc(n_out  * sizeof(half));
    float *h_out_ref = (float*)malloc(n_out  * sizeof(float));
    float *h_out_f32 = (float*)malloc(n_out  * sizeof(float));

    srand(42);
    for (int i = 0; i < n_inp;  ++i) h_inp[i]    = __float2half(randf_11());
    for (int i = 0; i < n_w;    ++i) h_w_gate[i] = __float2half(randf_11());
    for (int i = 0; i < n_norm; ++i) {
        h_w_norm[i] = __float2half(randf_11());
        h_b_norm[i] = __float2half((float)rand()/(float)RAND_MAX * 0.2f - 0.1f);
    }
    for (int i = 0; i < n_out; ++i) h_target[i] = __float2half(randf_11());

    // Device arrays
    half  *d_inp, *d_w_gate, *d_w_norm, *d_b_norm, *d_target;
    half  *d_out_pos, *d_out_neg;
    float *d_loss_pos, *d_loss_neg;
    checkCuda(cudaMalloc(&d_inp,      n_inp  * sizeof(half)));
    checkCuda(cudaMalloc(&d_w_gate,   n_w    * sizeof(half)));
    checkCuda(cudaMalloc(&d_w_norm,   n_norm * sizeof(half)));
    checkCuda(cudaMalloc(&d_b_norm,   n_norm * sizeof(half)));
    checkCuda(cudaMalloc(&d_target,   n_out  * sizeof(half)));
    checkCuda(cudaMalloc(&d_out_pos,  n_out  * sizeof(half)));
    checkCuda(cudaMalloc(&d_out_neg,  n_out  * sizeof(half)));
    checkCuda(cudaMalloc(&d_loss_pos, sizeof(float)));
    checkCuda(cudaMalloc(&d_loss_neg, sizeof(float)));

    checkCuda(cudaMemcpy(d_inp,    h_inp,    n_inp  * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_w_gate, h_w_gate, n_w    * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_w_norm, h_w_norm, n_norm * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_b_norm, h_b_norm, n_norm * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_target, h_target, n_out  * sizeof(half), cudaMemcpyHostToDevice));

    const int shm_fwd = 2 * ZO_TILE_M * ZO_TILE_K * sizeof(float)   // inp_tile_p + inp_tile_n
                      + ZO_TILE_K * ZO_TILE_N * sizeof(half)          // w_tile (half)
                      + 2 * ZO_TILE_M * 2 * sizeof(float2)            // smem2_p + smem2_n
                      + 2 * ZO_TILE_M * 2 * sizeof(float);            // warp_mse_p + warp_mse_n
    dim3 grid((M + ZO_TILE_M - 1) / ZO_TILE_M, B);

    // -----------------------------------------------------------------------
    // Test 1: correctness — eps=0, perturbation vanishes, compare to CPU ref
    // -----------------------------------------------------------------------
    printf("\n[1] Correctness check (eps=0)\n");

    checkCuda(cudaMemset(d_loss_pos, 0, sizeof(float)));
    checkCuda(cudaMemset(d_loss_neg, 0, sizeof(float)));
    zo_fused_forward_kernel<<<grid, FWD_THREADS, shm_fwd>>>(
        d_out_pos, d_out_neg, d_loss_pos, d_loss_neg,
        d_inp, d_inp, d_w_gate, d_target, d_w_norm, d_b_norm,
        M, d, I, /*seed=*/0ULL, /*eps=*/0.f);
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaMemcpy(h_out_gpu, d_out_pos, n_out * sizeof(half), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n_out; ++i) h_out_f32[i] = __half2float(h_out_gpu[i]);
    cpu_reference(h_out_ref, h_inp, h_w_gate, h_w_norm, h_b_norm, B, M, d, I);

    float abs_err = 0.f, max_ref = 0.f;
    for (int i = 0; i < n_out; ++i) {
        abs_err = fmaxf(abs_err, fabsf(h_out_ref[i] - h_out_f32[i]));
        max_ref = fmaxf(max_ref, fabsf(h_out_ref[i]));
    }
    float rel_err = abs_err / (max_ref + 1e-8f);
    printf("  Max abs error: %.4e  Max rel error: %.4e  %s\n",
           abs_err, rel_err, rel_err < 5e-3f ? "PASS" : "FAIL");

    printf("  First 4 outputs — %4s  %16s  %16s\n", "idx", "CPU (fp32)", "GPU (fp16→fp32)");
    for (int i = 0; i < 4; ++i)
        printf("    %-4d  %+15.8f  %+15.8f\n",
               i, h_out_ref[i], h_out_f32[i]);

    // -----------------------------------------------------------------------
    // Test 2: ZO gradient estimate and weight update
    //
    // MeZO estimate:  grad_est = (f+ - f-) / (2*eps)
    // Weight update:  W_gate  -= lr * grad_est * z   (z regenerated from seed)
    // -----------------------------------------------------------------------
    printf("\n[2] ZO gradient estimate and update\n");

    const unsigned long long seed = 0xDEADBEEF42ULL;
    const float eps = 1e-2f;
    const float lr  = 1e-3f;

    // Unperturbed loss before update
    float h_loss_before;
    checkCuda(cudaMemset(d_loss_pos, 0, sizeof(float)));
    checkCuda(cudaMemset(d_loss_neg, 0, sizeof(float)));
    zo_fused_forward_kernel<<<grid, FWD_THREADS, shm_fwd>>>(
        d_out_pos, d_out_neg, d_loss_pos, d_loss_neg,
        d_inp, d_inp, d_w_gate, d_target, d_w_norm, d_b_norm,
        M, d, I, seed, 0.f);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaMemcpy(&h_loss_before, d_loss_pos, sizeof(float), cudaMemcpyDeviceToHost));

    // f+ and f- in one dual-perturbation kernel call — loads inp and w_gate once
    float h_fpos, h_fneg;
    checkCuda(cudaMemset(d_loss_pos, 0, sizeof(float)));
    checkCuda(cudaMemset(d_loss_neg, 0, sizeof(float)));
    zo_fused_forward_kernel<<<grid, FWD_THREADS, shm_fwd>>>(
        d_out_pos, d_out_neg, d_loss_pos, d_loss_neg,
        d_inp, d_inp, d_w_gate, d_target, d_w_norm, d_b_norm,
        M, d, I, seed, eps);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaMemcpy(&h_fpos, d_loss_pos, sizeof(float), cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(&h_fneg, d_loss_neg, sizeof(float), cudaMemcpyDeviceToHost));

    float grad_est = (h_fpos - h_fneg) / (2.f * eps);
    const float n_elems = (float)(B * M * I);
    printf("  loss (unperturbed, before):          %.6f\n", h_loss_before / n_elems);
    printf("  [Φ(θ+εz,b) - Φ(θ-εz,b)] / 2ε:  %.6f\n", grad_est);

    // Apply:  W_gate -= lr * grad_est * z  (z regenerated from seed)
    int upd_groups = (d * I) / PHILOX_WIDTH;  // exact since I % PHILOX_WIDTH == 0
    int upd_blocks = (upd_groups + UPD_THREADS - 1) / UPD_THREADS;
    zo_update_kernel<<<upd_blocks, UPD_THREADS>>>(
        d_w_gate, d, I, seed, lr, grad_est);
    checkCuda(cudaDeviceSynchronize());

    // Unperturbed loss after update
    float h_loss_after;
    checkCuda(cudaMemset(d_loss_pos, 0, sizeof(float)));
    checkCuda(cudaMemset(d_loss_neg, 0, sizeof(float)));
    zo_fused_forward_kernel<<<grid, FWD_THREADS, shm_fwd>>>(
        d_out_pos, d_out_neg, d_loss_pos, d_loss_neg,
        d_inp, d_inp, d_w_gate, d_target, d_w_norm, d_b_norm,
        M, d, I, seed, 0.f);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaMemcpy(&h_loss_after, d_loss_pos, sizeof(float), cudaMemcpyDeviceToHost));

    printf("  loss (unperturbed, after):  %.6f\n", h_loss_after / n_elems);
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
            d_out_pos, d_out_neg, d_loss_pos, d_loss_neg,
            d_inp, d_inp, d_w_gate, d_target, d_w_norm, d_b_norm,
            M, d, I, seed, eps);

    cudaEventRecord(t0);
    for (int i = 0; i < RUNS; ++i)
        zo_fused_forward_kernel<<<grid, FWD_THREADS, shm_fwd>>>(
            d_out_pos, d_out_neg, d_loss_pos, d_loss_neg,
            d_inp, d_inp, d_w_gate, d_target, d_w_norm, d_b_norm,
            M, d, I, seed, eps);
    cudaEventRecord(t1);
    checkCuda(cudaDeviceSynchronize());
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("  zo_fused_forward: %.4f ms/call (%d calls)\n", ms / RUNS, RUNS);

    cudaEventRecord(t0);
    for (int i = 0; i < RUNS; ++i)
        zo_update_kernel<<<upd_blocks, UPD_THREADS>>>(
            d_w_gate, d, I, seed, lr, grad_est);
    cudaEventRecord(t1);
    checkCuda(cudaDeviceSynchronize());
    cudaEventElapsedTime(&ms, t0, t1);
    printf("  zo_update:        %.4f ms/call (%d calls)\n", ms / RUNS, RUNS);

    // Cleanup
    free(h_inp); free(h_w_gate); free(h_w_norm); free(h_b_norm);
    free(h_target); free(h_out_gpu); free(h_out_ref); free(h_out_f32);
    cudaFree(d_inp); cudaFree(d_w_gate); cudaFree(d_w_norm); cudaFree(d_b_norm);
    cudaFree(d_target); cudaFree(d_out_pos); cudaFree(d_out_neg);
    cudaFree(d_loss_pos); cudaFree(d_loss_neg);
    cudaEventDestroy(t0); cudaEventDestroy(t1);

    return rel_err < 5e-3f ? 0 : 1;
}
