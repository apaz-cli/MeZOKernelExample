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
 *   t = m_local * ZO_TPR + n_quarter.  Thread t owns 4 consecutive output
 *   columns n0 = n_start + 4*n_quarter for its own row m_local only.
 *   One Philox call per (weight-row k, n_quarter) yields 4 N(0,1) samples —
 *   all ZO_TILE_M rows use the same z for the same weight (shared perturbation).
 *   Register accumulators a[4] hold the partial dot products for m_local.
 *   Requires I % ZO_TILE_N == 0.
 *
 * Philox ILP:
 *   counter = k*(I/4) + n_start/4 + n_quarter uniquely addresses group (k, n0..n0+3).
 *   #pragma unroll 4 on the ki inner loop issues 4 independent Philox chains
 *   (distinct counters → no data dependence). The GPU pipelines their 7-round
 *   integer multiply chains in parallel, hiding Philox latency behind itself.
 *
 * Compile: nvcc -arch=sm_75 -O2 fused_zo_example.cu -o fused_zo_example
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
//   256 threads per block are arranged as a logical 4×64 grid:
//     m_local   = t / ZO_TPR  — which of the 4 output rows this thread owns
//     n_quarter = t % ZO_TPR  — which group of 4 consecutive columns (0–63)
//   Thread t accumulates dot products for row m_local, columns n_quarter*4
//   through n_quarter*4+3, and owns those same columns during LayerNorm.
//
// N-outer, K-inner (unlike the reference kernel): register accumulators a[4]
//   stay alive across all K-strips for a fixed N-strip.  This is required
//   because the Philox counter depends on both k and n, so the weight and z
//   for a given output column must both be resolved inside the K loop.
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
// No silu_buf: each thread normalizes the same 4-consecutive columns it
// computed in the GEMM (matching the Philox grouping), so post-SiLU values
// live in the v_cache register array and never touch shared memory.
// ============================================================================
#define ZO_TILE_M     4
#define ZO_TILE_K     64
#define ZO_TILE_N     256   // = ZO_TPR * 4
#define FWD_THREADS   256   // = ZO_TILE_M * ZO_TPR
#define ZO_TPR        64    // threads per row = FWD_THREADS / ZO_TILE_M = ZO_TILE_N / 4
#define I_DIM         4096  // hidden (intermediate) dimension
#define LAYERNORM_EPS 1e-5f

// ---------------------------------------------------------------------------
// Hyperparameter constraints
//
// FWD_THREADS == ZO_TILE_M * ZO_TPR
//   Thread decomposition t = m_local * ZO_TPR + n_quarter.  ZO_TILE_M rows,
//   ZO_TPR threads/row.
//
// ZO_TILE_N == ZO_TPR * 4
//   One Philox call per (thread, k-step) yields 4 N(0,1) samples, covering
//   4 consecutive columns.  Each N-tile must contain exactly ZO_TPR groups
//   of 4 columns — one group per thread — so ZO_TILE_N = 4 * ZO_TPR.
//
// ZO_TPR == ZO_TILE_K   (follows from the two above + A-tile load)
//   The A-tile [ZO_TILE_M × ZO_TILE_K] is loaded as inp_tile[t], one
//   element per thread.  FWD_THREADS = ZO_TILE_M * ZO_TPR threads must
//   cover ZO_TILE_M * ZO_TILE_K elements, so ZO_TPR must equal ZO_TILE_K.
//
// ZO_TPR == 64   (warp-shuffle epilogue hardcodes 2 warps/row)
//   Both stats and MSE reductions use smem2[m_local*2 + t_r/32] and
//   warp_mse[m_local*2 + t_r/32].  This assumes exactly 2 warps per row.
//   Changing ZO_TPR to 32 (1 warp) or 128 (4 warps) requires rewriting
//   both epilogue reductions.
//
// I % ZO_TILE_N == 0   (checked in main)
//   Same reason as the reference kernel: column-direction bounds guards are
//   absent in the N-tile loop, so w_gate and v_cache go out of bounds if I
//   is not a multiple of ZO_TILE_N.  Also ensures n_start is a multiple of
//   4, keeping the Philox counter (n_start >> 2) lossless.
//
// ZO_TILE_K % 4 == 0   (Philox ILP)
//   #pragma unroll 4 replicates the ki loop body 4 times, issuing 4
//   independent Philox chains (distinct counters k*IQ+...).  If ZO_TILE_K
//   is not divisible by 4, the compiler emits a remainder iteration whose
//   counter is not independent of the preceding group, collapsing the 4
//   chains into a dependency and eliminating the ILP benefit.
//
// I_DIM / ZO_TILE_N == I at runtime   (v_cache compile-time size)
//   float v_cache[(I_DIM/ZO_TILE_N)*4] is compile-time sized: I_DIM/ZO_TILE_N
//   N-tiles, 4 columns per thread per tile.  I_DIM must match the I passed
//   to the kernel; if they differ the cache is undersized or wastes registers.
//
// d % ZO_TILE_K: not required   (the check in main is overly conservative)
//   A-tile and B-tile loads are guarded by (k < d), so partial K-tiles are
//   correctly zero-padded.  The update kernel uses a flat linear group index
//   (group = k*(I/4) + n/4) independent of ZO_TILE_K, so partial tiles
//   cause no aliasing in the Philox counter space.
//
// Philox counter uniqueness
//   counter = k*(I/4) + n_start/4 + n_quarter.  The per-row group offset
//   n_start/4 + n_quarter is always < I/4, so no two weight rows share a
//   counter value.  This is guaranteed once I % ZO_TILE_N == 0.
//
// Hardware limits
//   FWD_THREADS <= 1024 (max threads/block).
//   shm_fwd <= 49152 by default (48 KB); up to 98304 on sm_75+ with
//   cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, N).
// ---------------------------------------------------------------------------

__global__ void zo_fused_forward_kernel(
    half       *out_pos,   // [B*M, I] output for W + eps*z
    half       *out_neg,   // [B*M, I] output for W - eps*z
    float      *loss_pos,  // MSE accumulator for + perturbation
    float      *loss_neg,  // MSE accumulator for - perturbation
    const half *inp_pos,   // [B, M, d] — input for + perturbation (ZO: same as inp_neg)
    const half *inp_neg,   // [B, M, d] — input for - perturbation (ZO: same as inp_pos)
    const half *w_gate,    // [d, I]
    const half *target,    // [B*M, I]
    const half *w_norm,    // [I]
    const half *b_norm,    // [I]
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
    const int IQ       = I >> 2;

    // Thread decomposition: t = m_local * ZO_TPR + n_quarter
    const int m_local  = t / ZO_TPR;
    const int n_quarter = t % ZO_TPR;
    const int row      = tile_row * ZO_TILE_M + m_local;

    const half *inp_pos_base = inp_pos + (size_t)batch * M * d;
    const half *inp_neg_base = inp_neg + (size_t)batch * M * d;

    // ----------------------------------------------------------------
    // Step 1: tiled GEMM — outer N-tile, inner K-tile.
    // All 256 threads cooperate loading inp_tile and w_tile from DRAM into
    // shared memory; each thread then computes dot products only for its own
    // m_local row (4 columns at a time matching Philox's output width).
    // After all K-strips, a[0..3] holds the complete dot products; SiLU is
    // applied and results go into v_cache registers.
    // ----------------------------------------------------------------
    float v_cache_p[(I_DIM / ZO_TILE_N) * 4];  // post-SiLU values for + perturbation
    float v_cache_n[(I_DIM / ZO_TILE_N) * 4];  // post-SiLU values for - perturbation
    float2 local_p = {0.f, 0.f};               // (Σv, Σv²) for + perturbation
    float2 local_n = {0.f, 0.f};               // (Σv, Σv²) for - perturbation
    int ci = 0;

    for (int n_start = 0; n_start < I; n_start += ZO_TILE_N, ci += 4) {
        float a_p[4] = {};   // dot product accumulators for + perturbation
        float a_n[4] = {};   // dot product accumulators for - perturbation

        for (int k_start = 0; k_start < d; k_start += ZO_TILE_K) {
            // Load both A tiles (one element per thread each).
            // In ZO, inp_pos and inp_neg point to the same array; the L1 cache
            // handles the duplicate load transparently.
            {
                int mi = t / ZO_TILE_K, ki = t % ZO_TILE_K;
                int r  = tile_row * ZO_TILE_M + mi;
                int k  = k_start + ki;
                bool valid = (r < M && k < d);
                inp_tile_p[t] = valid ? __half2float(inp_pos_base[r * d + k]) : 0.f;
                inp_tile_n[t] = valid ? __half2float(inp_neg_base[r * d + k]) : 0.f;
            }

            // Load B tile: [ZO_TILE_K × ZO_TILE_N] halfs, coalesced
            for (int idx = t; idx < ZO_TILE_K * ZO_TILE_N; idx += FWD_THREADS) {
                int ki = idx / ZO_TILE_N, ni = idx % ZO_TILE_N;
                int k  = k_start + ki;
                w_tile[idx] = (k < d) ? w_gate[(size_t)k * I + n_start + ni]
                                      : __float2half(0.f);
            }
            __syncthreads();

            // Dot product loop.  #pragma unroll 4 makes the compiler emit 4
            // copies with ki=0,1,2,3 simultaneously — each has a different
            // counter (k varies), so the 4 Philox calls are independent and
            // the GPU can pipeline their ~10-cycle latency chains in parallel.
            #pragma unroll 4
            for (int ki = 0; ki < ZO_TILE_K; ++ki) {
                int k = k_start + ki;
                // counter = k*(I/4) + column_group; same formula as zo_update_kernel,
                // so z is reproduced identically without storing it.
                float4 z4 = philox_normal_4(seed,
                    (unsigned long long)k * IQ + (n_start >> 2) + n_quarter);
                const half *wt = w_tile + ki * ZO_TILE_N + (n_quarter << 2);
                float w0 = __half2float(wt[0]), p0 = eps * z4.x;
                float w1 = __half2float(wt[1]), p1 = eps * z4.y;
                float w2 = __half2float(wt[2]), p2 = eps * z4.z;
                float w3 = __half2float(wt[3]), p3 = eps * z4.w;
                float iv_p = inp_tile_p[m_local * ZO_TILE_K + ki];
                float iv_n = inp_tile_n[m_local * ZO_TILE_K + ki];
                a_p[0] += iv_p * (w0 + p0);  a_n[0] += iv_n * (w0 - p0);
                a_p[1] += iv_p * (w1 + p1);  a_n[1] += iv_n * (w1 - p1);
                a_p[2] += iv_p * (w2 + p2);  a_n[2] += iv_n * (w2 - p2);
                a_p[3] += iv_p * (w3 + p3);  a_n[3] += iv_n * (w3 - p3);
            }
            __syncthreads();
        }

        // SiLU both perturbations; cache in registers; collect stats for both.
        float vp0 = a_p[0] / (1.f + expf(-a_p[0]));
        float vp1 = a_p[1] / (1.f + expf(-a_p[1]));
        float vp2 = a_p[2] / (1.f + expf(-a_p[2]));
        float vp3 = a_p[3] / (1.f + expf(-a_p[3]));
        float vn0 = a_n[0] / (1.f + expf(-a_n[0]));
        float vn1 = a_n[1] / (1.f + expf(-a_n[1]));
        float vn2 = a_n[2] / (1.f + expf(-a_n[2]));
        float vn3 = a_n[3] / (1.f + expf(-a_n[3]));
        v_cache_p[ci    ] = vp0; v_cache_p[ci + 1] = vp1;
        v_cache_p[ci + 2] = vp2; v_cache_p[ci + 3] = vp3;
        v_cache_n[ci    ] = vn0; v_cache_n[ci + 1] = vn1;
        v_cache_n[ci + 2] = vn2; v_cache_n[ci + 3] = vn3;
        local_p.x += vp0 + vp1 + vp2 + vp3;
        local_p.y += vp0*vp0 + vp1*vp1 + vp2*vp2 + vp3*vp3;
        local_n.x += vn0 + vn1 + vn2 + vn3;
        local_n.y += vn0*vn0 + vn1*vn1 + vn2*vn2 + vn3*vn3;
    }

    // ----------------------------------------------------------------
    // Step 2: LayerNorm stats — warp shuffle reduction for mean and variance.
    // Each thread has accumulated (Σv, Σv²) across its columns in 'local'.
    // ----------------------------------------------------------------
    const int t_r = n_quarter;

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
    // Step 3: normalize both outputs from register caches; write both;
    // accumulate MSE for each vs the shared target.
    // ----------------------------------------------------------------
    float local_loss_p = 0.f, local_loss_n = 0.f;
    if (row < M) {
        half       *out_row_p = out_pos + (size_t)(batch * M + row) * I;
        half       *out_row_n = out_neg + (size_t)(batch * M + row) * I;
        const half *tgt_row   = target  + (size_t)(batch * M + row) * I;
        ci = 0;
        for (int n_start = 0; n_start < I; n_start += ZO_TILE_N, ci += 4) {
            int n0 = n_start + (n_quarter << 2);
            for (int j = 0; j < 4; ++j) {
                float gamma_j = __half2float(w_norm[n0 + j]);
                float beta_j  = __half2float(b_norm[n0 + j]);
                float tgt_j   = __half2float(tgt_row[n0 + j]);
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
//   counter = k*(I/4) + n/4 = group  (linear index of the 4-weight group)
//
// Since weights are stored row-major and I % 4 == 0, aligned groups of 4
// consecutive weights always fall within the same weight row k — so the
// counter formula is equivalent to the forward kernel's counter.
// All 4 Philox outputs are consumed per thread, same as the forward kernel.
//
// Requires d*I % 4 == 0 (satisfied when I % 4 == 0).
// ============================================================================
#define UPD_THREADS 256

__global__ void zo_update_kernel(
    half *w_gate,      // [d, I], updated in-place
    int d, int I,
    unsigned long long seed, float lr, float grad_est)
{
    // group = k*(I/4) + t  where  t = n/4,  same counter as zo_fused_forward
    int group = blockIdx.x * UPD_THREADS + threadIdx.x;
    if ((size_t)group * 4 >= (size_t)d * I) return;

    float4 z4    = philox_normal_4(seed, (unsigned long long)group);
    float  scale = lr * grad_est;

    half *w = w_gate + (size_t)group * 4;
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

    printf("=== zo_fused_forward + zo_update ===\n");
    printf("  B=%d  M=%d  d=%d  I=%d\n", B, M, d, I);
    printf("  ZO_TILE_M=%d  ZO_TILE_K=%d  ZO_TILE_N=%d  FWD_THREADS=%d  UPD_THREADS=%d\n",
           ZO_TILE_M, ZO_TILE_K, ZO_TILE_N, FWD_THREADS, UPD_THREADS);

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
                      + ZO_TILE_K * ZO_TILE_N * sizeof(half)          // w_tile (shared)
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
        printf("               — %-4d  %+15.8f  %+15.8f\n",
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
    int upd_groups = (d * I) / 4;  // exact since I % 4 == 0
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
