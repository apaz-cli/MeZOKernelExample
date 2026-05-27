/*
 * fused_example.cu
 *
 * Computes:  out = layernorm(silu(inp @ W_gate))
 *            inp    : [B, M, d]
 *            W_gate : [d, I]   (shared across batch)
 *            out    : [B, M, I]
 *
 * Compile:  nvcc -arch=sm_89 -O2 fused_example.cu -o fused_example
 * Run:      ./fused_example
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define checkCuda(ans) gpuAssert((ans), __FILE__, __LINE__)

inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d — %s\n", file, line, cudaGetErrorString(code));
        exit(code);
    }
}

// ---------------------------------------------------------------------------
// Fused kernel: layernorm(silu(inp @ W_gate))
//
// Tile hierarchy (CUTLASS framing)
//   Block tile  TILE_M × TILE_N:  what this threadblock is responsible for
//   Thread tile 1     × THREAD_N: what each thread holds in registers
//   BLOCK_THREADS     = TILE_M × (TILE_N / THREAD_N)   (derived, not set directly)
//
//   In a production kernel the thread tile would be replaced by a warp tile
//   (Ampere mma.sync) or warpgroup tile (Hopper wgmma / Blackwell tcgen05).
//   The block tile and K-loop structure are identical across all three.
//
// Thread assignment
//   256 threads per block, laid out as TILE_M rows × (TILE_N/THREAD_N) cols:
//     m_local = t / (TILE_N/THREAD_N)  — which of the 4 output rows
//     n_local = t % (TILE_N/THREAD_N)  — which column group (0–63)
//   Thread t owns output columns  n_local*THREAD_N .. n_local*THREAD_N+THREAD_N-1
//   within each N-tile, and the same columns during LayerNorm.
//
// Grid:  dim3(ceil(M/TILE_M), B)
//
// Shared memory layout (~35 KB total):
//   inp_tile [TILE_M × TILE_K floats]      input strip      (1 KB)
//   w_tile   [TILE_K × TILE_N halfs]       weight strip     (32 KB)
//   smem_red [2 × BLOCK_THREADS floats]    tree-reduce buf  (2 KB)
//
// w_tile is stored as half: with THREAD_N=4 each thread reads four consecutive
// halfs per ki step (stride-4 access), giving 2-way bank conflicts.  Storing
// as float would give 4-way conflicts for the same access pattern.  Half is
// the better choice here; this is the same trade-off as in the ZO kernel.
//
// Steps:
//   1. Tiled GEMM (K-outer, N-inner): load inp strip once per K-tile; for each
//      N-tile load the weight strip and accumulate THREAD_N partial dot products
//      per thread into register array v_cache.
//   2. SiLU in-place on v_cache; smem tree reduce for per-row (Σv, Σv²).
//   3. Normalize from v_cache; write fp16 output.
// ---------------------------------------------------------------------------
#define TILE_M        4
#define TILE_K        64
#define TILE_N        256
#define THREAD_N      4
#define BLOCK_THREADS (TILE_M * TILE_N / THREAD_N)   // = 256
#define I_DIM         4096  // hidden (intermediate) dimension
#define LAYERNORM_EPS 1e-5f

// ---------------------------------------------------------------------------
// Constraints
//
// TILE_N / THREAD_N must be a power of 2   (smem tree reduce)
//   The epilogue halves the active-thread count each step.
//
// I % TILE_N == 0
//   The N-tile loop has no column-direction bounds guard.
//
// I_DIM == I at runtime   (v_cache compile-time size)
//   float v_cache[(I_DIM/TILE_N)*THREAD_N] is compile-time sized.
//
// d % TILE_K: not required
//   A-tile and B-tile loads are guarded by (k < d).
//
// M % TILE_M: not required
//   Output write is guarded by (row < M).
// ---------------------------------------------------------------------------
static_assert(((TILE_N / THREAD_N) & (TILE_N / THREAD_N - 1)) == 0,
              "TILE_N / THREAD_N must be a power of 2 (smem tree reduce)");

__global__ void layernorm_silu_matmul_kernel(
    half       * __restrict__ out,
    const half * __restrict__ inp,
    const half * __restrict__ w_gate,
    const half * __restrict__ w_norm,
    const half * __restrict__ b_norm,
    int B, int M, int d, int I)
{
    // One dynamic shared memory allocation, manually partitioned:
    extern __shared__ char smem_raw[];
    float  *inp_tile = (float *)smem_raw;
    half   *w_tile   = (half  *)(smem_raw + TILE_M * TILE_K * sizeof(float));
    float  *smem_red = (float *)(smem_raw + TILE_M * TILE_K * sizeof(float)
                                          + TILE_K * TILE_N * sizeof(half));
    // smem_red[0..BLOCK_THREADS) = per-thread Σv; smem_red[BLOCK_THREADS..2*) = per-thread Σv²

    const int tile_row        = blockIdx.x;
    const int batch           = blockIdx.y;
    const int t               = threadIdx.x;
    const int threads_per_row = TILE_N / THREAD_N;   // = 64

    // Thread tile decomposition.
    const int m_local = t / threads_per_row;
    const int n_local = t % threads_per_row;   // column-group index within row
    const int row     = tile_row * TILE_M + m_local;

    // ----------------------------------------------------------------
    // Step 1: tiled GEMM — K-outer, N-inner.
    //
    // v_cache[(I/TILE_N)*THREAD_N] holds the full dot products for this
    // thread's THREAD_N output columns across all N-tiles.  ci advances by
    // THREAD_N each N-tile; within a tile j=0..THREAD_N-1 indexes the columns.
    //
    // The A-tile is loaded as a strided cooperative loop so that TILE_K is
    // independent of BLOCK_THREADS and the thread-tile decomposition.
    // ----------------------------------------------------------------
    float v_cache[(I_DIM / TILE_N) * THREAD_N] = {};

    for (int k_start = 0; k_start < d; k_start += TILE_K) {
        // Cooperatively load TILE_M × TILE_K input strip.
        for (int idx = t; idx < TILE_M * TILE_K; idx += BLOCK_THREADS) {
            int mi = idx / TILE_K, ki = idx % TILE_K;
            int r  = tile_row * TILE_M + mi;
            int k  = k_start + ki;
            inp_tile[idx] = (r < M && k < d)
                ? __half2float(__ldg(&inp[(batch * M + r) * d + k])) : 0.f;
        }
        __syncthreads();

        for (int n_start = 0, ci = 0; n_start < I; n_start += TILE_N, ci += THREAD_N) {
            // Cooperatively load TILE_K × TILE_N weight strip.
            for (int idx = t; idx < TILE_K * TILE_N; idx += BLOCK_THREADS) {
                int ki = idx / TILE_N, ni = idx % TILE_N;
                int k  = k_start + ki;
                w_tile[idx] = (k < d) ? __ldg(&w_gate[k * I + n_start + ni])
                                      : __float2half(0.f);
            }
            __syncthreads();

            // Each thread accumulates THREAD_N dot products for its column group.
            float dot[THREAD_N] = {};
            #pragma unroll 4
            for (int ki = 0; ki < TILE_K; ki++) {
                float a = inp_tile[m_local * TILE_K + ki];
                const half *wt = w_tile + ki * TILE_N + n_local * THREAD_N;
                #pragma unroll
                for (int j = 0; j < THREAD_N; j++)
                    dot[j] += a * __half2float(wt[j]);
            }
            #pragma unroll
            for (int j = 0; j < THREAD_N; j++)
                v_cache[ci + j] += dot[j];

            __syncthreads();
        }
    }

    // ----------------------------------------------------------------
    // Step 2: SiLU in-place on v_cache; collect per-row stats (Σv, Σv²).
    // ----------------------------------------------------------------
    const int t_r = n_local;

    float2 local = {0.f, 0.f};
    for (int ci = 0; ci < (I_DIM / TILE_N) * THREAD_N; ++ci) {
        float g = v_cache[ci];
        float v = g / (1.f + expf(-g));
        v_cache[ci] = v;
        local.x += v;
        local.y += v * v;
    }

    // Smem tree reduce across threads_per_row threads per row.
    smem_red[t]                = local.x;
    smem_red[t + BLOCK_THREADS] = local.y;
    __syncthreads();
    for (int stride = threads_per_row >> 1; stride > 0; stride >>= 1) {
        if (t_r < stride) {
            smem_red[t]                += smem_red[t + stride];
            smem_red[t + BLOCK_THREADS] += smem_red[t + stride + BLOCK_THREADS];
        }
        __syncthreads();
    }
    float2 agg = { smem_red[m_local * threads_per_row],
                   smem_red[m_local * threads_per_row + BLOCK_THREADS] };
    float mean    = agg.x / I;
    float inv_std = rsqrtf(agg.y / I - mean * mean + LAYERNORM_EPS);

    // ----------------------------------------------------------------
    // Step 3: normalize from v_cache; write fp16 output.
    // ----------------------------------------------------------------
    if (row < M) {
        half *out_row = out + (batch * M + row) * I;
        for (int n_start = 0, ci = 0; n_start < I; n_start += TILE_N, ci += THREAD_N) {
            int n_base = n_start + n_local * THREAD_N;
            #pragma unroll
            for (int j = 0; j < THREAD_N; j++) {
                int n = n_base + j;
                float norm = (v_cache[ci + j] - mean) * inv_std;
                out_row[n] = __float2half(norm * __half2float(__ldg(&w_norm[n]))
                                              + __half2float(__ldg(&b_norm[n])));
            }
        }
    }
}

void cpu_reference(
    float      *out,
    const half *inp,
    const half *w_gate,
    const half *w_norm,
    const half *b_norm,
    int B, int M, int d, int I)
{
    float *tmp = (float*)malloc(M * I * sizeof(float));

    for (int b = 0; b < B; b++) {
        const half *inp_b = inp + b * M * d;
        float      *out_b = out + b * M * I;

        for (int m = 0; m < M; m++) {
            for (int i = 0; i < I; i++) {
                float dot = 0.0f;
                for (int k = 0; k < d; k++)
                    dot += __half2float(inp_b[m * d + k]) * __half2float(w_gate[k * I + i]);
                tmp[m * I + i] = dot / (1.0f + expf(-dot));
            }

            float sum = 0.0f, sum_sq = 0.0f;
            for (int i = 0; i < I; i++) {
                float v = tmp[m * I + i];
                sum    += v;
                sum_sq += v * v;
            }
            float mean    = sum / I;
            float inv_std = 1.0f / sqrtf(sum_sq / I - mean * mean + LAYERNORM_EPS);

            for (int i = 0; i < I; i++) {
                float norm = (tmp[m * I + i] - mean) * inv_std;
                out_b[m * I + i] = norm * __half2float(w_norm[i]) + __half2float(b_norm[i]);
            }
        }
    }

    free(tmp);
}

int main() {
    const int B = 4;
    const int M = 128;
    const int d = 256;
    const int I = I_DIM;

    const int n_inp  = B * M * d;
    const int n_w    = d * I;
    const int n_norm = I;
    const int n_out  = B * M * I;

    printf("\n");
    printf("=== layernorm(silu(inp @ W_gate)) ===\n");
    printf("  B=%d  M=%d  d=%d  I=%d\n", B, M, d, I);
    printf("  TILE_M=%d  TILE_K=%d  TILE_N=%d  THREAD_N=%d  BLOCK_THREADS=%d\n\n",
           TILE_M, TILE_K, TILE_N, THREAD_N, BLOCK_THREADS);

    half  *h_inp     = (half*) malloc(n_inp  * sizeof(half));
    half  *h_w_gate  = (half*) malloc(n_w    * sizeof(half));
    half  *h_w_norm  = (half*) malloc(n_norm * sizeof(half));
    half  *h_b_norm  = (half*) malloc(n_norm * sizeof(half));
    half  *h_out_gpu = (half*) malloc(n_out  * sizeof(half));
    float *h_out_ref = (float*)malloc(n_out  * sizeof(float));
    float *h_out_f32 = (float*)malloc(n_out  * sizeof(float));

    srand(42);
    for (int i = 0; i < n_inp; i++)
        h_inp[i] = __float2half(((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f);
    for (int i = 0; i < n_w; i++)
        h_w_gate[i] = __float2half(((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f);
    for (int i = 0; i < n_norm; i++) {
        h_w_norm[i] = __float2half(((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f);
        h_b_norm[i] = __float2half(((float)rand() / (float)RAND_MAX) * 0.2f - 0.1f);
    }

    half *d_inp, *d_w_gate, *d_w_norm, *d_b_norm, *d_out;
    checkCuda(cudaMalloc(&d_inp,    n_inp  * sizeof(half)));
    checkCuda(cudaMalloc(&d_w_gate, n_w    * sizeof(half)));
    checkCuda(cudaMalloc(&d_w_norm, n_norm * sizeof(half)));
    checkCuda(cudaMalloc(&d_b_norm, n_norm * sizeof(half)));
    checkCuda(cudaMalloc(&d_out,    n_out  * sizeof(half)));

    checkCuda(cudaMemcpy(d_inp,    h_inp,    n_inp  * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_w_gate, h_w_gate, n_w    * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_w_norm, h_w_norm, n_norm * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_b_norm, h_b_norm, n_norm * sizeof(half), cudaMemcpyHostToDevice));

    const int shm_bytes = TILE_M * TILE_K * sizeof(float)      // inp_tile
                        + TILE_K * TILE_N * sizeof(half)     // w_tile
                        + 2 * BLOCK_THREADS * sizeof(float); // smem_red
    dim3 grid((M + TILE_M - 1) / TILE_M, B);

    // -----------------------------------------------------------------------
    // [1] Correctness check
    // -----------------------------------------------------------------------
    printf("[1] Correctness check\n");

    layernorm_silu_matmul_kernel<<<grid, BLOCK_THREADS, shm_bytes>>>(
        d_out, d_inp, d_w_gate, d_w_norm, d_b_norm, B, M, d, I);
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaMemcpy(h_out_gpu, d_out, n_out * sizeof(half), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n_out; i++)
        h_out_f32[i] = __half2float(h_out_gpu[i]);
    cpu_reference(h_out_ref, h_inp, h_w_gate, h_w_norm, h_b_norm, B, M, d, I);

    float abs_err = 0.0f, max_ref = 0.0f;
    for (int i = 0; i < n_out; i++) {
        abs_err = fmaxf(abs_err, fabsf(h_out_ref[i] - h_out_f32[i]));
        max_ref = fmaxf(max_ref, fabsf(h_out_ref[i]));
    }
    float rel_err = abs_err / (max_ref + 1e-8f);

    printf("  Max abs error: %.4e  Max rel error: %.4e  %s\n",
           abs_err, rel_err, rel_err < 5e-3f ? "PASS" : "FAIL");
    printf("  First 4 outputs — %4s  %16s  %16s\n", "idx", "CPU (fp32)", "GPU (fp16→fp32)");
    for (int i = 0; i < 4; i++)
        printf("    %-4d  %+15.8f  %+15.8f\n", i, h_out_ref[i], h_out_f32[i]);

    // -----------------------------------------------------------------------
    // [2] Timing
    // -----------------------------------------------------------------------
    printf("\n[2] Timing\n");
    cudaEvent_t t_start, t_stop;
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_stop);

    const int WARMUP = 5, RUNS = 50;
    for (int i = 0; i < WARMUP; ++i)
        layernorm_silu_matmul_kernel<<<grid, BLOCK_THREADS, shm_bytes>>>(
            d_out, d_inp, d_w_gate, d_w_norm, d_b_norm, B, M, d, I);

    cudaEventRecord(t_start);
    for (int i = 0; i < RUNS; ++i)
        layernorm_silu_matmul_kernel<<<grid, BLOCK_THREADS, shm_bytes>>>(
            d_out, d_inp, d_w_gate, d_w_norm, d_b_norm, B, M, d, I);
    cudaEventRecord(t_stop);
    checkCuda(cudaDeviceSynchronize());

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, t_start, t_stop);
    printf("  layernorm_silu_matmul: %.4f ms/call (%d calls)\n", elapsed_ms / RUNS, RUNS);

    free(h_inp); free(h_w_gate); free(h_w_norm); free(h_b_norm);
    free(h_out_gpu); free(h_out_ref); free(h_out_f32);
    cudaFree(d_inp); cudaFree(d_w_gate); cudaFree(d_w_norm); cudaFree(d_b_norm); cudaFree(d_out);
    cudaEventDestroy(t_start); cudaEventDestroy(t_stop);

    return rel_err < 5e-3f ? 0 : 1;
}
