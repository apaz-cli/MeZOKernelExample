/*
 * fused_example.cu
 *
 * Computes:  out = layernorm(silu(inp @ W_gate))
 *            inp    : [B, M, d]
 *            W_gate : [d, I]   (shared across batch)
 *            out    : [B, M, I]
 *
 * Compile:  nvcc -arch=sm_75 -O2 fused_example.cu -o fused_example
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
// Why tiling?
//   GPU global memory (DRAM) costs ~200 cycles per access; shared memory —
//   a small, fast scratchpad on-chip shared by all threads in a block — costs
//   ~4 cycles.  The strategy: all 256 threads cooperatively load a narrow
//   strip of inp rows and W_gate columns into shared memory at once, compute
//   partial dot products from those cached copies, then advance to the next
//   strip.  Repeating across all strips accumulates the full dot product
//   while reading each weight from DRAM only once.
//
// Thread assignment
//   256 threads per block are arranged as a logical 4×64 grid:
//     m_local = threadIdx.x / 64  — which of the 4 output rows this thread owns
//     n_local = threadIdx.x % 64  — which column within that row (0–63)
//   Thread t is solely responsible for output element (m_local, n_local) in
//   each N-strip, and for the same column during LayerNorm.
//
// Grid:  dim3(ceil(M/TILE_M), B)  — one block per (row-tile, batch element)
// Block: BLOCK_THREADS = TILE_M * TILE_N threads (256 = 4 rows × 64 cols)
//
// Shared memory layout:
//   inp_tile [TILE_M × TILE_K floats]  input strip      (1 KB)
//   w_tile   [TILE_K × TILE_N halfs]   weight strip     (8 KB)
//   silu_buf [TILE_M × I floats]       dot product sums (8 KB)
//   smem2    [TILE_M*2 float2]         warp-leader stats (64 B)
//
// Steps:
//   1. Tiled GEMM: load inp and W_gate strips into shared memory; accumulate
//      partial dot products into silu_buf; repeat across all K and N strips.
//   2. SiLU + stats: apply SiLU; each thread accumulates partial (Σv, Σv²)
//      for its row using warp shuffles — threads within a 32-thread warp
//      exchange register values directly, no memory needed.
//   3. Normalize: apply (v − mean)/std * γ + β from register-cached values;
//      write fp16 output.
// ---------------------------------------------------------------------------
#define TILE_M        4
#define TILE_K        64
#define TILE_N        64
#define BLOCK_THREADS 256
#define I_DIM         4096  // hidden (intermediate) dimension
#define LAYERNORM_EPS 1e-5f

// ---------------------------------------------------------------------------
// Hyperparameter constraints
//
// BLOCK_THREADS == TILE_M * TILE_N
//   Thread decomposition t = m_local * TILE_N + n_local maps each thread to
//   exactly one output element.  TILE_M rows × TILE_N threads/row = total
//   block size.
//
// TILE_K == TILE_N   (follows from the above + A-tile load)
//   The A-tile [TILE_M × TILE_K] is loaded as inp_tile[t], one element per
//   thread.  The tile has TILE_M * TILE_K elements; the block has
//   BLOCK_THREADS = TILE_M * TILE_N threads, so TILE_K must equal TILE_N.
//   Violating this overflows or underloads inp_tile, corrupting w_tile.
//
// TILE_N == 64   (warp-shuffle epilogue hardcodes 2 warps/row)
//   After the intra-warp butterfly, exactly 2 warp leaders per row write to
//   smem2[m_local*2 + t_r/32].  Changing TILE_N to 32 (1 warp) or 128
//   (4 warps) requires rewriting the epilogue indices and smem2 sizing.
//
// I % TILE_N == 0
//   The N-tile loop loads w_gate[k*I + n_start+ni] and writes
//   silu_buf[m_local*I + n_start+n_local] with no column-direction bounds
//   guard.  If I is not a multiple of TILE_N, the last N-tile accesses
//   w_gate and silu_buf out of bounds.
//
// I_DIM / TILE_N == I at runtime   (v_cache compile-time size)
//   float v_cache[I_DIM / TILE_N] is compile-time sized.  I_DIM must match
//   the I passed to the kernel; if they differ the cache is undersized or
//   wastes registers.
//
// d % TILE_K: not required
//   Both the A-tile load (r < M && k < d) and B-tile load (k < d) are
//   guarded, so partial K-tiles are correctly zero-padded.
//
// M % TILE_M: not required
//   The output write is guarded by (row < M); out-of-bounds tiles produce
//   zero-padded stats but never write output.
//
// Hardware limits
//   BLOCK_THREADS <= 1024 (max threads/block).
//   shm_bytes <= 49152 by default (48 KB); up to 98304 on sm_75+ with
//   cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, N).
// ---------------------------------------------------------------------------

__global__ void layernorm_silu_matmul_kernel(
    half       *out,
    const half *inp,
    const half *w_gate,
    const half *w_norm,
    const half *b_norm,
    int B, int M, int d, int I)
{
    // One dynamic shared memory allocation, manually partitioned into regions:
    extern __shared__ char smem_raw[];
    float  *inp_tile = (float *)smem_raw;
    half   *w_tile   = (half  *)(smem_raw + TILE_M * TILE_K * sizeof(float));
    float  *silu_buf = (float *)(smem_raw + TILE_M * TILE_K * sizeof(float)
                                          + TILE_K * TILE_N * sizeof(half));
    float2 *smem2    = (float2 *)(smem_raw + TILE_M * TILE_K * sizeof(float)
                                           + TILE_K * TILE_N * sizeof(half)
                                           + TILE_M * I * sizeof(float));

    const int tile_row = blockIdx.x;
    const int batch    = blockIdx.y;
    const int t        = threadIdx.x;

    // Map flat thread index to a 2D position within the output tile.
    // m_local: which of the TILE_M rows  (0 – TILE_M-1)
    // n_local: which column within a row (0 – TILE_N-1)
    const int m_local = t / TILE_N;
    const int n_local = t % TILE_N;
    const int row     = tile_row * TILE_M + m_local;  // global row in inp/out

    // ----------------------------------------------------------------
    // Step 1: tiled GEMM
    // ----------------------------------------------------------------

    // Zero silu_buf before accumulating partial dot products into it.
    for (int idx = t; idx < TILE_M * I; idx += BLOCK_THREADS)
        silu_buf[idx] = 0.f;
    __syncthreads();  // all writes must finish before any thread reads silu_buf

    for (int k_start = 0; k_start < d; k_start += TILE_K) {
        // Cooperatively load a TILE_M×TILE_K strip of inp into shared memory.
        // With BLOCK_THREADS = TILE_M*TILE_K, each thread loads exactly one element.
        {
            int mi = t / TILE_K, ki = t % TILE_K;
            int r  = tile_row * TILE_M + mi;
            int k  = k_start + ki;
            inp_tile[t] = (r < M && k < d)
                ? __half2float(inp[(batch * M + r) * d + k]) : 0.f;
        }
        __syncthreads();  // inp_tile must be fully written before anyone reads it

        for (int n_start = 0; n_start < I; n_start += TILE_N) {
            // Cooperatively load a TILE_K×TILE_N strip of W_gate into shared memory.
            // Adjacent threads load adjacent columns — "coalesced" access merges
            // them into fewer, wider DRAM transactions.
            for (int idx = t; idx < TILE_K * TILE_N; idx += BLOCK_THREADS) {
                int ki = idx / TILE_N, ni = idx % TILE_N;
                int k  = k_start + ki;
                w_tile[idx] = (k < d) ? w_gate[k * I + n_start + ni]
                                      : __float2half(0.f);
            }
            __syncthreads();  // w_tile must be ready before the dot product loop

            // Each thread accumulates its partial dot product for element
            // (m_local, n_start+n_local) of the output.
            float dot = 0.f;
            for (int ki = 0; ki < TILE_K; ki++)
                dot += inp_tile[m_local * TILE_K + ki]
                     * __half2float(w_tile[ki * TILE_N + n_local]);
            silu_buf[m_local * I + n_start + n_local] += dot;
            __syncthreads();  // silu_buf writes must finish before next w_tile load
        }
    }

    // ----------------------------------------------------------------
    // Step 2: SiLU + per-row stats for LayerNorm
    // Read each accumulated dot product, apply SiLU, and collect the
    // per-row sums (Σv) and (Σv²) needed to compute mean and variance.
    // v_cache keeps the post-SiLU values in registers so step 3 doesn't
    // need to re-read silu_buf.
    // ----------------------------------------------------------------
    const int t_r = n_local;   // this thread's index within its row's 64-thread slice

    float v_cache[I_DIM / TILE_N];
    float2 local = {0.f, 0.f};
    for (int ci = 0, n = t_r; n < I; n += TILE_N, ++ci) {
        float g = silu_buf[m_local * I + n];
        float v = g / (1.f + expf(-g));
        v_cache[ci] = v;
        local.x += v;
        local.y += v * v;
    }

    // Warp reduction: the GPU executes threads in hardware groups of 32 called
    // warps.  __shfl_xor_sync lets threads within a warp exchange register
    // values directly — no shared memory, no synchronization needed.
    // The XOR butterfly (offsets 16, 8, 4, 2, 1) is a standard parallel sum:
    // after 5 steps every lane holds the sum of all 32 lanes in its warp.
    for (int s = 16; s > 0; s >>= 1) {
        local.x += __shfl_xor_sync(0xffffffff, local.x, s);
        local.y += __shfl_xor_sync(0xffffffff, local.y, s);
    }
    // Each row spans 2 warps.  Lane 0 of each warp (t_r % 32 == 0) holds the
    // warp's total and writes it to shared memory.  After one __syncthreads(),
    // every thread reads both slots and independently computes mean/variance.
    if (t_r % 32 == 0)
        smem2[m_local * 2 + t_r / 32] = local;
    __syncthreads();
    float2 agg    = { smem2[m_local * 2].x + smem2[m_local * 2 + 1].x,
                      smem2[m_local * 2].y + smem2[m_local * 2 + 1].y };
    float mean    = agg.x / I;
    float inv_std = rsqrtf(agg.y / I - mean * mean + LAYERNORM_EPS);

    // ----------------------------------------------------------------
    // Step 3: normalize from register cache; guard out-of-bounds rows
    // ----------------------------------------------------------------
    if (row < M) {
        half *out_row = out + (batch * M + row) * I;
        for (int ci = 0, n = t_r; n < I; n += TILE_N, ++ci) {
            float norm = (v_cache[ci] - mean) * inv_std;
            out_row[n] = __float2half(norm * __half2float(w_norm[n])
                                          + __half2float(b_norm[n]));
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

    printf("=== layernorm(silu(inp @ W_gate)) ===\n");
    printf("  B=%d  M=%d  d=%d  I=%d\n", B, M, d, I);
    printf("  TILE_M=%d  TILE_K=%d  TILE_N=%d  BLOCK_THREADS=%d\n\n",
           TILE_M, TILE_K, TILE_N, BLOCK_THREADS);

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

    const int shm_bytes = TILE_M * TILE_K * sizeof(float)
                        + TILE_K * TILE_N * sizeof(half)
                        + TILE_M * I * sizeof(float)
                        + TILE_M * 2 * sizeof(float2);
    dim3 grid((M + TILE_M - 1) / TILE_M, B);
    checkCuda(cudaFuncSetAttribute(layernorm_silu_matmul_kernel,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   shm_bytes));

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
        printf("               — %-4d  %+15.8f  %+15.8f\n", i, h_out_ref[i], h_out_f32[i]);

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
