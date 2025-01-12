#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

struct timeval t_start, t_end;
void cputimer_start(){
  gettimeofday(&t_start, 0);
}
void cputimer_stop(const char* info){
  gettimeofday(&t_end, 0);
  double time = (1000000.0*(t_end.tv_sec-t_start.tv_sec) + t_end.tv_usec-t_start.tv_usec);
  printf("Timing - %s. \t\tElasped %.0f microseconds \n", info, time);
}

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

using namespace nvcuda;

__global__ void convertFp32ToFp16(half *out, float *in, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(in[idx]);
    }
}

__global__ void convertFp16ToFp32(float *out, half *in, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __half2float(in[idx]);
    }
}

__global__ void gemm(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// WMMA kernel
__global__ void wmma_gemm(half *a, half *b, float *c, 
                         int M, int N, int K,
                         int WARPS_ROW) {
#if __CUDA_ARCH__ >= 700

    const int WARPS_COL = WARPS_ROW;  
    
    const int BLOCK_ROW_TILES = WARPS_ROW * WMMA_M;
    const int BLOCK_COL_TILES = WARPS_COL * WMMA_N;

  
    int warpId = threadIdx.x / warpSize;
    int warpM = warpId / WARPS_ROW;
    int warpN = warpId % WARPS_COL;
    
    
    int blockM = blockIdx.y * BLOCK_ROW_TILES;
    int blockN = blockIdx.x * BLOCK_COL_TILES;

    
    int warpRow = blockM + (warpM * WMMA_M);
    int warpCol = blockN + (warpN * WMMA_N);

    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, 
                  wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                  wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    
    wmma::fill_fragment(c_frag, 0.0f);

    
    for (int k = 0; k < K; k += WMMA_K) {
        if (warpRow < M && k < K) {
            
            wmma::load_matrix_sync(a_frag, a + warpRow * K + k, K);
        }
        
        if (k < K && warpCol < N) {
            
            wmma::load_matrix_sync(b_frag, b + k * N + warpCol, N);
        }

       
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

   
    if (warpRow < M && warpCol < N) {
        wmma::store_matrix_sync(c + warpRow * N + warpCol, c_frag, N, 
                              wmma::mem_row_major);
    }
#endif
}

int main() {
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major < 7) {
        printf("Error: Tensor Cores are not available on this device\n");
        return -1;
    }

    
    int M = 4096;
    int N = 4096;
    int K = 4096;

    
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    float *h_CCPU = new float[M * N];

    
    for (int i = 0; i < M * K; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = rand() / (float)RAND_MAX;
    }

    half *d_A_half, *d_B_half;
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    cudaMalloc(&d_A_half, M * K * sizeof(half));
    cudaMalloc(&d_B_half, K * N * sizeof(half));

  
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    
    int threads = 256;
    int blocks = (M * K + threads - 1) / threads;
    convertFp32ToFp16<<<blocks, threads>>>(d_A_half, d_A, M * K);
    blocks = (K * N + threads - 1) / threads;
    convertFp32ToFp16<<<blocks, threads>>>(d_B_half, d_B, K * N);

    const int THREADS_PER_BLOCK = 128;  
    const int WARPS_PER_BLOCK = THREADS_PER_BLOCK / 32;
    const int WARPS_ROW = (int)sqrt(WARPS_PER_BLOCK); 
    
    
    const int BLOCK_ROW_TILES = WARPS_ROW * WMMA_M;
    const int BLOCK_COL_TILES = WARPS_ROW * WMMA_N;

    dim3 blockDim(THREADS_PER_BLOCK, 1);
    dim3 gridDim(
        (N + BLOCK_COL_TILES - 1) / BLOCK_COL_TILES,
        (M + BLOCK_ROW_TILES - 1) / BLOCK_ROW_TILES
    );

    
    printf("\nConfiguration:\n");
    printf("Threads per block: %d\n", THREADS_PER_BLOCK);
    printf("Warps per block: %d (%dx%d)\n", WARPS_PER_BLOCK, WARPS_ROW, WARPS_ROW);
    printf("Block tiles: %dx%d\n", BLOCK_ROW_TILES, BLOCK_COL_TILES);
    printf("Grid: %dx%d\n", gridDim.x, gridDim.y);
    printf("Matrix: %dx%d * %dx%d\n", M, K, K, N);

    
  

    cputimer_start();
    wmma_gemm<<<gridDim, blockDim>>>(d_A_half, d_B_half, d_C, M, N, K, WARPS_ROW);
    cputimer_stop("wmma");

    
   

    
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cputimer_start();
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += h_A[i * K + k] * h_B[k * N + j];
            }
            h_CCPU[i * N + j] = sum;
        }
    }
    cputimer_stop("cpu");
    
    bool correct = true;
    int errors = 0;
    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    float sum_diff = 0.0f;

    for (int i = 0; i < M * N; i++) {
        float diff = fabs(h_C[i] - h_CCPU[i]);
        sum_diff += diff;
        max_diff = max(max_diff, diff);
        if (diff > 0.1f) {
            errors++;
            if (errors < 10) {
                printf("Mismatch at %d: GPU=%f, CPU=%f (diff=%f)\n", 
                       i, h_C[i], h_CCPU[i], diff);
            }
            correct = false;
        }
    }

    avg_diff = sum_diff / (M * N);
    printf("\nAccuracy Analysis:\n");
    printf("Max difference: %f\n", max_diff);
    printf("Average difference: %f\n", avg_diff);
    printf("Total errors: %d\n", errors);
    printf("Error rate: %f%%\n", (float)errors * 100 / (M * N));
    printf("total warps: %d\n", M * N / (WMMA_M * WMMA_N));
    printf("Matrix multiplication %s\n", correct ? "PASSED" : "FAILED");
   
    if (!correct) printf("Total errors: %d\n", errors);

    
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_CCPU;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A_half);
    cudaFree(d_B_half);

    // invoke the gemm kernel
    
    
    return 0;
}