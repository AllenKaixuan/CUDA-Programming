#include <stdio.h>

#define N 2048 * 2048 // Number of elements in each vector

/*
 * Optimize this already-accelerated codebase. Work iteratively,
 * and use nsys to support your work.
 *
 * Aim to profile `saxpy` (without modifying `N`) running under
 * 200,000 ns.
 *
 * Some bugs have been placed in this codebase for your edification.
 */

__global__ void saxpy(int * a, int * b, int * c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if ( tid < N )
        a[tid] = 2;
        b[tid] = 1;
        c[tid] = 2 * a[tid] + b[tid];
}

int main()
{
    int *a, *b, *c;
    int deviceId;
    cudaGetDevice(&deviceId);
    int size = N * sizeof (int); // The total number of bytes per vector

    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);

    // Initialize memory
    // for( int i = 0; i < N; ++i )
    // {
    //     a[i] = 2;
    //     b[i] = 1;
    //     c[i] = 0;
    // }

    int threads_per_block = 1024;
    int number_of_blocks = (N / threads_per_block) + 1;
    cudaMemPrefetchAsync(a, size, deviceId);
    cudaMemPrefetchAsync(b, size, deviceId);
    saxpy <<< number_of_blocks, threads_per_block >>> ( a, b, c );
    cudaDeviceSynchronize();
    // Print out the first and last 5 values of c for a quality check
    cudaMemPrefetchAsync(c, size, cudaCpuDeviceId);
    for( int i = 0; i < 5; ++i )
        printf("c[%d] = %d, ", i, c[i]);
    printf ("\n");
    for( int i = N-5; i < N; ++i )
        printf("c[%d] = %d, ", i, c[i]);
    printf ("\n");

    cudaFree( a ); cudaFree( b ); cudaFree( c );
}
