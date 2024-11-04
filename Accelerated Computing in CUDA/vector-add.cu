#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

inline cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

void initWith(float num, float* a, int N)
{
    for (int i = 0; i < N; ++i)
    {
        a[i] = num;
    }
}

__global__ void addVectorsInto(float* result, float* a, float* b, int N)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += gridDim.x * blockDim.x)
    {
        result[i] = a[i] + b[i];
    }
}

void checkElementsAre(float target, float* array, int N)
{
    for (int i = 0; i < N; i++)
    {
        if (array[i] != target)
        {
            printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
			return;
        }
    }
    printf("SUCCESS! All values added correctly.\n");
}

int main()
{
    const int N = 2 << 20;
    size_t size = N * sizeof(float);

    float* a;
    float* b;
    float* c;
    cudaError_t sync_error;


    checkCuda(cudaMallocManaged(&a, size));
    checkCuda(cudaMallocManaged(&b, size));
    checkCuda(cudaMallocManaged(&c, size));


    initWith(3, a, N);
    initWith(4, b, N);
    initWith(0, c, N);
    int threads_per_block = 1024;
    int number_of_blocks = (N - 1 + threads_per_block) / threads_per_block;

    addVectorsInto <<<number_of_blocks, threads_per_block >> > (c, a, b, N);

    sync_error = cudaGetLastError();
    if (sync_error != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(sync_error));

    checkCuda(cudaDeviceSynchronize());
    checkElementsAre(7, c, N);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}
