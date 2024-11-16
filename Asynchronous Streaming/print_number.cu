#include <stdio.h>
#include <unistd.h>

__global__ void printNumber(int number)
{
  printf("%d\n", number);
}

int main()
{
  cudaStream_t* stream;
  stream = (cudaStream_t*)malloc(5 * sizeof(cudaStream_t));
  for (int i = 0; i < 5; ++i)
  {
    cudaStreamCreate(&stream[i]);
    printNumber<<<1, 1, 0, stream[i]>>>(i);
    cudaStreamDestroy(stream[i]);
  }
  cudaDeviceSynchronize();
}

