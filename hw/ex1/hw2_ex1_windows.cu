#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <Windows.h>

#define DataType double

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len)
{
	//@@ Insert code to implement vector addition here
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len)
		out[i] = in1[i] + in2[i];
}

int main(int argc, char **argv)
{

	int inputLength;
	DataType *hostInput1;
	DataType *hostInput2;
	DataType *hostOutput;
	DataType *resultRef;
	DataType *deviceInput1;
	DataType *deviceInput2;
	DataType *deviceOutput;
	// Timer
	LARGE_INTEGER frequency;
	LARGE_INTEGER start, end;
	double elapsedTime;
	QueryPerformanceFrequency(&frequency);

	//@@ Insert code below to read in inputLength from args
	inputLength = atoi(argv[1]);
	printf("The input length is %d\n", inputLength);

	//@@ Insert code below to allocate Host memory for input and output
	size_t size = inputLength * sizeof(DataType);
	hostInput1 = (DataType *)malloc(size);
	hostInput2 = (DataType *)malloc(size);
	hostOutput = (DataType *)malloc(size);
	resultRef = (DataType *)malloc(size);

	//@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
	for (int i = 0; i < inputLength; ++i)
	{
		hostInput1[i] = (DataType)rand() / (DataType)RAND_MAX;
		hostInput2[i] = (DataType)rand() / (DataType)RAND_MAX;
		resultRef[i] = hostInput1[i] + hostInput2[i];
	}

	//@@ Insert code below to allocate GPU memory here

	cudaMalloc(&deviceInput1, size);
	cudaMalloc(&deviceInput2, size);
	cudaMalloc(&deviceOutput, size);

	//@@ Insert code to below to Copy memory to the GPU here
	QueryPerformanceCounter(&start);
	cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice);
	QueryPerformanceCounter(&end);
	elapsedTime = (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart;
	printf("data copy from host to device: %f seconds\n", elapsedTime);

	//@@ Initialize the 1D grid and block dimensions here
	int threads_per_block = 256;
	int number_of_blocks = (inputLength + threads_per_block - 1) / threads_per_block;

	//@@ Launch the GPU Kernel here
	QueryPerformanceCounter(&start);
	vecAdd<<<number_of_blocks, threads_per_block>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
	QueryPerformanceCounter(&end);
	elapsedTime = (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart;
	printf("CUDA kernel: %f seconds\n", elapsedTime);

	//@@ Copy the GPU memory back to the CPU here
	QueryPerformanceCounter(&start);
	cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);
	QueryPerformanceCounter(&end);
	elapsedTime = (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart;
	printf("data copy from device to host: %f seconds\n", elapsedTime);

	//@@ Insert code below to compare the output with the reference
	DataType max_error = 0;
	for (int i = 0; i < inputLength; ++i)
	{
		if (fabs(hostOutput[i] - resultRef[i]) > max_error)
			max_error = fabs(hostOutput[i] - resultRef[i]);
	}

	if (max_error > 1e-5)
		printf("Problem! The Max Error of %.5f is NOT within acceptable bounds.\n", max_error);
	else
		printf("The Max Error of %.5f is within acceptable bounds.\n", max_error);
	//@@ Free the GPU memory here
	cudaFree(deviceInput1);
	cudaFree(deviceInput2);
	cudaFree(deviceOutput);
	//@@ Free the CPU memory here
	free(hostInput1);
	free(hostInput2);
	free(hostOutput);

	return 0;
}