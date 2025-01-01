#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <sys/time.h>

#define DataType double

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len)
{
	//@@ Insert code to implement vector addition here
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len)
		out[i] = in1[i] + in2[i];
}

double get_time_in_seconds()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + tv.tv_usec / 1000000.0;
}

int main(int argc, char **argv)
{

	int inputLength;
	double start_time, end_time;
	DataType *hostInput1;
	DataType *hostInput2;
	DataType *hostOutput;
	DataType *resultRef;
	DataType *deviceInput1;
	DataType *deviceInput2;
	DataType *deviceOutput;

	//@@ Insert code below to read in inputLength from args
	inputLength = atoi(argv[1]);
	printf("The input length is %d\n", inputLength);

	//@@ Insert code below to read in segment size (S_seg) from args
	int S_seg = 1024; 
	if (argc > 2) {
		S_seg = atoi(argv[2]);
	}
	printf("The segment size is %d\n", S_seg);

	//@@ Insert code below to allocate Host memory for input and output
	size_t size = inputLength * sizeof(DataType);
	hostInput1 = (DataType *)malloc(size);
	hostInput2 = (DataType *)malloc(size);
	hostOutput = (DataType *)malloc(size);
	resultRef  = (DataType *)malloc(size);

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

	//non-stream
	start_time = get_time_in_seconds();
non-stream
	// Host -> Device
	cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice);

	int threads_per_block = 256;
	int number_of_blocks  = (inputLength + threads_per_block - 1) / threads_per_block;

	vecAdd<<<number_of_blocks, threads_per_block>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
	cudaDeviceSynchronize();

	cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);

	end_time = get_time_in_seconds();
	double nonStreamTime = end_time - start_time;
	printf("Non-stream version total time: %f seconds\n", nonStreamTime);

	// check answer
	double max_error_nonStream = 0;
	for (int i = 0; i < inputLength; ++i)
	{
		double err = fabs(hostOutput[i] - resultRef[i]);
		if (err > max_error_nonStream) {
			max_error_nonStream = err;
		}
	}
	printf("Non-stream version Max Error: %.5f\n", max_error_nonStream);

	//stream version
	//
	//@@ Insert code below to create multiple CUDA streams
	const int NUM_STREAMS = 4;
	cudaStream_t streams[NUM_STREAMS];
	for (int i = 0; i < NUM_STREAMS; i++)
	{
		cudaStreamCreate(&streams[i]);
	}

	start_time = get_time_in_seconds();

	int totalSegments = (inputLength + S_seg - 1) / S_seg;  

	for (int seg = 0; seg < totalSegments; seg++)
	{
		int streamIdx = seg % NUM_STREAMS;

		int offset = seg * S_seg;
		int len = ((offset + S_seg) < inputLength) ? S_seg : (inputLength - offset);
		size_t segmentSize = len * sizeof(DataType);

		cudaMemcpyAsync(deviceInput1 + offset,
		                hostInput1   + offset,
		                segmentSize,
		                cudaMemcpyHostToDevice,
		                streams[streamIdx]);

		cudaMemcpyAsync(deviceInput2 + offset,
		                hostInput2   + offset,
		                segmentSize,
		                cudaMemcpyHostToDevice,
		                streams[streamIdx]);

		// Kernel
		int threads_per_block_stream = 256;
		int number_of_blocks_stream  = (len + threads_per_block_stream - 1) / threads_per_block_stream;

		vecAdd<<<number_of_blocks_stream, threads_per_block_stream, 0, streams[streamIdx]>>>(
		    deviceInput1 + offset,
		    deviceInput2 + offset,
		    deviceOutput + offset,
		    len
		);

		cudaMemcpyAsync(hostOutput + offset,
		                deviceOutput + offset,
		                segmentSize,
		                cudaMemcpyDeviceToHost,
		                streams[streamIdx]);
	}

	// wait for all streams to be done
	for (int i = 0; i < NUM_STREAMS; i++)
	{
		cudaStreamSynchronize(streams[i]);
	}

	end_time = get_time_in_seconds();
	double streamTime = end_time - start_time;
	printf("Streaming version total time (%d segments, %d streams): %f seconds\n", 
	       totalSegments, NUM_STREAMS, streamTime);

	double max_error_stream = 0;
	for (int i = 0; i < inputLength; ++i)
	{
		double err = fabs(hostOutput[i] - resultRef[i]);
		if (err > max_error_stream) {
			max_error_stream = err;
		}
	}
	printf("Streaming version Max Error: %.5f\n", max_error_stream);


	printf("\n--- Performance comparison ---\n");
	printf("Non-stream time      = %f s\n", nonStreamTime);
	printf("Streaming time       = %f s\n", streamTime);
	printf("Speedup (non/stream) = %f\n", nonStreamTime / streamTime);
	printf("------------------------------\n");

	//@@ Insert code below to destroy multiple streams
	for (int i = 0; i < NUM_STREAMS; i++)
	{
		cudaStreamDestroy(streams[i]);
	}

	//@@ Free the GPU memory here
	cudaFree(deviceInput1);
	cudaFree(deviceInput2);
	cudaFree(deviceOutput);

	//@@ Free the CPU memory here
	free(hostInput1);
	free(hostInput2);
	free(hostOutput);
	free(resultRef);

	return 0;
}
