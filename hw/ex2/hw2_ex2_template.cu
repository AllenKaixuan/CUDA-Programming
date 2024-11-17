#include <stdio.h>
#include <sys/time.h>

#define DataType double

double get_time_in_seconds()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + tv.tv_usec / 1000000.0;
}

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                     int numAColumns, int numBRows, int numBColumns)
{
    //@@ Insert code to implement matrix multiplication here
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i < numARows && j < numBColumns)
    {
        DataType val = 0;
        for (int k = 0; k < numAColumns; k++)
        {
            val += A[i * numAColumns + k] * B[k * numBColumns + j];
        }
        C[i * numBColumns + j] = val;
    }
}

int main(int argc, char **argv)
{

    DataType *hostA;     // The A matrix
    DataType *hostB;     // The B matrix
    DataType *hostC;     // The output C matrix
    DataType *resultRef; // The reference result
    DataType *deviceA;
    DataType *deviceB;
    DataType *deviceC;
    int numARows;    // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows;    // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows;
    int numCColumns;

    double start_time, end_time;

    //@@ Insert code below to read in numARows, numAColumns, numBColumns from args
    numARows = atoi(argv[1]);
    numAColumns = atoi(argv[2]);
    numBColumns = atoi(argv[3]);
    numBRows = numAColumns;
    numCRows = numARows;
    numCColumns = numBColumns;

    printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

    //@@ Insert code below to allocate Host memory for input and output
    size_t size_A = numARows * numAColumns * sizeof(DataType);
    size_t size_B = numBRows * numBColumns * sizeof(DataType);
    size_t size_C = numCRows * numCColumns * sizeof(DataType);
    hostA = (DataType *)malloc(size_A);
    hostB = (DataType *)malloc(size_B);
    hostC = (DataType *)malloc(size_C);
    resultRef = (DataType *)malloc(size_C);

    //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
    for (int i = 0; i < numARows * numAColumns; i++)
        hostA[i] = (DataType)rand() / (DataType)RAND_MAX;

    for (int i = 0; i < numBRows * numBColumns; i++)
        hostB[i] = (DataType)rand() / (DataType)RAND_MAX;

    for (int i = 0; i < numCRows; ++i)
    {
        for (int j = 0; j < numCColumns; ++j)
        {
            DataType val = 0;
            for (int k = 0; k < numAColumns; ++k)
                val += hostA[i * numAColumns + k] * hostB[k * numBColumns + j];
            resultRef[i * numCColumns + j] = val;
        }
    }

    //@@ Insert code below to allocate GPU memory here
    cudaMalloc(&deviceA, size_A);
    cudaMalloc(&deviceB, size_B);
    cudaMalloc(&deviceC, size_C);

    //@@ Insert code to below to Copy memory to the GPU here
    start_time = get_time_in_seconds();
    cudaMemcpy(deviceA, hostA, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, size_B, cudaMemcpyHostToDevice);
    end_time = get_time_in_seconds();
	printf("data copy from host to device: %f seconds\n", end_time - start_time);
    //@@ Initialize the grid and block dimensions here
    dim3 threads_per_block(16, 16);
    dim3 number_of_blocks((numCRows + threads_per_block.x - 1) / threads_per_block.x,
                          (numCColumns + threads_per_block.y - 1) / threads_per_block.y);

    //@@ Launch the GPU Kernel here
    start_time = get_time_in_seconds();
    gemm<<<number_of_blocks, threads_per_block>>>(deviceA, deviceB, deviceC, numARows,
                                                  numAColumns, numBRows, numBColumns);
    end_time = get_time_in_seconds();
	printf("CUDA kernel: %f seconds\n", end_time - start_time);	
    //@@ Copy the GPU memory back to the CPU here
    start_time = get_time_in_seconds();
    cudaMemcpy(hostC, deviceC, size_C, cudaMemcpyDeviceToHost);
    end_time = get_time_in_seconds();
	printf("data copy from device to host: %f seconds\n", end_time - start_time);
    //@@ Insert code below to compare the output with the reference
    DataType max_error = 0;
    for (int i = 0; i < numCRows * numCColumns; ++i)
    {
        if (fabs(hostC[i] - resultRef[i]) > max_error)
            max_error = fabs(hostC[i] - resultRef[i]);
    }

    if (max_error > 1e-4)
        printf("Problem! The Max Error of %.5f is NOT within acceptable bounds.\n", max_error);
    else
        printf("The Max Error of %.5f is within acceptable bounds.\n", max_error);
    //@@ Free the GPU memory here
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    //@@ Free the CPU memory here
    free(hostA);
    free(hostB);
    free(hostC);
    free(resultRef);

    return 0;
}
