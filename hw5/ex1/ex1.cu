#include <stdio.h>
#include <sys/time.h>

#define DataType float

struct timeval t_start, t_end;
void cputimer_start(){
  gettimeofday(&t_start, 0);
}
void cputimer_stop(const char* info){
  gettimeofday(&t_end, 0);
  double time = (1000000.0*(t_end.tv_sec-t_start.tv_sec) + t_end.tv_usec-t_start.tv_usec);
  printf("Timing - %s. \t\tElasped %.0f microseconds \n", info, time);
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


__global__ void tiled_gemm(DataType *A, DataType *B, DataType *C,
                          int numARows, int numAColumns,
                          int numBRows, int numBColumns,
                          int TILE_WIDTH) {
    
    extern __shared__ DataType shared_mem[];
    DataType* ds_A = shared_mem;
    DataType* ds_B = &shared_mem[TILE_WIDTH * TILE_WIDTH];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    DataType sum = 0.0;

    int numTiles = (numAColumns + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; t++) {
        
        if (row < numARows && (t * TILE_WIDTH + tx) < numAColumns) {
            ds_A[ty * TILE_WIDTH + tx] = A[row * numAColumns + t * TILE_WIDTH + tx];
        } else {
            ds_A[ty * TILE_WIDTH + tx] = 0.0;
        }

        
        if ((t * TILE_WIDTH + ty) < numBRows && col < numBColumns) {
            ds_B[ty * TILE_WIDTH + tx] = B[(t * TILE_WIDTH + ty) * numBColumns + col];
        } else {
            ds_B[ty * TILE_WIDTH + tx] = 0.0;
        }

        __syncthreads();

        if (row < numARows && col < numBColumns) {
            for (int k = 0; k < TILE_WIDTH && (t * TILE_WIDTH + k) < numAColumns; k++) {
                sum += ds_A[ty * TILE_WIDTH + k] * ds_B[k * TILE_WIDTH + tx];
            }
        }

        __syncthreads();
    }

    if (row < numARows && col < numBColumns) {
        C[row * numBColumns + col] = sum;
    }
}

int main(int argc, char **argv)
{

    DataType *hostA;     // The A matrix
    DataType *hostB;     // The B matrix
    DataType *hostC;     // The output C matrix
    DataType *resultRef; // The reference result
    DataType *hostC_tiled;
    DataType *deviceA;
    DataType *deviceB;
    DataType *deviceC;
    DataType *deviceC_tiled;
    int numARows;    // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows;    // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows;
    int numCColumns;

    // double start_time, end_time;

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
    hostC_tiled = (DataType *)malloc(size_C);
    resultRef = (DataType *)malloc(size_C);

    //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
    for (int i = 0; i < numARows * numAColumns; i++)
        hostA[i] = (DataType)rand() / (DataType)RAND_MAX;

    for (int i = 0; i < numBRows * numBColumns; i++)
        hostB[i] = (DataType)rand() / (DataType)RAND_MAX;
    cputimer_start();
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
    cputimer_stop("CPU");
    //@@ Insert code below to allocate GPU memory here
    cudaMalloc(&deviceA, size_A);
    cudaMalloc(&deviceB, size_B);
    cudaMalloc(&deviceC, size_C);
    cudaMalloc(&deviceC_tiled, size_C);
    //@@ Insert code to below to Copy memory to the GPU here
    
    
    cudaMemcpy(deviceA, hostA, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, size_B, cudaMemcpyHostToDevice);
    

    //@@ Initialize the grid and block dimensions here
    dim3 threads_per_block(16, 16);
    dim3 number_of_blocks((numCRows + threads_per_block.x - 1) / threads_per_block.x,
                          (numCColumns + threads_per_block.y - 1) / threads_per_block.y);

    //@@ Launch the GPU Kernel here
    cputimer_start();
    gemm<<<number_of_blocks, threads_per_block>>>(deviceA, deviceB, deviceC, numARows,
                                                  numAColumns, numBRows, numBColumns);
    cputimer_stop("CUDA gemm");
   
    cudaDeviceSynchronize();
	
    //@@ Copy the GPU memory back to the CPU here
   
    cudaMemcpy(hostC, deviceC, size_C, cudaMemcpyDeviceToHost);
    
    //@@ Insert code below to compare the output with the reference
    DataType max_error = 0;
    for (int i = 0; i < numCRows * numCColumns; ++i)
    {
        if (fabs(hostC[i] - resultRef[i]) > max_error)
            max_error = fabs(hostC[i] - resultRef[i]);
    }

    if (max_error > 1e-3)
        printf("Problem! The Max Error of %.5f is NOT within acceptable bounds.\n", max_error);
    else
        printf("The Max Error of %.5f is within acceptable bounds.\n", max_error);
    
    

   
    int TILE_X[] = {8, 16, 32};
    int TILE_Y[] = {16, 16, 16};
    
 
    
    printf("\n------- Testing different tile sizes -------\n");
    for(int i = 0; i < 3; i++) {
        int TILE_WIDTH = ceil(sqrt(TILE_X[i]*TILE_Y[i]));
        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
        dim3 dimGrid((numBColumns + TILE_WIDTH - 1) / TILE_WIDTH,
                     (numARows + TILE_WIDTH - 1) / TILE_WIDTH);
        size_t sharedMemSize = 2 * TILE_WIDTH * TILE_WIDTH * sizeof(DataType);

        
        // printf("\nTesting TILE_WIDTH = %d:\n", TILE_WIDTH);
        // printf("Grid Dim: (%d, %d)\n", dimGrid.x, dimGrid.y);
        // printf("Block Dim: (%d, %d)\n", dimBlock.x, dimBlock.y);
        // printf("Shared Memory: %zu bytes\n", sharedMemSize);

        printf("CUDA kernel (Tiled) with tile size %dx%d\n", TILE_X[i], TILE_Y[i]);
        cputimer_start();
        tiled_gemm<<<dimGrid, dimBlock, sharedMemSize>>>(
            deviceA, deviceB, deviceC_tiled,
            numARows, numAColumns,
            numBRows, numBColumns,
            TILE_WIDTH
        );
       
        cputimer_stop("");
        
        cudaDeviceSynchronize();

       
        cudaMemcpy(hostC_tiled, deviceC_tiled, size_C, cudaMemcpyDeviceToHost);

        
        DataType max_error = 0;
        for (int j = 0; j < numCRows * numCColumns; ++j) {
            if (fabs(hostC_tiled[j] - resultRef[j]) > max_error)
                max_error = fabs(hostC_tiled[j] - resultRef[j]);
        }

        printf("Max Error = %.5f %s\n", 
            max_error,
            max_error > 1e-3 ? "NOT ACCEPTABLE" : "ACCEPTABLE");
    }

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    cudaFree(deviceC_tiled);
    //@@ Free the CPU memory here
    free(hostA);
    free(hostB);
    free(hostC);
    free(hostC_tiled);
    free(resultRef);

    return 0;
}
