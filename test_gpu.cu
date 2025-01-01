#include <stdio.h>
void printGPUProperties() {
    cudaDeviceProp prop;
    int count;
    
    // Get number of devices
    cudaGetDeviceCount(&count);
    printf("Found %d CUDA devices\n", count);
    
    // Get properties for each device
    for(int i = 0; i < count; i++) {
        cudaGetDeviceProperties(&prop, i);
        printf("\nDevice %d: %s\n", i, prop.name);
        printf("  Max grid dimensions: (%d, %d, %d)\n", 
            prop.maxGridSize[0], 
            prop.maxGridSize[1], 
            prop.maxGridSize[2]);
        printf("  Max block dimensions: (%d, %d, %d)\n", 
            prop.maxThreadsDim[0], 
            prop.maxThreadsDim[1], 
            prop.maxThreadsDim[2]);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    }
}

int main() {
    printGPUProperties();
    return 0;
}   