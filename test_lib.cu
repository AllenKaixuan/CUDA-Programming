#include <iostream>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call) \
    if((call) != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return EXIT_FAILURE; \
    }

#define CHECK_CUSPARSE(call) \
    if((call) != CUSPARSE_STATUS_SUCCESS) { \
        std::cerr << "cuSPARSE error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return EXIT_FAILURE; \
    }

#define CHECK_CUBLAS(call) \
    if((call) != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return EXIT_FAILURE; \
    }

int main() {
    // Initialize cuSPARSE and cuBLAS
    cusparseHandle_t cusparseHandle;
    cublasHandle_t cublasHandle;

    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    // Your cuSPARSE and cuBLAS code here

    // Clean up
    CHECK_CUSPARSE(cusparseDestroy(cusparseHandle));
    CHECK_CUBLAS(cublasDestroy(cublasHandle));

    return EXIT_SUCCESS;
}