#include <cublas_v2.h>
#include <cuComplex.h>
#include <cstdio>


static const char *cublasGetErrorString(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        default:
            return "<unknown>";
    }
    return "<unknown>";
}

#define checkCudaErrors(stmt) {                                 \
    cudaError_t err = stmt;                            \
    if (err != cudaSuccess) {                          \
      fprintf(stderr, "%s in file %s, function %s, line %i: %04d %s\n", #stmt, __FILE__, __FUNCTION__, __LINE__, err, cudaGetErrorString(err)); \
      exit(1); \
    }                                                  \
}

#define checkCuttErrors(stmt) {                                 \
    cuttResult err = stmt;                            \
    if (err != CUTT_SUCCESS) {                          \
      fprintf(stderr, "%s in file %s, function %s, line %i.\n", #stmt, __FILE__, __FUNCTION__, __LINE__); \
      exit(1); \
    }                                                  \
}

#define checkBlasErrors(stmt) { \
    cublasStatus_t err = stmt; \
    if (err != CUBLAS_STATUS_SUCCESS) {                          \
      fprintf(stderr, "%s in file %s, function %s, line %i: %04d %s\n", #stmt, __FILE__, __FUNCTION__, __LINE__, err, cublasGetErrorString(err)); \
      exit(1); \
    } \
}

int main() {
    int nq = N_QUBIT;
    cuDoubleComplex* arr;
    cuDoubleComplex* mat;
    cuDoubleComplex* result;
    checkCudaErrors(cudaMalloc(&arr, sizeof(cuDoubleComplex) << nq));
    checkCudaErrors(cudaMalloc(&mat, sizeof(cuDoubleComplex) * 1024 * 1024));
    checkCudaErrors(cudaMalloc(&result, sizeof(cuDoubleComplex) << nq));
    cublasHandle_t handle;
    checkBlasErrors(cublasCreate(&handle));
    // checkBlasErrors(cublasSetMathMode(handle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
    int numElements = 1 << nq;
    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0), beta = make_cuDoubleComplex(0.0, 0.0);
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    for (int K = 2; K < 1024; K <<= 1) {
        printf("K = %d\n", K);
        for (int i = 0; i < 100; i++) {
            checkCudaErrors(cudaEventRecord(start));
            
            checkBlasErrors(cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                K, numElements / K, K, // M, N, K
                &alpha, mat, K, // alpha, a, lda
                arr, K, // b, ldb
                &beta, result, K // beta, c, ldc
            ));

            float time;
            checkCudaErrors(cudaEventRecord(stop));
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time, start, stop);
            printf("%.10f ", time);
        }
        printf("\n");
    }
    return 0;
}
