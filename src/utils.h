#pragma once

#include <cstdio>
#include <cuComplex.h>
#include <cuda.h>
#include <cutt.h>
#include <cuda_runtime.h>
#include <memory>
#include <cublas_v2.h>

#if USE_MPI
#include <mpi.h>
#include <nccl.h>
#endif

#ifdef USE_DOUBLE
typedef double qreal;
typedef long long qindex;
typedef cuDoubleComplex qComplex;
#define make_qComplex make_cuDoubleComplex
#define MPI_Complex MPI_C_DOUBLE_COMPLEX
#define cublasGEMM cublasZgemm
#define NCCL_FLOAT_TYPE ncclDouble
#else
typedef float qreal;
typedef long long qindex;
typedef cuFloatComplex qComplex;
#define make_qComplex make_cuFloatComplex
#define MPI_Complex MPI_C_COMPLEX
#define cublasGEMM cublasCgemm
#define NCCL_FLOAT_TYPE ncclFloat
#endif

#define SERIALIZE_STEP(x) { *reinterpret_cast<decltype(x)*>(arr + cur) = x; cur += sizeof(x); }
#define DESERIALIZE_STEP(x) { x = *reinterpret_cast<const decltype(x)*>(arr + cur); cur += sizeof(x); }

#define SERIALIZE_VECTOR(x, result) { \
    auto tmp_chars = reinterpret_cast<const unsigned char*>(x.data()); \
    result.insert(result.end(), tmp_chars, tmp_chars + sizeof(decltype(x)::value_type) * x.size()); \
}

#define DESERIALIZE_VECTOR(x, size) { \
    x.resize(size); \
    auto tmp_size = sizeof(decltype(x)::value_type) * size; \
    memcpy(x.data(), arr + cur, tmp_size); \
    cur += tmp_size; \
}


#define UNREACHABLE() { \
    printf("file %s line %i: unreachable!\n", __FILE__, __LINE__); \
    fflush(stdout); \
    exit(1); \
}

const int LOCAL_QUBIT_SIZE = 10; // is hardcoded
const int BLAS_MAT_LIMIT = BLAS_MAT_LIMIT_DEFINED;
const int THREAD_DEP = THREAD_DEP_DEFINED; // 1 << THREAD_DEP threads per block
const int COALESCE_GLOBAL = COALESCE_GLOBAL_DEFINED;
const int MAX_GATE = 600;
const int MIN_MAT_SIZE = MIN_MAT_SIZE_DEFINED;

static const char *cublasGetErrorString(cublasStatus_t error) {
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
    UNREACHABLE()
}

static const char *cuttGetErrorString(cuttResult error) {
    switch (error) {
        case CUTT_INVALID_PLAN:
            return "CUTT_INVALID_PLAN";
        case CUTT_INVALID_PARAMETER:
            return "CUTT_INVALID_PARAMETER";
        case CUTT_INVALID_DEVICE:
            return "CUTT_INVALID_DEVICE";
        case CUTT_INTERNAL_ERROR:
            return "CUTT_INTERNAL_ERROR";
        case CUTT_UNDEFINED_ERROR:
            return "CUTT_UNDEFINED_ERROR";
        default:
            return "<unknown>";
    }
    UNREACHABLE()
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
      fprintf(stderr, "%s in file %s, function %s, line %i: %04d %s\n", #stmt, __FILE__, __FUNCTION__, __LINE__, err, cuttGetErrorString(err)); \
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

#define checkMPIErrors(stmt) {                          \
  int err = stmt;                                      \
  if(err != MPI_SUCCESS) {                          \
    fprintf(stderr, "%s in file %s, function %s, line %i: %04d\n", #stmt, __FILE__, __FUNCTION__, __LINE__, err); \
      exit(1); \
  }                                                 \
}

#define checkNCCLErrors(stmt) {                         \
  ncclResult_t err= stmt;                             \
  if (err != ncclSuccess) {                            \
    fprintf(stderr, "%s in file %s, function %s, line %i: %04d %s\n", #stmt, __FILE__, __FUNCTION__, __LINE__, err, ncclGetErrorString(err)); \
      exit(1); \
  }                                                 \
}

namespace MyGlobalVars {
    extern int numGPUs;
    extern int localGPUs;
    extern int bit;
    extern std::unique_ptr<cudaStream_t[]> streams;
    extern std::unique_ptr<cudaStream_t[]> streams_comm;
    extern std::unique_ptr<cublasHandle_t[]> blasHandles;
#if USE_MPI
    extern std::unique_ptr<ncclComm_t[]> ncclComms;
#endif
    void init();
};

namespace MyMPI {
    extern int rank;
    extern int commSize;
    extern int commBit;
    void init();
};

template<typename T>
int bitCount(T x) {
    int ret = 0;
    for (T i = x; i; i -= i & (-i)) {
        ret++;
    }
    return ret;
}

qreal zero_wrapper(qreal x);

qComplex operator * (const qComplex& a, const qComplex& b);
qComplex operator + (const qComplex& a, const qComplex& b);

bool isUnitary(std::unique_ptr<qComplex[]>& mat, int n);

qComplex make_qComplex(qreal x);
bool operator < (const qComplex& a, const qComplex& b);

int get_bit(int n);