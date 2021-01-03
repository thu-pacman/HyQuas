#pragma once

#include <cstdio>
#include <cuComplex.h>
#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>
#include <cublas_v2.h>

#ifdef USE_DOUBLE
typedef double qreal;
typedef int qindex;
typedef cuDoubleComplex qComplex;
#define make_qComplex make_cuDoubleComplex
#define MPI_Complex MPI_C_DOUBLE_COMPLEX
#define cublasGEMM cublasDgemm
#else
typedef float qreal;
typedef int qindex;
typedef cuFloatComplex qComplex;
#define make_qComplex make_cuFloatComplex
#define MPI_Complex MPI_C_COMPLEX
#define cublasGEMM cublasSgemm
#endif

#define SERIALIZE_STEP(x) { *reinterpret_cast<decltype(x)*>(arr + cur) = x; cur += sizeof(x); }
#define DESERIALIZE_STEP(x) { x = *reinterpret_cast<const decltype(x)*>(arr + cur); cur += sizeof(x); }

#define UNREACHABLE() { \
    printf("file %s line %i: unreachable!\n", __FILE__, __LINE__); \
    fflush(stdout); \
    exit(1); \
}

const int LOCAL_QUBIT_SIZE = 10; // is hardcoded
const int THREAD_DEP = 7; // 1 << THREAD_DEP threads per block
const int MAX_GATE = 600;

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

namespace MyGlobalVars {
    extern int numGPUs;
    extern int bit;
    extern std::unique_ptr<cudaStream_t[]> streams;
    void init();
};

struct Complex {
    qreal real;
    qreal imag;

    Complex() = default;
    Complex(qreal x): real(x), imag(0) {}
    Complex(qreal real, qreal imag): real(real), imag(imag) {}
    Complex(const Complex&) = default;
    Complex(const qComplex& x): real(x.x), imag(x.y) {}
    operator qComplex() { return make_qComplex(real, imag); }

    Complex& operator = (qreal x) {
        real = x;
        imag = 0;
        return *this;
    }
    bool operator < (const Complex& b) const {
        return real == b.real ? imag < b.imag : real < b.real;
    }
    qreal len() const { return real * real + imag * imag; }
};

struct ComplexArray {
    qreal *real;
    qreal *imag;
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