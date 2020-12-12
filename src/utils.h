#pragma once

#include <cstdio>
#include <cuComplex.h>
#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>

typedef float qreal;
typedef int qindex;
typedef cuFloatComplex qComplex;
#define make_qComplex make_cuFloatComplex
#define MPI_Complex MPI_C_COMPLEX

#define SERIALIZE_STEP(x) { *reinterpret_cast<decltype(x)*>(arr + cur) = x; cur += sizeof(x); }
#define DESERIALIZE_STEP(x) { x = *reinterpret_cast<const decltype(x)*>(arr + cur); cur += sizeof(x); }

const int LOCAL_QUBIT_SIZE = 10; // is hardcoded

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