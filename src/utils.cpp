#include "utils.h"

#include <cstring>
#include "logger.h"

namespace MyGlobalVars {
int numGPUs;
int bit;
std::unique_ptr<cudaStream_t[]> streams;
std::unique_ptr<cublasHandle_t[]> blasHandles;
void init() {
    checkCudaErrors(cudaGetDeviceCount(&numGPUs));
    Logger::add("Total GPU: %d", numGPUs);
    bit = -1;
    int x = numGPUs;
    while (x) {
        bit ++;
        x >>= 1;
    }
    if ((1 << bit) != numGPUs) {
        printf("GPU num must be power of two! %d %d\n", numGPUs, bit);
        exit(1);
    }

    streams = std::make_unique<cudaStream_t[]>(MyGlobalVars::numGPUs);
    blasHandles = std::make_unique<cublasHandle_t[]>(MyGlobalVars::numGPUs);
    for (int i = 0; i < numGPUs; i++) {
        checkCudaErrors(cudaSetDevice(i));
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        Logger::add("[%d] %s", i, prop.name);
        for (int j = 0; j < numGPUs; j++)
            if (i != j && (i ^ j) < 4) {
                checkCudaErrors(cudaDeviceEnablePeerAccess(j, 0));
            }
        checkCudaErrors(cudaStreamCreate(&streams[i]);)
        checkBlasErrors(cublasCreate(&blasHandles[i]));
        checkBlasErrors(cublasSetStream(blasHandles[i], streams[i]));
    }
}
};

qreal zero_wrapper(qreal x) {
    const qreal eps = 1e-14;
    if (x > -eps && x < eps) {
        return 0;
    } else {
        return x;
    }
}

qComplex operator * (const qComplex& a, const qComplex& b) {
    return make_qComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

qComplex operator + (const qComplex& a, const qComplex& b) {
    return make_qComplex(a.x + b.x, a.y + b.y);
}

bool isUnitary(std::unique_ptr<qComplex[]>& mat, int n) {
    qComplex result[n * n];
    memset(result, 0, sizeof(result));
    for (int k = 0; k < n; k++)
        #pragma omp parallel for
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                qComplex v1 = mat[k * n + i];
                v1.y = - v1.y;
                result[i * n + j] = result[i * n + j] + v1 * mat[k * n + j];
            }
    bool wa = 0;
    qreal eps = 1e-8;
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        qComplex val = result[i * n + i];
        if (fabs(val.x - 1) > eps || fabs(val.y) > eps) {
            wa = 1;
        }
        for (int j = 0; j < n; j++) {
            if (i == j)
                continue;
            qComplex val = result[i * n + j];
            if (fabs(val.x) > eps || fabs(val.y) > eps)
                wa = 1;
        }
    }
    if (wa) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++)
                printf("(%.2f %.2f) ", result[i * n + j].x, result[i * n + j].y);
            printf("\n");
        }
        exit(1);
    }
    return 1;
}

qComplex make_qComplex(qreal x) {
    return make_qComplex(x, 0.0);
}

bool operator < (const qComplex& a, const qComplex& b) {
        return a.x == b.x ? a.y < b.y : a.x < b.x;
}