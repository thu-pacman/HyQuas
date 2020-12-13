#include "utils.h"
#include "logger.h"

namespace MyGlobalVars {
int numGPUs;
int bit;
std::unique_ptr<cudaStream_t[]> streams;

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
    for (int i = 0; i < numGPUs; i++) {
        checkCudaErrors(cudaSetDevice(i));
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        Logger::add("[%d] %s", i, prop.name);
        checkCudaErrors(cudaStreamCreate(&streams[i]);)
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
