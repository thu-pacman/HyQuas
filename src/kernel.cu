#include "kernel.h"
#include <cstdio>

#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)
inline void __checkCudaErrors(cudaError_t err, const char *file, const int line) {
    if (cudaSuccess != err)
    {
        fprintf(stderr, "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, line %i.\n", err, cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

const int SINGLE_SIZE_DEP = 5; // handle 1 << SINGLE_SIZE_DEP items per thread
const int THREAD_DEP = 8; // 1 << THREAD_DEP threads per block

__global__ void initZeroState(ComplexArray& a) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = (idx << SINGLE_SIZE_DEP); i < ((idx + 1) << SINGLE_SIZE_DEP); i++)
        a.real[i] = 1;
    for (int i = (idx << SINGLE_SIZE_DEP); i < ((idx + 1) << SINGLE_SIZE_DEP); i++)
        a.imag[i] = 0;
}


void kernelInit(ComplexArray& deviceStateVec, int numQubits) {
    int nVec = 1 << numQubits;
    checkCudaErrors(cudaMalloc(&deviceStateVec.real, sizeof(qreal) << numQubits));
    checkCudaErrors(cudaMalloc(&deviceStateVec.imag, sizeof(qreal) << numQubits));
    initZeroState<<<nVec>>(SINGLE_SIZE_DEP + THREAD_DEP), 1<<THREAD_DEP>>>(deviceStateVec);
    printf("init end\n");
}