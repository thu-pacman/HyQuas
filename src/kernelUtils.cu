#include "kernel.h"
#include <assert.h>
#include <cstdio>
#include <cuda.h>

__global__ void isnanTestKernel(qComplex *data, int n) { // with grimDim == 1
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (isnan(data[i].x) || isnan(data[i].y)) {
            printf("nan at %d\n", i);
            asm("trap;");
        }
    }
}

__global__ void printVectorKernel(qComplex *data, int n) { // with gridDim == 1 && blockDim == 1
    for (int i = 0; i < n; i++)
        printf("(%f, %f)", data[i].x, data[i].y);
    printf("\n");
}

__global__ void whileTrueKernel() {
    while (true);
}

void isnanTest(qComplex* data, int n, cudaStream_t& stream) {
    isnanTestKernel<<<1, 32, 0, stream>>>(data, n / 32);
}

void printVector(qComplex* data, int n, cudaStream_t& stream) {
    printVectorKernel<<<1, 1, 0, stream>>>(data, n);
}

void whileTrue() {
    whileTrueKernel<<<1,1>>>();
}
