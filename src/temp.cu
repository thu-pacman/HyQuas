#include "kernel.h"
#include <cstdio>
#include <assert.h>
using namespace std;

#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)
inline void __checkCudaErrors(cudaError_t err, const char *file, const int line) {
    if (cudaSuccess != err)
    {
        fprintf(stderr, "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, line %i.\n", err, cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

struct KernelGate {
    Complex alpha;
    Complex beta;
    int targetQubit;
    int controlQubit;
    GateType type;
};

__device__ __constant__ double recRoot2 = 1.4142135623730950488016887242097; // WARNING
__constant__ KernelGate deviceGates[1024];

const int THREAD_DEP = 7; // 1 << THREAD_DEP threads per block
const int MAX_GATE = 1024;
extern __shared__ qreal real[1<<LOCAL_QUBIT_SIZE];
extern __shared__ qreal imag[1<<LOCAL_QUBIT_SIZE];

__device__ inline void alphaBetaGate(int lo, int hi, Complex alpha, Complex beta) {
    qreal loReal = real[lo];
    qreal loImag = imag[lo];
    qreal hiReal = real[hi];
    qreal hiImag = imag[hi];
    real[lo] = alpha.real * loReal - alpha.imag * loImag - beta.real * hiReal - beta.imag * hiImag;
    imag[lo] = alpha.real * loImag + alpha.imag * loReal - beta.real * hiImag + beta.imag * hiReal;
    real[hi] = beta.real * loReal - beta.imag * loImag + alpha.real * hiReal + alpha.imag * hiImag;
    imag[hi] = beta.real * loImag + beta.imag * loReal + alpha.real * hiImag - alpha.imag * hiReal;
}

__device__ inline void hadamardGate(int lo, int hi) {
    qreal loReal = real[lo];
    qreal loImag = imag[lo];
    qreal hiReal = real[hi];
    qreal hiImag = imag[hi];
    real[lo] = recRoot2 * (loReal + hiReal);
    imag[lo] = recRoot2 * (loImag + hiImag);
    real[hi] = recRoot2 * (loReal - hiReal);
    imag[hi] = recRoot2 * (loImag - hiImag);
}

__device__ inline void pauliXGate(int lo, int hi) {
    qreal Real = real[lo];
    qreal Imag = imag[lo];
    real[lo] = real[hi];
    imag[lo] = imag[hi];
    real[hi] = Real;
    imag[hi] = Imag;
}

__device__ inline void pauliYGate(int lo, int hi) {
    qreal loReal = real[lo];
    qreal loImag = imag[lo];
    qreal hiReal = real[hi];
    qreal hiImag = imag[hi];
    real[lo] = hiImag;
    imag[lo] = -hiReal;
    real[hi] = -loImag;
    imag[hi] = loReal;
}

__device__ inline void pauliZGate(int hi) {
    real[hi] = -real[hi];
    imag[hi] = -imag[hi];
}

__device__ inline void sGate(int hi) {
    qreal hiReal = real[hi];
    qreal hiImag = imag[hi];
    real[hi] = -hiImag;
    imag[hi] = hiReal;
}

__device__ inline void tGate(int hi) {
    qreal hiReal = real[hi];
    qreal hiImag = imag[hi];
    real[hi] = recRoot2 * (hiReal - hiImag);
    imag[hi] = recRoot2 * (hiReal + hiImag);
}

template <unsigned int blockSize>
__global__ void run(ComplexArray a, int numGates, KernelGate* gates) {
    int idx = blockIdx.x * blockSize + threadIdx.x;
    int n = 1 << LOCAL_QUBIT_SIZE; // no need for long long
    // fetch data
    for (int i = idx; i < n; i += blockSize) {
        real[i] = a.real[i];
        imag[i] = a.imag[i];
    }
    __syncthreads();
    return;
    // compute
    for (int i = 0; i < numGates; i++) {
        int controlQubit = gates[i].controlQubit;
        int targetQubit = gates[i].targetQubit;
        if (controlQubit != -1) {
            int m = 1 << (LOCAL_QUBIT_SIZE - 2);
            int maskTarget = (1 << targetQubit) - 1;
            int maskControl = (1 << controlQubit) - 1;
            for (int j = idx; j < m; j += blockSize) {
                int lo;
                if (controlQubit > gates[i].targetQubit) {
                    lo = ((i >> targetQubit) << (targetQubit + 1)) | (i & maskTarget);
                    lo = ((lo >> controlQubit) << (controlQubit + 1) | 1) | (lo & maskControl);
                } else {
                    lo = ((i >> controlQubit) << (controlQubit + 1) | 1) | (i & maskTarget);
                    lo = ((lo >> targetQubit) << (targetQubit + 1)) | (lo & maskControl);
                }
                int hi = lo | (1 << targetQubit);
                switch (gates[i].type) {
                    case GateCNot: {
                        pauliXGate(lo, hi);
                        break;
                    }
                    case GateCPauliY: {
                        pauliYGate(lo, hi);
                        break;
                    }
                    case GateCRotateX: // no break
                    case GateCRotateY: // no break
                    case GateCRotateZ: {
                        alphaBetaGate(lo, hi, gates[i].alpha, gates[i].beta);
                        break;
                    }
                    default: {
                        assert(false);
                    }
                }
            }
        } else {
            int m = 1 << (LOCAL_QUBIT_SIZE - 1);
            int maskTarget = (1 << targetQubit) - 1;
            for (int j = idx; j < m; j += blockSize) {
                int lo = ((i >> targetQubit) << (targetQubit + 1)) | (i & maskTarget);
                int hi = lo | (1 << targetQubit);
                switch (gates[i].type) {
                    case GateHadamard: {
                        hadamardGate(lo, hi);
                        break;
                    }
                    case GatePauliX: {
                        pauliXGate(lo, hi);
                        break;
                    }
                    case GatePauliY: {
                        pauliYGate(lo, hi);
                        break;
                    }
                    case GatePauliZ: {
                        pauliZGate(hi);
                        break;
                    }
                    case GateRotateX:
                    case GateRotateY:
                    case GateRotateZ: {
                        alphaBetaGate(lo, hi, gates[i].alpha, gates[i].beta);
                        break;
                    }
                }
            }
        }
        __syncthreads();
    }
    // write back
    for (int i = 0; i < n; i += blockSize) {
        a.real[i] = real[i];
        a.imag[i] = imag[i];
    }
}


void kernelExecSmall(ComplexArray& deviceStateVec, int numQubits, const vector<Gate>& gates) {
    KernelGate hostGates[gates.size()];
    assert(gates.size() < MAX_GATE);
    for (int i = 0; i < gates.size(); i++) {
        hostGates[i].alpha = gates[i].mat[0][0];
        hostGates[i].beta = gates[i].mat[1][0];
        hostGates[i].targetQubit = gates[i].targetQubit;
        hostGates[i].controlQubit = gates[i].controlQubit;
        hostGates[i].type = gates[i].type;
    }
    checkCudaErrors(cudaMemcpyToSymbol(deviceGates, hostGates, gates.size()));
    run<1<<THREAD_DEP><<<(1<<numQubits)>>LOCAL_QUBIT_SIZE, 1<<THREAD_DEP>>>(deviceStateVec, gates.size(), deviceGates);
    checkCudaErrors(cudaDeviceSynchronize()); // WARNING: for time measure!
}