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
    qreal alpha;
    qreal beta;
    int targetQubit;
    int controlQubit;
    GateType type;
};

__device__ __constant__ double recRoot2 = 0.70710678118654752440084436210485; // WARNING
__constant__ KernelGate deviceGates[1024];

const int THREAD_DEP = 7; // 1 << THREAD_DEP threads per block
const int MAX_GATE = 1024;
extern __shared__ qreal real[1<<LOCAL_QUBIT_SIZE];
extern __shared__ qreal imag[1<<LOCAL_QUBIT_SIZE];

__device__ inline void rotateXGate(int lo, int hi, qreal alpha, qreal beta) {
    qreal loReal = real[lo];
    qreal loImag = imag[lo];
    qreal hiReal = real[hi];
    qreal hiImag = imag[hi];
    real[lo] = alpha * loReal + beta * hiImag;
    imag[lo] = alpha * loImag - beta * hiReal;
    real[hi] = alpha * hiReal + beta * loImag;
    imag[hi] = alpha * hiImag - beta * loReal;
}

__device__ inline void rotateYGate(int lo, int hi, qreal alpha, qreal beta) {
    qreal loReal = real[lo];
    qreal loImag = imag[lo];
    qreal hiReal = real[hi];
    qreal hiImag = imag[hi];
    real[lo] = alpha * loReal - beta * hiReal;
    imag[lo] = alpha * loImag - beta * hiImag;
    real[hi] = beta * loReal + alpha * hiReal;
    imag[hi] = beta * loImag + alpha * hiImag;
}

__device__ inline void rotateZGate(int lo, int hi, qreal alpha, qreal beta){
    qreal loReal = real[lo];
    qreal loImag = imag[lo];
    qreal hiReal = real[hi];
    qreal hiImag = imag[hi];
    real[lo] = alpha * loReal + beta * loImag;
    imag[lo] = alpha * loImag - beta * loReal;
    real[hi] = alpha * hiReal - beta * hiImag;
    imag[hi] = alpha * hiImag + beta * hiReal;
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
__global__ void run(ComplexArray a, int numGates) {
    int idx = blockIdx.x * blockSize + threadIdx.x;
    int n = 1 << LOCAL_QUBIT_SIZE; // no need for long long
    qindex prefix = blockIdx.x << LOCAL_QUBIT_SIZE;
    // fetch data
    for (qindex i = idx; i < n; i += blockSize) {
        real[i] = a.real[i | prefix];
        imag[i] = a.imag[i | prefix];
    }
    __syncthreads();
    // compute
    for (int i = 0; i < numGates; i++) {
        int controlQubit = deviceGates[i].controlQubit;
        int targetQubit = deviceGates[i].targetQubit;
        if (controlQubit != -1) {
            int m = 1 << (LOCAL_QUBIT_SIZE - 2);
            int maskTarget = (1 << targetQubit) - 1;
            int maskControl = (1 << controlQubit) - 1;
            for (int j = idx; j < m; j += blockSize) {
                int lo;
                if (controlQubit > targetQubit) {
                    lo = ((j >> targetQubit) << (targetQubit + 1)) | (j & maskTarget);
                    lo = ((lo >> controlQubit) << (controlQubit + 1)) | (lo & maskControl) | (1 << controlQubit);
                } else {
                    lo = ((j >> controlQubit) << (controlQubit + 1)) | (j & maskControl)  | (1 << controlQubit);
                    lo = ((lo >> targetQubit) << (targetQubit + 1)) | (lo & maskTarget);
                }
                int hi = lo | (1 << targetQubit);
                switch (deviceGates[i].type) {
                    case GateCNot: {
                        pauliXGate(lo, hi);
                        break;
                    }
                    case GateCPauliY: {
                        pauliYGate(lo, hi);
                        break;
                    }
                    case GateCRotateX: {
                        rotateXGate(lo, hi, deviceGates[i].alpha, deviceGates[i].beta);
                        break;
                    }
                    case GateCRotateY: {
                        rotateYGate(lo, hi, deviceGates[i].alpha, deviceGates[i].beta);
                        break;
                    }
                    case GateCRotateZ: {
                        rotateZGate(lo, hi, deviceGates[i].alpha, deviceGates[i].beta);
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
                int lo = ((j >> targetQubit) << (targetQubit + 1)) | (j & maskTarget);
                int hi = lo | (1 << targetQubit);
                switch (deviceGates[i].type) {
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
                    case GateRotateX: {
                        rotateXGate(lo, hi, deviceGates[i].alpha, deviceGates[i].beta);
                        break;
                    }
                    case GateRotateY: {
                        rotateYGate(lo, hi, deviceGates[i].alpha, deviceGates[i].beta);
                        break;
                    }
                    case GateRotateZ: {
                        rotateZGate(lo, hi, deviceGates[i].alpha, deviceGates[i].beta);
                        break;
                    }
                    case GateS: {
                        sGate(hi);
                        break;
                    }
                    case GateT: {
                        tGate(hi);
                        break;
                    }
                    default: {
                        assert(false);
                    }
                }
            }
        }
        __syncthreads();
        // if (idx == 0) {
        //     printf("%d(%d->%d %d):\n", i, controlQubit, targetQubit, deviceGates[i].type);
        //     for (int j = 0; j < 32; j++) {
        //         printf("(%f %f) ", real[j], imag[j]);
        //         if (j % 4 == 3)
        //             printf("\n");
        //     }
        // }
    }
    // write back
    for (qindex i = idx; i < n; i += blockSize) {
        a.real[i | prefix] = real[i];
        a.imag[i | prefix] = imag[i];
    }
}


void kernelExecSmall(ComplexArray& deviceStateVec, int numQubits, const vector<Gate>& gates) {
    KernelGate hostGates[gates.size()];
    assert(gates.size() < MAX_GATE);
    for (size_t i = 0; i < gates.size(); i++) {
        switch (gates[i].type) {
            case GateRotateX: // no break
            case GateCRotateX: {
                hostGates[i].alpha = gates[i].mat[0][0].real;
                hostGates[i].beta = gates[i].mat[0][1].imag;
                break;
            }
            case GateRotateY: // no break
            case GateCRotateY: {
                hostGates[i].alpha = gates[i].mat[0][0].real;
                hostGates[i].beta = gates[i].mat[1][0].real;
                break;
            }
            case GateRotateZ: // no break
            case GateCRotateZ: {
                hostGates[i].alpha = gates[i].mat[0][0].real;
                hostGates[i].beta = - gates[i].mat[0][0].imag;
                break;
            }
            default: {
                hostGates[i].alpha = hostGates[i].beta = 0;
            }
        }
        hostGates[i].targetQubit = gates[i].targetQubit;
        hostGates[i].controlQubit = gates[i].controlQubit;
        hostGates[i].type = gates[i].type;
    }
    checkCudaErrors(cudaMemcpyToSymbol(deviceGates, hostGates, sizeof(hostGates)));
    run<1<<THREAD_DEP><<<(1<<numQubits)>>LOCAL_QUBIT_SIZE, 1<<THREAD_DEP>>>(deviceStateVec, gates.size());
    checkCudaErrors(cudaDeviceSynchronize()); // WARNING: for time measure!
}