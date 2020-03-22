#include "kernel.h"
#include <cstdio>
#include <assert.h>
#include <map>
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
    char targetIsGlobal;  // 0-local 1-global
    char controlIsGlobal; // 0-local 1-global 2-not control 
};

const int THREAD_DEP = 7; // 1 << THREAD_DEP threads per block
const int MAX_GATE = 1024;
extern __shared__ qreal real[1<<LOCAL_QUBIT_SIZE];
extern __shared__ qreal imag[1<<LOCAL_QUBIT_SIZE];
extern __shared__ qindex blockBias;

__device__ __constant__ double recRoot2 = 0.70710678118654752440084436210485; // more elegant way?
__constant__ KernelGate deviceGates[1024];

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

__device__ inline void rotateZlo(int lo, qreal alpha, qreal beta) {
    qreal loReal = real[lo];
    qreal loImag = imag[lo];
    real[lo] = alpha * loReal + beta * loImag;
    imag[lo] = alpha * loImag - beta * loReal;
}

__device__ inline void rotateZhi(int hi, qreal alpha, qreal beta){
    qreal hiReal = real[hi];
    qreal hiImag = imag[hi];
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
__device__ void doCompute(int numGates) {
    for (int i = 0; i < numGates; i++) {
        int controlQubit = deviceGates[i].controlQubit;
        int targetQubit = deviceGates[i].targetQubit;
        char controlIsGlobal = deviceGates[i].controlIsGlobal;
        char targetIsGlobal = deviceGates[i].targetIsGlobal;
        if (!controlIsGlobal) {
            if (!targetIsGlobal) {
                int m = 1 << (LOCAL_QUBIT_SIZE - 2);
                int maskTarget = (1 << targetQubit) - 1;
                int maskControl = (1 << controlQubit) - 1;
                for (int j = threadIdx.x; j < m; j += blockSize) {
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
                assert(deviceGates[i].type == GateCRotateZ);
                bool isHighBlock = (blockIdx.x >> targetQubit) & 1;
                int m = 1 << (LOCAL_QUBIT_SIZE - 1);
                int maskControl = (1 << controlQubit) - 1;
                if (!isHighBlock){
                    for (int j = threadIdx.x; j < m; j += blockSize) {
                        int x = ((j >> controlQubit) << (controlQubit + 1)) | (j & maskControl)  | (1 << controlQubit);
                        rotateZlo(x, deviceGates[i].alpha, deviceGates[i].beta);
                    }
                } else {
                    for (int j = threadIdx.x; j < m; j += blockSize) {
                        int x = ((j >> controlQubit) << (controlQubit + 1)) | (j & maskControl)  | (1 << controlQubit);
                        rotateZhi(x, deviceGates[i].alpha, deviceGates[i].beta);
                    }
                }
            }
        } else {
            if (controlIsGlobal == 1 && !((blockIdx.x >> controlQubit) & 1)) {
                continue;
            }
            if (!targetIsGlobal) {
                int m = 1 << (LOCAL_QUBIT_SIZE - 1);
                int maskTarget = (1 << targetQubit) - 1;
                for (int j = threadIdx.x; j < m; j += blockSize) {
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
                        case GateCRotateZ: // no break
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
            } else {
                bool isHighBlock = (blockIdx.x >> targetQubit) & 1;
                switch (deviceGates[i].type) {
                    case GateCRotateZ: // no break
                    case GateRotateZ: {
                        int m = 1 << LOCAL_QUBIT_SIZE;
                        if (!isHighBlock){
                            for (int j = threadIdx.x; j < m; j += blockSize) {
                                rotateZlo(j, deviceGates[i].alpha, deviceGates[i].beta);
                            }
                        } else {
                            for (int j = threadIdx.x; j < m; j += blockSize) {
                                rotateZhi(j, deviceGates[i].alpha, deviceGates[i].beta);
                            }
                        }
                        break;
                    }
                    case GatePauliZ: {
                        if (!isHighBlock) continue;
                        int m = 1 << LOCAL_QUBIT_SIZE;
                        for (int j = threadIdx.x; j < m; j += blockSize) {
                            pauliZGate(j);
                        }
                        break;
                    }
                    case GateS: {
                        if (!isHighBlock) continue;
                        int m = 1 << LOCAL_QUBIT_SIZE;
                        for (int j = threadIdx.x; j < m; j += blockSize) {
                            sGate(j);
                        }
                        break;
                    }
                    case GateT: {
                        if (!isHighBlock) continue;
                        int m = 1 << LOCAL_QUBIT_SIZE;
                        for (int j = threadIdx.x; j < m; j += blockSize) {
                            tGate(j);
                        }
                        break;
                    }
                    default: {
                        assert(false);
                    }
                }
            }
        }
        __syncthreads();
    }
}

__device__ void fetchData(ComplexArray a, qindex* threadBias,  qindex idx, qindex blockHot, qindex enumerate, int numQubits) {
    if (threadIdx.x == 0) {
        int bid = blockIdx.x;
        qindex bias = 0;
        for (qindex bit = 1; bit < (qindex(1) << numQubits); bit <<= 1) {
            if (blockHot & bit) {
                if (bid & 1)
                    bias |= bit;
                bid >>= 1;
            }
        }
        blockBias = bias;
    }
    __syncthreads();
    qindex bias = blockBias | threadBias[threadIdx.x];
    for (int x = ((1 << (LOCAL_QUBIT_SIZE - THREAD_DEP)) - 1) << THREAD_DEP | threadIdx.x, y = enumerate;
        x >= 0;
        x -= (1 << THREAD_DEP), y = enumerate & (y - 1)) {
            
        real[x] = a.real[bias | y];
        imag[x] = a.imag[bias | y];
    }
}

__device__ void saveData(ComplexArray a, qindex* threadBias, qindex idx, qindex blockHot, qindex enumerate) {
    qindex bias = blockBias | threadBias[threadIdx.x];
    for (int x = ((1 << (LOCAL_QUBIT_SIZE - THREAD_DEP)) - 1) << THREAD_DEP | threadIdx.x, y = enumerate;
        x >= 0;
        x -= (1 << THREAD_DEP), y = enumerate & (y - 1)) {
        
        a.real[bias | y] = real[x];
        a.imag[bias | y] = imag[x];
    }
}

template <unsigned int blockSize>
__global__ void run(ComplexArray a, qindex* threadBias, int numQubits, int numGates, qindex blockHot, qindex enumerate) {
    qindex idx = blockIdx.x * blockSize + threadIdx.x;
    fetchData(a, threadBias, idx, blockHot, enumerate, numQubits);
    __syncthreads();
    doCompute<blockSize>(numGates);
    __syncthreads();
    saveData(a, threadBias, idx, blockHot, enumerate);
}

void kernelExecOpt(ComplexArray& deviceStateVec, int numQubits, const Schedule& schedule) {
    qindex hostThreadBias[1 << THREAD_DEP];
    qindex* threadBias;
    checkCudaErrors(cudaMalloc(&threadBias, sizeof(hostThreadBias)));
    for (auto& gg: schedule.gateGroups) {
        // initialize blockHot, enumerate, threadBias
        qindex relatedQubits = gg.relatedQubits;
        int cnt = bitCount(relatedQubits);
        if (cnt < LOCAL_QUBIT_SIZE) {
            int cnt = bitCount(relatedQubits);
            for (int i = 0; i < LOCAL_QUBIT_SIZE; i++) {
                if (!(relatedQubits & (1 << i))) {
                    cnt++;
                    relatedQubits |= (1 << i);
                    if (cnt == LOCAL_QUBIT_SIZE)
                    break;
                }
            }
        }
        qindex blockHot = (qindex(1) << numQubits) - 1 - relatedQubits;
        qindex enumerate = relatedQubits;
        qindex threadHot = 0;
        for (int i = 0; i < THREAD_DEP; i++) {
            qindex x = enumerate & (-enumerate);
            threadHot += x;
            enumerate -= x;
        }
        assert((threadHot | enumerate) == relatedQubits);
        for (int i = (1 << THREAD_DEP) - 1, j = threadHot; i >= 0; i--, j = threadHot & (j - 1)) {
            hostThreadBias[i] = j;
        }
        checkCudaErrors(cudaMemcpy(threadBias, hostThreadBias, sizeof(hostThreadBias), cudaMemcpyHostToDevice));
        // printf("related %x blockHot %x enumerate %x hostThreadBias[5] %x\n", relatedQubits, blockHot, enumerate, hostThreadBias[5]);

        // initialize gates
        std::map<int, int> toID;
        int localCnt = 0;
        int globalCnt = 0;
        for (int i = 0; i < numQubits; i++) {
            if (relatedQubits & (qindex(1) << i)) {
                toID[i] = localCnt++;
            } else {
                toID[i] = globalCnt++;
            }
        }
        auto isLocalQubit = [relatedQubits] (int x) {
            return relatedQubits >> x & 1;
        };
        KernelGate hostGates[gg.gates.size()];
        assert(gg.gates.size() < MAX_GATE);
        for (size_t i = 0; i < gg.gates.size(); i++) {
            switch (gg.gates[i].type) {
                case GateRotateX: // no break
                case GateCRotateX: {
                    hostGates[i].alpha = gg.gates[i].mat[0][0].real;
                    hostGates[i].beta = gg.gates[i].mat[0][1].imag;
                    break;
                }
                case GateRotateY: // no break
                case GateCRotateY: {
                    hostGates[i].alpha = gg.gates[i].mat[0][0].real;
                    hostGates[i].beta = gg.gates[i].mat[1][0].real;
                    break;
                }
                case GateRotateZ: // no break
                case GateCRotateZ: {
                    hostGates[i].alpha = gg.gates[i].mat[0][0].real;
                    hostGates[i].beta = - gg.gates[i].mat[0][0].imag;
                    break;
                }
                default: {
                    hostGates[i].alpha = hostGates[i].beta = 0;
                }
            }
            
            if (gg.gates[i].controlQubit == -1) {
                hostGates[i].controlQubit = -1;
                hostGates[i].controlIsGlobal = 2;
            } else {
                hostGates[i].controlQubit = toID[gg.gates[i].controlQubit];
                hostGates[i].controlIsGlobal = 1 - isLocalQubit(gg.gates[i].controlQubit);
            }

            hostGates[i].targetQubit = toID[gg.gates[i].targetQubit];
            hostGates[i].targetIsGlobal = 1 - isLocalQubit(gg.gates[i].targetQubit);
            
            hostGates[i].type = gg.gates[i].type;
        }
        checkCudaErrors(cudaMemcpyToSymbol(deviceGates, hostGates, sizeof(hostGates)));

        // execute
        run<1<<THREAD_DEP><<<(1<<numQubits)>>LOCAL_QUBIT_SIZE, 1<<THREAD_DEP>>>(deviceStateVec, threadBias, numQubits, gg.gates.size(), blockHot, enumerate);
        // checkCudaErrors(cudaDeviceSynchronize()); // WARNING: for time measure!
        // printf("Group End\n");
    }
    // checkCudaErrors(cudaDeviceSynchronize()); // WARNING: for time measure!
}