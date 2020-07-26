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

const int SINGLE_SIZE_DEP = 0; // handle 1 << SINGLE_SIZE_DEP items per thread
const int THREAD_DEP = 7; // 1 << THREAD_DEP threads per block
const int REDUCE_BLOCK_DEP = 6; // 1 << REDUCE_BLOCK_DEP blocks in final reduction

void kernelInit(ComplexArray& deviceStateVec, int numQubits) {
    cudaError_t cuda_status;
	cuda_status = cudaSetDevice(0);
	if (cuda_status != cudaSuccess) {
        printf("cudaSetDevice failed! ");
        exit(1);
	}
    assert(numQubits > (SINGLE_SIZE_DEP +THREAD_DEP + 1 + REDUCE_BLOCK_DEP + THREAD_DEP + 1));
    assert(numQubits < 31);
    size_t size = sizeof(qreal) << numQubits;
    checkCudaErrors(cudaMalloc(&deviceStateVec.real, size));
    checkCudaErrors(cudaMalloc(&deviceStateVec.imag, size));
    checkCudaErrors(cudaMemset(deviceStateVec.real, 0, size));
    checkCudaErrors(cudaMemset(deviceStateVec.imag, 0, size));
    qreal one = 1;
    checkCudaErrors(cudaMemcpy(deviceStateVec.real, &one, sizeof(qreal), cudaMemcpyHostToDevice)); // state[0] = 1
}


#define SINGLE_GATE_BEGIN \
    qindex idx = blockIdx.x * blockSize + threadIdx.x; \
    qindex mask = (qindex(1) << targetQubit) - 1; \
    for (qindex i = (idx << SINGLE_SIZE_DEP); i < ((idx + 1) << SINGLE_SIZE_DEP); i++) { \
        qindex lo = ((i >> targetQubit) << (targetQubit + 1)) | (i & mask); \
        qindex hi = lo | (qindex(1) << targetQubit);

#define SINGLE_GATE_END }

#define CONTROL_GATE_BEGIN \
    qindex idx = blockIdx.x * blockSize + threadIdx.x; \
    qindex mask = (qindex(1) << targetQubit) - 1; \
    for (qindex i = (idx << SINGLE_SIZE_DEP); i < ((idx + 1) << SINGLE_SIZE_DEP); i++) { \
        qindex lo = ((i >> targetQubit) << (targetQubit + 1)) | (i & mask); \
        if (!((lo >> controlQubit) & 1)) \
            continue; \
        qindex hi = lo | (qindex(1) << targetQubit);

#define CONTROL_GATE_END }

template <unsigned int blockSize>
__global__ void controlledNotGate(ComplexArray a, int numQubit_, int controlQubit, int targetQubit) {
    CONTROL_GATE_BEGIN {
        qreal real = a.real[lo];
        qreal imag = a.imag[lo];
        a.real[lo] = a.real[hi];
        a.imag[lo] = a.imag[hi];
        a.real[hi] = real;
        a.imag[hi] = imag;
    } CONTROL_GATE_END
}

template <unsigned int blockSize>
__global__ void controlledPauliYGate(ComplexArray a, int numQubit_, int controlQubit, int targetQubit) {
    CONTROL_GATE_BEGIN {
        qreal loReal = a.real[lo];
        qreal loImag = a.imag[lo];
        qreal hiReal = a.real[hi];
        qreal hiImag = a.imag[hi];
        a.real[lo] = hiImag;
        a.imag[lo] = -hiReal;
        a.real[hi] = -loImag;
        a.imag[hi] = loReal;
    } CONTROL_GATE_END
}

template <unsigned int blockSize>
__global__ void controlledRotateXGate(ComplexArray a, int numQubit_, int controlQubit, int targetQubit, qreal alpha, qreal beta) {
    CONTROL_GATE_BEGIN {
        qreal loReal = a.real[lo];
        qreal loImag = a.imag[lo];
        qreal hiReal = a.real[hi];
        qreal hiImag = a.imag[hi];
        a.real[lo] = alpha * loReal + beta * hiImag;
        a.imag[lo] = alpha * loImag - beta * hiReal;
        a.real[hi] = alpha * hiReal + beta * loImag;
        a.imag[hi] = alpha * hiImag - beta * loReal;
    } CONTROL_GATE_END
}

template <unsigned int blockSize>
__global__ void controlledRotateYGate(ComplexArray a, int numQubit_, int controlQubit, int targetQubit, qreal alpha, qreal beta) {
    CONTROL_GATE_BEGIN {
        qreal loReal = a.real[lo];
        qreal loImag = a.imag[lo];
        qreal hiReal = a.real[hi];
        qreal hiImag = a.imag[hi];
        a.real[lo] = alpha * loReal - beta * hiReal;
        a.imag[lo] = alpha * loImag - beta * hiImag;
        a.real[hi] = beta * loReal + alpha * hiReal;
        a.imag[hi] = beta * loImag + alpha * hiImag;
    } CONTROL_GATE_END
}

template <unsigned int blockSize>
__global__ void controlledRotateZGate(ComplexArray a, int numQubit_, int controlQubit, int targetQubit, qreal alpha, qreal beta) {
    CONTROL_GATE_BEGIN {
        qreal loReal = a.real[lo];
        qreal loImag = a.imag[lo];
        qreal hiReal = a.real[hi];
        qreal hiImag = a.imag[hi];
        a.real[lo] = alpha * loReal + beta * loImag;
        a.imag[lo] = alpha * loImag - beta * loReal;
        a.real[hi] = alpha * hiReal - beta * hiImag;
        a.imag[hi] = alpha * hiImag + beta * hiReal;
    } CONTROL_GATE_END
}

template <unsigned int blockSize>
__global__ void hadamardGate(ComplexArray a, int numQubit_, int targetQubit, qreal recRoot2) {
    SINGLE_GATE_BEGIN {
        qreal loReal = a.real[lo];
        qreal loImag = a.imag[lo];
        qreal hiReal = a.real[hi];
        qreal hiImag = a.imag[hi];
        a.real[lo] = recRoot2 * (loReal + hiReal);
        a.imag[lo] = recRoot2 * (loImag + hiImag);
        a.real[hi] = recRoot2 * (loReal - hiReal);
        a.imag[hi] = recRoot2 * (loImag - hiImag);
    } SINGLE_GATE_END
}

template <unsigned int blockSize>
__global__ void pauliXGate(ComplexArray a, int numQubit_, int targetQubit) {
    SINGLE_GATE_BEGIN {
        qreal real = a.real[lo];
        qreal imag = a.imag[lo];
        a.real[lo] = a.real[hi];
        a.imag[lo] = a.imag[hi];
        a.real[hi] = real;
        a.imag[hi] = imag;
    } SINGLE_GATE_END
}

template <unsigned int blockSize>
__global__ void pauliYGate(ComplexArray a, int numQubit_, int targetQubit) {
    SINGLE_GATE_BEGIN {
        qreal loReal = a.real[lo];
        qreal loImag = a.imag[lo];
        qreal hiReal = a.real[hi];
        qreal hiImag = a.imag[hi];
        a.real[lo] = hiImag;
        a.imag[lo] = -hiReal;
        a.real[hi] = -loImag;
        a.imag[hi] = loReal;
    } SINGLE_GATE_END
}

template <unsigned int blockSize>
__global__ void pauliZGate(ComplexArray a, int numQubit_, int targetQubit) {
    SINGLE_GATE_BEGIN {
        a.real[hi] = -a.real[hi];
        a.imag[hi] = -a.imag[hi];
    } SINGLE_GATE_END
}

template <unsigned int blockSize>
__global__ void rotateXGate(ComplexArray a, int numQubit_, int targetQubit, qreal alpha, qreal beta) {
    SINGLE_GATE_BEGIN {
        qreal loReal = a.real[lo];
        qreal loImag = a.imag[lo];
        qreal hiReal = a.real[hi];
        qreal hiImag = a.imag[hi];
        a.real[lo] = alpha * loReal + beta * hiImag;
        a.imag[lo] = alpha * loImag - beta * hiReal;
        a.real[hi] = alpha * hiReal + beta * loImag;
        a.imag[hi] = alpha * hiImag - beta * loReal;
    } SINGLE_GATE_END
}

template <unsigned int blockSize>
__global__ void rotateYGate(ComplexArray a, int numQubit_, int targetQubit, qreal alpha, qreal beta) {
    SINGLE_GATE_BEGIN {
        qreal loReal = a.real[lo];
        qreal loImag = a.imag[lo];
        qreal hiReal = a.real[hi];
        qreal hiImag = a.imag[hi];
        a.real[lo] = alpha * loReal - beta * hiReal;
        a.imag[lo] = alpha * loImag - beta * hiImag;
        a.real[hi] = beta * loReal + alpha * hiReal;
        a.imag[hi] = beta * loImag + alpha * hiImag;
    } SINGLE_GATE_END
}

template <unsigned int blockSize>
__global__ void rotateZGate(ComplexArray a, int numQubit_, int targetQubit, qreal alpha, qreal beta) {
    SINGLE_GATE_BEGIN {
        qreal loReal = a.real[lo];
        qreal loImag = a.imag[lo];
        qreal hiReal = a.real[hi];
        qreal hiImag = a.imag[hi];
        a.real[lo] = alpha * loReal + beta * loImag;
        a.imag[lo] = alpha * loImag - beta * loReal;
        a.real[hi] = alpha * hiReal - beta * hiImag;
        a.imag[hi] = alpha * hiImag + beta * hiReal;
    } SINGLE_GATE_END
}

template <unsigned int blockSize>
__global__ void sGate(ComplexArray a, int numQubit_, int targetQubit) {
    SINGLE_GATE_BEGIN {
        qreal hiReal = a.real[hi];
        qreal hiImag = a.imag[hi];
        a.real[hi] = -hiImag;
        a.imag[hi] = hiReal;
    } SINGLE_GATE_END
}

template <unsigned int blockSize>
__global__ void tGate(ComplexArray a, int numQubit_, int targetQubit, qreal recRoot2) {
    SINGLE_GATE_BEGIN {
        qreal hiReal = a.real[hi];
        qreal hiImag = a.imag[hi];
        a.real[hi] = recRoot2 * (hiReal - hiImag);
        a.imag[hi] = recRoot2 * (hiReal + hiImag);
    } SINGLE_GATE_END
}

void kernelExecSimple(ComplexArray& deviceStateVec, int numQubits, const Schedule& schedule) {
    int numQubit_ = numQubits - 1;
    int nVec = 1 << numQubit_;
    for (auto& gg: schedule.gateGroups) {
        for (auto& gate: gg.gates) {
            switch (gate.type) {
                case GateCNot: {
                    controlledNotGate<1<<THREAD_DEP><<<nVec>>(SINGLE_SIZE_DEP + THREAD_DEP), 1<<THREAD_DEP>>>(deviceStateVec, numQubit_, gate.controlQubit, gate.targetQubit);
                    break;
                }
                case GateCPauliY: {
                    controlledPauliYGate<1<<THREAD_DEP><<<nVec>>(SINGLE_SIZE_DEP + THREAD_DEP), 1<<THREAD_DEP>>>(deviceStateVec, numQubit_, gate.controlQubit, gate.targetQubit);
                    break;
                }
                case GateCRotateX: {
                    controlledRotateXGate<1<<THREAD_DEP><<<nVec>>(SINGLE_SIZE_DEP + THREAD_DEP), 1<<THREAD_DEP>>>(
                        deviceStateVec, numQubit_, gate.controlQubit, gate.targetQubit, gate.mat[0][0].real, gate.mat[0][1].imag);
                    break;
                }
                case GateCRotateY: {
                    controlledRotateYGate<1<<THREAD_DEP><<<nVec>>(SINGLE_SIZE_DEP + THREAD_DEP), 1<<THREAD_DEP>>>(
                        deviceStateVec, numQubit_, gate.controlQubit, gate.targetQubit, gate.mat[0][0].real, gate.mat[1][0].real);
                    break;
                }
                case GateCRotateZ: {
                    controlledRotateZGate<1<<THREAD_DEP><<<nVec>>(SINGLE_SIZE_DEP + THREAD_DEP), 1<<THREAD_DEP>>>(
                        deviceStateVec, numQubit_, gate.controlQubit, gate.targetQubit, gate.mat[0][0].real, - gate.mat[0][0].imag);
                    break;
                }
                case GateHadamard: {
                    hadamardGate<1<<THREAD_DEP><<<nVec>>(SINGLE_SIZE_DEP + THREAD_DEP), 1<<THREAD_DEP>>>(deviceStateVec, numQubit_, gate.targetQubit, 1/sqrt(2));
                    break;
                }
                case GatePauliX: {
                    pauliXGate<1<<THREAD_DEP><<<nVec>>(SINGLE_SIZE_DEP + THREAD_DEP), 1<<THREAD_DEP>>>(deviceStateVec, numQubit_, gate.targetQubit);
                    break;
                }
                case GatePauliY: {
                    pauliYGate<1<<THREAD_DEP><<<nVec>>(SINGLE_SIZE_DEP + THREAD_DEP), 1<<THREAD_DEP>>>(deviceStateVec, numQubit_, gate.targetQubit);
                    break;
                }
                case GatePauliZ: {
                    pauliZGate<1<<THREAD_DEP><<<nVec>>(SINGLE_SIZE_DEP + THREAD_DEP), 1<<THREAD_DEP>>>(deviceStateVec, numQubit_, gate.targetQubit);
                    break;
                }
                case GateRotateX: {
                    rotateXGate<1<<THREAD_DEP><<<nVec>>(SINGLE_SIZE_DEP + THREAD_DEP), 1<<THREAD_DEP>>>(
                        deviceStateVec, numQubit_, gate.targetQubit, gate.mat[0][0].real, gate.mat[0][1].imag);
                    break;
                }
                case GateRotateY: {
                    rotateYGate<1<<THREAD_DEP><<<nVec>>(SINGLE_SIZE_DEP + THREAD_DEP), 1<<THREAD_DEP>>>(
                        deviceStateVec, numQubit_, gate.targetQubit, gate.mat[0][0].real, gate.mat[1][0].real);
                    break;
                }
                case GateRotateZ: {
                    rotateZGate<1<<THREAD_DEP><<<nVec>>(SINGLE_SIZE_DEP + THREAD_DEP), 1<<THREAD_DEP>>>(
                        deviceStateVec, numQubit_, gate.targetQubit, gate.mat[0][0].real, - gate.mat[0][0].imag);
                    break;
                }
                case GateS: {
                    sGate<1<<THREAD_DEP><<<nVec>>(SINGLE_SIZE_DEP + THREAD_DEP), 1<<THREAD_DEP>>>(deviceStateVec, numQubit_, gate.targetQubit);
                    break;
                }
                case GateT: {
                    tGate<1<<THREAD_DEP><<<nVec>>(SINGLE_SIZE_DEP + THREAD_DEP), 1<<THREAD_DEP>>>(deviceStateVec, numQubit_, gate.targetQubit, 1/sqrt(2));
                    break;
                }
                default: {
                    assert(false);
                }
            }
        }
    }
}

template <unsigned int blockSize>
__device__ void warpReduce(volatile qreal *sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__device__ void blockReduce(volatile qreal *sdata, unsigned int tid) {
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) warpReduce<blockSize>(sdata, tid);
}

template <unsigned int blockSize>
__global__ void reduce(qreal* g_idata, qreal *g_odata, unsigned int n, unsigned int gridSize) {
    __shared__ qreal sdata[blockSize];
    unsigned tid = threadIdx.x;
    unsigned idx = blockIdx.x * blockSize + threadIdx.x;
    unsigned twoGrid = gridSize << 1;
    sdata[tid] = 0;
    for (int i = idx; i < n; i += twoGrid) {
        sdata[tid] += g_idata[i] + g_idata[i + gridSize];
    }
    __syncthreads();
    blockReduce<blockSize>(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize>
__global__ void measure(ComplexArray a, qreal* ans, int numQubit_, int targetQubit) {
    __shared__ qreal sdata[blockSize];
    qindex idx = blockIdx.x * blockSize + threadIdx.x;
    int tid = threadIdx.x;
    qindex mask = (qindex(1) << targetQubit) - 1;
    sdata[tid] = 0;
    for (qindex i = (idx << SINGLE_SIZE_DEP); i < ((idx + 1) << SINGLE_SIZE_DEP); i++) {
        qindex lo = ((i >> targetQubit) << (targetQubit + 1)) | (i & mask);
        qreal loReal = a.real[lo];
        qreal loImag = a.imag[lo];
        sdata[tid] += loReal * loReal + loImag * loImag;
    }
    __syncthreads();
    blockReduce<blockSize>(sdata, tid);
    if (tid == 0) ans[blockIdx.x] = sdata[0];
}

qreal kernelMeasure(ComplexArray& deviceStateVec, int numQubits, int targetQubit) {
    int numQubit_ = numQubits - 1;
    qindex nVec = 1 << numQubit_;
    qindex totalBlocks = nVec >> THREAD_DEP >> SINGLE_SIZE_DEP;
    qreal *ans1, *ans2, *ans3;
    checkCudaErrors(cudaMalloc(&ans1, sizeof(qreal) * totalBlocks));
    measure<1<<THREAD_DEP><<<totalBlocks, 1<<THREAD_DEP>>>(deviceStateVec, ans1, numQubit_, targetQubit);
    checkCudaErrors(cudaMalloc(&ans2, sizeof(qreal) * (1<<REDUCE_BLOCK_DEP)));
    reduce<1<<THREAD_DEP><<<1<<REDUCE_BLOCK_DEP, 1<<THREAD_DEP>>>
        (ans1, ans2, totalBlocks, 1 << (THREAD_DEP + REDUCE_BLOCK_DEP));
    checkCudaErrors(cudaMallocHost(&ans3, sizeof(qreal) * (1<<REDUCE_BLOCK_DEP)));
    checkCudaErrors(cudaMemcpy(ans3, ans2, sizeof(qreal) * (1<<REDUCE_BLOCK_DEP), cudaMemcpyDeviceToHost));
    qreal ret = 0;
    for (int i = 0; i < (1<<REDUCE_BLOCK_DEP); i++)
        ret += ans3[i];
    checkCudaErrors(cudaFree(ans1));
    checkCudaErrors(cudaFree(ans2));
    checkCudaErrors(cudaFreeHost(ans3));
    return ret;
}

Complex kernelGetAmp(ComplexArray& deviceStateVec, qindex idx) {
    Complex ret;
    cudaMemcpy(&ret.real, deviceStateVec.real + idx, sizeof(qreal), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ret.imag, deviceStateVec.imag + idx, sizeof(qreal), cudaMemcpyDeviceToHost);
    return ret;
}