#include "kernel.h"
#include <cstdio>
#include <assert.h>
using namespace std;

const int SINGLE_SIZE_DEP = 0; // handle 1 << SINGLE_SIZE_DEP items per thread
const int REDUCE_BLOCK_DEP = 6; // 1 << REDUCE_BLOCK_DEP blocks in final reduction

void kernelInit(std::vector<qComplex*> &deviceStateVec, int numQubits) {
    size_t size = (sizeof(qComplex) << numQubits) >> MyGlobalVars::bit;
    if (MyGlobalVars::numGPUs > 1 || BACKEND == 3 || BACKEND == 4) {
        size <<= 1;
    }
#if BACKEND == 2
    deviceStateVec.resize(1);
    cudaSetDevice(0);
    checkCudaErrors(cudaMalloc(&deviceStateVec[0], sizeof(qComplex) << numQubits));
    checkCudaErrors(cudaMemsetAsync(deviceStateVec[0], 0, sizeof(qComplex) << numQubits, MyGlobalVars::streams[0]));
#else
    deviceStateVec.resize(MyGlobalVars::localGPUs);
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        cudaSetDevice(g);
        checkCudaErrors(cudaMalloc(&deviceStateVec[g], size));
        checkCudaErrors(cudaMemsetAsync(deviceStateVec[g], 0, size, MyGlobalVars::streams[g]));
    }
#endif
    qComplex one = make_qComplex(1.0, 0.0);
    checkCudaErrors(cudaMemcpyAsync(deviceStateVec[0], &one, sizeof(qComplex), cudaMemcpyHostToDevice, MyGlobalVars::streams[0])); // state[0] = 1
#if BACKEND == 1 || BACKEND == 3 || BACKEND == 4 || BACKEND == 5
    initControlIdx();
#endif
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaStreamSynchronize(MyGlobalVars::streams[g]));
    }
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

#define CC_GATE_BEGIN \
    qindex idx = blockIdx.x * blockSize + threadIdx.x; \
    qindex mask = (qindex(1) << targetQubit) - 1; \
    for (qindex i = (idx << SINGLE_SIZE_DEP); i < ((idx + 1) << SINGLE_SIZE_DEP); i++) { \
        qindex lo = ((i >> targetQubit) << (targetQubit + 1)) | (i & mask); \
        if (!((lo >> c1) & 1)) \
            continue; \
        if (!((lo >> c2) & 1)) \
            continue; \
        qindex hi = lo | (qindex(1) << targetQubit);
#define CC_GATE_END }



template <unsigned int blockSize>
__global__ void CCXKernel(qComplex* a, int numQubit_, int c1, int c2, int targetQubit) {
    CC_GATE_BEGIN {
        qreal real = a[lo].x;
        qreal imag = a[lo].y;
        a[lo].x = a[hi].x;
        a[lo].y = a[hi].y;
        a[hi].x = real;
        a[hi].y = imag;
    } CC_GATE_END
}


template <unsigned int blockSize>
__global__ void CNOTKernel(qComplex* a, int numQubit_, int controlQubit, int targetQubit) {
    CONTROL_GATE_BEGIN {
        qreal real = a[lo].x;
        qreal imag = a[lo].y;
        a[lo].x = a[hi].x;
        a[lo].y = a[hi].y;
        a[hi].x = real;
        a[hi].y = imag;
    } CONTROL_GATE_END
}

template <unsigned int blockSize>
__global__ void CYKernel(qComplex* a, int numQubit_, int controlQubit, int targetQubit) {
    CONTROL_GATE_BEGIN {
        qreal loReal = a[lo].x;
        qreal loImag = a[lo].y;
        qreal hiReal = a[hi].x;
        qreal hiImag = a[hi].y;
        a[lo].x = hiImag;
        a[lo].y = -hiReal;
        a[hi].x = -loImag;
        a[hi].y = loReal;
    } CONTROL_GATE_END
}

template <unsigned int blockSize>
__global__ void CZKernel(qComplex* a, int numQubit_, int controlQubit, int targetQubit) {
    CONTROL_GATE_BEGIN {
        a[hi].x = -a[hi].x;
        a[hi].y = -a[hi].y;
    } CONTROL_GATE_END
}

template <unsigned int blockSize>
__global__ void CRXKernel(qComplex* a, int numQubit_, int controlQubit, int targetQubit, qreal alpha, qreal beta) {
    CONTROL_GATE_BEGIN {
        qreal loReal = a[lo].x;
        qreal loImag = a[lo].y;
        qreal hiReal = a[hi].x;
        qreal hiImag = a[hi].y;
        a[lo].x = alpha * loReal + beta * hiImag;
        a[lo].y = alpha * loImag - beta * hiReal;
        a[hi].x = alpha * hiReal + beta * loImag;
        a[hi].y = alpha * hiImag - beta * loReal;
    } CONTROL_GATE_END
}

template <unsigned int blockSize>
__global__ void CRYKernel(qComplex* a, int numQubit_, int controlQubit, int targetQubit, qreal alpha, qreal beta) {
    CONTROL_GATE_BEGIN {
        qreal loReal = a[lo].x;
        qreal loImag = a[lo].y;
        qreal hiReal = a[hi].x;
        qreal hiImag = a[hi].y;
        a[lo].x = alpha * loReal - beta * hiReal;
        a[lo].y = alpha * loImag - beta * hiImag;
        a[hi].x = beta * loReal + alpha * hiReal;
        a[hi].y = beta * loImag + alpha * hiImag;
    } CONTROL_GATE_END
}

template <unsigned int blockSize>
__global__ void CRZKernel(qComplex* a, int numQubit_, int controlQubit, int targetQubit, qreal alpha, qreal beta) {
    CONTROL_GATE_BEGIN {
        qreal loReal = a[lo].x;
        qreal loImag = a[lo].y;
        qreal hiReal = a[hi].x;
        qreal hiImag = a[hi].y;
        a[lo].x = alpha * loReal + beta * loImag;
        a[lo].y = alpha * loImag - beta * loReal;
        a[hi].x = alpha * hiReal - beta * hiImag;
        a[hi].y = alpha * hiImag + beta * hiReal;
    } CONTROL_GATE_END
}

template <unsigned int blockSize>
__global__ void CU1Kernel(qComplex* a, int numQubit_, int controlQubit, int targetQubit, qreal alpha, qreal beta) {
    CONTROL_GATE_BEGIN {
        qreal hiReal = a[hi].x;
        qreal hiImag = a[hi].y;
        a[hi].x = alpha * hiReal - beta * hiImag;
        a[hi].y = alpha * hiImag + beta * hiReal;
    } CONTROL_GATE_END
}

template <unsigned int blockSize>
__global__ void U1Kernel(qComplex* a, int numQubit_, int targetQubit, qreal alpha, qreal beta) {
    SINGLE_GATE_BEGIN {
        qreal hiReal = a[hi].x;
        qreal hiImag = a[hi].y;
        a[hi].x = alpha * hiReal - beta * hiImag;
        a[hi].y = alpha * hiImag + beta * hiReal;
    } SINGLE_GATE_END
}

#define COMPLEX_MULTIPLY_REAL(i0, r0, i1, r1) (i0 * i1 - r0 * r1)
#define COMPLEX_MULTIPLY_IMAG(i0, r0, i1, r1) (i0 * r1 + i1 * r0)

template <unsigned int blockSize>
__global__ void UKernel(qComplex* a, int numQubit_, int targetQubit, qreal r00, qreal i00, qreal r01, qreal i01, qreal r10, qreal i10, qreal r11, qreal i11) {
    SINGLE_GATE_BEGIN {
        qreal loReal = a[lo].x;
        qreal loImag = a[lo].y;
        qreal hiReal = a[hi].x;
        qreal hiImag = a[hi].y;
        a[lo].x = COMPLEX_MULTIPLY_REAL(loReal, loImag, r00, i00) + COMPLEX_MULTIPLY_REAL(hiReal, hiImag, r01, i01);
        a[lo].y = COMPLEX_MULTIPLY_IMAG(loReal, loImag, r00, i00) + COMPLEX_MULTIPLY_IMAG(hiReal, hiImag, r01, i01);
        a[hi].x = COMPLEX_MULTIPLY_REAL(loReal, loImag, r10, i10) + COMPLEX_MULTIPLY_REAL(hiReal, hiImag, r11, i11);
        a[hi].y = COMPLEX_MULTIPLY_IMAG(loReal, loImag, r10, i10) + COMPLEX_MULTIPLY_IMAG(hiReal, hiImag, r11, i11);
    } SINGLE_GATE_END
}

#undef COMPLEX_MULTIPLY_REAL
#undef COMPLEX_MULTIPLY_IMAG

template <unsigned int blockSize>
__global__ void HKernel(qComplex* a, int numQubit_, int targetQubit, qreal recRoot2) {
    SINGLE_GATE_BEGIN {
        qreal loReal = a[lo].x;
        qreal loImag = a[lo].y;
        qreal hiReal = a[hi].x;
        qreal hiImag = a[hi].y;
        a[lo].x = recRoot2 * (loReal + hiReal);
        a[lo].y = recRoot2 * (loImag + hiImag);
        a[hi].x = recRoot2 * (loReal - hiReal);
        a[hi].y = recRoot2 * (loImag - hiImag);
    } SINGLE_GATE_END
}

template <unsigned int blockSize>
__global__ void XKernel(qComplex* a, int numQubit_, int targetQubit) {
    SINGLE_GATE_BEGIN {
        qreal real = a[lo].x;
        qreal imag = a[lo].y;
        a[lo].x = a[hi].x;
        a[lo].y = a[hi].y;
        a[hi].x = real;
        a[hi].y = imag;
    } SINGLE_GATE_END
}

template <unsigned int blockSize>
__global__ void YKernel(qComplex* a, int numQubit_, int targetQubit) {
    SINGLE_GATE_BEGIN {
        qreal loReal = a[lo].x;
        qreal loImag = a[lo].y;
        qreal hiReal = a[hi].x;
        qreal hiImag = a[hi].y;
        a[lo].x = hiImag;
        a[lo].y = -hiReal;
        a[hi].x = -loImag;
        a[hi].y = loReal;
    } SINGLE_GATE_END
}

template <unsigned int blockSize>
__global__ void ZKernel(qComplex* a, int numQubit_, int targetQubit) {
    SINGLE_GATE_BEGIN {
        a[hi].x = -a[hi].x;
        a[hi].y = -a[hi].y;
    } SINGLE_GATE_END
}


template <unsigned int blockSize>
__global__ void SKernel(qComplex* a, int numQubit_, int targetQubit) {
    SINGLE_GATE_BEGIN {
        qreal hiReal = a[hi].x;
        qreal hiImag = a[hi].y;
        a[hi].x = -hiImag;
        a[hi].y = hiReal;
    } SINGLE_GATE_END
}

template <unsigned int blockSize>
__global__ void SDGKernel(qComplex* a, int numQubit_, int targetQubit) {
    SINGLE_GATE_BEGIN {
        qreal hiReal = a[hi].x;
        qreal hiImag = a[hi].y;
        a[hi].x = hiImag;
        a[hi].y = -hiReal;
    } SINGLE_GATE_END
}

template <unsigned int blockSize>
__global__ void TKernel(qComplex* a, int numQubit_, int targetQubit, qreal recRoot2) {
    SINGLE_GATE_BEGIN {
        qreal hiReal = a[hi].x;
        qreal hiImag = a[hi].y;
        a[hi].x = recRoot2 * (hiReal - hiImag);
        a[hi].y = recRoot2 * (hiReal + hiImag);
    } SINGLE_GATE_END
}

template <unsigned int blockSize>
__global__ void TDGKernel(qComplex* a, int numQubit_, int targetQubit, qreal recRoot2) {
    SINGLE_GATE_BEGIN {
        qreal hiReal = a[hi].x;
        qreal hiImag = a[hi].y;
        a[hi].x = recRoot2 * (hiReal + hiImag);
        a[hi].y = recRoot2 * (hiImag - hiReal);
    } SINGLE_GATE_END
}

template <unsigned int blockSize>
__global__ void RXKernel(qComplex* a, int numQubit_, int targetQubit, qreal alpha, qreal beta) {
    SINGLE_GATE_BEGIN {
        qreal loReal = a[lo].x;
        qreal loImag = a[lo].y;
        qreal hiReal = a[hi].x;
        qreal hiImag = a[hi].y;
        a[lo].x = alpha * loReal + beta * hiImag;
        a[lo].y = alpha * loImag - beta * hiReal;
        a[hi].x = alpha * hiReal + beta * loImag;
        a[hi].y = alpha * hiImag - beta * loReal;
    } SINGLE_GATE_END
}

template <unsigned int blockSize>
__global__ void RYKernel(qComplex* a, int numQubit_, int targetQubit, qreal alpha, qreal beta) {
    SINGLE_GATE_BEGIN {
        qreal loReal = a[lo].x;
        qreal loImag = a[lo].y;
        qreal hiReal = a[hi].x;
        qreal hiImag = a[hi].y;
        a[lo].x = alpha * loReal - beta * hiReal;
        a[lo].y = alpha * loImag - beta * hiImag;
        a[hi].x = beta * loReal + alpha * hiReal;
        a[hi].y = beta * loImag + alpha * hiImag;
    } SINGLE_GATE_END
}

template <unsigned int blockSize>
__global__ void RZKernel(qComplex* a, int numQubit_, int targetQubit, qreal alpha, qreal beta) {
    SINGLE_GATE_BEGIN {
        qreal loReal = a[lo].x;
        qreal loImag = a[lo].y;
        qreal hiReal = a[hi].x;
        qreal hiImag = a[hi].y;
        a[lo].x = alpha * loReal + beta * loImag;
        a[lo].y = alpha * loImag - beta * loReal;
        a[hi].x = alpha * hiReal - beta * hiImag;
        a[hi].y = alpha * hiImag + beta * hiReal;
    } SINGLE_GATE_END
}


void kernelExecSimple(qComplex* deviceStateVec, int numQubits, const std::vector<Gate> & gates) {
    checkCudaErrors(cudaSetDevice(0));
    int numQubit_ = numQubits - 1;
    int nVec = 1 << numQubit_;
    for (auto& gate: gates) {
        switch (gate.type) {
            case GateType::CCX: {
                CCXKernel<1<<THREAD_DEP><<<nVec>>(SINGLE_SIZE_DEP + THREAD_DEP), 1<<THREAD_DEP>>>(deviceStateVec, numQubit_, gate.controlQubit, gate.controlQubit2, gate.targetQubit);
                break;
            }
            case GateType::CNOT: {
                CNOTKernel<1<<THREAD_DEP><<<nVec>>(SINGLE_SIZE_DEP + THREAD_DEP), 1<<THREAD_DEP>>>(deviceStateVec, numQubit_, gate.controlQubit, gate.targetQubit);
                break;
            }
            case GateType::CY: {
                CYKernel<1<<THREAD_DEP><<<nVec>>(SINGLE_SIZE_DEP + THREAD_DEP), 1<<THREAD_DEP>>>(deviceStateVec, numQubit_, gate.controlQubit, gate.targetQubit);
                break;
            }
            case GateType::CZ: {
                CZKernel<1<<THREAD_DEP><<<nVec>>(SINGLE_SIZE_DEP + THREAD_DEP), 1<<THREAD_DEP>>>(deviceStateVec, numQubit_, gate.controlQubit, gate.targetQubit);
                break;
            }
            case GateType::CRX: {
                CRXKernel<1<<THREAD_DEP><<<nVec>>(SINGLE_SIZE_DEP + THREAD_DEP), 1<<THREAD_DEP>>>(
                    deviceStateVec, numQubit_, gate.controlQubit, gate.targetQubit, gate.mat[0][0].x, -gate.mat[0][1].y);
                break;
            }
            case GateType::CRY: {
                CRYKernel<1<<THREAD_DEP><<<nVec>>(SINGLE_SIZE_DEP + THREAD_DEP), 1<<THREAD_DEP>>>(
                    deviceStateVec, numQubit_, gate.controlQubit, gate.targetQubit, gate.mat[0][0].x, gate.mat[1][0].x);
                break;
            }
            case GateType::CU1: {
                CU1Kernel<1<<THREAD_DEP><<<nVec>>(SINGLE_SIZE_DEP + THREAD_DEP), 1<<THREAD_DEP>>>(
                    deviceStateVec, numQubit_, gate.controlQubit, gate.targetQubit, gate.mat[1][1].x, gate.mat[1][1].y);
                break;
            }
            case GateType::CRZ: {
                CRZKernel<1<<THREAD_DEP><<<nVec>>(SINGLE_SIZE_DEP + THREAD_DEP), 1<<THREAD_DEP>>>(
                    deviceStateVec, numQubit_, gate.controlQubit, gate.targetQubit, gate.mat[0][0].x, - gate.mat[0][0].y);
                break;
            }
            case GateType::U1: {
                U1Kernel<1<<THREAD_DEP><<<nVec>>(SINGLE_SIZE_DEP + THREAD_DEP), 1<<THREAD_DEP>>>(
                    deviceStateVec, numQubit_, gate.targetQubit, gate.mat[1][1].x, gate.mat[1][1].y);
                break;
            }
            case GateType::U2: // no break
            case GateType::U3: {
                UKernel<1<<THREAD_DEP><<<nVec>>(SINGLE_SIZE_DEP + THREAD_DEP), 1<<THREAD_DEP>>>(
                    deviceStateVec, numQubit_, gate.targetQubit,
                    gate.mat[0][0].x, gate.mat[0][0].y,
                    gate.mat[0][1].x, gate.mat[0][1].y,
                    gate.mat[1][0].x, gate.mat[1][0].y,
                    gate.mat[1][1].x, gate.mat[1][1].y
                );
                break;
            }
            case GateType::H: {
                HKernel<1<<THREAD_DEP><<<nVec>>(SINGLE_SIZE_DEP + THREAD_DEP), 1<<THREAD_DEP>>>(deviceStateVec, numQubit_, gate.targetQubit, 1/sqrt(2));
                break;
            }
            case GateType::X: {
                XKernel<1<<THREAD_DEP><<<nVec>>(SINGLE_SIZE_DEP + THREAD_DEP), 1<<THREAD_DEP>>>(deviceStateVec, numQubit_, gate.targetQubit);
                break;
            }
            case GateType::Y: {
                YKernel<1<<THREAD_DEP><<<nVec>>(SINGLE_SIZE_DEP + THREAD_DEP), 1<<THREAD_DEP>>>(deviceStateVec, numQubit_, gate.targetQubit);
                break;
            }
            case GateType::Z: {
                ZKernel<1<<THREAD_DEP><<<nVec>>(SINGLE_SIZE_DEP + THREAD_DEP), 1<<THREAD_DEP>>>(deviceStateVec, numQubit_, gate.targetQubit);
                break;
            }
            case GateType::S: {
                SKernel<1<<THREAD_DEP><<<nVec>>(SINGLE_SIZE_DEP + THREAD_DEP), 1<<THREAD_DEP>>>(deviceStateVec, numQubit_, gate.targetQubit);
                break;
            }
            case GateType::SDG: {
                SKernel<1<<THREAD_DEP><<<nVec>>(SINGLE_SIZE_DEP + THREAD_DEP), 1<<THREAD_DEP>>>(deviceStateVec, numQubit_, gate.targetQubit);
                break;
            }
            case GateType::T: {
                TKernel<1<<THREAD_DEP><<<nVec>>(SINGLE_SIZE_DEP + THREAD_DEP), 1<<THREAD_DEP>>>(deviceStateVec, numQubit_, gate.targetQubit, 1/sqrt(2));
                break;
            }
            case GateType::TDG: {
                TKernel<1<<THREAD_DEP><<<nVec>>(SINGLE_SIZE_DEP + THREAD_DEP), 1<<THREAD_DEP>>>(deviceStateVec, numQubit_, gate.targetQubit, 1/sqrt(2));
                break;
            }
            case GateType::RX: {
                RXKernel<1<<THREAD_DEP><<<nVec>>(SINGLE_SIZE_DEP + THREAD_DEP), 1<<THREAD_DEP>>>(
                    deviceStateVec, numQubit_, gate.targetQubit, gate.mat[0][0].x, -gate.mat[0][1].y);
                break;
            }
            case GateType::RY: {
                RYKernel<1<<THREAD_DEP><<<nVec>>(SINGLE_SIZE_DEP + THREAD_DEP), 1<<THREAD_DEP>>>(
                    deviceStateVec, numQubit_, gate.targetQubit, gate.mat[0][0].x, gate.mat[1][0].x);
                break;
            }
            case GateType::RZ: {
                RZKernel<1<<THREAD_DEP><<<nVec>>(SINGLE_SIZE_DEP + THREAD_DEP), 1<<THREAD_DEP>>>(
                    deviceStateVec, numQubit_, gate.targetQubit, gate.mat[0][0].x, - gate.mat[0][0].y);
                break;
            }
            default: {
                assert(false);
            }
        }
    }
    checkCudaErrors(cudaDeviceSynchronize());
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
__global__ void measure(qComplex* a, qreal* ans, int numQubit_, int targetQubit) {
    __shared__ qreal sdata[blockSize];
    qindex idx = blockIdx.x * blockSize + threadIdx.x;
    int tid = threadIdx.x;
    qindex mask = (qindex(1) << targetQubit) - 1;
    sdata[tid] = 0;
    for (qindex i = (idx << SINGLE_SIZE_DEP); i < ((idx + 1) << SINGLE_SIZE_DEP); i++) {
        qindex lo = ((i >> targetQubit) << (targetQubit + 1)) | (i & mask);
        sdata[tid] += a[lo].x * a[lo].x + a[lo].y * a[lo].y;
    }
    __syncthreads();
    blockReduce<blockSize>(sdata, tid);
    if (tid == 0) ans[blockIdx.x] = sdata[0];
}

qreal kernelMeasure(qComplex* deviceStateVec, int numQubits, int targetQubit) {
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

qComplex kernelGetAmp(qComplex* deviceStateVec, qindex idx) {
    qComplex ret;
    cudaMemcpy(&ret, deviceStateVec + idx, sizeof(qComplex), cudaMemcpyDeviceToHost);
    return ret;
}

void kernelDeviceToHost(qComplex* hostStateVec, qComplex* deviceStateVec, int numQubits) {
    cudaMemcpy(hostStateVec, deviceStateVec, sizeof(qComplex) << numQubits, cudaMemcpyDeviceToHost);
}

void kernelDestroy(qComplex* deviceStateVec) {
    cudaFree(deviceStateVec);
}