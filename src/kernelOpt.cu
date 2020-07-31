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
    int targetQubit;
    int controlQubit;
    GateType type;
    char targetIsGlobal;  // 0-local 1-global
    char controlIsGlobal; // 0-local 1-global 2-not control 
    qreal r00, i00, r01, i01, r10, i10, r11, i11;
};

const int THREAD_DEP = 7; // 1 << THREAD_DEP threads per block
const int MAX_GATE = 152;
const int MAX_QUBIT = 30;
extern __shared__ qreal real[1<<LOCAL_QUBIT_SIZE];
extern __shared__ qreal imag[1<<LOCAL_QUBIT_SIZE];
extern __shared__ qindex blockBias;

__device__ __constant__ double recRoot2 = 0.70710678118654752440084436210485; // more elegant way?
__constant__ KernelGate deviceGates[MAX_GATE];


__device__ inline void XSingle(int lo, int hi) {
    qreal Real = real[lo];
    qreal Imag = imag[lo];
    real[lo] = real[hi];
    imag[lo] = imag[hi];
    real[hi] = Real;
    imag[hi] = Imag;
}

__device__ inline void YSingle(int lo, int hi) {
    qreal loReal = real[lo];
    qreal loImag = imag[lo];
    qreal hiReal = real[hi];
    qreal hiImag = imag[hi];
    real[lo] = hiImag;
    imag[lo] = -hiReal;
    real[hi] = -loImag;
    imag[hi] = loReal;
}

__device__ inline void ZHi(int hi) {
    real[hi] = -real[hi];
    imag[hi] = -imag[hi];
}


__device__ inline void RXSingle(int lo, int hi, qreal alpha, qreal beta) {
    qreal loReal = real[lo];
    qreal loImag = imag[lo];
    qreal hiReal = real[hi];
    qreal hiImag = imag[hi];
    real[lo] = alpha * loReal + beta * hiImag;
    imag[lo] = alpha * loImag - beta * hiReal;
    real[hi] = alpha * hiReal + beta * loImag;
    imag[hi] = alpha * hiImag - beta * loReal;
}

__device__ inline void RYSingle(int lo, int hi, qreal alpha, qreal beta) {
    qreal loReal = real[lo];
    qreal loImag = imag[lo];
    qreal hiReal = real[hi];
    qreal hiImag = imag[hi];
    real[lo] = alpha * loReal - beta * hiReal;
    imag[lo] = alpha * loImag - beta * hiImag;
    real[hi] = beta * loReal + alpha * hiReal;
    imag[hi] = beta * loImag + alpha * hiImag;
}

__device__ inline void RZSingle(int lo, int hi, qreal alpha, qreal beta){
    qreal loReal = real[lo];
    qreal loImag = imag[lo];
    qreal hiReal = real[hi];
    qreal hiImag = imag[hi];
    real[lo] = alpha * loReal + beta * loImag;
    imag[lo] = alpha * loImag - beta * loReal;
    real[hi] = alpha * hiReal - beta * hiImag;
    imag[hi] = alpha * hiImag + beta * hiReal;
}

__device__ inline void RZLo(int lo, qreal alpha, qreal beta) {
    qreal loReal = real[lo];
    qreal loImag = imag[lo];
    real[lo] = alpha * loReal + beta * loImag;
    imag[lo] = alpha * loImag - beta * loReal;
}

__device__ inline void RZHi(int hi, qreal alpha, qreal beta){
    qreal hiReal = real[hi];
    qreal hiImag = imag[hi];
    real[hi] = alpha * hiReal - beta * hiImag;
    imag[hi] = alpha * hiImag + beta * hiReal;
}

__device__ inline void U1Hi(int hi, qreal alpha, qreal beta) {
    qreal hiReal = real[hi];
    qreal hiImag = imag[hi];
    real[hi] = alpha * hiReal - beta * hiImag;
    imag[hi] = alpha * hiImag + beta * hiReal;
}

#define COMPLEX_MULTIPLY_REAL(i0, r0, i1, r1) (i0 * i1 - r0 * r1)
#define COMPLEX_MULTIPLY_IMAG(i0, r0, i1, r1) (i0 * r1 + i1 * r0)
__device__ inline void USingle(int lo, int hi, qreal r00, qreal i00, qreal r01, qreal i01, qreal r10, qreal i10, qreal r11, qreal i11) {
    qreal loReal = real[lo];
    qreal loImag = imag[lo];
    qreal hiReal = real[hi];
    qreal hiImag = imag[hi];
    real[lo] = COMPLEX_MULTIPLY_REAL(loReal, loImag, r00, i00) + COMPLEX_MULTIPLY_REAL(hiReal, hiImag, r01, i01);
    imag[lo] = COMPLEX_MULTIPLY_IMAG(loReal, loImag, r00, i00) + COMPLEX_MULTIPLY_IMAG(hiReal, hiImag, r01, i01);
    real[hi] = COMPLEX_MULTIPLY_REAL(loReal, loImag, r10, i10) + COMPLEX_MULTIPLY_REAL(hiReal, hiImag, r11, i11);
    real[hi] = COMPLEX_MULTIPLY_IMAG(loReal, loImag, r10, i10) + COMPLEX_MULTIPLY_IMAG(hiReal, hiImag, r11, i11);
}

__device__ inline void HSingle(int lo, int hi) {
    qreal loReal = real[lo];
    qreal loImag = imag[lo];
    qreal hiReal = real[hi];
    qreal hiImag = imag[hi];
    real[lo] = recRoot2 * (loReal + hiReal);
    imag[lo] = recRoot2 * (loImag + hiImag);
    real[hi] = recRoot2 * (loReal - hiReal);
    imag[hi] = recRoot2 * (loImag - hiImag);
}

__device__ inline void SHi(int hi) {
    qreal hiReal = real[hi];
    qreal hiImag = imag[hi];
    real[hi] = -hiImag;
    imag[hi] = hiReal;
}

__device__ inline void THi(int hi) {
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
                        // controlled gates' base type
                        case GateType::CNOT: {
                            XSingle(lo, hi);
                            break;
                        }
                        case GateType::CY: {
                            YSingle(lo, hi);
                            break;
                        }
                        case GateType::CZ: {
                            ZHi(hi);
                            break;
                        }
                        case GateType::CRX: {
                            RXSingle(lo, hi, deviceGates[i].r00, -deviceGates[i].i00);
                            break;
                        }
                        case GateType::CRY: {
                            RYSingle(lo, hi, deviceGates[i].r00, deviceGates[i].r10);
                            break;
                        }
                        case GateType::CRZ: {
                            RZSingle(lo, hi, deviceGates[i].r00, deviceGates[i].i00);
                            break;
                        }
                        default: {
                            assert(false);
                        }
                    }
                }
            } else {
                assert(deviceGates[i].type == GateType::CZ || deviceGates[i].type == GateType::CRZ);
                bool isHighBlock = (blockIdx.x >> targetQubit) & 1;
                int m = 1 << (LOCAL_QUBIT_SIZE - 1);
                int maskControl = (1 << controlQubit) - 1;
                if (!isHighBlock){
                    for (int j = threadIdx.x; j < m; j += blockSize) {
                        int x = ((j >> controlQubit) << (controlQubit + 1)) | (j & maskControl)  | (1 << controlQubit);
                        if (deviceGates[i].type == GateType::CRZ) {
                            RZLo(x, deviceGates[i].r00, - deviceGates[i].i00);
                        }
                    }
                } else {
                    for (int j = threadIdx.x; j < m; j += blockSize) {
                        int x = ((j >> controlQubit) << (controlQubit + 1)) | (j & maskControl)  | (1 << controlQubit);
                        if (deviceGates[i].type == GateType::CRZ) {
                            RZHi(x, deviceGates[i].r00, - deviceGates[i].i00);
                        } else {
                            ZHi(x);
                        }
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
                        case GateType::U1: {
                            U1Hi(hi, deviceGates[i].r11, deviceGates[i].i11);
                            break;
                        }
                        case GateType::U2:
                        case GateType::U3: {
                            USingle(lo, hi, deviceGates[i].r00, deviceGates[i].i00, deviceGates[i].r01, deviceGates[i].i01, deviceGates[i].r10, deviceGates[i].i10, deviceGates[i].r11, deviceGates[i].i11);
                            break;
                        }
                        case GateType::H: {
                            HSingle(lo, hi);
                            break;
                        }
                        case GateType::X: {
                            XSingle(lo, hi);
                            break;
                        }
                        case GateType::Y: {
                            YSingle(lo, hi);
                            break;
                        }
                        case GateType::Z: // no break
                        case GateType::CZ: {
                            ZHi(hi);
                            break;
                        }
                        case GateType::RX: {
                            RXSingle(lo, hi, deviceGates[i].r00, -deviceGates[i].i01);
                            break;
                        }
                        case GateType::RY: {
                            RYSingle(lo, hi, deviceGates[i].r00, deviceGates[i].r10);
                            break;
                        }
                        case GateType::RZ: // no break
                        case GateType::CRZ: {
                            RZSingle(lo, hi, deviceGates[i].r00, -deviceGates[i].i00);
                            break;
                        }
                        case GateType::S: {
                            SHi(hi);
                            break;
                        }
                        case GateType::T: {
                            THi(hi);
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
                    case GateType::CZ: // no break
                    case GateType::Z: {
                        int m = 1 << LOCAL_QUBIT_SIZE;
                        if (!isHighBlock){
                            for (int j = threadIdx.x; j < m; j += blockSize) {
                                RZLo(j, deviceGates[i].i00, - deviceGates[i].r00);
                            }
                        } else {
                            for (int j = threadIdx.x; j < m; j += blockSize) {
                                RZHi(j, deviceGates[i].i00, - deviceGates[i].r00);
                            }
                        }
                        break;
                    }
                    case GateType::RZ:
                    case GateType::CRZ: {
                        if (!isHighBlock) continue;
                        int m = 1 << LOCAL_QUBIT_SIZE;
                        for (int j = threadIdx.x; j < m; j += blockSize) {
                            ZHi(j);
                        }
                        break;
                    }
                    case GateType::S: {
                        if (!isHighBlock) continue;
                        int m = 1 << LOCAL_QUBIT_SIZE;
                        for (int j = threadIdx.x; j < m; j += blockSize) {
                            SHi(j);
                        }
                        break;
                    }
                    case GateType::T: {
                        if (!isHighBlock) continue;
                        int m = 1 << LOCAL_QUBIT_SIZE;
                        for (int j = threadIdx.x; j < m; j += blockSize) {
                            THi(j);
                        }
                        break;
                    }
                    case GateType::U1: {
                        if (!isHighBlock) continue;
                        int m = 1 << LOCAL_QUBIT_SIZE;
                        for (int j = threadIdx.x; j < m; j += blockSize) {
                            U1Hi(j, deviceGates[i].r11, deviceGates[i].i11);
                        }
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

__device__ void saveData(ComplexArray a, qindex* threadBias, qindex enumerate) {
    qindex bias = blockBias | threadBias[threadIdx.x];
    for (int x = ((1 << (LOCAL_QUBIT_SIZE - THREAD_DEP)) - 1) << THREAD_DEP | threadIdx.x, y = enumerate;
        x >= 0;
        x -= (1 << THREAD_DEP), y = enumerate & (y - 1)) {
        
        a.real[bias | y] = real[x];
        a.imag[bias | y] = imag[x];
    }
}

#define REDUCE_QUBIT_STEP(x) {\
if (blockSize >= x * 2) { \
    if (tid < x) { \
            sdata[tid] += sdata[tid + x]; \
    } \
} \
__syncthreads();\
}

#define REDUCE_SINGLE_STEP(x) {\
if (blockSize >= x * 2) { \
    if (tid < x) { \
        sdata[tid] += sdata[tid + x]; \
    } \
} \
__syncthreads(); \
}

template <unsigned int blockSize>
__device__ void measure(qreal* result, qindex* threadBias, int numQubits, qindex enumerate) {
    int tid = threadIdx.x;
    qindex bias = blockBias | threadBias[tid];
    __shared__ qreal sdata[blockSize];
    for (int x = ((1 << (LOCAL_QUBIT_SIZE - THREAD_DEP)) - 1) << THREAD_DEP | tid, y = enumerate;
        x >= 0;
        x -= (1 << THREAD_DEP), y = enumerate & (y - 1)) {
            real[x] = real[x] * real[x] + imag[x] * imag[x];
    }
    for (int j = 0; j < numQubits; j++) {
        sdata[tid] = 0;
        for (int x = ((1 << (LOCAL_QUBIT_SIZE - THREAD_DEP)) - 1) << THREAD_DEP | tid, y = enumerate;
            x >= 0;
            x -= (1 << THREAD_DEP), y = enumerate & (y - 1)) {
            
            qindex target = bias | y;
            if ((target >> j) & 1) {
                sdata[tid] += real[x];
            }
        }
        __syncthreads();
        REDUCE_QUBIT_STEP(512);
        REDUCE_QUBIT_STEP(256);
        REDUCE_QUBIT_STEP(128);
        REDUCE_QUBIT_STEP(64);
        REDUCE_QUBIT_STEP(32);
        REDUCE_QUBIT_STEP(16);
        REDUCE_QUBIT_STEP(8);
        REDUCE_QUBIT_STEP(4);
        REDUCE_QUBIT_STEP(2);
        REDUCE_QUBIT_STEP(1);
        if (tid == 0) {
            result[j * gridDim.x + blockIdx.x] = sdata[0];
        }
    }
}

template <unsigned int blockSize>
__global__ void run(ComplexArray a, qindex* threadBias, int numQubits, int numGates, qindex blockHot, qindex enumerate) {
    qindex idx = blockIdx.x * blockSize + threadIdx.x;
    fetchData(a, threadBias, idx, blockHot, enumerate, numQubits);
    __syncthreads();
    doCompute<blockSize>(numGates);
    __syncthreads();
    saveData(a, threadBias, enumerate);
}

template <unsigned int blockSize>
__global__ void runLast(ComplexArray a, qreal* result, qindex* threadBias, int numQubits, int numGates, qindex blockHot, qindex enumerate) {
    qindex idx = blockIdx.x * blockSize + threadIdx.x;
    fetchData(a, threadBias, idx, blockHot, enumerate, numQubits);
    __syncthreads();
    doCompute<blockSize>(numGates);
    __syncthreads();
    saveData(a, threadBias, enumerate);
    measure<blockSize>(result, threadBias, numQubits, enumerate); // measure must behind save
}

template <unsigned int blockSize, typename T>
__global__ void reduceSum(T* g_idata, T* g_odata, int n) {
    __shared__ T sdata[blockSize];
    int tid = threadIdx.x;
    qindex i = blockIdx.x * n + tid;
    int gridSize = blockSize * 2;
    sdata[tid] = 0;
    while (i < (blockIdx.x + 1) * n) {
        sdata[tid] += g_idata[i] + g_idata[i + blockSize]; i += gridSize;
    }
    __syncthreads();
    REDUCE_SINGLE_STEP(512);
    REDUCE_SINGLE_STEP(256);
    REDUCE_SINGLE_STEP(128);
    REDUCE_SINGLE_STEP(64);
    REDUCE_SINGLE_STEP(32);
    REDUCE_SINGLE_STEP(16);
    REDUCE_SINGLE_STEP(8);
    REDUCE_SINGLE_STEP(4);
    REDUCE_SINGLE_STEP(2);
    REDUCE_SINGLE_STEP(1);
    // if (tid == 0) printf("%d: sdata %f\n", blockIdx.x, sdata[0]);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

std::vector<qreal> kernelExecOpt(ComplexArray& deviceStateVec, int numQubits, const Schedule& schedule) {
    assert(numQubits <= MAX_QUBIT);
    qindex hostThreadBias[1 << THREAD_DEP];
    qindex* threadBias;
    checkCudaErrors(cudaMalloc(&threadBias, sizeof(hostThreadBias)));
    std::vector<qreal> ret;
    for (size_t g = 0; g < schedule.gateGroups.size(); g++) {
        cudaEvent_t start, stop;
        if (MEASURE_STAGE) {
            checkCudaErrors(cudaEventCreate(&start));
            checkCudaErrors(cudaEventCreate(&stop));
            checkCudaErrors(cudaEventRecord(start, 0));
        }
        auto& gates = schedule.gateGroups[g].gates;
        // initialize blockHot, enumerate, threadBias
        qindex relatedQubits = schedule.gateGroups[g].relatedQubits;
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
        KernelGate hostGates[gates.size()];
        assert(gates.size() < MAX_GATE);
        for (size_t i = 0; i < gates.size(); i++) {
            hostGates[i].r00 = gates[i].mat[0][0].real;
            hostGates[i].i00 = gates[i].mat[0][0].imag;
            hostGates[i].r01 = gates[i].mat[0][1].real;
            hostGates[i].i01 = gates[i].mat[0][1].imag;
            hostGates[i].r10 = gates[i].mat[1][0].real;
            hostGates[i].i10 = gates[i].mat[1][0].imag;
            hostGates[i].r11 = gates[i].mat[1][1].real;
            hostGates[i].i11 = gates[i].mat[1][1].imag;
            if (gates[i].controlQubit == -1) {
                hostGates[i].controlQubit = -1;
                hostGates[i].controlIsGlobal = 2;
            } else {
                hostGates[i].controlQubit = toID[gates[i].controlQubit];
                hostGates[i].controlIsGlobal = 1 - isLocalQubit(gates[i].controlQubit);
            }

            hostGates[i].targetQubit = toID[gates[i].targetQubit];
            hostGates[i].targetIsGlobal = 1 - isLocalQubit(gates[i].targetQubit);
            
            hostGates[i].type = gates[i].type;
        }
        checkCudaErrors(cudaMemcpyToSymbol(deviceGates, hostGates, sizeof(hostGates)));

        // execute
        qindex gridDim = (1 << numQubits) >> LOCAL_QUBIT_SIZE;
        if (g == schedule.gateGroups.size() - 1) {
            qreal *deviceResultRaw, *deviceResult;
            // raw format:
            // ---------------------------------------------------
            // | qubit 0 (* gridDim) | qubit 1 (* gridDim) | ...
            // ---------------------------------------------------
            checkCudaErrors(cudaMalloc(&deviceResultRaw, gridDim * sizeof(qreal) * numQubits));
            runLast<1<<THREAD_DEP><<<gridDim, 1<<THREAD_DEP>>>
                (deviceStateVec, deviceResultRaw, threadBias, numQubits, gates.size(), blockHot, enumerate);
            checkCudaErrors(cudaMalloc(&deviceResult, sizeof(qreal) * numQubits));
            reduceSum<1<<THREAD_DEP><<<numQubits, 1<<THREAD_DEP>>>(deviceResultRaw, deviceResult, gridDim);
            ret.resize(numQubits);
            checkCudaErrors(cudaMemcpy(ret.data(), deviceResult, sizeof(qreal) * numQubits, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaFree(deviceResult));
            checkCudaErrors(cudaFree(deviceResultRaw));
        }else {
            run<1<<THREAD_DEP><<<gridDim, 1<<THREAD_DEP>>>
                (deviceStateVec, threadBias, numQubits, gates.size(), blockHot, enumerate);
        }
        if (MEASURE_STAGE) {
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float time;
            cudaEventElapsedTime(&time, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            printf("[Group %d] time for %x: %f\n", int(g), relatedQubits, time);
        }
        // printf("Group End\n");
    }
    checkCudaErrors(cudaFree(threadBias));
    if (MEASURE_STAGE) {
        checkCudaErrors(cudaDeviceSynchronize()); // WARNING: for time measure!
    }
    return ret;
}

