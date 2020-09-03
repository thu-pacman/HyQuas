#include "kernel.h"
#include <cstdio>
#include <assert.h>
#include <map>
using namespace std;

#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)
inline void __checkCudaErrors(cudaError_t err, const char *file, const int line) {
    if (cudaSuccess != err)
    {
        fprintf(stderr, "checkCudaErrors() Driver API error = %04d \"%s\" from file %s, line %i.\n", err, cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

struct KernelGate {
    int targetQubit;
    int controlQubit;
    int controlQubit2;
    GateType type;
    char targetIsGlobal;  // 0-local 1-global
    char controlIsGlobal; // 0-local 1-global 2-not control 
    char control2IsGlobal; // 0-local 1-global 2-not control
    qreal r00, i00, r01, i01, r10, i10, r11, i11;
};

const int THREAD_DEP = 7; // 1 << THREAD_DEP threads per block
const int MAX_GATE = 600;
const int MAX_QUBIT = 30;
extern __shared__ qreal real[1<<LOCAL_QUBIT_SIZE];
extern __shared__ qreal imag[1<<LOCAL_QUBIT_SIZE];
extern __shared__ qindex blockBias;

__device__ __constant__ qreal recRoot2 = 0.70710678118654752440084436210485; // more elegant way?
__constant__ KernelGate deviceGates[MAX_GATE];
#ifdef USE_GROUP
int* loIdx;
#endif

__device__ __forceinline__ void XSingle(int lo, int hi) {
    qreal Real = real[lo];
    qreal Imag = imag[lo];
    real[lo] = real[hi];
    imag[lo] = imag[hi];
    real[hi] = Real;
    imag[hi] = Imag;
}

__device__ __forceinline__ void YSingle(int lo, int hi) {
    qreal loReal = real[lo];
    qreal loImag = imag[lo];
    qreal hiReal = real[hi];
    qreal hiImag = imag[hi];
    real[lo] = hiImag;
    imag[lo] = -hiReal;
    real[hi] = -loImag;
    imag[hi] = loReal;
}

__device__ __forceinline__ void ZHi(int hi) {
    real[hi] = -real[hi];
    imag[hi] = -imag[hi];
}


__device__ __forceinline__ void RXSingle(int lo, int hi, qreal alpha, qreal beta) {
    qreal loReal = real[lo];
    qreal loImag = imag[lo];
    qreal hiReal = real[hi];
    qreal hiImag = imag[hi];
    real[lo] = alpha * loReal + beta * hiImag;
    imag[lo] = alpha * loImag - beta * hiReal;
    real[hi] = alpha * hiReal + beta * loImag;
    imag[hi] = alpha * hiImag - beta * loReal;
}

__device__ __forceinline__ void RYSingle(int lo, int hi, qreal alpha, qreal beta) {
    qreal loReal = real[lo];
    qreal loImag = imag[lo];
    qreal hiReal = real[hi];
    qreal hiImag = imag[hi];
    real[lo] = alpha * loReal - beta * hiReal;
    imag[lo] = alpha * loImag - beta * hiImag;
    real[hi] = beta * loReal + alpha * hiReal;
    imag[hi] = beta * loImag + alpha * hiImag;
}

__device__ __forceinline__ void RZSingle(int lo, int hi, qreal alpha, qreal beta){
    qreal loReal = real[lo];
    qreal loImag = imag[lo];
    qreal hiReal = real[hi];
    qreal hiImag = imag[hi];
    real[lo] = alpha * loReal + beta * loImag;
    imag[lo] = alpha * loImag - beta * loReal;
    real[hi] = alpha * hiReal - beta * hiImag;
    imag[hi] = alpha * hiImag + beta * hiReal;
}

__device__ __forceinline__ void RZLo(int lo, qreal alpha, qreal beta) {
    qreal loReal = real[lo];
    qreal loImag = imag[lo];
    real[lo] = alpha * loReal + beta * loImag;
    imag[lo] = alpha * loImag - beta * loReal;
}

__device__ __forceinline__ void RZHi(int hi, qreal alpha, qreal beta){
    qreal hiReal = real[hi];
    qreal hiImag = imag[hi];
    real[hi] = alpha * hiReal - beta * hiImag;
    imag[hi] = alpha * hiImag + beta * hiReal;
}

__device__ __forceinline__ void U1Hi(int hi, qreal alpha, qreal beta) {
    qreal hiReal = real[hi];
    qreal hiImag = imag[hi];
    real[hi] = alpha * hiReal - beta * hiImag;
    imag[hi] = alpha * hiImag + beta * hiReal;
}

#define COMPLEX_MULTIPLY_REAL(i0, r0, i1, r1) (i0 * i1 - r0 * r1)
#define COMPLEX_MULTIPLY_IMAG(i0, r0, i1, r1) (i0 * r1 + i1 * r0)
__device__ __forceinline__ void USingle(int lo, int hi, qreal r00, qreal i00, qreal r01, qreal i01, qreal r10, qreal i10, qreal r11, qreal i11) {
    qreal loReal = real[lo];
    qreal loImag = imag[lo];
    qreal hiReal = real[hi];
    qreal hiImag = imag[hi];
    real[lo] = COMPLEX_MULTIPLY_REAL(loReal, loImag, r00, i00) + COMPLEX_MULTIPLY_REAL(hiReal, hiImag, r01, i01);
    imag[lo] = COMPLEX_MULTIPLY_IMAG(loReal, loImag, r00, i00) + COMPLEX_MULTIPLY_IMAG(hiReal, hiImag, r01, i01);
    real[hi] = COMPLEX_MULTIPLY_REAL(loReal, loImag, r10, i10) + COMPLEX_MULTIPLY_REAL(hiReal, hiImag, r11, i11);
    imag[hi] = COMPLEX_MULTIPLY_IMAG(loReal, loImag, r10, i10) + COMPLEX_MULTIPLY_IMAG(hiReal, hiImag, r11, i11);
}

__device__ __forceinline__ void HSingle(int lo, int hi) {
    qreal loReal = real[lo];
    qreal loImag = imag[lo];
    qreal hiReal = real[hi];
    qreal hiImag = imag[hi];
    real[lo] = recRoot2 * (loReal + hiReal);
    imag[lo] = recRoot2 * (loImag + hiImag);
    real[hi] = recRoot2 * (loReal - hiReal);
    imag[hi] = recRoot2 * (loImag - hiImag);
}

__device__ __forceinline__ void SHi(int hi) {
    qreal hiReal = real[hi];
    qreal hiImag = imag[hi];
    real[hi] = -hiImag;
    imag[hi] = hiReal;
}

__device__ __forceinline__ void THi(int hi) {
    qreal hiReal = real[hi];
    qreal hiImag = imag[hi];
    real[hi] = recRoot2 * (hiReal - hiImag);
    imag[hi] = recRoot2 * (hiReal + hiImag);
}

#define FOLLOW_NEXT(TYPE) \
case GateType::TYPE: // no break

#define CASE_CONTROL(TYPE, OP) \
case GateType::TYPE: { \
    OP; \
    lo += shift; hi += shift; \
    lo ^= shift >> 5; hi ^= shift >> 5; \
    OP; \
    break; \
}

#define CASE_CTR_SMALL_SMALL(TYPE, OP) \
case GateType::TYPE: { \
    OP; \
    lo += shift; hi += shift; \
    OP; \
    break; \
}

#define CASE_SINGLE(TYPE, OP) \
case GateType::TYPE: { \
    for (int task = 0; task < 4; task++) { \
        OP; \
        lo += add[task]; hi += add[task]; \
    } \
    break; \
}


#define CASE_LO_HI(TYPE, OP_LO, OP_HI) \
case GateType::TYPE: { \
    int m = 1 << LOCAL_QUBIT_SIZE; \
    if (!isHighBlock){ \
        for (int k = threadIdx.x; k < m; k += blockSize) { \
            int j = k ^ (k >> 5); \
            OP_LO; \
        } \
    } else { \
        for (int k = threadIdx.x; k < m; k += blockSize) { \
            int j = k ^ (k >> 5); \
            OP_HI; \
        } \
    } \
    break; \
}

#define CASE_SKIPLO_HI(TYPE, OP_HI) \
case GateType::TYPE: { \
    if (!isHighBlock) continue; \
    int m = 1 << LOCAL_QUBIT_SIZE; \
    for (int k = threadIdx.x; k < m; k += blockSize) { \
        int j = k ^ (k >> 5); \
        OP_HI; \
    } \
    break; \
}

template <unsigned int blockSize>
__device__ void doCompute(int numGates, int* loArr) {
    for (int i = 0; i < numGates; i++) {
        int controlQubit = deviceGates[i].controlQubit;
        int targetQubit = deviceGates[i].targetQubit;
        char controlIsGlobal = deviceGates[i].controlIsGlobal;
        char targetIsGlobal = deviceGates[i].targetIsGlobal;
        if (deviceGates[i].type == GateType::CCX) {
            int controlQubit2 = deviceGates[i].controlQubit2;
            int control2IsGlobal = deviceGates[i].control2IsGlobal;
            if (!control2IsGlobal) {
                int m = 1 << (LOCAL_QUBIT_SIZE - 1);
                assert(!controlIsGlobal && !targetIsGlobal);
                assert(deviceGates[i].type == GateType::CCX);
                int maskTarget = (1 << targetQubit) - 1;
                for (int j = threadIdx.x; j < m; j += blockSize) {
                    int lo = ((j >> targetQubit) << (targetQubit + 1)) | (j & maskTarget);
                    if (!(lo >> controlQubit & 1) || !(lo >> controlQubit2 & 1))
                        continue;
                    int hi = lo | (1 << targetQubit);
                    lo = lo ^ (lo >> 5); hi = hi ^ (hi >> 5);
                    XSingle(lo, hi);
                }
                continue;
            }
            if (control2IsGlobal == 1 && !((blockIdx.x >> controlQubit2) & 1)) {
                continue;
            }
        }
        if (!controlIsGlobal) {
            if (!targetIsGlobal) {
                int smallQubit = controlQubit > targetQubit ? targetQubit : controlQubit;
                int largeQubit = controlQubit > targetQubit ? controlQubit : targetQubit;
                int maskSmall = (1 << smallQubit) - 1;
                int maskLarge = (1 << largeQubit) - 1;
                if (controlQubit < 5 && targetQubit < 5) {
                    int lo = loArr[(controlQubit * 5 + targetQubit) << THREAD_DEP | threadIdx.x];
                    int hi = lo ^ (1 << targetQubit);
                    int shift;
                    if (largeQubit == 4) {
                        if (smallQubit == 3) {
                            shift = 1 << 7;
                        } else {
                            shift = 1 << 8;
                        }
                    } else {
                        shift = 1 << 9;
                    }
                    switch (deviceGates[i].type) {
                        CASE_CTR_SMALL_SMALL(CNOT, XSingle(lo, hi))
                        CASE_CTR_SMALL_SMALL(CY, YSingle(lo, hi))
                        CASE_CTR_SMALL_SMALL(CZ, ZHi(hi))
                        CASE_CTR_SMALL_SMALL(CRX, RXSingle(lo, hi, deviceGates[i].r00, deviceGates[i].i01))
                        CASE_CTR_SMALL_SMALL(CRY, RYSingle(lo, hi, deviceGates[i].r00, deviceGates[i].r10))
                        CASE_CTR_SMALL_SMALL(CRZ, RZSingle(lo, hi, deviceGates[i].r00, -deviceGates[i].i00))
                        default: {
                            assert(false);
                        }
                    }
                } else if (controlQubit >= 5 && targetQubit < 5 && controlQubit - targetQubit != 5) {
                    int tid = threadIdx.x;
                    int smallQubit = controlQubit - 5 > targetQubit ? targetQubit : controlQubit - 5;
                    int largeQubit = controlQubit - 5 > targetQubit ? controlQubit - 5 : targetQubit;
                    int maskSmall = (1 << smallQubit) - 1;
                    int maskLarge = (1 << largeQubit) - 1;
                    int maskTarget = (1 << targetQubit) - 1;
                    int x_id = tid >> 5;
                    x_id = x_id >> smallQubit << (smallQubit + 1) | (x_id & maskSmall);
                    x_id = x_id >> largeQubit << (largeQubit + 1) | (x_id & maskLarge);
                    x_id |= (1 << controlQubit - 5);
                    int y_id = tid & 15;
                    y_id = y_id >> targetQubit << (targetQubit + 1) | (y_id & maskTarget);
                    y_id ^= x_id;
                    int lo = x_id << 5 | y_id;
                    if (tid & (1 << 4)) {
                        lo += 33 << targetQubit;
                    }
                    int hi = lo ^ (1 << targetQubit);
                    int shift;
                    if (largeQubit == 4) {
                        if (smallQubit == 3) {
                            shift = 1 << 7;
                        } else {
                            shift = 1 << 8;
                        }
                    } else {
                        shift = 1 << 9;
                    }
                    switch (deviceGates[i].type) {
                        CASE_CTR_SMALL_SMALL(CNOT, XSingle(lo, hi))
                        CASE_CTR_SMALL_SMALL(CY, YSingle(lo, hi))
                        CASE_CTR_SMALL_SMALL(CZ, ZHi(hi))
                        CASE_CTR_SMALL_SMALL(CRX, RXSingle(lo, hi, deviceGates[i].r00, deviceGates[i].i01))
                        CASE_CTR_SMALL_SMALL(CRY, RYSingle(lo, hi, deviceGates[i].r00, deviceGates[i].r10))
                        CASE_CTR_SMALL_SMALL(CRZ, RZSingle(lo, hi, deviceGates[i].r00, -deviceGates[i].i00))
                        default: {
                            assert(false);
                        }
                    }
                } else if (controlQubit < 5 && targetQubit >= 5 && targetQubit - controlQubit != 5) {
                    int tid = threadIdx.x;
                    int smallQubit = controlQubit > targetQubit - 5 ? targetQubit - 5 : controlQubit;
                    int largeQubit = controlQubit > targetQubit - 5 ? controlQubit : targetQubit - 5;
                    int maskSmall = (1 << smallQubit) - 1;
                    int maskLarge = (1 << largeQubit) - 1;
                    int targetMod = (1 << targetQubit) - 1;
                    int maskTarget = (1 << targetMod) - 1;
                    int maskControl = (1 << controlQubit) - 1;
                    int x_id = tid >> 5;
                    x_id = x_id >> smallQubit << (smallQubit + 1) | (x_id & maskSmall);
                    x_id = x_id >> largeQubit << (largeQubit + 1) | (x_id & maskLarge);
                    int y_id = tid & 15;
                    y_id = y_id >> controlQubit << (controlQubit + 1) | (y_id & maskControl);
                    y_id |= 1 << controlQubit;
                    y_id ^= x_id;
                    int lo = x_id << 5 | y_id;
                    if (tid & (1 << 4)) {
                        lo += 31 << controlQubit;
                    }
                    int hi = lo ^ (1 << targetQubit) ^ (1 << (targetQubit - 5));
                    int shift;
                    if (largeQubit == 4) {
                        if (smallQubit == 3) {
                            shift = 1 << 7;
                        } else {
                            shift = 1 << 8;
                        }
                    } else {
                        shift = 1 << 9;
                    }
                    switch (deviceGates[i].type) {
                        CASE_CTR_SMALL_SMALL(CNOT, XSingle(lo, hi))
                        CASE_CTR_SMALL_SMALL(CY, YSingle(lo, hi))
                        CASE_CTR_SMALL_SMALL(CZ, ZHi(hi))
                        CASE_CTR_SMALL_SMALL(CRX, RXSingle(lo, hi, deviceGates[i].r00, deviceGates[i].i01))
                        CASE_CTR_SMALL_SMALL(CRY, RYSingle(lo, hi, deviceGates[i].r00, deviceGates[i].r10))
                        CASE_CTR_SMALL_SMALL(CRZ, RZSingle(lo, hi, deviceGates[i].r00, -deviceGates[i].i00))
                        default: {
                            assert(false);
                        }
                    }
                } else {
                    int m = 1 << (LOCAL_QUBIT_SIZE - 2);
                    int lo = ((threadIdx.x >> smallQubit) << (smallQubit + 1)) | (threadIdx.x & maskSmall);
                    lo = ((lo >> largeQubit) << (largeQubit + 1)) | (lo & maskLarge);
                    lo |= 1 << controlQubit;
                    int hi = lo | (1 << targetQubit);
                    lo ^= lo >> 5;
                    hi ^= hi >> 5;
                    int shift;
                    if (largeQubit == 9) {
                        if (smallQubit == 8) {
                            shift = 1 << 7;
                        } else {
                            shift = 1 << 8;
                        }
                    } else {
                        shift = 1 << 9;
                    }
                    switch (deviceGates[i].type) {
                        CASE_CONTROL(CNOT, XSingle(lo, hi))
                        CASE_CONTROL(CY, YSingle(lo, hi))
                        CASE_CONTROL(CZ, ZHi(hi))
                        CASE_CONTROL(CRX, RXSingle(lo, hi, deviceGates[i].r00, deviceGates[i].i01))
                        CASE_CONTROL(CRY, RYSingle(lo, hi, deviceGates[i].r00, deviceGates[i].r10))
                        CASE_CONTROL(CRZ, RZSingle(lo, hi, deviceGates[i].r00, -deviceGates[i].i00))
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
                    if (deviceGates[i].type == GateType::CRZ) {
                        for (int j = threadIdx.x; j < m; j += blockSize) {
                            int x = ((j >> controlQubit) << (controlQubit + 1)) | (j & maskControl)  | (1 << controlQubit);
                            x ^= x >> 5;
                            RZLo(x, deviceGates[i].r00, - deviceGates[i].i00);
                        }
                    }
                } else {
                    if (deviceGates[i].type == GateType::CRZ) {
                        for (int j = threadIdx.x; j < m; j += blockSize) {
                            int x = ((j >> controlQubit) << (controlQubit + 1)) | (j & maskControl)  | (1 << controlQubit);
                            x ^= x >> 5;
                            RZHi(x, deviceGates[i].r00, - deviceGates[i].i00);
                        }
                    } else {
                        for (int j = threadIdx.x; j < m; j += blockSize) {
                            int x = ((j >> controlQubit) << (controlQubit + 1)) | (j & maskControl)  | (1 << controlQubit);
                            x ^= x >> 5;
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
                int x_id = threadIdx.x >> 5; \
                switch (targetQubit) {
                    case 0: case 5: x_id <<= 1; break;
                    case 1: case 6: x_id = (x_id & 2) << 1 | (x_id & 1); break;
                    default: break;
                }
                int lo, hi;
                if (targetQubit < 5) {
                    int y_id = threadIdx.x & 15;
                    y_id = (y_id >> targetQubit) << (targetQubit + 1) | (y_id & maskTarget);
                    lo = x_id << 5 | y_id;
                    lo += (threadIdx.x & 31) < 16 ? 0 : 33 << targetQubit;
                    hi = lo ^ (1 << targetQubit);
                } else {
                    int y_id = threadIdx.x & 31;
                    lo = x_id << 5 | y_id;
                    hi = lo ^ (33 << (targetQubit - 5));
                }
                int add[4];
                switch (targetQubit) {
                    case 0: case 1: case 2:
                    case 5: case 6: case 7: {
                        add[0] = add[1] = add[2] = 256;
                        break;
                    }
                    case 3: case 8: {
                        add[0] = 128; add[1] = 384; add[2] = 128;
                        break;
                    }
                    case 4: case 9: {
                        add[0] = 128; add[1] = 128; add[2] = 128;
                        break;
                    }
                }
                switch (deviceGates[i].type) {
                    CASE_SINGLE(U1, U1Hi(hi, deviceGates[i].r11, deviceGates[i].i11))
                    FOLLOW_NEXT(U2)
                    CASE_SINGLE(U3, USingle(lo, hi, deviceGates[i].r00, deviceGates[i].i00, deviceGates[i].r01, deviceGates[i].i01, deviceGates[i].r10, deviceGates[i].i10, deviceGates[i].r11, deviceGates[i].i11))
                    CASE_SINGLE(H, HSingle(lo, hi))
                    FOLLOW_NEXT(X)
                    FOLLOW_NEXT(CNOT)
                    CASE_SINGLE(CCX, XSingle(lo, hi))
                    FOLLOW_NEXT(Y)
                    CASE_SINGLE(CY, YSingle(lo, hi))
                    FOLLOW_NEXT(Z)
                    CASE_SINGLE(CZ, ZHi(hi))
                    FOLLOW_NEXT(RX)
                    CASE_SINGLE(CRX, RXSingle(lo, hi, deviceGates[i].r00, -deviceGates[i].i01))
                    FOLLOW_NEXT(RY)
                    CASE_SINGLE(CRY, RYSingle(lo, hi, deviceGates[i].r00, deviceGates[i].r10))
                    FOLLOW_NEXT(RZ)
                    CASE_SINGLE(CRZ, RZSingle(lo, hi, deviceGates[i].r00, -deviceGates[i].i00))
                    CASE_SINGLE(S, SHi(hi))
                    CASE_SINGLE(T, THi(hi))
                    default: {
                        assert(false);
                    }
                }
            } else {
                bool isHighBlock = (blockIdx.x >> targetQubit) & 1;
                switch (deviceGates[i].type) {
                    FOLLOW_NEXT(RZ)
                    CASE_LO_HI(CRZ, RZLo(j, deviceGates[i].i00, - deviceGates[i].r00), RZHi(j, deviceGates[i].i00, - deviceGates[i].r00))
                    FOLLOW_NEXT(Z)
                    CASE_SKIPLO_HI(CZ, ZHi(j))
                    CASE_SKIPLO_HI(S, SHi(j))
                    CASE_SKIPLO_HI(T, THi(j))
                    CASE_SKIPLO_HI(U1, U1Hi(j, deviceGates[i].r11, deviceGates[i].i11))
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
            
        real[x ^ (x >> 5)] = a.real[bias | y];
        imag[x ^ (x >> 5)] = a.imag[bias | y];
    }
}

__device__ void saveData(ComplexArray a, qindex* threadBias, qindex enumerate) {
    qindex bias = blockBias | threadBias[threadIdx.x];
    for (int x = ((1 << (LOCAL_QUBIT_SIZE - THREAD_DEP)) - 1) << THREAD_DEP | threadIdx.x, y = enumerate;
        x >= 0;
        x -= (1 << THREAD_DEP), y = enumerate & (y - 1)) {
        
        a.real[bias | y] = real[x ^ (x >> 5)];
        a.imag[bias | y] = imag[x ^ (x >> 5)];
    }
}

template <unsigned int blockSize>
__global__ void run(ComplexArray a, qindex* threadBias, int* loArr, int numQubits, int numGates, qindex blockHot, qindex enumerate) {
    qindex idx = blockIdx.x * blockSize + threadIdx.x;
    fetchData(a, threadBias, idx, blockHot, enumerate, numQubits);
    __syncthreads();
    doCompute<blockSize>(numGates, loArr);
    __syncthreads();
    saveData(a, threadBias, enumerate);
}

#ifdef USE_GROUP
void initControlIdx() {
    cudaMalloc(&loIdx, sizeof(int) * 25 * (1 << THREAD_DEP));
    int loIdx_host[5][5][1 << THREAD_DEP];
    for (int controlQubit = 0; controlQubit < 5; controlQubit ++)
        for (int targetQubit = 0; targetQubit < 5; targetQubit ++)
            for (int tid = 0; tid < (1 << THREAD_DEP); tid++) {
                if (controlQubit == targetQubit) continue;
                int smallQubit = controlQubit > targetQubit ? targetQubit : controlQubit;
                int largeQubit = controlQubit > targetQubit ? controlQubit : targetQubit;
                int maskSmall = (1 << smallQubit) - 1;
                int maskLarge = (1 << largeQubit) - 1;
                int x_id = tid >> 5;
                x_id = x_id >> smallQubit << (smallQubit + 1) | (x_id & maskSmall);
                x_id = x_id >> largeQubit << (largeQubit + 1) | (x_id & maskLarge);
                int y_id = tid & 7;
                y_id = y_id >> smallQubit << (smallQubit + 1) | (y_id & maskSmall);
                y_id = y_id >> largeQubit << (largeQubit + 1) | (y_id & maskLarge);
                y_id |= 1 << controlQubit;
                int lo = x_id << 5 | y_id;
                if (tid & (1 << 3)) {
                    lo += 33 << targetQubit;
                }
                if (tid & (1 << 4)) {
                    lo += 31 << controlQubit;
                }
                loIdx_host[controlQubit][targetQubit][tid] = lo;
            }
    checkCudaErrors(cudaMemcpy(loIdx, loIdx_host[0][0], sizeof(int) * 25 * (1 << THREAD_DEP), cudaMemcpyHostToDevice));
}
#endif

std::vector<qreal> kernelExecOpt(ComplexArray& deviceStateVec, int numQubits, const Schedule& schedule) {
    assert(numQubits <= MAX_QUBIT);
    qindex hostThreadBias[1 << THREAD_DEP];
    qindex* threadBias;
    checkCudaErrors(cudaMalloc(&threadBias, sizeof(hostThreadBias)));
    std::vector<qreal> ret;
    for (size_t g = 0; g < schedule.gateGroups.size(); g++) {
#ifdef MEASURE_STAGE
        cudaEvent_t start, stop;
            checkCudaErrors(cudaEventCreate(&start));
            checkCudaErrors(cudaEventCreate(&stop));
            checkCudaErrors(cudaEventRecord(start, 0));
#endif
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
            if (gates[i].controlQubit2 != -1) {
                int c1 = gates[i].controlQubit;
                int c2 = gates[i].controlQubit2;
                if (isLocalQubit(c2) && !isLocalQubit(c1)) {
                    int c = c1; c1 = c2; c2 = c;
                }
                hostGates[i].controlQubit2 = toID[c2];
                hostGates[i].control2IsGlobal = 1 - isLocalQubit(c2);
                hostGates[i].controlQubit = toID[c1];
                hostGates[i].controlIsGlobal = 1 - isLocalQubit(c1);
            } else if (gates[i].controlQubit != -1) {
                hostGates[i].controlQubit2 = -1;
                hostGates[i].control2IsGlobal = 2;
                hostGates[i].controlQubit = toID[gates[i].controlQubit];
                hostGates[i].controlIsGlobal = 1 - isLocalQubit(gates[i].controlQubit);
            } else {
                hostGates[i].controlQubit2 = -1;
                hostGates[i].control2IsGlobal = 2;
                hostGates[i].controlQubit = -1;
                hostGates[i].controlIsGlobal = 2;
            }

            hostGates[i].targetQubit = toID[gates[i].targetQubit];
            hostGates[i].targetIsGlobal = 1 - isLocalQubit(gates[i].targetQubit);
            
            hostGates[i].type = gates[i].type;
        }
        checkCudaErrors(cudaMemcpyToSymbol(deviceGates, hostGates, sizeof(hostGates)));

        // execute
        qindex gridDim = (1 << numQubits) >> LOCAL_QUBIT_SIZE;
        run<1<<THREAD_DEP><<<gridDim, 1<<THREAD_DEP>>>
            (deviceStateVec, threadBias, loIdx, numQubits, gates.size(), blockHot, enumerate);
#ifdef MEASURE_STAGE
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float time;
            cudaEventElapsedTime(&time, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            printf("[Group %d] time for %x: %f\n", int(g), relatedQubits, time);
#endif
        // printf("Group End\n");
    }
    checkCudaErrors(cudaFree(threadBias));
    checkCudaErrors(cudaDeviceSynchronize()); // WARNING: for time measure!
    return ret;
}

