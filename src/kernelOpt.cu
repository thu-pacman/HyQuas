#include "kernel.h"
#include <cstdio>
#include <assert.h>
#include <map>
#include <omp.h>
#include "gate.h"
#include "executor.h"
using namespace std;

extern __shared__ qComplex shm[1<<LOCAL_QUBIT_SIZE];
extern __shared__ qindex blockBias;

__device__ __constant__ qreal recRoot2 = 0.70710678118654752440084436210485; // more elegant way?
__constant__ KernelGate deviceGates[MAX_GATE];

std::vector<int*> loIdx_device;
std::vector<int*> shiftAt_device;


__device__ __forceinline__ void XSingle(int loIdx, int hiIdx) {
    qComplex v = shm[loIdx];
    shm[loIdx] = shm[hiIdx];
    shm[hiIdx] = v;
}

__device__ __forceinline__ void YSingle(int loIdx, int hiIdx) {
    qComplex lo = shm[loIdx];
    qComplex hi = shm[hiIdx];
    
    shm[loIdx] = make_qComplex(hi.y, -hi.x);
    shm[hiIdx] = make_qComplex(-lo.y, lo.x);
}

__device__ __forceinline__ void ZHi(int hiIdx) {
    qComplex v = shm[hiIdx];
    shm[hiIdx] = make_qComplex(-v.x, -v.y);
}


__device__ __forceinline__ void RXSingle(int loIdx, int hiIdx, qreal alpha, qreal beta) {
    qComplex lo = shm[loIdx];
    qComplex hi = shm[hiIdx];
    shm[loIdx] = make_qComplex(alpha * lo.x + beta * hi.y, alpha * lo.y - beta * hi.x);
    shm[hiIdx] = make_qComplex(alpha * hi.x + beta * lo.y, alpha * hi.y - beta * lo.x);
}

__device__ __forceinline__ void RYSingle(int loIdx, int hiIdx, qreal alpha, qreal beta) {
    qComplex lo = shm[loIdx];
    qComplex hi = shm[hiIdx];
    shm[loIdx] = make_qComplex(alpha * lo.x - beta * hi.x, alpha * lo.y - beta * hi.y);
    shm[hiIdx] = make_qComplex(beta * lo.x + alpha * hi.x, beta * lo.y + alpha * hi.y);
}

__device__ __forceinline__ void RZSingle(int loIdx, int hiIdx, qreal alpha, qreal beta){
    qComplex lo = shm[loIdx];
    qComplex hi = shm[hiIdx];
    shm[loIdx] = make_qComplex(alpha * lo.x + beta * lo.y, alpha * lo.y - beta * lo.x);
    shm[hiIdx] = make_qComplex(alpha * hi.x - beta * hi.y, alpha * hi.y + beta * hi.x);
}

__device__ __forceinline__ void RZLo(int loIdx, qreal alpha, qreal beta) {
    qComplex lo = shm[loIdx];
    shm[loIdx] = make_qComplex(alpha * lo.x + beta * lo.y, alpha * lo.y - beta * lo.x);
}

__device__ __forceinline__ void RZHi(int hiIdx, qreal alpha, qreal beta){
    qComplex hi = shm[hiIdx];
    shm[hiIdx] = make_qComplex(alpha * hi.x - beta * hi.y, alpha * hi.y + beta * hi.x);
}

#define COMPLEX_MULTIPLY_REAL(v0, v1) (v0.x * v1.x - v0.y * v1.y)
#define COMPLEX_MULTIPLY_IMAG(v0, v1) (v0.x * v1.y + v0.y * v1.x)

__device__ __forceinline__ void U1Hi(int hiIdx, qComplex p) {
    qComplex hi = shm[hiIdx];
    shm[hiIdx] = make_qComplex(COMPLEX_MULTIPLY_REAL(hi, p), COMPLEX_MULTIPLY_IMAG(hi, p));
}

__device__ __forceinline__ void USingle(int loIdx, int hiIdx, qComplex v00, qComplex v01, qComplex v10, qComplex v11) {
    qComplex lo = shm[loIdx];
    qComplex hi = shm[hiIdx];
    shm[loIdx] = make_qComplex(COMPLEX_MULTIPLY_REAL(lo, v00) + COMPLEX_MULTIPLY_REAL(hi, v01),
                               COMPLEX_MULTIPLY_IMAG(lo, v00) + COMPLEX_MULTIPLY_IMAG(hi, v01));
    shm[hiIdx] = make_qComplex(COMPLEX_MULTIPLY_REAL(lo, v10) + COMPLEX_MULTIPLY_REAL(hi, v11),
                               COMPLEX_MULTIPLY_IMAG(lo, v10) + COMPLEX_MULTIPLY_IMAG(hi, v11));
}

__device__ __forceinline__ void HSingle(int loIdx, int hiIdx) {
    qComplex lo = shm[loIdx];
    qComplex hi = shm[hiIdx];
    shm[loIdx] = make_qComplex(recRoot2 * (lo.x + hi.x), recRoot2 * (lo.y + hi.y));
    shm[hiIdx] = make_qComplex(recRoot2 * (lo.x - hi.x), recRoot2 * (lo.y - hi.y));
}

__device__ __forceinline__ void SHi(int hiIdx) {
    qComplex hi = shm[hiIdx];
    shm[hiIdx] = make_qComplex(-hi.y, hi.x);
}

__device__ __forceinline__ void THi(int hiIdx) {
    qComplex hi = shm[hiIdx];
    shm[hiIdx] = make_qComplex(recRoot2 * (hi.x - hi.y), recRoot2 * (hi.x + hi.y));
}

__device__ __forceinline__ void GIISingle(int loIdx, int hiIdx) {
    qComplex lo = shm[loIdx];
    shm[loIdx] = make_qComplex(-lo.y, lo.x);
    qComplex hi = shm[hiIdx];
    shm[hiIdx] = make_qComplex(-hi.y, hi.x);
}

__device__ __forceinline__ void GII(int idx) {
    qComplex v = shm[idx];
    shm[idx] = make_qComplex(-v.y, v.x);
}

__device__ __forceinline__ void GZZSingle(int loIdx, int hiIdx) {
    qComplex lo = shm[loIdx];
    shm[loIdx] = make_qComplex(-lo.x, -lo.y);
    qComplex hi = shm[hiIdx];
    shm[hiIdx] = make_qComplex(-hi.x, -hi.y);
}

__device__ __forceinline__ void GZZ(int idx) { 
    qComplex v = shm[idx];
    shm[idx] = make_qComplex(-v.x, -v.y);
}

__device__ __forceinline__ void GCCSingle(int loIdx, int hiIdx, qComplex p) {
    qComplex lo = shm[loIdx];
    shm[loIdx] = make_qComplex(COMPLEX_MULTIPLY_REAL(lo, p), COMPLEX_MULTIPLY_IMAG(lo, p));
    qComplex hi = shm[hiIdx];
    shm[hiIdx] = make_qComplex(COMPLEX_MULTIPLY_REAL(hi, p), COMPLEX_MULTIPLY_IMAG(hi, p));
}

__device__ __forceinline__ void GCC(int idx, qComplex p) {
    qComplex v = shm[idx];
    shm[idx] = make_qComplex(COMPLEX_MULTIPLY_REAL(v, p), COMPLEX_MULTIPLY_IMAG(v, p));
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

#define LOHI_SAME(TYPE, OP) \
case GateType::TYPE: { \
    int m = 1 << LOCAL_QUBIT_SIZE; \
    for (int k = threadIdx.x; k < m; k += blockSize) { \
        int j = k ^ (k >> 5); \
        OP; \
    } \
    break; \
}

#define ID_BREAK() \
case GateType::ID: { \
    break; \
}

template <unsigned int blockSize>
__device__ void doCompute(int numGates, int* loArr, int* shiftAt) {
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
                if ((controlQubit < 5 || targetQubit < 5) && (controlQubit - targetQubit != 5) && (targetQubit - controlQubit != 5)) {
                    int lo = loArr[(controlQubit * 10 + targetQubit) << THREAD_DEP | threadIdx.x];
                    int hi = lo ^ (1 << targetQubit);
                    if (targetQubit >= 5) {
                        hi ^= 1 << (targetQubit - 5);
                    }
                    int shift = shiftAt[controlQubit * 10 + targetQubit];
                    switch (deviceGates[i].type) {
                        FOLLOW_NEXT(CCX)
                        CASE_CTR_SMALL_SMALL(CNOT, XSingle(lo, hi))
                        CASE_CTR_SMALL_SMALL(CY, YSingle(lo, hi))
                        CASE_CTR_SMALL_SMALL(CZ, ZHi(hi))
                        CASE_CTR_SMALL_SMALL(CRX, RXSingle(lo, hi, deviceGates[i].r00, -deviceGates[i].i01))
                        CASE_CTR_SMALL_SMALL(CRY, RYSingle(lo, hi, deviceGates[i].r00, deviceGates[i].r10))
                        CASE_CTR_SMALL_SMALL(CRZ, RZSingle(lo, hi, deviceGates[i].r00, -deviceGates[i].i00))
                        default: {
                            assert(false);
                        }
                    }
                } else {
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
                        FOLLOW_NEXT(CCX)
                        CASE_CONTROL(CNOT, XSingle(lo, hi))
                        CASE_CONTROL(CY, YSingle(lo, hi))
                        CASE_CONTROL(CZ, ZHi(hi))
                        CASE_CONTROL(CRX, RXSingle(lo, hi, deviceGates[i].r00, -deviceGates[i].i01))
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
                            RZLo(x, deviceGates[i].r00, -deviceGates[i].i00);
                        }
                    }
                } else {
                    if (deviceGates[i].type == GateType::CRZ) {
                        for (int j = threadIdx.x; j < m; j += blockSize) {
                            int x = ((j >> controlQubit) << (controlQubit + 1)) | (j & maskControl)  | (1 << controlQubit);
                            x ^= x >> 5;
                            RZHi(x, deviceGates[i].r00, -deviceGates[i].i00);
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
                    FOLLOW_NEXT(GOC)
                    CASE_SINGLE(U1, U1Hi(hi, make_qComplex(deviceGates[i].r11, deviceGates[i].i11)))
                    FOLLOW_NEXT(U2)
                    CASE_SINGLE(U3, USingle(lo, hi, make_qComplex(deviceGates[i].r00, deviceGates[i].i00), make_qComplex(deviceGates[i].r01, deviceGates[i].i01), make_qComplex(deviceGates[i].r10, deviceGates[i].i10), make_qComplex(deviceGates[i].r11, deviceGates[i].i11)));
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
                    CASE_SINGLE(GII, GIISingle(lo, hi))
                    CASE_SINGLE(GZZ, GZZSingle(lo, hi))
                    CASE_SINGLE(GCC, GCCSingle(lo, hi, make_qComplex(deviceGates[i].r00, deviceGates[i].i00)))
                    ID_BREAK()
                    default: {
                        assert(false);
                    }
                }
            } else {
                bool isHighBlock = (blockIdx.x >> targetQubit) & 1;
                switch (deviceGates[i].type) {
                    FOLLOW_NEXT(RZ)
                    CASE_LO_HI(CRZ, RZLo(j, deviceGates[i].r00, -deviceGates[i].i00), RZHi(j, deviceGates[i].r00, -deviceGates[i].i00))
                    FOLLOW_NEXT(Z)
                    CASE_SKIPLO_HI(CZ, ZHi(j))
                    CASE_SKIPLO_HI(S, SHi(j))
                    CASE_SKIPLO_HI(T, THi(j))
                    FOLLOW_NEXT(GOC)
                    CASE_SKIPLO_HI(U1, U1Hi(j, make_qComplex(deviceGates[i].r11, deviceGates[i].i11)))
                    LOHI_SAME(GII, GII(j))
                    LOHI_SAME(GZZ, GZZ(j))
                    LOHI_SAME(GCC, GCC(j, make_qComplex(deviceGates[i].r00, deviceGates[i].i00)))
                    ID_BREAK()
                    default: {
                        assert(false);
                    }
                }
            }
        }
        __syncthreads();
    }
}

__device__ void fetchData(qComplex* a, qindex* threadBias,  qindex idx, qindex blockHot, qindex enumerate, int numLocalQubits) {
    if (threadIdx.x == 0) {
        int bid = blockIdx.x;
        qindex bias = 0;
        for (qindex bit = 1; bit < (qindex(1) << numLocalQubits); bit <<= 1) {
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
        
        shm[x ^ (x >> 5)] = a[bias | y];
    }
}

__device__ void saveData(qComplex* a, qindex* threadBias, qindex enumerate) {
    qindex bias = blockBias | threadBias[threadIdx.x];
    for (int x = ((1 << (LOCAL_QUBIT_SIZE - THREAD_DEP)) - 1) << THREAD_DEP | threadIdx.x, y = enumerate;
        x >= 0;
        x -= (1 << THREAD_DEP), y = enumerate & (y - 1)) {
        
        a[bias | y] = shm[x ^ (x >> 5)];
    }
}

template <unsigned int blockSize>
__global__ void run(qComplex* a, qindex* threadBias, int* loArr, int* shiftAt, int numLocalQubits, int numGates, qindex blockHot, qindex enumerate) {
    qindex idx = blockIdx.x * blockSize + threadIdx.x;
    fetchData(a, threadBias, idx, blockHot, enumerate, numLocalQubits);
    __syncthreads();
    doCompute<blockSize>(numGates, loArr, shiftAt);
    __syncthreads();
    saveData(a, threadBias, enumerate);
}

#if BACKEND == 1 || BACKEND == 3
void initControlIdx() {
    int loIdx_host[10][10][1 << THREAD_DEP];
    int shiftAt_host[10][10];
    loIdx_device.resize(MyGlobalVars::numGPUs);
    shiftAt_device.resize(MyGlobalVars::numGPUs);
    for (int i = 0; i < MyGlobalVars::numGPUs; i++) {
        cudaSetDevice(i);
        cudaMalloc(&loIdx_device[i], sizeof(loIdx_host));
        cudaMalloc(&shiftAt_device[i], sizeof(shiftAt_host));
    }
    for (int controlQubit = 0; controlQubit < 5; controlQubit ++)
        for (int targetQubit = 0; targetQubit < 5; targetQubit ++) {
            if (controlQubit == targetQubit) continue;
            int smallQubit = controlQubit > targetQubit ? targetQubit : controlQubit;
            int largeQubit = controlQubit > targetQubit ? controlQubit : targetQubit;
            int maskSmall = (1 << smallQubit) - 1;
            int maskLarge = (1 << largeQubit) - 1;
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
            shiftAt_host[controlQubit][targetQubit] = shift;
            for (int tid = 0; tid < (1 << THREAD_DEP); tid++) {
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
        }

    for (int controlQubit = 5; controlQubit < 10; controlQubit ++)
        for (int targetQubit = 0; targetQubit < 5; targetQubit ++) {
            if (targetQubit - controlQubit == 5) continue;
            int smallQubit = controlQubit - 5 > targetQubit ? targetQubit : controlQubit - 5;
            int largeQubit = controlQubit - 5 > targetQubit ? controlQubit - 5 : targetQubit;
            int maskSmall = (1 << smallQubit) - 1;
            int maskLarge = (1 << largeQubit) - 1;
            int maskTarget = (1 << targetQubit) - 1;
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
            shiftAt_host[controlQubit][targetQubit] = shift;
            for (int tid = 0; tid < (1 << THREAD_DEP); tid++) {
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
                loIdx_host[controlQubit][targetQubit][tid] = lo;
            }
        }
    for (int controlQubit = 0; controlQubit < 5; controlQubit ++)
        for (int targetQubit = 5; targetQubit < 10; targetQubit ++) {
            int smallQubit = controlQubit > targetQubit - 5 ? targetQubit - 5 : controlQubit;
            int largeQubit = controlQubit > targetQubit - 5 ? controlQubit : targetQubit - 5;
            int maskSmall = (1 << smallQubit) - 1;
            int maskLarge = (1 << largeQubit) - 1;
            int maskControl = (1 << controlQubit) - 1;
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
            shiftAt_host[controlQubit][targetQubit] = shift;
            for (int tid = 0; tid < (1 << THREAD_DEP); tid++) {
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
                loIdx_host[controlQubit][targetQubit][tid] = lo;
            }        
        }
    loIdx_device.resize(MyGlobalVars::numGPUs);
    shiftAt_device.resize(MyGlobalVars::numGPUs);
    for (int g = 0; g < MyGlobalVars::numGPUs; g++) {
        checkCudaErrors(cudaMemcpyAsync(loIdx_device[g], loIdx_host[0][0], sizeof(loIdx_host), cudaMemcpyHostToDevice, MyGlobalVars::streams[g]));
        checkCudaErrors(cudaMemcpyAsync(shiftAt_device[g], shiftAt_host[0], sizeof(shiftAt_host), cudaMemcpyHostToDevice, MyGlobalVars::streams[g]));
    }
}
#endif

void copyGatesToSymbol(KernelGate* hostGates, int numGates, cudaStream_t& stream, int gpuID) {
    checkCudaErrors(cudaMemcpyToSymbolAsync(deviceGates, hostGates + gpuID * numGates, sizeof(KernelGate) * numGates, 0, cudaMemcpyDefault, stream));
}

void launchExecutor(int gridDim, qComplex* deviceStateVec, qindex* threadBias, int numLocalQubits, int numGates, qindex blockHot, qindex enumerate, cudaStream_t& stream, int gpuID) {
    run<1<<THREAD_DEP><<<gridDim, 1<<THREAD_DEP, 0, stream>>>
        (deviceStateVec, threadBias, loIdx_device[gpuID], shiftAt_device[gpuID], numLocalQubits, numGates, blockHot, enumerate);
}