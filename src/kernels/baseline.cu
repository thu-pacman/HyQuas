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

__device__ __forceinline__ void SDGHi(int hiIdx) {
    qComplex hi = shm[hiIdx];
    shm[hiIdx] = make_qComplex(hi.y, -hi.x);
}

__device__ __forceinline__ void THi(int hiIdx) {
    qComplex hi = shm[hiIdx];
    shm[hiIdx] = make_qComplex(recRoot2 * (hi.x - hi.y), recRoot2 * (hi.x + hi.y));
}

__device__ __forceinline__ void TDGHi(int hiIdx) {
    qComplex hi = shm[hiIdx];
    shm[hiIdx] = make_qComplex(recRoot2 * (hi.x + hi.y), recRoot2 * (hi.x - hi.y));
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
    for (int j = threadIdx.x; j < m; j += blockSize) { \
        int lo = ((j >> smallQubit) << (smallQubit + 1)) | (j & maskSmall); \
        lo = ((lo >> largeQubit) << (largeQubit + 1)) | (lo & maskLarge); \
        lo |= 1 << controlQubit; \
        int hi = lo | (1 << targetQubit); \
        OP; \
    } \
    break; \
}

#define CASE_SINGLE(TYPE, OP) \
case GateType::TYPE: { \
    for (int j = threadIdx.x; j < m; j += blockSize) { \
        int lo = ((j >> targetQubit) << (targetQubit + 1)) | (j & maskTarget); \
        int hi = lo | (1 << targetQubit); \
        OP; \
    } \
    break;\
}

#define CASE_LO_HI(TYPE, OP_LO, OP_HI) \
case GateType::TYPE: { \
    int m = 1 << LOCAL_QUBIT_SIZE; \
    if (!isHighBlock){ \
        for (int j = threadIdx.x; j < m; j += blockSize) { \
            OP_LO; \
        } \
    } else { \
        for (int j = threadIdx.x; j < m; j += blockSize) { \
            OP_HI; \
        } \
    } \
    break; \
}

#define CASE_SKIPLO_HI(TYPE, OP_HI) \
case GateType::TYPE: { \
    if (!isHighBlock) continue; \
    int m = 1 << LOCAL_QUBIT_SIZE; \
    for (int j = threadIdx.x; j < m; j += blockSize) { \
        OP_HI; \
    } \
    break; \
}

#define LOHI_SAME(TYPE, OP) \
case GateType::TYPE: { \
    int m = 1 << LOCAL_QUBIT_SIZE; \
    for (int j = threadIdx.x; j < m; j += blockSize) { \
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
                int m = 1 << (LOCAL_QUBIT_SIZE - 2);
                int smallQubit = controlQubit > targetQubit ? targetQubit : controlQubit;
                int largeQubit = controlQubit > targetQubit ? controlQubit : targetQubit;
                int maskSmall = (1 << smallQubit) - 1;
                int maskLarge = (1 << largeQubit) - 1;
                switch (deviceGates[i].type) {
                    FOLLOW_NEXT(CCX)
                    CASE_CONTROL(CNOT, XSingle(lo, hi))
                    CASE_CONTROL(CY, YSingle(lo, hi))
                    CASE_CONTROL(CZ, ZHi(hi))
                    CASE_CONTROL(CRX, RXSingle(lo, hi, deviceGates[i].r00, -deviceGates[i].i01))
                    CASE_CONTROL(CRY, RYSingle(lo, hi, deviceGates[i].r00, deviceGates[i].r10))
                    CASE_CONTROL(CU1, U1Hi(hi, make_qComplex(deviceGates[i].r11, deviceGates[i].i11)))
                    CASE_CONTROL(CRZ, RZSingle(lo, hi, deviceGates[i].r00, -deviceGates[i].i00))
                    default: {
                        assert(false);
                    }
                }
            } else {
                assert(deviceGates[i].type == GateType::CZ || deviceGates[i].type == GateType::CU1 || deviceGates[i].type == GateType::CRZ);
                bool isHighBlock = (blockIdx.x >> targetQubit) & 1;
                int m = 1 << (LOCAL_QUBIT_SIZE - 1);
                int maskControl = (1 << controlQubit) - 1;
                if (!isHighBlock){
                    if (deviceGates[i].type == GateType::CRZ) {
                        for (int j = threadIdx.x; j < m; j += blockSize) {
                            int x = ((j >> controlQubit) << (controlQubit + 1)) | (j & maskControl)  | (1 << controlQubit);
                            RZLo(x, deviceGates[i].r00, -deviceGates[i].i00);
                        }
                    }
                } else {
                    switch (deviceGates[i].type) {
                        case GateType::CZ: {
                            for (int j = threadIdx.x; j < m; j += blockSize) {
                                int x = ((j >> controlQubit) << (controlQubit + 1)) | (j & maskControl)  | (1 << controlQubit);
                                ZHi(x);
                            }
                            break;    
                        }
                        case GateType::CU1: {
                            for (int j = threadIdx.x; j < m; j += blockSize) {
                                int x = ((j >> controlQubit) << (controlQubit + 1)) | (j & maskControl)  | (1 << controlQubit);
                                U1Hi(x, make_qComplex(deviceGates[i].r11, deviceGates[i].i11));
                            }
                            break;
                        }
                        case GateType::CRZ: {
                            for (int j = threadIdx.x; j < m; j += blockSize) {
                                int x = ((j >> controlQubit) << (controlQubit + 1)) | (j & maskControl)  | (1 << controlQubit);
                                RZHi(x, deviceGates[i].r00, -deviceGates[i].i00);
                            }
                            break;
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
                switch (deviceGates[i].type) {
                    FOLLOW_NEXT(GOC)
                    FOLLOW_NEXT(CU1)
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
                    CASE_SINGLE(SDG, SDGHi(hi))
                    CASE_SINGLE(T, THi(hi))
                    CASE_SINGLE(TDG, TDGHi(hi))
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
                    CASE_SKIPLO_HI(SDG, SDGHi(j))
                    CASE_SKIPLO_HI(T, THi(j))
                    CASE_SKIPLO_HI(TDG, TDGHi(j))
                    FOLLOW_NEXT(GOC)
                    FOLLOW_NEXT(CU1)
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
        
        shm[x] = a[bias | y];
    }
}

__device__ void saveData(qComplex* a, qindex* threadBias, qindex enumerate) {
    qindex bias = blockBias | threadBias[threadIdx.x];
    for (int x = ((1 << (LOCAL_QUBIT_SIZE - THREAD_DEP)) - 1) << THREAD_DEP | threadIdx.x, y = enumerate;
        x >= 0;
        x -= (1 << THREAD_DEP), y = enumerate & (y - 1)) {
        
        a[bias | y] = shm[x];
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

#if BACKEND == 1 || BACKEND == 3 || BACKEND == 4
void initControlIdx() {
    loIdx_device.resize(MyGlobalVars::numGPUs);
    shiftAt_device.resize(MyGlobalVars::numGPUs);
}
#endif

void copyGatesToSymbol(KernelGate* hostGates, int numGates, cudaStream_t& stream, int gpuID) {
    checkCudaErrors(cudaMemcpyToSymbolAsync(deviceGates, hostGates + gpuID * numGates, sizeof(KernelGate) * numGates, 0, cudaMemcpyDefault, stream));
}

void launchExecutor(int gridDim, qComplex* deviceStateVec, qindex* threadBias, int numLocalQubits, int numGates, qindex blockHot, qindex enumerate, cudaStream_t& stream, int gpuID) {
    run<1<<THREAD_DEP><<<gridDim, 1<<THREAD_DEP, 0, stream>>>
        (deviceStateVec, threadBias, loIdx_device[gpuID], shiftAt_device[gpuID], numLocalQubits, numGates, blockHot, enumerate);
}