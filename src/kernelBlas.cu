#include "kernel.h"
#include <cuda.h>
#include <cublas_v2.h>
#include <assert.h>
#include <cstdio>

__global__ void isnanTest(qComplex *data, int elePerBlock) {
    int l = elePerBlock * blockIdx.x;
    int r = l + elePerBlock;
    for (int i = l + threadIdx.x; i < r; i += blockDim.x) {
        if (isnan(data[i].x) || isnan(data[i].y)) {
            printf("nan at %d\n", i);
            asm("trap;");
        }
    }
}

__global__ void printVector(qComplex *data, int n) { // with gridDim == 1 && blockDim == 1
    for (int i = 0; i < n; i++)
        printf("(%f, %f)", data[i].x, data[i].y);
    printf("\n");
}

void kernelExecBlas(std::vector<qComplex*> deviceStateVec, int numQubits, const Schedule& schedule, const std::vector<std::vector<qComplex*>>& deviceMats) {
    assert(MyGlobalVars::numGPUs == 1);
    assert(schedule.localGroups.size() == 1);
    assert(MyGlobalVars::bit == 0);
    int numLocalQubits = numQubits - MyGlobalVars::bit;
    int numElements = 1 << numLocalQubits;
    std::vector<qComplex*> deviceBuffer;
    deviceBuffer.resize(MyGlobalVars::numGPUs);
    for (int g = 0; g < MyGlobalVars::numGPUs; g++) {
        deviceBuffer[g] = deviceStateVec[g] + numElements;        
    }
    cublasHandle_t handle;
    checkBlasErrors(cublasCreate(&handle));
    checkBlasErrors(cublasSetStream(handle, MyGlobalVars::streams[0]));
    auto& gateGroups = schedule.localGroups[0].gateGroups;
    qComplex alpha = make_qComplex(1.0, 0.0), beta = make_qComplex(0.0, 0.0);
    for (size_t i = 0; i < gateGroups.size(); i++) {
        if (i > 0) {
            checkCuttErrors(cuttExecute(schedule.cuttPlans[0][i], deviceStateVec[0], deviceBuffer[0]));
        } else {
            checkCudaErrors(cudaMemcpyAsync(deviceBuffer[0], deviceStateVec[0], numElements * sizeof(qComplex), cudaMemcpyDeviceToDevice, MyGlobalVars::streams[0]));
        }
        int K = 1 << bitCount(gateGroups[i].relatedQubits);
        // printVector<<<1, 1, 0, MyGlobalVars::streams[0]>>>(deviceMats[0][i], K*K);
        // printVector<<<1, 1, 0, MyGlobalVars::streams[0]>>>(deviceBuffer[0], 32);
#ifdef CHECK_NAN_BEFORE_GEMM
        // isnanTest<<<1, 32, 0, MyGlobalVars::streams[0]>>>(deviceMats[0][i], K * K);
        // checkCudaErrors(cudaStreamSynchronize(MyGlobalVars::streams[0]));
        // isnanTest<<<numElements / 1024, 32, 0, MyGlobalVars::streams[0]>>>(deviceBuffer[0], 1024);
        // checkCudaErrors(cudaStreamSynchronize(MyGlobalVars::streams[0]));
#endif
        checkBlasErrors(cublasGEMM(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            K, numElements / K , K, // M, N, K
            &alpha, deviceMats[0][i], K, // alpha, a, lda
            deviceBuffer[0], K, // b, ldb
            &beta, deviceStateVec[0], K // beta, c, ldc
        ));
        // printVector<<<1, 1, 0, MyGlobalVars::streams[0]>>>(deviceStateVec[0], 32);
    }
    checkCudaErrors(cudaStreamSynchronize(MyGlobalVars::streams[0]));
    checkBlasErrors(cublasDestroy(handle));
}

void kernelMatInit(const Schedule& schedule, std::vector<std::vector<qComplex*>>& deviceMats) {
    deviceMats.clear();
    deviceMats.resize(MyGlobalVars::numGPUs);
    for (auto& lg: schedule.localGroups) {
        for (auto& gg: lg.gateGroups) {
            int n = 1 << bitCount(gg.relatedQubits);
            for (int g = 0; g < MyGlobalVars::numGPUs; g++) {
                checkCudaErrors(cudaSetDevice(g));
                qComplex* mat;
                cudaMalloc(&mat, n * n * sizeof(qComplex));
                int i = deviceMats[g].size();
                cudaMemcpyAsync(mat, schedule.matrix[i].get(), n * n * sizeof(qComplex), cudaMemcpyHostToDevice, MyGlobalVars::streams[g]);
                deviceMats[g].push_back(mat);
            }
        }
    }
}

void kernelMatDestroy(std::vector<std::vector<qComplex*>>& deviceMats) {
    for (int g = 0; g < MyGlobalVars::numGPUs; g++) {
        checkCudaErrors(cudaSetDevice(g));
        for (auto& mat: deviceMats[g]) {
            checkCudaErrors(cudaFree(mat));
        }
    }
    deviceMats.clear();
}