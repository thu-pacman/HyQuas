#include "kernel.h"
#include <cuda.h>
#include <cublas_v2.h>
#include <assert.h>

void kernelExecBlas(std::vector<qComplex*> deviceStateVec, int numQubits, const Schedule& schedule, const std::vector<std::vector<qComplex*>>& deviceMats) {
    assert(MyGlobalVars::numGPUs == 1);
    assert(schedule.localGroups.size() == 1);
    int numLocalQubits = numQubits - MyGlobalVars::bit;
    int numElements = 1 << numLocalQubits;
    std::vector<qComplex*> deviceBuffer;
    deviceBuffer.resize(MyGlobalVars::numGPUs);
    for (int g = 0; g < MyGlobalVars::numGPUs; g++) {
        deviceBuffer[g] = deviceStateVec[g] + numElements;        
    }
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    cublasSetStream(handle, MyGlobalVars::streams[0]);
    auto& gateGroups = schedule.localGroups[0].gateGroups;
    qComplex alpha = make_qComplex(1.0, 0.0), beta = make_qComplex(0.0, 0.0);
    for (size_t i = 0; i < gateGroups.size(); i++) {
        if (i > 0) {
            checkCuttErrors(cuttExecute(schedule.cuttPlans[0][i], deviceStateVec[0], deviceBuffer[0]));
        } else {
            checkCudaErrors(cudaMemcpyAsync(deviceBuffer[0], deviceStateVec[0], numElements * sizeof(qComplex), cudaMemcpyDeviceToDevice, MyGlobalVars::streams[0]));
        }
        int K = 1 << bitCount(gateGroups[i].relatedQubits);
        cublasGEMM(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            K, (1 << numQubits) / K , K, // M, N, K
            &alpha, deviceMats[0][i], K, // alpha, a, lda
            deviceBuffer[0], K, // b, ldb
            &beta, deviceStateVec[0], K // c, ldc
        );
    }
    cublasDestroy(handle);
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
                int i = deviceMats.size();
                cudaMemcpy(mat, schedule.matrix[i].get(), n * n * sizeof(qComplex), cudaMemcpyHostToDevice);
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