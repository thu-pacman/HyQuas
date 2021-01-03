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

void kernelExecBlas(std::vector<qComplex*> deviceStateVec, int numQubits, const Schedule& schedule, const std::vector<std::vector<qreal*>>& deviceMats) {
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
    qreal alpha = 1.0, beta = 0.0;
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
            K * 2, numElements / K, K * 2, // M, N, K
            &alpha, deviceMats[0][i], K * 2, // alpha, a, lda
            reinterpret_cast<qreal*>(deviceBuffer[0]), K * 2, // b, ldb
            &beta, reinterpret_cast<qreal*>(deviceStateVec[0]), K * 2 // beta, c, ldc
        ));
        // printVector<<<1, 1, 0, MyGlobalVars::streams[0]>>>(deviceStateVec[0], 32);
    }
    checkCudaErrors(cudaStreamSynchronize(MyGlobalVars::streams[0]));
    checkBlasErrors(cublasDestroy(handle));
}

void kernelMatInit(const Schedule& schedule, std::vector<std::vector<qreal*>>& deviceMats) {
    deviceMats.clear();
    deviceMats.resize(MyGlobalVars::numGPUs);
    for (auto& lg: schedule.localGroups) {
        for (int ggID = 0; ggID < lg.gateGroups.size(); ggID ++) {
            auto& gg = lg.gateGroups[ggID];
            int n = 1 << bitCount(gg.relatedQubits);
            qreal realMat[2 * n][2 * n];
            #pragma omp parallel for
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) {
                    qComplex val = schedule.matrix[ggID][i * n + j];
                    realMat[i * 2][j * 2] = val.x;
                    realMat[i * 2][j * 2 + 1] = val.y;
                    realMat[i * 2 + 1][j * 2] = -val.y;
                    realMat[i * 2 + 1][j * 2 + 1] = val.x;
                }
            for (int g = 0; g < MyGlobalVars::numGPUs; g++) {
                checkCudaErrors(cudaSetDevice(g));
                qreal* mat;
                cudaMalloc(&mat, n * n * 4 * sizeof(qreal));
                cudaMemcpyAsync(mat, realMat, n * n * 4 * sizeof(qreal), cudaMemcpyHostToDevice, MyGlobalVars::streams[g]);
                deviceMats[g].push_back(mat);
            }
        }
    }
}

void kernelMatDestroy(std::vector<std::vector<qreal*>>& deviceMats) {
    for (int g = 0; g < MyGlobalVars::numGPUs; g++) {
        checkCudaErrors(cudaSetDevice(g));
        for (auto& mat: deviceMats[g]) {
            checkCudaErrors(cudaFree(mat));
        }
    }
    deviceMats.clear();
}