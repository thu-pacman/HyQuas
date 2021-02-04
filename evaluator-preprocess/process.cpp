#include <assert.h>
#include <fstream>
#include <cstring>
#include <regex>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include "circuit.h"
#include "logger.h"
using namespace std;

#define DIFF_QUBIT_NUMS 7
int qubit_nums[DIFF_QUBIT_NUMS] = {22, 23, 24, 25, 26, 27, 28};

FILE* curr_file;

void procPerGateSingle(int numQubits) {
    int num_gates = 512;
    for (int i = int(GateType::U1); i < int(GateType::TOTAL); i++) {
        printf("single gate %s\n", Gate::get_name(GateType(i)).c_str());
        for (int j = 0; j < LOCAL_QUBIT_SIZE; j++) {
            Circuit c(numQubits);
            for (int k = 0; k < num_gates; k++) {
                c.addGate(Gate::random(j, j + 1, GateType(i)));
            }
            c.compile();
            int time = c.run(false);
            fprintf(curr_file, "%d ", time);
        }
        fprintf(curr_file, "\n");
    }
    fprintf(curr_file, "\n");
}

void procPerGateCtr(int numQubits) {
    int num_gates = 512;
    for (int g = int(GateType::CNOT); g <= int(GateType::CRZ); g++) {
        printf("control gate %s\n", Gate::get_name(GateType(g)).c_str());
        for (int i = 0; i < LOCAL_QUBIT_SIZE; i++) {
            for (int j = 0; j < LOCAL_QUBIT_SIZE; j++) {
                if (i == j) { fprintf(curr_file, "0 "); continue; }
                Circuit c(numQubits);
                for (int k = 0; k < num_gates; k++) {
                    c.addGate(Gate::control(i, j, GateType(g)));
                }
                c.compile();
                int time = c.run(false);
                fprintf(curr_file, "%d ", time);
            }
            fprintf(curr_file, "\n");
        }
        fprintf(curr_file, "\n");
    }
}

void procBLAS(int numQubits) {
    cuDoubleComplex* arr;
    cuDoubleComplex* mat;
    cuDoubleComplex* result;
    checkCudaErrors(cudaMalloc(&arr, sizeof(cuDoubleComplex) << numQubits));
    checkCudaErrors(cudaMalloc(&mat, sizeof(cuDoubleComplex) << numQubits));
    checkCudaErrors(cudaMalloc(&result, sizeof(cuDoubleComplex) << numQubits));
    cublasHandle_t handle;
    checkBlasErrors(cublasCreate(&handle));
    int numElements = 1 << numQubits;
    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0), beta = make_cuDoubleComplex(0.0, 0.0);
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));    
    for (int K = 1; K < 1024; K <<= 1) {
        printf("blas calculating K = %d\n", K);
        double sum_time = 0.0;
        for (int i = 0; i < 100; i++) {
            checkCudaErrors(cudaEventRecord(start));
            
            checkBlasErrors(cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                K, numElements / K, K, // M, N, K
                &alpha, mat, K, // alpha, a, lda
                arr, K, // b, ldb
                &beta, result, K // beta, c, ldc
            ));

            float time;
            checkCudaErrors(cudaEventRecord(stop));
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time, start, stop);
            sum_time += time;
            //printf("%.10f ", time);
            
        }
        //printf("\n");
        fprintf(curr_file, "%d %f\n", K, sum_time / 100);
    }
    fprintf(curr_file, "\n");
    checkCudaErrors(cudaFree(arr));
    checkCudaErrors(cudaFree(mat));
    checkCudaErrors(cudaFree(result));
}

void procCutt(int numQbits) {
    numQbits += 1;
    double *in, *out;
    checkCudaErrors(cudaMalloc(&in, sizeof(double) * (1 << numQbits)));
    checkCudaErrors(cudaMalloc(&out, sizeof(double) * (1 << numQbits)));
    int dim[numQbits];
    for (int i = 0; i < numQbits; i++) dim[i] = 2;
    int total = 0;
    double sum_time = 0.0;
    for (int change = 1; change <= 20; change ++) {
        int perm[numQbits];
        printf("Cutt calculating  change = %d\n", change);
        for (int tt = 0; tt < 100; tt++) {
            for (int i = 0; i < numQbits; i++) perm[i] = i;
            for (int i = 0; i < change; i++) {
                std::swap(perm[rand() % numQbits], perm[rand() % numQbits]);
            }
            cuttHandle plan;
            checkCuttErrors(cuttPlan(&plan, numQbits, dim, perm, sizeof(double), 0));
            cudaEvent_t start, stop;
            float time;
            checkCudaErrors(cudaEventCreate(&start));
            checkCudaErrors(cudaEventCreate(&stop));
            checkCudaErrors(cudaEventRecord(start, 0));
            checkCuttErrors(cuttExecute(plan, in, out));
            checkCudaErrors(cudaEventRecord(stop, 0));
            checkCudaErrors(cudaEventSynchronize(stop));
            checkCudaErrors(cudaEventElapsedTime(&time, start, stop));
            //printf("%.10f ms ", time);
            total ++;
            sum_time += time;
        }
        //printf("\n");
    }
    fprintf(curr_file, "%f\n", sum_time / total);
    checkCudaErrors(cudaFree(in));
    checkCudaErrors(cudaFree(out));
}

void process(int numQubits) {
    printf("processing qubit number : %d\n", numQubits);
    string file_name = string("../evaluator-preprocess/parameter-files/") + to_string(numQubits) + string("qubits.out"); 
    curr_file = fopen(file_name.c_str(), "w");
    procPerGateSingle(numQubits);
    procPerGateCtr(numQubits);
    procBLAS(numQubits);
    procCutt(numQubits);
    fclose(curr_file);
}

int main()
{
    MyGlobalVars::init();
    for(int i = 0; i < DIFF_QUBIT_NUMS; i++) {
        process(qubit_nums[i]);
    }
}
