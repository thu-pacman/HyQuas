#pragma once

#include <vector>
#include <cutt.h>

#include "gate.h"
#include "utils.h"
#include "compiler.h"
#include "circuit.h"

// kernelSimple
void kernelInit(std::vector<qComplex*> &deviceStateVec, int numQubits);
void kernelExecSimple(qComplex* deviceStateVec, int numQubits, const std::vector<Gate> & gates);
qreal kernelMeasure(qComplex* deviceStateVec, int numQubits, int targetQubit);
void kernelMeasureAll(std::vector<qComplex*>& deviceStateVec, qreal* results, int numLocalQubits);
qComplex kernelGetAmp(qComplex* deviceStateVec, qindex idx);
void kernelDeviceToHost(qComplex* hostStateVec, qComplex* deviceStateVec, int numQubits);
void kernelDestroy(qComplex* deviceStateVec);
void cuttPlanInit(std::vector<cuttHandle>& plans);

// kernelOpt
void initControlIdx();
// call cudaSetDevice() before this function
void copyGatesToSymbol(KernelGate* hostGates, int numGates, cudaStream_t& stream, int gpuID);

// call cudaSetDevice() before this function
void launchExecutor(int gridDim, qComplex* deviceStateVec, unsigned int* threadBias, int numLocalQubits, int numGates, unsigned int blockHot, unsigned int enumerate, cudaStream_t& stream, int gpuID);


// kernelUtils
void isnanTest(qComplex* data, int n, cudaStream_t& stream);
void printVector(qComplex* data, int n, cudaStream_t& stream);
void whileTrue();