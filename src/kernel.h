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
qComplex kernelGetAmp(qComplex* deviceStateVec, qindex idx);
void kernelDeviceToHost(qComplex* hostStateVec, qComplex* deviceStateVec, int numQubits);
void kernelDestroy(qComplex* deviceStateVec);
void cuttPlanInit(std::vector<cuttHandle>& plans);

// kernelOpt
void initControlIdx();
void copyGatesToSymbol(KernelGate* hostGates, int numGates);
void launchExecutor(int gridDim, std::vector<qComplex*> &deviceStateVec, std::vector<qindex*> threadBias, int numLocalQubits, int numGates, qindex blockHot, qindex enumerate);

// kernelBlas
void kernelMatInit(const Schedule& schedule, std::vector<std::vector<qreal*>>& deviceMats);
void kernelExecBlas(std::vector<qComplex*> deviceStateVec, int numQubits, const Schedule& schedule, const std::vector<std::vector<qreal*>>& deviceMats);
void kernelMatDestroy(std::vector<std::vector<qreal*>>& deviceMats);
