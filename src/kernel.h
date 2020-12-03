#pragma once

#include <vector>
#include <cutt.h>

#include "gate.h"
#include "utils.h"
#include "compiler.h"
#include "circuit.h"

void kernelInit(qComplex* &deviceStateVec, int numQubits);
void kernelExecSimple(qComplex* deviceStateVec, int numQubits, const std::vector<Gate> & gates);
std::vector<qreal> kernelExecOpt(qComplex* deviceStateVec, int numQubits, const Schedule& schedule);
qreal kernelMeasure(qComplex* deviceStateVec, int numQubits, int targetQubit);
Complex kernelGetAmp(qComplex* deviceStateVec, qindex idx);
void kernelDeviceToHost(qComplex* hostStateVec, qComplex* deviceStateVec, int numQubits);
void kernelDestroy(qComplex* deviceStateVec);
void cuttPlanInit(std::vector<cuttHandle>& plans);

// internal
void initControlIdx();