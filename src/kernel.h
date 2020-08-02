#pragma once

#include <vector>
#include "gate.h"
#include "utils.h"
#include "compiler.h"
#include "circuit.h"

#define MEASURE_STAGE 0

void kernelInit(ComplexArray& deviceStateVec, int numQubits);
void kernelExecSimple(ComplexArray& deviceStateVec, int numQubits, const std::vector<Gate> & gates);
std::vector<qreal> kernelExecOpt(ComplexArray& deviceStateVec, int numQubits, const Schedule& schedule);
qreal kernelMeasure(ComplexArray& deviceStateVec, int numQubits, int targetQubit);
Complex kernelGetAmp(ComplexArray& deviceStateVec, qindex idx);