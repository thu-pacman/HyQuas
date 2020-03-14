#pragma once

#include <vector>
#include "gate.h"
#include "utils.h"

void kernelInit(ComplexArray& deviceStateVec, int numQubits);
void kernelExec(ComplexArray& deviceStateVec, int numQubits, const std::vector<Gate>& gates);
qreal kernelMeasure(ComplexArray& deviceStateVec, int numQubits, int targetQubit);
Complex kernelGetAmp(ComplexArray& deviceStateVec, qindex idx);