#pragma once

#include <vector>
#include "gate.h"
#include "utils.h"
#include "compiler.h"

void kernelInit(ComplexArray& deviceStateVec, int numQubits);
void kernelExecSimple(ComplexArray& deviceStateVec, int numQubits, const Schedule& schedule);
void kernelExecOpt(ComplexArray& deviceStateVec, int numQubits, const Schedule& schedule);
qreal kernelMeasure(ComplexArray& deviceStateVec, int numQubits, int targetQubit);
Complex kernelGetAmp(ComplexArray& deviceStateVec, qindex idx);