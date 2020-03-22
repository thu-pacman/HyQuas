#pragma once

#include <vector>
#include "gate.h"
#include "utils.h"
#include "compiler.h"

void kernelInit(ComplexArray& deviceStateVec, int numQubits);
void kernelExec(ComplexArray& deviceStateVec, int numQubits, const Schedule& schedule);
void kernelExecSmall(ComplexArray& deviceStateVec, int numQubits, const Schedule& schedule);
qreal kernelMeasure(ComplexArray& deviceStateVec, int numQubits, int targetQubit);
Complex kernelGetAmp(ComplexArray& deviceStateVec, qindex idx);