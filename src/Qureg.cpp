#include "Qureg.h"

#include <cstdio>
#include <assert.h>
#include "QuESTEnv.h"
#include "utils.h"
#include "kernel.h"

Qureg createQureg(int numQubits, const QuESTEnv& env) {
    return Qureg(numQubits, env);
}

qreal calcProbOfOutcome(Qureg& q, int measureQubit, int outcome) {
    return 0;
}

Complex getAmp(Qureg& q, long long int index) {
    return Complex(0, 0);
}

void destroyQureg(Qureg& q, const QuESTEnv& env) {
    printf("destroy qureg: just return!");
}

void Qureg::run() {
    kernelInit(deviceStateVec, numQubits);
    assert(false);
}
