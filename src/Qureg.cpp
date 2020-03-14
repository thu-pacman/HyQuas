#include "Qureg.h"

#include <cstdio>
#include "QuESTEnv.h"
#include "utils.h"

Qureg createQureg(int numQubits, QuESTEnv& env) {
    return Qureg();
}

qreal calcProbOfOutcome(Qureg& q, int measureQubit, int outcome) {
    return 0;
}

Complex getAmp(Qureg& q, long long int index) {
    return Complex(0, 0);
}

void destroyQureg(Qureg& q, QuESTEnv& env) {
    printf("destroy qureg: just return!");
}