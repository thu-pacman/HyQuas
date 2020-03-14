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
    qreal probZero = q.measure(measureQubit);
    return outcome ? 1 - probZero : probZero;
}

Complex getAmp(Qureg& q, qindex index) {
    return q.ampAt(index);
}

void destroyQureg(Qureg& q, const QuESTEnv& env) {
    printf("destroy qureg: just return!\n");
}

void Qureg::run() {
    kernelInit(deviceStateVec, numQubits);
    kernelExec(deviceStateVec, numQubits, gates);
}

void Qureg::dumpGates() {
    int totalGates = gates.size();
    const int GATE_PER_LINE = 40;
    for (int l = 0; l < totalGates; l += GATE_PER_LINE) {
        for (int i = 0; i < numQubits; i++) {
            printf("%2d:", i);
            for (int j = l; j < std::min(totalGates, l + GATE_PER_LINE); j++) {
                const Gate& gate = gates[j];
                int l = gate.name.length() + 1;
                if (i == gate.controlQubit) {
                    printf(".");
                    for (int j = 1; j < l; j++) printf("-");
                } else if (i == gate.targetQubit) {
                    printf("%s-", gate.name.c_str());
                } else {
                    for (int j = 0; j < l; j++) printf("-");
                }
            }
            printf("\n");
        }
        printf("\n");
    }
}

qreal Qureg::measure(int targetQubit) {
    return kernelMeasure(deviceStateVec, numQubits, targetQubit);
}

Complex Qureg::ampAt(qindex idx) {
    return kernelGetAmp(deviceStateVec, idx);
}