#include "circuit.h"

#include <cstdio>
#include <assert.h>
#include "utils.h"
#include "kernel.h"
#include "compiler.h"
#define USE_OPT 1
using namespace std;

void Circuit::run() {
    kernelInit(deviceStateVec, numQubits);
    if (USE_OPT) {
        result = kernelExecOpt(deviceStateVec, numQubits, schedule);
    } else {
        kernelExecOpt(deviceStateVec, numQubits, schedule);
    }
}

void Circuit::dumpGates() {
    int totalGates = gates.size();
    printf("total Gates: %d\n", totalGates);
    int L = 3;
    for (const Gate& gate: gates) {
        for (int i = 0; i < numQubits; i++) {
            if (i == gate.controlQubit) {
                printf(".");
                for (int j = 1; j < L; j++) printf(" ");
            } else if (i == gate.targetQubit) {
                printf("%s", gate.name.c_str());
                for (int j = gate.name.length(); j < L; j++)
                    printf(" ");
            } else {
                printf("|");
                for (int j = 1; j < L; j++) printf(" ");
            }
        }
        printf("\n");
    }
}

qreal Circuit::measure(int targetQubit, int state) {
    qreal prob = USE_OPT ? (1 - result[targetQubit]) : kernelMeasure(deviceStateVec, numQubits, targetQubit);
    if (state == 1) {
        prob = 1 - prob;
    }
    return prob;
}

Complex Circuit::ampAt(qindex idx) {
    return kernelGetAmp(deviceStateVec, idx);
}

void Circuit::compile() {
    printf("before compiler %d\n", int(gates.size()));
    Compiler compiler(numQubits, LOCAL_QUBIT_SIZE, gates);
    schedule = compiler.run();
    printf("Total Groups: %d\n", int(schedule.gateGroups.size()));
}