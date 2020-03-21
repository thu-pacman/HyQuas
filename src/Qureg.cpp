#include "Qureg.h"

#include <cstdio>
#include <assert.h>
#include "QuESTEnv.h"
#include "utils.h"
#include "kernel.h"
using namespace std;

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
    printf("total Gates: %d\n", totalGates);
    int L = 4;
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

qreal Qureg::measure(int targetQubit) {
    return kernelMeasure(deviceStateVec, numQubits, targetQubit);
}

Complex Qureg::ampAt(qindex idx) {
    return kernelGetAmp(deviceStateVec, idx);
}

void Qureg::compile() {
    vector<Gate> cur[numQubits];
    vector<GateGroup> groups;
    for (auto& gate: gates) {
        if (gate.isControlGate()) {
            GateGroup gg;
            gg.insert(gg.end(), cur[gate.controlQubit].begin(), cur[gate.controlQubit].end());
            gg.insert(gg.end(), cur[gate.targetQubit].begin(), cur[gate.targetQubit].end());
            cur[gate.controlQubit].clear();
            cur[gate.targetQubit].clear();
            gg.push_back(gate);
            groups.push_back(gg);
        } else {
            cur[gate.targetQubit].push_back(gate);
        }
    }
    for (int i = 0; i < numQubits; i++) {
        groups.push_back(cur[i]);
    }
    gates.clear();
    for (auto& group: groups) {
        for (auto& gate: group) {
            gates.push_back(gate);
        }
    }
    dumpGates();
}