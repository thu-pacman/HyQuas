#include "circuit.h"

#include <cstdio>
#include <assert.h>
#include <chrono>
#include "utils.h"
#include "kernel.h"
#include "compiler.h"
using namespace std;

void Circuit::run() {
    kernelInit(deviceStateVec, numQubits);
    auto start = chrono::system_clock::now();
#ifdef USE_GROUP
    kernelExecOpt(deviceStateVec, numQubits, schedule);
#else
    kernelExecSimple(deviceStateVec, numQubits, gates);
#endif
    auto end = chrono::system_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    printf("time: %d ms\n", int(duration.count()));
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

Complex Circuit::ampAt(qindex idx) {
    return kernelGetAmp(deviceStateVec, idx);
}

void Circuit::compile() {
#ifdef USE_GROUP
    printf("before compiler %d\n", int(gates.size()));
    Compiler compiler(numQubits, LOCAL_QUBIT_SIZE, gates);
    schedule = compiler.run();
    printf("Total Groups: %d\n", int(schedule.gateGroups.size()));
    fflush(stdout);
#endif
}