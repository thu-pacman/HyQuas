#include "circuit.h"

#include <cstdio>
#include <assert.h>
#include <chrono>
#include "utils.h"
#include "kernel.h"
#include "compiler.h"
#include "logger.h"
using namespace std;

void Circuit::run() {
    kernelInit(deviceStateVec, numQubits);
    auto start = chrono::system_clock::now();
#ifdef USE_GROUP
    kernelExecOpt(deviceStateVec, numQubits, schedule);
#else
    kernelExecSimple(deviceStateVec, numQubits, gates);
#endif
    // gates.clear();
    // for (auto& gg: schedule.gateGroups)
    //     for (auto& g: gg.gates)
    //         gates.push_back(g);
    // kernelExecSimple(deviceStateVec, numQubits, gates);
    auto end = chrono::system_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    Logger::add("Time Cost: %d ms", int(duration.count()));
    resultReal.resize(1ll << numQubits);
    resultImag.resize(1ll << numQubits);
    kernelDeviceToHost((ComplexArray){resultReal.data(), resultImag.data()}, deviceStateVec, numQubits);
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
    return Complex(resultReal[idx], resultImag[idx]);
}

void Circuit::compile() {
    Logger::add("Total Gates %d", int(gates.size()));
#ifdef USE_GROUP
    Compiler compiler(numQubits, LOCAL_QUBIT_SIZE, gates);
    schedule = compiler.run();
    Logger::add("Total Groups: %d", int(schedule.gateGroups.size()));
    fflush(stdout);
#endif
}