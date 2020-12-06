#include "circuit.h"

#include <cstdio>
#include <assert.h>
#include <chrono>
#include <mpi.h>
#include "utils.h"
#include "kernel.h"
#include "compiler.h"
#include "logger.h"
using namespace std;

int Circuit::run(bool copy_back) {
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
    result.resize(1ll << numQubits);
    if (copy_back) {
        int elements = 1ll << (numQubits - MyGlobalVars::bit);
        for (int g = 0; g < MyGlobalVars::numGPUs; g++) {
            kernelDeviceToHost((qComplex*)result.data() + elements * g, deviceStateVec[g], numQubits - MyGlobalVars::bit);
            kernelDestroy(deviceStateVec[g]);
        }
    }
    return duration.count();
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
    int id = 0;
    for (int i = 0; i < numQubits; i++) {
        if (idx >> i & 1)
            id |= qindex(1) << schedule.finalPos[i];
    }
    return Complex(result[idx].x, result[idx].y);
}

void Circuit::compile() {
    Logger::add("Total Gates %d", int(gates.size()));
#ifdef USE_GROUP
    Compiler compiler(numQubits, numQubits - 3, LOCAL_QUBIT_SIZE, gates);
    schedule = compiler.run();
    int totalGroups = 0;
    for (auto& lg: schedule.localGroups) totalGroups += lg.gateGroups.size();
    Logger::add("Total Groups: %d %d", int(schedule.localGroups.size()), totalGroups);
    schedule.initCuttPlans(numQubits);
#ifdef SHOW_SCHEDULE
    if (MyMPI::rank == 0) {
        schedule.dump(numQubits);
    }
#endif
#endif
}