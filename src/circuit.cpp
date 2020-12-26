#include "circuit.h"

#include <cstdio>
#include <assert.h>
#include <chrono>
#include <mpi.h>
#include <algorithm>
#include "utils.h"
#include "kernel.h"
#include "compiler.h"
#include "logger.h"
using namespace std;

int Circuit::run(bool copy_back) {
    kernelInit(deviceStateVec, numQubits);
    auto start = chrono::system_clock::now();
#if BACKEND == 0
    kernelExecSimple(deviceStateVec[0], numQubits, gates);
#elif BACKEND == 1
    kernelExecOpt(deviceStateVec, numQubits, schedule);
#elif BACKEND == 2
    gates.clear();
    for (size_t lgID = 0; lgID < schedule.localGroups.size(); lgID++) {
        auto& lg = schedule.localGroups[lgID];
        for (size_t ggID = 0; ggID < lg.gateGroups.size(); ggID++) {
            auto& gg = lg.gateGroups[ggID];
            for (auto& g: gg.gates)
                gates.push_back(g);
        }
    }
    schedule.finalPos.clear();
    for (int i = 0; i < numQubits; i++) {
        schedule.finalPos.push_back(i);
    }
    kernelExecSimple(deviceStateVec[0], numQubits, gates);
#elif BACKEND == 3
    kernelExecOpt(deviceStateVec, numQubits, schedule);
#endif
    auto end = chrono::system_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    Logger::add("Time Cost: %d ms", int(duration.count()));
    result.resize(1ll << numQubits);
    if (copy_back) {
        qindex elements = 1ll << (numQubits - MyGlobalVars::bit);
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

qindex Circuit::toPhysicalID(qindex idx) {
    int id = 0;
    for (int i = 0; i < numQubits; i++) {
        if (idx >> i & 1)
            id |= qindex(1) << schedule.finalPos[i];
    }
    return id;
}

qindex Circuit::toLogicID(qindex idx) {
    int id = 0;
    for (int i = 0; i < numQubits; i++) {
        if (idx >> schedule.finalPos[i] & 1)
            id |= qindex(1) << i;
    }
    return id;
}

Complex Circuit::ampAt(qindex idx) {
    qindex id = toPhysicalID(idx);
    return Complex(result[id].x, result[id].y);
}

void Circuit::compile() {
    Logger::add("Total Gates %d", int(gates.size()));
#if BACKEND == 1 || BACKEND == 2
    Compiler compiler(numQubits, numQubits - MyGlobalVars::bit, LOCAL_QUBIT_SIZE, gates, true);
    schedule = compiler.run();
    int totalGroups = 0;
    for (auto& lg: schedule.localGroups) totalGroups += lg.gateGroups.size();
    Logger::add("Total Groups: %d %d", int(schedule.localGroups.size()), totalGroups);
    schedule.initCuttPlans(numQubits);
#ifdef SHOW_SCHEDULE
    schedule.dump(numQubits);
#endif
#elif BACKEND == 3
    Compiler compiler(numQubits, numQubits - MyGlobalVars::bit, LOCAL_QUBIT_SIZE, gates, false);
    schedule = compiler.run();
    int totalGroups = 0;
    for (auto& lg: schedule.localGroups) totalGroups += lg.gateGroups.size();
    Logger::add("Total Groups: %d %d", int(schedule.localGroups.size()), totalGroups);
#ifdef SHOW_SCHEDULE
    schedule.dump(numQubits);
#endif
    schedule.initCuttPlans(numQubits);
#else
    schedule.finalPos.clear();
    for (int i = 0; i < numQubits; i++) {
        schedule.finalPos.push_back(i);
    }
#endif
}

void Circuit::printState() {
    for (int i = 0; i < 128; i++) {
        Complex x = ampAt(i);
        printf("%d %.12f: %.12f %.12f\n", i, x.real * x.real + x.imag * x.imag, zero_wrapper(x.real), zero_wrapper(x.imag));
    }
    std::vector<std::pair<qindex, Complex>> largeAmps;
    for (int i = 0; i < (1 << numQubits); i++) {
        if (result[i].x * result[i].x + result[i].y * result[i].y > 0.001) {
            int logicID = toLogicID(i);
            if (logicID >= 128) {
                largeAmps.push_back(make_pair(toLogicID(i), Complex(result[i])));
            }
        }
    }
    sort(largeAmps.begin(), largeAmps.end());
    for (auto& amp: largeAmps) {
        auto& x = amp.second;
        printf("%d %.12f: %.12f %.12f\n", amp.first, x.real * x.real + x.imag * x.imag, zero_wrapper(x.real), zero_wrapper(x.imag));
    }
}