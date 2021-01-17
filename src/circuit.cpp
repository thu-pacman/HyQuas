#include "circuit.h"

#include <cstdio>
#include <assert.h>
#include <chrono>
#include <mpi.h>
#include <algorithm>
#include <cuda_profiler_api.h>
#include "utils.h"
#include "kernel.h"
#include "compiler.h"
#include "logger.h"
#include "executor.h"
using namespace std;

int Circuit::run(bool copy_back) {
    kernelInit(deviceStateVec, numQubits);
    for (int i = 0; i < MyGlobalVars::numGPUs; i++) {
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaProfilerStart());
    }
    auto start = chrono::system_clock::now();
#if BACKEND == 0
    kernelExecSimple(deviceStateVec[0], numQubits, gates);
#elif BACKEND == 1 || BACKEND == 3 || BACKEND == 4
    Executor(deviceStateVec, numQubits, schedule).run();
#elif BACKEND == 2
    gates.clear();
    for (size_t lgID = 0; lgID < schedule.localGroups.size(); lgID++) {
        auto& lg = schedule.localGroups[lgID];
        for (size_t ggID = 0; ggID < lg.fullGroups.size(); ggID++) {
            auto& gg = lg.fullGroups[ggID];
            for (auto& g: gg.gates)
                gates.push_back(g);
        }
    }
    schedule.finalState = State(numQubits);
    kernelExecSimple(deviceStateVec[0], numQubits, gates);
#endif
    auto end = chrono::system_clock::now();
    for (int i = 0; i < MyGlobalVars::numGPUs; i++) {
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaProfilerStop());
    }
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    Logger::add("Time Cost: %d ms", int(duration.count()));
    result.resize(1ll << numQubits);
    if (copy_back) {
        qindex elements = 1ll << (numQubits - MyGlobalVars::bit);
        for (int g = 0; g < MyGlobalVars::numGPUs; g++) {
            kernelDeviceToHost((qComplex*)result.data() + elements * g, deviceStateVec[g], numQubits - MyGlobalVars::bit);
        }
    }
    for (int g = 0; g < MyGlobalVars::numGPUs; g++) {
        kernelDestroy(deviceStateVec[g]);
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
    auto& pos = schedule.finalState.pos;
    for (int i = 0; i < numQubits; i++) {
        if (idx >> i & 1)
            id |= qindex(1) << pos[i];
    }
    return id;
}

qindex Circuit::toLogicID(qindex idx) {
    int id = 0;
    auto& pos = schedule.finalState.pos;
    for (int i = 0; i < numQubits; i++) {
        if (idx >> pos[i] & 1)
            id |= qindex(1) << i;
    }
    return id;
}

qComplex Circuit::ampAt(qindex idx) {
    qindex id = toPhysicalID(idx);
    return make_qComplex(result[id].x, result[id].y);
}

void Circuit::compile() {
    Logger::add("Total Gates %d", int(gates.size()));
#if BACKEND == 1 || BACKEND == 2 || BACKEND == 3 || BACKEND == 4
    Compiler compiler(numQubits, gates);
    schedule = compiler.run();
    int totalGroups = 0;
    for (auto& lg: schedule.localGroups) totalGroups += lg.fullGroups.size();
    Logger::add("Total Groups: %d %d", int(schedule.localGroups.size()), totalGroups);
    schedule.initMatrix(numQubits);
#ifdef SHOW_SCHEDULE
    schedule.dump(numQubits);
#endif
#else
    schedule.finalState = State(numQubits);
#endif
}

void Circuit::printState() {
    for (int i = 0; i < 128; i++) {
        qComplex x = ampAt(i);
        printf("%d %.12f: %.12f %.12f\n", i, x.x * x.x + x.y * x.y, zero_wrapper(x.x), zero_wrapper(x.y));
    }
    std::vector<std::pair<qindex, qComplex>> largeAmps;
    for (int i = 0; i < (1 << numQubits); i++) {
        if (result[i].x * result[i].x + result[i].y * result[i].y > 0.001) {
            int logicID = toLogicID(i);
            if (logicID >= 128) {
                largeAmps.push_back(make_pair(toLogicID(i), result[i]));
            }
        }
    }
    sort(largeAmps.begin(), largeAmps.end());
    for (auto& amp: largeAmps) {
        auto& x = amp.second;
        printf("%d %.12f: %.12f %.12f\n", amp.first, x.x * x.x + x.y * x.y, zero_wrapper(x.x), zero_wrapper(x.y));
    }
}