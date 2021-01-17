#pragma once
#include <vector>
#include "schedule.h"
#include "utils.h"
#include "gate.h"

class Compiler {
public:
    Compiler(int numQubits, std::vector<Gate> inputGates);
    Schedule run();
private:
    void fillLocals(LocalGroup& lg);
    std::vector<std::vector<Gate>> moveToNext(LocalGroup& lg);
    int numQubits;
    int localSize;
    int shareSize;
    bool enableGlobal;
    std::vector<Gate> gates;
};

class OneLayerCompiler {
public:
    OneLayerCompiler(int numQubits, int localSize, qindex localQubits, std::vector<Gate> inputGates, bool enableGlobal, qindex whiteList = 0, qindex required = 0);
    // OneLayerCompiler(int numQubits, qindex localQubits, std::vector<Gate> inputGates);
    LocalGroup run();
    // LocalGroup run(State s, bool usePerGate, bool useBLAS);
private:
    int numQubits;
    int localSize;
    qindex localQubits;
    bool enableGlobal;
    qindex whiteList;
    qindex required;
    std::vector<Gate> remainGates;
    bool advance;
    GateGroup getGroup(bool full[], qindex related[], bool enableGlobal);
    void remove(GateGroup& gg);
};