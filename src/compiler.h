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
    LocalGroup run();
private:
    int numQubits;
    int localSize;
    qindex localQubits;
    bool enableGlobal;
    qindex whiteList;
    qindex required;
    std::vector<Gate> remainGates;
    GateGroup getGroup();
    void remove(GateGroup& gg);
};