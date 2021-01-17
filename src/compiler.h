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
    OneLayerCompiler(int numQubits, const std::vector<Gate>& inputGates);
protected:
    int numQubits;
    std::vector<Gate> remainGates;
    GateGroup getGroup(bool full[], qindex related[], bool enableGlobal, int localSize, qindex localQubits);
};

class SimpleCompiler: public OneLayerCompiler {
public:
    SimpleCompiler(int numQubits, int localSize, qindex localQubits, const std::vector<Gate>& inputGates, bool enableGlobal, qindex whiteList = 0, qindex required = 0);
    LocalGroup run();
private:
    int localSize;
    qindex localQubits;
    bool enableGlobal;
    qindex whiteList;
    qindex required;
};

class AdvanceCompiler: public OneLayerCompiler {
public:
    AdvanceCompiler(int numQubits, qindex localQubits, std::vector<Gate> inputGates);
    LocalGroup run(State &state, bool usePerGate, bool useBLAS, int preGateSize, int blasSize);
private:
    qindex localQubits;
};