#pragma once
#include <vector>
#include "schedule.h"
#include "utils.h"
#include "gate.h"

class Compiler {
public:
    Compiler(int numQubits, int localSize, int shareSize, std::vector<Gate> inputGates);
    Schedule run();
private:
    int numQubits;
    int localSize;
    int shareSize;
    std::vector<Gate> gates;
};

class OneLayerCompiler {
public:
    OneLayerCompiler(int numQubits, int localSize, std::vector<Gate> inputGates);
    LocalGroup run();
private:
    int numQubits;
    int localSize;
    std::vector<Gate> remainGates;
    GateGroup getGroup();
    void remove(GateGroup& gg);
};