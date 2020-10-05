#pragma once
#include <vector>
#include "schedule.h"
#include "utils.h"
#include "gate.h"

class Compiler {
public:
    Compiler(int numQubits, int localSize, std::vector<Gate> inputGates);
    Schedule run();
private:
    int numQubits;
    int localSize;
    std::vector<Gate> remainGates;
    GateGroup getGroup();
    void removeFromSchedule(GateGroup& gg);
    Schedule schedule;
};