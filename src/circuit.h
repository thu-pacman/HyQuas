#pragma once

#include <string>
#include <vector>
#include "utils.h"
#include "gate.h"
#include "schedule.h"

class Circuit {
public:
    Circuit(int numQubits): numQubits(numQubits) {}
    void compile();
    int run(bool copy_back = true);
    void addGate(const Gate& gate) {
        gates.push_back(gate);
    }
    void dumpGates();
    void printState();
    Complex ampAt(qindex idx);
    const int numQubits;

private:
    qindex toPhysicalID(qindex idx);
    qindex toLogicID(qindex idx);
    std::vector<Gate> gates;
    std::vector<qComplex*> deviceStateVec;
    std::vector<std::vector<qreal*>> deviceMats;
    Schedule schedule;
    std::vector<qComplex> result;
};