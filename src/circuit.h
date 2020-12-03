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
    Complex ampAt(qindex idx);
    const int numQubits;

private:
    std::vector<Gate> gates;
    qComplex* deviceStateVec;
    Schedule schedule;
    std::vector<qComplex> result;
};