#pragma once

#include <string>
#include <vector>
#include "utils.h"
#include "gate.h"
#include "compiler.h"

class Circuit {
public:
    Circuit(int numQubits): numQubits(numQubits) {}
    void compile();
    void run();
    void addGate(const Gate& gate) {
        gates.push_back(gate);
    }
    qreal measure(int targetQubit, int state = 0); // probability of zero state
    void dumpGates();
    Complex ampAt(qindex idx);

private:
    int numQubits;
    std::vector<Gate> gates;
    ComplexArray deviceStateVec;
    Schedule schedule;
    std::vector<qreal> result;
};