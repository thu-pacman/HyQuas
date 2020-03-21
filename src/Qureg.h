#pragma once

#include <string>
#include <vector>
#include "utils.h"
#include "QuESTEnv.h"
#include "gate.h"

class Qureg {
public:
    Qureg(int numQubits, const QuESTEnv& env): numQubits(numQubits), env(env) {}
    void run();
    void addGate(const Gate& gate) {
        // WARNING
        gates.push_back(gate);
    }
    qreal measure(int targetQubit); // probability of zero state
    void dumpGates();
    Complex ampAt(qindex idx);

private:
    int numQubits;
    const QuESTEnv& env;
    std::vector<Gate> gates;
    ComplexArray deviceStateVec;
};