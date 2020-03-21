#pragma once

#include <string>
#include <vector>
#include "utils.h"
#include "QuESTEnv.h"
#include "gate.h"
#include "compiler.h"

class Qureg {
public:
    Qureg(int numQubits, const QuESTEnv& env): numQubits(numQubits), env(env) {}
    void compile();
    void run();
    void addGate(const Gate& gate) {
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
    Schedule schedule;
};