#pragma once

#include <string>
#include <vector>
#include "utils.h"
#include "QuESTEnv.h"


enum GateType {
    GateNormal,
    GateDiagonal,
    GateSwap // [0, 1; 1, 0], not two-qubit swap gate
};

struct Gate {
    GateType type;
    Complex mat[2][2];
    std::string name;
    int targetQubit;
    int controlQubit; // -1 if no control
    Gate() = default;
    Gate(const Gate&) = default;
};

class Qureg {
public:
    Qureg(int numQubits, const QuESTEnv& env): numQubits(numQubits), env(env) {}
    void run();
    void addGate(const Gate& gate) {
        gates.push_back(gate);
    }
    void dumpGates() {
        int totalGates = gates.size();
        const int GATE_PER_LINE = 40;
        for (int l = 0; l < totalGates; l += GATE_PER_LINE) {
            for (int i = 0; i < numQubits; i++) {
                printf("%2d:", i);
                for (int j = l; j < std::min(totalGates, l + GATE_PER_LINE); j++) {
                    const Gate& gate = gates[j];
                    int l = gate.name.length() + 1;
                    if (i == gate.controlQubit) {
                        printf(".");
                        for (int j = 1; j < l; j++) printf("-");
                    } else if (i == gate.targetQubit) {
                        printf("%s-", gate.name.c_str());
                    } else {
                        for (int j = 0; j < l; j++) printf("-");
                    }
                }
                printf("\n");
            }
            printf("\n");
        }
    }

private:
    int numQubits;
    const QuESTEnv& env;
    std::vector<Gate> gates;
    ComplexArray deviceStateVec;
};