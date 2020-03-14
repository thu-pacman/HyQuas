#pragma once

#include <string>
#include "utils.h"

enum GateType {
    GateHadamard,
    GateCAlphaBeta,
    GateCNot
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
