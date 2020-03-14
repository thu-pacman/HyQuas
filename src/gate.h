#pragma once

#include <string>
#include "utils.h"

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
