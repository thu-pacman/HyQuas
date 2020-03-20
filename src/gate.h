#pragma once

#include <string>
#include "utils.h"

enum GateType {
    GateHadamard,
    GateCNot,
    GateCPauliY,
    GateCRotateX,
    GateCRotateY,
    GateCRotateZ,
    GatePauliX,
    GatePauliY,
    GatePauliZ,
    GateRotateX,
    GateRotateY,
    GateRotateZ,
    GateS,
    GateT
};

struct Gate {
    GateType type;
    Complex mat[2][2];
    std::string name;
    int targetQubit;
    int controlQubit; // -1 if no control
    Gate() = default;
    Gate(const Gate&) = default;
    bool isControlGate() {
        return controlQubit != -1;
    }
};
