#pragma once

#include <string>
#include "utils.h"

enum GateType {
    GateHadamard,
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
    GateT,
    GateCNot,
    GateU1,
    GateU3,
    GateCZ,
};

struct Gate {
    int gateID;
    GateType type;
    Complex mat[2][2];
    std::string name;
    int targetQubit;
    int controlQubit; // -1 if no control
    Gate() = default;
    Gate(const Gate&) = default;
    bool isControlGate() const {
        return controlQubit != -1;
    }
    bool isDiagonal() const {
        return type == GateCRotateZ || type == GatePauliZ || type == GateRotateZ || type == GateS || type == GateT || type == GateCZ || type == GateU1;
    }

    static Gate controlledPauliY(int controlQubit, int targetQubit);
    static Gate controlledRotateX(int controlQubit, int targetQubit, qreal angle);
    static Gate controlledRotateY(int controlQubit, int targetQubit, qreal angle);
    static Gate controlledRotateZ(int controlQubit, int targetQubit, qreal angle);
    static Gate pauliY(int targetQubit);
    static Gate pauliZ(int targetQubit);
    static Gate sGate(int targetQubit);
    static Gate tGate(int targetQubit);
    
    static Gate CNOT(int controlQubit, int targetQubit);
    static Gate U1(int targetQubit, qreal lambda);
    static Gate U3(int targetQubit, qreal theta, qreal phi, qreal lambda);
    static Gate CZ(int controlQubit, int targetQubit);
    static Gate Hadamard(int targetQubit);
    static Gate PauliX(int targetQubit);
    static Gate RotateX(int targetQubit, qreal angle);
    static Gate RotateY(int targetQubit, qreal angle);
    static Gate RotateZ(int targetQubit, qreal angle);

};
