#include "gate.h"

#include <cmath>
#include <assert.h>

static int globalGateID = 0;

Gate Gate::CNOT(int controlQubit, int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateCNot;
    g.mat[0][0] = 0; g.mat[0][1] = 1;
    g.mat[1][0] = 1; g.mat[1][1] = 0;
    g.name = "CN";
    g.targetQubit = targetQubit;
    g.controlQubit = controlQubit;
    return g;
}

Gate Gate::U1(int targetQubit, qreal lambda) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateU1;
    g.mat[0][0] = 1;
    g.mat[0][1] = 0;
    g.mat[1][0] = 0;
    g.mat[1][1] = Complex(cos(lambda), sin(lambda));
    g.name = "U1";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::U3(int targetQubit, qreal theta, qreal phi, qreal lambda) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateU3;
    g.mat[0][0] = cos(theta / 2);
    g.mat[0][1] = Complex(-cos(lambda) * sin(theta / 2), -sin(lambda) * sin(theta / 2));
    g.mat[1][0] = Complex(cos(phi) * sin(theta / 2), sin(phi) * sin(theta / 2));
    g.mat[1][1] = Complex(cos(phi + lambda) * cos(theta / 2), sin(phi + lambda) * cos(theta / 2));
    g.name = "U3";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}


Gate Gate::CZ(int controlQubit, int targetQubit) {
    assert(controlQubit != targetQubit);
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateCZ;
    g.mat[0][0] = 1; g.mat[0][1] = 0;
    g.mat[1][0] = 0; g.mat[1][1] = -1;
    g.name = "CZ";
    g.targetQubit = targetQubit;
    g.controlQubit = controlQubit;
    return g;
}

Gate Gate::controlledPauliY(int controlQubit, int targetQubit) {
    assert(controlQubit != targetQubit);
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateCPauliY;
    g.mat[0][0] = 0; g.mat[0][1] = Complex(0, -1);
    g.mat[1][0] = Complex(0, 1); g.mat[1][1] = Complex(0, 0);
    g.name = "CP";
    g.targetQubit = targetQubit;
    g.controlQubit = controlQubit;
    return g;
}

Gate Gate::controlledRotateX(int controlQubit, int targetQubit, qreal angle) {
    assert(controlQubit != targetQubit);
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateCRotateX;
    g.mat[0][0] = cos(angle/2.0); g.mat[0][1] = Complex(0, sin(angle/2.0));
    g.mat[1][0] = Complex(0, -sin(angle/2.0)); g.mat[1][1] = cos(angle/2.0);
    g.name = "CX";
    g.targetQubit = targetQubit;
    g.controlQubit = controlQubit;
    return g;
}

Gate Gate::controlledRotateY(int controlQubit, int targetQubit, qreal angle) {
    assert(controlQubit != targetQubit);
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateCRotateY;
    g.mat[0][0] = cos(angle/2.0); g.mat[0][1] = -sin(angle/2.0);
    g.mat[1][0] = sin(angle/2.0); g.mat[1][1] = cos(angle/2.0);
    g.name = "CY";
    g.targetQubit = targetQubit;
    g.controlQubit = controlQubit;
    return g;
}

Gate Gate::controlledRotateZ(int controlQubit, int targetQubit, qreal angle) {
    assert(controlQubit != targetQubit);
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateCRotateZ;
    g.mat[0][0] = Complex(cos(angle/2), -sin(angle/2)); g.mat[0][1] = 0;
    g.mat[1][0] = 0; g.mat[1][1] = Complex(cos(angle/2), sin(angle/2));
    g.name = "CZ";
    g.targetQubit = targetQubit;
    g.controlQubit = controlQubit;
    return g;
}

Gate Gate::Hadamard(int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateHadamard;
    g.mat[0][0] = 1/sqrt(2); g.mat[0][1] = 1/sqrt(2);
    g.mat[1][0] = 1/sqrt(2); g.mat[1][1] = -1/sqrt(2);
    g.name = "H";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::PauliX(int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GatePauliX;
    g.mat[0][0] = 0; g.mat[0][1] = 1;
    g.mat[1][0] = 1; g.mat[1][1] = 0;
    g.name = "X";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::pauliY(int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GatePauliY;
    g.mat[0][0] = 0; g.mat[0][1] = Complex(0, -1);
    g.mat[1][0] = Complex(0, 1); g.mat[1][1] = Complex(0, 0);
    g.name = "Y";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::pauliZ(int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GatePauliZ;
    g.mat[0][0] = 1; g.mat[0][1] = 0;
    g.mat[1][0] = 0; g.mat[1][1] = -1;
    g.name = "Z";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::RotateX(int targetQubit, qreal angle) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateRotateX;
    g.mat[0][0] = cos(angle/2.0); g.mat[0][1] = Complex(0, -sin(angle/2.0));
    g.mat[1][0] = Complex(0, -sin(angle/2.0)); g.mat[1][1] = cos(angle/2.0);
    g.name = "RX";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::RotateY(int targetQubit, qreal angle) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateRotateY;
    g.mat[0][0] = cos(angle/2.0); g.mat[0][1] = -sin(angle/2.0);
    g.mat[1][0] = sin(angle/2.0); g.mat[1][1] = cos(angle/2.0);
    g.name = "RY";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::RotateZ(int targetQubit, qreal angle) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateRotateZ;
    g.mat[0][0] = Complex(cos(angle/2), -sin(angle/2)); g.mat[0][1] = 0;
    g.mat[1][0] = 0; g.mat[1][1] = Complex(cos(angle/2), sin(angle/2));
    g.name = "RZ";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::sGate(int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateS;
    g.mat[0][0] = 1; g.mat[0][1] = 0;
    g.mat[1][0] = 0; g.mat[1][1] = Complex(0, 1);
    g.name = "S";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::tGate(int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateT;
    g.mat[0][0] = 1; g.mat[0][1] = 0;
    g.mat[1][0] = 0; g.mat[1][1] = Complex(1/sqrt(2), 1/sqrt(2));
    g.name = "T";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}
