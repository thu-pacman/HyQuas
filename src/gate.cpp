#include "gate.h"

#include <cmath>
#include <assert.h>

static int globalGateID = 0;

Gate Gate::CCX(int controlQubit, int controlQubit2, int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::CCX;
    g.mat[0][0] = 0; g.mat[0][1] = 1;
    g.mat[1][0] = 1; g.mat[1][1] = 0;
    g.name = "CCX";
    g.targetQubit = targetQubit;
    g.controlQubit = controlQubit;
    g.controlQubit2 = controlQubit2;
    return g;

}

Gate Gate::CNOT(int controlQubit, int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::CNOT;
    g.mat[0][0] = 0; g.mat[0][1] = 1;
    g.mat[1][0] = 1; g.mat[1][1] = 0;
    g.name = "CN";
    g.targetQubit = targetQubit;
    g.controlQubit = controlQubit;
    return g;
}

Gate Gate::CY(int controlQubit, int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::CY;
    g.mat[0][0] = 0; g.mat[0][1] = Complex(0, -1);
    g.mat[1][0] = Complex(0, 1); g.mat[1][1] = Complex(0, 0);
    g.name = "CY";
    g.targetQubit = targetQubit;
    g.controlQubit = controlQubit;
    return g;
}

Gate Gate::CZ(int controlQubit, int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::CZ;
    g.mat[0][0] = 1; g.mat[0][1] = 0;
    g.mat[1][0] = 0; g.mat[1][1] = -1;
    g.name = "CZ";
    g.targetQubit = targetQubit;
    g.controlQubit = controlQubit;
    return g;
}

Gate Gate::CRX(int controlQubit, int targetQubit, qreal angle) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::CRX;
    g.mat[0][0] = cos(angle/2.0); g.mat[0][1] = Complex(0, -sin(angle/2.0));
    g.mat[1][0] = Complex(0, -sin(angle/2.0)); g.mat[1][1] = cos(angle/2.0);
    g.name = "CRX";
    g.targetQubit = targetQubit;
    g.controlQubit = controlQubit;
    return g;
}

Gate Gate::CRY(int controlQubit, int targetQubit, qreal angle) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::CRY;
    g.mat[0][0] = cos(angle/2.0); g.mat[0][1] = -sin(angle/2.0);
    g.mat[1][0] = sin(angle/2.0); g.mat[1][1] = cos(angle/2.0);
    g.name = "CRY";
    g.targetQubit = targetQubit;
    g.controlQubit = controlQubit;
    return g;
}

Gate Gate::CRZ(int controlQubit, int targetQubit, qreal angle) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::CRZ;
    g.mat[0][0] = Complex(cos(angle/2), -sin(angle/2)); g.mat[0][1] = 0;
    g.mat[1][0] = 0; g.mat[1][1] = Complex(cos(angle/2), sin(angle/2));
    g.name = "CRZ";
    g.targetQubit = targetQubit;
    g.controlQubit = controlQubit;
    return g;
}


Gate Gate::U1(int targetQubit, qreal lambda) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::U1;
    g.mat[0][0] = 1;
    g.mat[0][1] = 0;
    g.mat[1][0] = 0;
    g.mat[1][1] = Complex(cos(lambda), sin(lambda));
    g.name = "U1";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::U2(int targetQubit, qreal phi, qreal lambda) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::U2;
    g.mat[0][0] = 1 / sqrt(2);
    g.mat[0][1] = Complex(-cos(lambda), -sin(lambda));
    g.mat[1][0] = Complex(cos(lambda), sin(lambda));
    g.mat[1][1] = Complex(cos(lambda + phi), sin(lambda + phi));
    g.name = "U2";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::U3(int targetQubit, qreal theta, qreal phi, qreal lambda) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::U3;
    g.mat[0][0] = cos(theta / 2);
    g.mat[0][1] = Complex(-cos(lambda) * sin(theta / 2), -sin(lambda) * sin(theta / 2));
    g.mat[1][0] = Complex(cos(phi) * sin(theta / 2), sin(phi) * sin(theta / 2));
    g.mat[1][1] = Complex(cos(phi + lambda) * cos(theta / 2), sin(phi + lambda) * cos(theta / 2));
    g.name = "U3";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::H(int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::H;
    g.mat[0][0] = 1/sqrt(2); g.mat[0][1] = 1/sqrt(2);
    g.mat[1][0] = 1/sqrt(2); g.mat[1][1] = -1/sqrt(2);
    g.name = "H";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::X(int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::X;
    g.mat[0][0] = 0; g.mat[0][1] = 1;
    g.mat[1][0] = 1; g.mat[1][1] = 0;
    g.name = "X";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::Y(int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::Y;
    g.mat[0][0] = 0; g.mat[0][1] = Complex(0, -1);
    g.mat[1][0] = Complex(0, 1); g.mat[1][1] = Complex(0, 0);
    g.name = "Y";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::Z(int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::Z;
    g.mat[0][0] = 1; g.mat[0][1] = 0;
    g.mat[1][0] = 0; g.mat[1][1] = -1;
    g.name = "Z";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::S(int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::S;
    g.mat[0][0] = 1; g.mat[0][1] = 0;
    g.mat[1][0] = 0; g.mat[1][1] = Complex(0, 1);
    g.name = "S";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::T(int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::T;
    g.mat[0][0] = 1; g.mat[0][1] = 0;
    g.mat[1][0] = 0; g.mat[1][1] = Complex(1/sqrt(2), 1/sqrt(2));
    g.name = "T";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}


Gate Gate::RX(int targetQubit, qreal angle) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::RX;
    g.mat[0][0] = cos(angle/2.0); g.mat[0][1] = Complex(0, -sin(angle/2.0));
    g.mat[1][0] = Complex(0, -sin(angle/2.0)); g.mat[1][1] = cos(angle/2.0);
    g.name = "RX";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::RY(int targetQubit, qreal angle) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::RY;
    g.mat[0][0] = cos(angle/2.0); g.mat[0][1] = -sin(angle/2.0);
    g.mat[1][0] = sin(angle/2.0); g.mat[1][1] = cos(angle/2.0);
    g.name = "RY";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::RZ(int targetQubit, qreal angle) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::RZ;
    g.mat[0][0] = Complex(cos(angle/2), -sin(angle/2)); g.mat[0][1] = 0;
    g.mat[1][0] = 0; g.mat[1][1] = Complex(cos(angle/2), sin(angle/2));
    g.name = "RZ";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

