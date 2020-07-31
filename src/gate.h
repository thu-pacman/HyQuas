#pragma once

#include <string>
#include "utils.h"

enum class GateType {
    CNOT, CY, CZ, CRX, CRY, CRZ, U1, U2, U3, H, X, Y, Z, S, T, RX, RY, RZ
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
        return type == GateType::CZ || type == GateType::CRZ || type == GateType::U1 || type == GateType::Z || type == GateType::S || type == GateType::T || type == GateType::RZ;
    }
    
    static Gate CNOT(int controlQubit, int targetQubit);
    static Gate CY(int controlQubit, int targetQubit);
    static Gate CZ(int controlQubit, int targetQubit);
    static Gate CRX(int controlQubit, int targetQubit, qreal angle);
    static Gate CRY(int controlQubit, int targetQubit, qreal angle);
    static Gate CRZ(int controlQubit, int targetQubit, qreal angle);
    static Gate U1(int targetQubit, qreal lambda);
    static Gate U2(int targetQubit, qreal phi, qreal lambda);
    static Gate U3(int targetQubit, qreal theta, qreal phi, qreal lambda);
    static Gate H(int targetQubit);
    static Gate X(int targetQubit);
    static Gate Y(int targetQubit);
    static Gate Z(int targetQubit);
    static Gate S(int targetQubit); 
    static Gate T(int targetQubit);
    static Gate RX(int targetQubit, qreal angle);
    static Gate RY(int targetQubit, qreal angle);
    static Gate RZ(int targetQubit, qreal angle);

};
