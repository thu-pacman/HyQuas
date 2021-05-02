#pragma once

#include <string>
#include <vector>
#include "utils.h"

enum class GateType {
    CCX, CNOT, CY, CZ, CP, CR, CRX, CRY, CU1, CRZ, U1, U2, U3, H, X, Y, Z, P, S, SDG, T, TDG, R, RX, RY, RZ, TOTAL, ID, GII, GZZ, GOC, GCC 
};

struct Gate {
    int gateID;
    GateType type;
    qComplex mat[2][2];
    std::string name;
    int targetQubit;
    int controlQubit; // -1 if no control
    int controlQubit2; // -1 if no control
    Gate(): controlQubit(-1), controlQubit2(-1) {};
    Gate(const Gate&) = default;
    bool isControlGate() const {
        return controlQubit != -1;
    }
    bool isC2Gate() const {
        return controlQubit2 != -1;
    }
    bool isDiagonal() const {
        return type == GateType::CZ || type == GateType::CP || type == GateType::CU1 || type == GateType::CRZ || type == GateType::U1 || type == GateType::P || type == GateType::Z || type == GateType::S || type == GateType::SDG || type == GateType::T || type == GateType::TDG || type == GateType::RZ;
    }

    static Gate CCX(int c1, int c2, int targetQubit);
    static Gate CNOT(int controlQubit, int targetQubit);
    static Gate CY(int controlQubit, int targetQubit);
    static Gate CZ(int controlQubit, int targetQubit);
    static Gate CP(int controlQubit, int targetQubit, qreal angle);
    static Gate CR(int controlQubit, int targetQubit, qreal angle, qreal axisX, qreal axisY, qreal axisZ);
    static Gate CRX(int controlQubit, int targetQubit, qreal angle);
    static Gate CRY(int controlQubit, int targetQubit, qreal angle);
    static Gate CU1(int controlQubit, int targetQubit, qreal lambda);
    static Gate CRZ(int controlQubit, int targetQubit, qreal angle);
    static Gate U1(int targetQubit, qreal lambda);
    static Gate U2(int targetQubit, qreal phi, qreal lambda);
    static Gate U3(int targetQubit, qreal theta, qreal phi, qreal lambda);
    static Gate H(int targetQubit);
    static Gate X(int targetQubit);
    static Gate Y(int targetQubit);
    static Gate Z(int targetQubit);
    static Gate P(int targetQubit, qreal angle);
    static Gate S(int targetQubit);
    static Gate SDG(int targetQubit); 
    static Gate T(int targetQubit);
    static Gate TDG(int targetQubit);
    static Gate R(int targetQubit, qreal angle, qreal axisX, qreal axisY, qreal axisZ);
    static Gate RX(int targetQubit, qreal angle);
    static Gate RY(int targetQubit, qreal angle);
    static Gate RZ(int targetQubit, qreal angle);
    static Gate ID(int targetQubit);
    static Gate GII(int targetQubit);
    static Gate GTT(int targetQubit);
    static Gate GZZ(int targetQubit);
    static Gate GOC(int targetQubit, qreal real, qreal imag);
    static Gate GCC(int targetQubit, qreal real, qreal imag);
    static Gate random(int lo, int hi);
    static Gate random(int lo, int hi, GateType type);
    static Gate control(int controlQubit, int targetQubit, GateType type);
    static GateType toCU(GateType type);
    static GateType toU(GateType type);
    static std::string get_name(GateType ty);
    std::vector<unsigned char> serialize() const;
    static Gate deserialize(const unsigned char* arr, int& cur);
};

struct KernelGate {
    int targetQubit;
    int controlQubit;
    int controlQubit2;
    GateType type;
    char targetIsGlobal;  // 0-local 1-global
    char controlIsGlobal; // 0-local 1-global 2-not control 
    char control2IsGlobal; // 0-local 1-global 2-not control
    qreal r00, i00, r01, i01, r10, i10, r11, i11;

    KernelGate(
        GateType type_,
        int controlQubit2_, char control2IsGlobal_, 
        int controlQubit_, char controlIsGlobal_,
        int targetQubit_, char targetIsGlobal_,
        const qComplex mat[2][2]
    ):
        targetQubit(targetQubit_), controlQubit(controlQubit_), controlQubit2(controlQubit2_),
        type(type_),
        targetIsGlobal(targetIsGlobal_), controlIsGlobal(controlIsGlobal_), control2IsGlobal(control2IsGlobal_),
        r00(mat[0][0].x), i00(mat[0][0].y), r01(mat[0][1].x), i01(mat[0][1].y),
        r10(mat[1][0].x), i10(mat[1][0].y), r11(mat[1][1].x), i11(mat[1][1].y) {}
    
    KernelGate(
        GateType type_,
        int controlQubit_, char controlIsGlobal_,
        int targetQubit_, char targetIsGlobal_,
        const qComplex mat[2][2]
    ): KernelGate(type_, 2, -1, controlQubit_, controlIsGlobal_, targetQubit_, targetIsGlobal_, mat) {}

    KernelGate(
        GateType type_,
        int targetQubit_, char targetIsGlobal_,
        const qComplex mat[2][2]
    ): KernelGate(type_, 2, -1, 2, -1, targetQubit_, targetIsGlobal_, mat) {}

    KernelGate() = default;

    static KernelGate ID() {
        qComplex mat[2][2] = {1, 0, 0, 1}; \
        return KernelGate(GateType::ID, 0, 0, mat);
    }
};