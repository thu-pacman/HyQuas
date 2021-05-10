#pragma once

#include <string>
#include <vector>
#include "utils.h"

enum class GateType {
    CNOT, CY, CZ, CU, CUC, CRX, CRY, CU1, CRZ, U1, U2, U, UC, U3, H, X, Y, Z, S, SDG, T, TDG, RX, RY, RZ, FSM, MU1, MZ, MU, TOTAL, ID, GII, GZZ, GOC, GCC, MCI
};

struct Gate {
    int gateID;
    GateType type;
    qComplex mat[2][2];
    std::string name;
    int targetQubit;
    int controlQubit; // -1 for single bit gate， -2 for MC gates, -3 for two qubit gates
    qindex encodeQubit; // bit map of the control qubits of MC gates, target2 for two qubit gate
    std::vector<int> controlQubits;
    Gate(): controlQubit(-1), encodeQubit(0) {};
    Gate(const Gate&) = default;
    
    bool isControlGate() const {
        return controlQubit >= 0;
    }

    bool isSingleQubitGate() const {
        return controlQubit == -1;
    }

    bool isMCGate() const {
        return controlQubit == -2;
    }

    bool isTwoQubitGate() const {
        return controlQubit == -3;
    }
    
    bool isDiagonal() const {
        return type == GateType::CZ || type == GateType::CU1 || type == GateType::CRZ || type == GateType::U1 || type == GateType::Z || type == GateType::S || type == GateType::SDG || type == GateType::T || type == GateType::TDG || type == GateType::RZ;
    }

    bool hasControl(int q) const {
        if (isControlGate()) return controlQubit == q;
        if (isMCGate()) return encodeQubit >> q & 1;
        return false;
    }

    bool hasTarget(int q) const {
        if (isTwoQubitGate()) return targetQubit == q || encodeQubit == q;
        return targetQubit == q;
    }

    // static Gate CCX(int c1, int c2, int targetQubit);
    static Gate CNOT(int controlQubit, int targetQubit);
    static Gate CY(int controlQubit, int targetQubit);
    static Gate CZ(int controlQubit, int targetQubit);
    static Gate CU(int controlQubit, int targetQubit, qComplex a0, qComplex a1, qComplex b0, qComplex b1);
    static Gate CUC(int controlQubit, int targetQubit, qComplex alpha, qComplex beta);
    static Gate CRX(int controlQubit, int targetQubit, qreal angle);
    static Gate CRY(int controlQubit, int targetQubit, qreal angle);
    static Gate CU1(int controlQubit, int targetQubit, qreal lambda);
    static Gate CRZ(int controlQubit, int targetQubit, qreal angle);
    static Gate U1(int targetQubit, qreal lambda);
    static Gate U2(int targetQubit, qreal phi, qreal lambda);
    static Gate U(int targetQubit, qComplex a0, qComplex a1, qComplex b0, qComplex b1);
    static Gate UC(int targetQubit, qComplex alpha, qComplex beta);
    static Gate U3(int targetQubit, qreal theta, qreal phi, qreal lambda);
    static Gate H(int targetQubit);
    static Gate X(int targetQubit);
    static Gate Y(int targetQubit);
    static Gate Z(int targetQubit);
    static Gate S(int targetQubit);
    static Gate SDG(int targetQubit); 
    static Gate T(int targetQubit);
    static Gate TDG(int targetQubit);
    static Gate RX(int targetQubit, qreal angle);
    static Gate RY(int targetQubit, qreal angle);
    static Gate RZ(int targetQubit, qreal angle);
    static Gate MU1(std::vector<int> controlQubits, int targetQubit, qreal lambda);
    static Gate MZ(std::vector<int> controlQubits, int targetQubit);
    static Gate MU(std::vector<int> controlQubits, int targetQubit, qComplex a0, qComplex a1, qComplex b0, qComplex b1);
    static Gate FSIM(int targetQubit1, int targetQubit2, qreal theta, qreal phi);
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
    qindex encodeQubit;
    GateType type;
    char targetIsGlobal;  // 0-local 1-global
    char controlIsGlobal; // 0-local 1-global 2-not control 
    qreal r00, i00, r01, i01, r10, i10, r11, i11;

    KernelGate(
        GateType type_,
        qindex encodeQubit_,
        int controlQubit_, char controlIsGlobal_,
        int targetQubit_, char targetIsGlobal_,
        const qComplex mat[2][2]
    ):
        targetQubit(targetQubit_), controlQubit(controlQubit_), encodeQubit(encodeQubit_),
        type(type_),
        targetIsGlobal(targetIsGlobal_), controlIsGlobal(controlIsGlobal_),
        r00(mat[0][0].x), i00(mat[0][0].y), r01(mat[0][1].x), i01(mat[0][1].y),
        r10(mat[1][0].x), i10(mat[1][0].y), r11(mat[1][1].x), i11(mat[1][1].y) {}

    KernelGate() = default;

    // control gate
    static KernelGate controlGate(
        GateType type,
        int controlQubit, char controlIsGlobal,
        int targetQubit, char targetIsGlobal,
        const qComplex mat[2][2]
    ) {
        return KernelGate(type, 0, controlQubit, controlIsGlobal, targetQubit, targetIsGlobal, mat);
    }

    // multi-control gate
    static KernelGate mcGate(
        GateType type,
        qindex mcQubits,
        int targetQubit, char targetIsGlobal,
        const qComplex mat[2][2]
    ) {
        return KernelGate(type, mcQubits, -2, 2, targetQubit, targetIsGlobal, mat);
    }
    
    // two qubit gate
    static KernelGate twoQubitGate(
        GateType type,
        int targetQubit1, char target1IsGlobal,
        int targetQubit2, char target2IsGlobal,
        const qComplex mat[2][2]
    ) {
        return KernelGate(type, targetQubit1, -3, target1IsGlobal, targetQubit2, target2IsGlobal, mat);
    }

    // single qubit gate
    static KernelGate singleQubitGate(
        GateType type,
        int targetQubit, char targetIsGlobal,
        const qComplex mat[2][2]
    ) {
        return KernelGate(type, 0, 2, -1, targetQubit, targetIsGlobal, mat);
    }

    static KernelGate ID() {
        qComplex mat[2][2] = {1, 0, 0, 1}; \
        return KernelGate::singleQubitGate(GateType::ID, 0, 0, mat);
    }
};