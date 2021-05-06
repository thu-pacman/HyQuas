#include "gate.h"

#include <cmath>
#include <cstring>
#include <assert.h>

static int globalGateID = 0;

// Gate Gate::CCX(int controlQubit, int controlQubit2, int targetQubit) {
//     Gate g;
//     g.gateID = ++ globalGateID;
//     g.type = GateType::CCX;
//     g.mat[0][0] = make_qComplex(0); g.mat[0][1] = make_qComplex(1);
//     g.mat[1][0] = make_qComplex(1); g.mat[1][1] = make_qComplex(0);
//     g.name = "CCX";
//     g.targetQubit = targetQubit;
//     g.controlQubit = controlQubit;
//     g.controlQubit2 = controlQubit2;
//     return g;

// }

Gate Gate::CNOT(int controlQubit, int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::CNOT;
    g.mat[0][0] = make_qComplex(0); g.mat[0][1] = make_qComplex(1);
    g.mat[1][0] = make_qComplex(1); g.mat[1][1] = make_qComplex(0);
    g.name = "CN";
    g.targetQubit = targetQubit;
    g.controlQubit = controlQubit;
    return g;
}

Gate Gate::CY(int controlQubit, int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::CY;
    g.mat[0][0] = make_qComplex(0); g.mat[0][1] = make_qComplex(0, -1);
    g.mat[1][0] = make_qComplex(0, 1); g.mat[1][1] = make_qComplex(0);
    g.name = "CY";
    g.targetQubit = targetQubit;
    g.controlQubit = controlQubit;
    return g;
}

Gate Gate::CZ(int controlQubit, int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::CZ;
    g.mat[0][0] = make_qComplex(1); g.mat[0][1] = make_qComplex(0);
    g.mat[1][0] = make_qComplex(0); g.mat[1][1] = make_qComplex(-1);
    g.name = "CZ";
    g.targetQubit = targetQubit;
    g.controlQubit = controlQubit;
    return g;
}

Gate Gate::CU(int controlQubit, int targetQubit, qComplex a0, qComplex a1, qComplex b0, qComplex b1) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::CU;
    g.mat[0][0] = a0; g.mat[0][1] = a1;
    g.mat[1][0] = b0; g.mat[1][1] = b1;
    g.name = "CU";
    g.targetQubit = targetQubit;
    g.controlQubit = controlQubit;
    return g;
}

Gate Gate::CUC(int controlQubit, int targetQubit, qComplex alpha, qComplex beta) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::CUC;
    g.mat[0][0] = make_qComplex(alpha.x, alpha.y); g.mat[0][1] = make_qComplex(-beta.x, beta.y);
    g.mat[1][0] = make_qComplex(beta.x, beta.y); g.mat[1][1] = make_qComplex(alpha.x, -alpha.y);
    g.name = "CUC";
    g.targetQubit = targetQubit;
    g.controlQubit = controlQubit;
    return g;
}

Gate Gate::CRX(int controlQubit, int targetQubit, qreal angle) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::CRX;
    g.mat[0][0] = make_qComplex(cos(angle/2.0)); g.mat[0][1] = make_qComplex(0, -sin(angle/2.0));
    g.mat[1][0] = make_qComplex(0, -sin(angle/2.0)); g.mat[1][1] = make_qComplex(cos(angle/2.0));
    g.name = "CRX";
    g.targetQubit = targetQubit;
    g.controlQubit = controlQubit;
    return g;
}

Gate Gate::CRY(int controlQubit, int targetQubit, qreal angle) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::CRY;
    g.mat[0][0] = make_qComplex(cos(angle/2.0)); g.mat[0][1] = make_qComplex(-sin(angle/2.0));
    g.mat[1][0] = make_qComplex(sin(angle/2.0)); g.mat[1][1] = make_qComplex(cos(angle/2.0));
    g.name = "CRY";
    g.targetQubit = targetQubit;
    g.controlQubit = controlQubit;
    return g;
}

Gate Gate::CU1(int controlQubit, int targetQubit, qreal lambda) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::CU1;
    g.mat[0][0] = make_qComplex(1);
    g.mat[0][1] = make_qComplex(0);
    g.mat[1][0] = make_qComplex(0);
    g.mat[1][1] = make_qComplex(cos(lambda), sin(lambda));
    g.name = "CU1";
    g.targetQubit = targetQubit;
    g.controlQubit = controlQubit;
    return g;
}

Gate Gate::CRZ(int controlQubit, int targetQubit, qreal angle) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::CRZ;
    g.mat[0][0] = make_qComplex(cos(angle/2), -sin(angle/2)); g.mat[0][1] = make_qComplex(0);
    g.mat[1][0] = make_qComplex(0); g.mat[1][1] = make_qComplex(cos(angle/2), sin(angle/2));
    g.name = "CRZ";
    g.targetQubit = targetQubit;
    g.controlQubit = controlQubit;
    return g;
}

Gate Gate::U1(int targetQubit, qreal lambda) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::U1;
    g.mat[0][0] = make_qComplex(1);
    g.mat[0][1] = make_qComplex(0);
    g.mat[1][0] = make_qComplex(0);
    g.mat[1][1] = make_qComplex(cos(lambda), sin(lambda));
    g.name = "U1";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::U2(int targetQubit, qreal phi, qreal lambda) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::U2;
    g.mat[0][0] = make_qComplex(1.0 / sqrt(2));
    g.mat[0][1] = make_qComplex(-cos(lambda), -sin(lambda));
    g.mat[1][0] = make_qComplex(cos(lambda), sin(lambda));
    g.mat[1][1] = make_qComplex(cos(lambda + phi), sin(lambda + phi));
    g.name = "U2";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::U3(int targetQubit, qreal theta, qreal phi, qreal lambda) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::U3;
    g.mat[0][0] = make_qComplex(cos(theta / 2));
    g.mat[0][1] = make_qComplex(-cos(lambda) * sin(theta / 2), -sin(lambda) * sin(theta / 2));
    g.mat[1][0] = make_qComplex(cos(phi) * sin(theta / 2), sin(phi) * sin(theta / 2));
    g.mat[1][1] = make_qComplex(cos(phi + lambda) * cos(theta / 2), sin(phi + lambda) * cos(theta / 2));
    g.name = "U3";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::H(int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::H;
    g.mat[0][0] = make_qComplex(1/sqrt(2)); g.mat[0][1] = make_qComplex(1/sqrt(2));
    g.mat[1][0] = make_qComplex(1/sqrt(2)); g.mat[1][1] = make_qComplex(-1/sqrt(2));
    g.name = "H";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::X(int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::X;
    g.mat[0][0] = make_qComplex(0); g.mat[0][1] = make_qComplex(1);
    g.mat[1][0] = make_qComplex(1); g.mat[1][1] = make_qComplex(0);
    g.name = "X";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::Y(int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::Y;
    g.mat[0][0] = make_qComplex(0); g.mat[0][1] = make_qComplex(0, -1);
    g.mat[1][0] = make_qComplex(0, 1); g.mat[1][1] = make_qComplex(0);
    g.name = "Y";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::Z(int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::Z;
    g.mat[0][0] = make_qComplex(1); g.mat[0][1] = make_qComplex(0);
    g.mat[1][0] = make_qComplex(0); g.mat[1][1] = make_qComplex(-1);
    g.name = "Z";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::S(int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::S;
    g.mat[0][0] = make_qComplex(1); g.mat[0][1] = make_qComplex(0);
    g.mat[1][0] = make_qComplex(0); g.mat[1][1] = make_qComplex(0, 1);
    g.name = "S";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::SDG(int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::SDG;
    g.mat[0][0] = make_qComplex(1); g.mat[0][1] = make_qComplex(0);
    g.mat[1][0] = make_qComplex(0); g.mat[1][1] = make_qComplex(0, -1);
    g.name = "SDG";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::T(int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::T;
    g.mat[0][0] = make_qComplex(1); g.mat[0][1] = make_qComplex(0);
    g.mat[1][0] = make_qComplex(0); g.mat[1][1] = make_qComplex(1/sqrt(2), 1/sqrt(2));
    g.name = "T";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::TDG(int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::T;
    g.mat[0][0] = make_qComplex(1); g.mat[0][1] = make_qComplex(0);
    g.mat[1][0] = make_qComplex(0); g.mat[1][1] = make_qComplex(1/sqrt(2), -1/sqrt(2));
    g.name = "TDG";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::RX(int targetQubit, qreal angle) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::RX;
    g.mat[0][0] = make_qComplex(cos(angle/2.0)); g.mat[0][1] = make_qComplex(0, -sin(angle/2.0));
    g.mat[1][0] = make_qComplex(0, -sin(angle/2.0)); g.mat[1][1] = make_qComplex(cos(angle/2.0));
    g.name = "RX";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::RY(int targetQubit, qreal angle) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::RY;
    g.mat[0][0] = make_qComplex(cos(angle/2.0)); g.mat[0][1] = make_qComplex(-sin(angle/2.0));
    g.mat[1][0] = make_qComplex(sin(angle/2.0)); g.mat[1][1] = make_qComplex(cos(angle/2.0));
    g.name = "RY";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::RZ(int targetQubit, qreal angle) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::RZ;
    g.mat[0][0] = make_qComplex(cos(angle/2), -sin(angle/2)); g.mat[0][1] = make_qComplex(0);
    g.mat[1][0] = make_qComplex(0); g.mat[1][1] = make_qComplex(cos(angle/2), sin(angle/2));
    g.name = "RZ";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::U(int targetQubit, qComplex a0, qComplex a1, qComplex b0, qComplex b1) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::U;
    g.mat[0][0] = a0; g.mat[0][1] = a1;
    g.mat[1][0] = b0; g.mat[1][1] = b1;
    g.name = "U";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::MU1(std::vector<int> controlQubits, int targetQubit, qreal lambda) {
    if (controlQubits.size() == 0) return Gate::U1(targetQubit, lambda);
    if (controlQubits.size() == 1) return Gate::CU1(controlQubits[0], targetQubit, lambda);
    Gate g;
    g.gateID = ++globalGateID;
    g.type = GateType::MU1;
    g.mat[0][0] = make_qComplex(1);
    g.mat[0][1] = make_qComplex(0);
    g.mat[1][0] = make_qComplex(0);
    g.mat[1][1] = make_qComplex(cos(lambda), sin(lambda));
    g.name = "MU1";
    g.encodeQubit = to_bitmap(controlQubits);
    g.targetQubit = targetQubit;
    g.controlQubit = -2;
    g.controlQubits = controlQubits;
    return g;
}

Gate Gate::MZ(std::vector<int> controlQubits, int targetQubit) {
    if (controlQubits.size() == 0) return Gate::Z(targetQubit);
    if (controlQubits.size() == 1) return Gate::CZ(controlQubits[0], targetQubit);
    Gate g;
    g.gateID = ++globalGateID;
    g.type = GateType::MZ;
    g.mat[0][0] = make_qComplex(1); g.mat[0][1] = make_qComplex(0);
    g.mat[1][0] = make_qComplex(0); g.mat[1][1] = make_qComplex(-1);
    g.name = "MZ";
    g.encodeQubit = to_bitmap(controlQubits);
    g.targetQubit = targetQubit;
    g.controlQubit = -2;
    g.controlQubits = controlQubits;
    return g;
}

Gate Gate::MU(std::vector<int> controlQubits, int targetQubit, qComplex a0, qComplex a1, qComplex b0, qComplex b1) {
    if (controlQubits.size() == 0) return Gate::U(targetQubit, a0, a1, b0, b1);
    if (controlQubits.size() == 1) return Gate::CU(controlQubits[0], targetQubit, a0, a1, b0, b1);
    Gate g;
    g.gateID = ++globalGateID;
    g.type = GateType::MU;
    g.mat[0][0] = a0; g.mat[0][1] = a1;
    g.mat[1][0] = b0; g.mat[1][1] = b1;
    g.name = "MU";
    g.encodeQubit = to_bitmap(controlQubits);
    g.targetQubit = targetQubit;
    g.controlQubit = -2;
    g.controlQubits = controlQubits;
    return g;
}

Gate Gate::FSIM(int targetQubit1, int targetQubit2, qreal theta, qreal phi) {
    Gate g;
    g.gateID = ++globalGateID;
    g.type = GateType::FSM;
    // a compressed matrix representation. be careful in blas backend
    g.mat[0][0] = make_qComplex(cos(theta)); g.mat[0][1] = make_qComplex(0, -sin(theta));
    g.mat[0][1] = make_qComplex(0, -sin(theta)); g.mat[1][1] = make_qComplex(cos(theta), -sin(theta));
    g.name = "FSM";
    g.encodeQubit = targetQubit1;
    g.targetQubit = targetQubit2;
    g.controlQubit = -3;
    return g;
}
    

Gate Gate::UC(int targetQubit, qComplex alpha, qComplex beta) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::UC;
    g.mat[0][0] = make_qComplex(alpha.x, alpha.y); g.mat[0][1] = make_qComplex(-beta.x, beta.y);
    g.mat[1][0] = make_qComplex(beta.x, beta.y); g.mat[1][1] = make_qComplex(alpha.x, -alpha.y);
    g.name = "UC";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::ID(int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::ID;
    g.mat[0][0] = make_qComplex(1); g.mat[0][1] = make_qComplex(0);
    g.mat[1][0] = make_qComplex(0); g.mat[1][1] = make_qComplex(1);
    g.name = "ID";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::GII(int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::GII;
    g.mat[0][0] = make_qComplex(0, 1); g.mat[0][1] = make_qComplex(0);
    g.mat[1][0] = make_qComplex(0); g.mat[1][1] = make_qComplex(0, 1);
    g.name = "GII";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::GZZ(int targetQubit) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::GZZ;
    g.mat[0][0] = make_qComplex(-1); g.mat[0][1] = make_qComplex(0);
    g.mat[1][0] = make_qComplex(0); g.mat[1][1] = make_qComplex(-1);
    g.name = "GZZ";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::GOC(int targetQubit, qreal real, qreal imag) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::GOC;
    g.mat[0][0] = make_qComplex(1); g.mat[0][1] = make_qComplex(0);
    g.mat[1][0] = make_qComplex(0); g.mat[1][1] = make_qComplex(real, imag);
    g.name = "GOC";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

Gate Gate::GCC(int targetQubit, qreal real, qreal imag) {
    Gate g;
    g.gateID = ++ globalGateID;
    g.type = GateType::GCC;
    g.mat[0][0] = make_qComplex(real, imag); g.mat[0][1] = make_qComplex(0);
    g.mat[1][0] = make_qComplex(0); g.mat[1][1] = make_qComplex(real, imag);
    g.name = "GCC";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    return g;
}

auto gen_01_float = []() {
    return rand() * 1.0 / RAND_MAX;
};
auto gen_0_2pi_float = []() {
        return gen_01_float() * acos(-1) * 2;
};

Gate Gate::random(int lo, int hi) {
    int type = rand() % int(GateType::TOTAL);
    return random(lo, hi, GateType(type));
}

Gate Gate::random(int lo, int hi, GateType type) {
    auto gen_c2_id = [lo, hi](int &t, int &c1, int &c2) {
        assert(hi - lo >= 3);
        do {
            c2 = rand() % (hi - lo) + lo;
            c1 = rand() % (hi - lo) + lo;
            t = rand() % (hi - lo) + lo;
        } while (c2 == c1 || c2 == t || c1 == t);
    };
    auto gen_c1_id = [lo, hi](int &t, int &c1) {
        assert(hi - lo >= 2);
        do {
            c1 = rand() % (hi - lo) + lo;
            t = rand() % (hi - lo) + lo;
        } while (c1 == t);
    };
    auto gen_single_id = [lo, hi](int &t) {
        t = rand() % (hi - lo) + lo;
    };
    switch (type) {
        // case GateType::CCX: {
        //     int t, c1, c2;
        //     gen_c2_id(t, c1, c2);
        //     return CCX(c1, c2, t);
        // }
        case GateType::CNOT: {
            int t, c1;
            gen_c1_id(t, c1);
            return CNOT(c1, t);
        }
        case GateType::CY: {
            int t, c1;
            gen_c1_id(t, c1);
            return CY(c1, t);
        }
        case GateType::CZ: {
            int t, c1;
            gen_c1_id(t, c1);
            return CZ(c1, t);
        }
        case GateType::CU: {
            int t, c1;
            gen_c1_id(t, c1);
            return CU(
                c1, t,
                make_qComplex(gen_01_float(), gen_01_float()),
                make_qComplex(gen_01_float(), gen_01_float()),
                make_qComplex(gen_01_float(), gen_01_float()),
                make_qComplex(gen_01_float(), gen_01_float())
            );
        }
        case GateType::CUC: {
            int t, c1;
            gen_c1_id(t, c1);
            return CUC(c1, t, make_qComplex(gen_01_float(), gen_01_float()), make_qComplex(gen_01_float(), gen_01_float()));
        }
        case GateType::CRX: {
            int t, c1;
            gen_c1_id(t, c1);
            return CRX(c1, t, gen_0_2pi_float());
        }
        case GateType::CRY: {
            int t, c1;
            gen_c1_id(t, c1);
            return CRY(c1, t, gen_0_2pi_float());
        }
        case GateType::CU1: {
            int t, c1;
            gen_c1_id(t, c1);
            return CU1(c1, t, gen_0_2pi_float());
        }
        case GateType::CRZ: {
            int t, c1;
            gen_c1_id(t, c1);
            return CRZ(c1, t, gen_0_2pi_float());
        }
        case GateType::U1: {
            int t;
            gen_single_id(t);
            return U1(t, gen_0_2pi_float());
        }
        case GateType::U2: {
            int t;
            gen_single_id(t);
            return U2(t, gen_0_2pi_float(), gen_0_2pi_float());
        }
        case GateType::U: {
            int t;
            gen_single_id(t);
            return U(
                t,
                make_qComplex(gen_01_float(), gen_01_float()),
                make_qComplex(gen_01_float(), gen_01_float()),
                make_qComplex(gen_01_float(), gen_01_float()),
                make_qComplex(gen_01_float(), gen_01_float())
            );
        }
        case GateType::UC: {
            int t;
            gen_single_id(t);
            return UC(t, make_qComplex(gen_01_float(), gen_01_float()), make_qComplex(gen_01_float(), gen_01_float()));
        }
        case GateType::U3: {
            int t;
            gen_single_id(t);
            return U3(t, gen_0_2pi_float(), gen_0_2pi_float(), gen_0_2pi_float());
        }
        case GateType::H: {
            int t;
            gen_single_id(t);
            return H(t);
        }
        case GateType::X: {
            int t;
            gen_single_id(t);
            return X(t);
        }
        case GateType::Y: {
            int t;
            gen_single_id(t);
            return Y(t);
        }
        case GateType::Z: {
            int t;
            gen_single_id(t);
            return Z(t);
        }
        case GateType::S: {
            int t;
            gen_single_id(t);
            return S(t);
        }
        case GateType::SDG: {
            int t;
            gen_single_id(t);
            return SDG(t);
        }
        case GateType::T: {
            int t;
            gen_single_id(t);
            return T(t);
        }
        case GateType::TDG: {
            int t;
            gen_single_id(t);
            return TDG(t);
        }
        case GateType::RX: {
            int t;
            gen_single_id(t);
            return RX(t, gen_0_2pi_float());
        }
        case GateType::RY: {
            int t;
            gen_single_id(t);
            return RY(t, gen_0_2pi_float());
        }
        case GateType::RZ: {
            int t;
            gen_single_id(t);
            return RZ(t, gen_0_2pi_float());
        }
        case GateType::FSM: // no break
        case GateType::MU1: // no break
        case GateType::MZ: // no break
        case GateType::MU: {
            UNIMPLEMENTAED();
        }
        default: {
            printf("invalid %d\n", (int) type);
            assert(false);
        }
    }
    exit(1);
}

Gate Gate::control(int controlQubit, int targetQubit, GateType type) {
    switch (type) {
        case GateType::CNOT: {
            return CNOT(controlQubit, targetQubit);
        }
        case GateType::CY: {
            return CY(controlQubit, targetQubit);
        }
        case GateType::CZ: {
            return CZ(controlQubit, targetQubit);
        }
        case GateType::CU: {
            return CU(
                controlQubit, targetQubit,
                make_qComplex(gen_01_float(), gen_01_float()),
                make_qComplex(gen_01_float(), gen_01_float()),
                make_qComplex(gen_01_float(), gen_01_float()),
                make_qComplex(gen_01_float(), gen_01_float())
            );
        }
        case GateType::CUC: {
            return CUC(controlQubit, targetQubit, make_qComplex(gen_01_float(), gen_01_float()), make_qComplex(gen_01_float(), gen_01_float()));
        }
        case GateType::CRX: {
            return CRX(controlQubit, targetQubit, gen_0_2pi_float());
        }
        case GateType::CRY: {
            return CRY(controlQubit, targetQubit, gen_0_2pi_float());
        }
        case GateType::CU1: {
            return CU1(controlQubit, targetQubit, gen_0_2pi_float());
        }
        case GateType::CRZ: {
            return CRZ(controlQubit, targetQubit, gen_0_2pi_float());
        }
        default: {
            assert(false);
        }
    }
    exit(1);
}

// GateType Gate::toCU(GateType type) {
//     if (type == GateType::CCX) {
//         return GateType::CNOT;
//     } else {
//         UNREACHABLE()
//     }
// }

GateType Gate::toU(GateType type) {
    switch (type) {
        // case GateType::CCX:
        case GateType::CNOT:
            return GateType::X;
        case GateType::CY:
            return GateType::Y;
        case GateType::CZ:
            return GateType::Z;
        case GateType::CU:
            return GateType::U;
        case GateType::CUC:
            return GateType::UC;
        case GateType::CRX:
            return GateType::RX;
        case GateType::CRY:
            return GateType::RY;
        case GateType::CU1:
            return GateType::U1;
        case GateType::CRZ:
            return GateType::RZ;
        default:
            UNREACHABLE()
    }
}

std::string Gate::get_name(GateType ty) {
    return random(0, 10, ty).name;
}

std::vector<unsigned char> Gate::serialize() const {
    auto name_len = name.length();
    int len =
        sizeof(name_len) + name.length() + 1 + sizeof(gateID) + sizeof(type) + sizeof(mat)
        + sizeof(targetQubit) + sizeof(controlQubit) + sizeof(encodeQubit);
    std::vector<unsigned char> ret; ret.resize(len);
    unsigned char* arr = ret.data();
    int cur = 0;
    SERIALIZE_STEP(gateID);
    SERIALIZE_STEP(type);
    memcpy(arr + cur, mat, sizeof(mat)); cur += sizeof(qComplex) * 4;
    SERIALIZE_STEP(name_len);
    strcpy(reinterpret_cast<char*>(arr) + cur, name.c_str()); cur += name_len + 1;
    SERIALIZE_STEP(targetQubit);
    SERIALIZE_STEP(controlQubit);
    SERIALIZE_STEP(encodeQubit);
    assert(cur == len);
    return ret;
}

Gate Gate::deserialize(const unsigned char* arr, int& cur) {
    Gate g;
    DESERIALIZE_STEP(g.gateID);
    DESERIALIZE_STEP(g.type);
    memcpy(g.mat, arr + cur, sizeof(g.mat)); cur += sizeof(qComplex) * 4;
    decltype(g.name.length()) name_len; DESERIALIZE_STEP(name_len);
    g.name = std::string(reinterpret_cast<const char*>(arr) + cur, name_len); cur += name_len + 1;
    DESERIALIZE_STEP(g.targetQubit);
    DESERIALIZE_STEP(g.controlQubit);
    DESERIALIZE_STEP(g.encodeQubit);
    if (g.controlQubit == -2) {
        g.controlQubits.clear();
        int qid = 0;
        qindex ctrs = g.encodeQubit;
        while (ctrs > 0) {
            if (ctrs & 1) g.controlQubits.push_back(qid);
            qid ++;
        }
    }
    return g;
}