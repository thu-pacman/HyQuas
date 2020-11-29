#include "gate.h"

#include <cmath>
#include <cstring>
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
        case GateType::CCX: {
            int t, c1, c2;
            gen_c2_id(t, c1, c2);
            return CCX(c1, c2, t);
        }
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
        case GateType::T: {
            int t;
            gen_single_id(t);
            return T(t);
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
        default: {
            printf("invalid %d\n", type);
            assert(false);
        }
    }
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
        case GateType::CRX: {
            return CRX(controlQubit, targetQubit, gen_0_2pi_float());
        }
        case GateType::CRY: {
            return CRY(controlQubit, targetQubit, gen_0_2pi_float());
        }
        case GateType::CRZ: {
            return CRZ(controlQubit, targetQubit, gen_0_2pi_float());
        }
        default: {
            assert(false);
        }
    }
}

std::string Gate::get_name(GateType ty) {
    return random(0, 10, ty).name;
}

std::vector<unsigned char> Gate::serialize() const {
    auto name_len = name.length();
    int len =
        sizeof(name_len) + name.length() + 1 + sizeof(gateID) + sizeof(type) + sizeof(mat)
        + sizeof(targetQubit) + sizeof(controlQubit) + sizeof(controlQubit2);
    std::vector<unsigned char> ret; ret.resize(len);
    unsigned char* arr = ret.data();
    int cur = 0;
    SERIALIZE_STEP(gateID);
    SERIALIZE_STEP(type);
    memcpy(arr + cur, mat, sizeof(mat)); cur += sizeof(Complex) * 4;
    SERIALIZE_STEP(name_len);
    strcpy(reinterpret_cast<char*>(arr) + cur, name.c_str()); cur += name_len + 1;
    SERIALIZE_STEP(targetQubit);
    SERIALIZE_STEP(controlQubit);
    SERIALIZE_STEP(controlQubit2);
    assert(cur == len);
    return ret;
}

Gate Gate::deserialize(const unsigned char* arr, int& cur) {
    Gate g;
    DESERIALIZE_STEP(g.gateID);
    DESERIALIZE_STEP(g.type);
    memcpy(g.mat, arr + cur, sizeof(g.mat)); cur += sizeof(Complex) * 4;
    decltype(g.name.length()) name_len; DESERIALIZE_STEP(name_len);
    g.name = std::string(reinterpret_cast<const char*>(arr) + cur, name_len); cur += name_len + 1;
    DESERIALIZE_STEP(g.targetQubit);
    DESERIALIZE_STEP(g.controlQubit);
    DESERIALIZE_STEP(g.controlQubit2);
    return g;
}