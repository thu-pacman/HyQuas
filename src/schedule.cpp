#include "schedule.h"
#include <cstring>
#include <algorithm>
#include <assert.h>

GateGroup GateGroup::merge(const GateGroup& a, const GateGroup& b) {
    GateGroup ret;
    ret.relatedQubits = a.relatedQubits | b.relatedQubits;
    ret.gates = a.gates;
    std::vector<int> usedID;
    for (auto& g: a.gates) {
        usedID.push_back(g.gateID);
    }
    std::sort(usedID.begin(), usedID.end());
    for (auto& g: b.gates) {
        if (!std::binary_search(usedID.begin(), usedID.end(), g.gateID)) {
            ret.gates.push_back(g);
        }
    }
    return ret;
}

void GateGroup::addGate(const Gate& gate) {
    gates.push_back(gate);
    if (!gate.isDiagonal()) {
        relatedQubits |= qindex(1) << gate.targetQubit;
    }
}

void Schedule::dump(int numQubits) {
    int L = 3;
    for (auto& lg: localGroups) {
        for (auto& gg: lg.gateGroups) {
            for (const Gate& gate: gg.gates) {
                for (int i = 0; i < numQubits; i++) {
                    if (i == gate.controlQubit) {
                        printf(".");
                        for (int j = 1; j < L; j++) printf(" ");
                    } else if (i == gate.targetQubit) {
                        printf("%s", gate.name.c_str());
                        for (int j = gate.name.length(); j < L; j++)
                            printf(" ");
                    } else {
                        if (gg.contains(i)) putchar('+');
                        else if (lg.contains(i)) putchar('/');
                        else putchar('|');
                        for (int j = 1; j < L; j++) printf(" ");
                    }
                }
                printf("\n");
            }
            printf("\n");
        }
        for (int i = 0; i < numQubits * L; i++) {
            printf("-");
        }
        printf("\n\n");
    }
    fflush(stdout);
}

std::vector<int> GateGroup::toID() const {
    std::vector<int> ret;
    for (auto& gate: gates) {
        ret.push_back(gate.gateID);
    }
    return ret;
}

std::vector<unsigned char> GateGroup::serialize() const {
    auto num_gates = gates.size();
    std::vector<unsigned char> result;
    result.resize(sizeof(relatedQubits) + sizeof(num_gates));
    auto arr = result.data();
    int cur = 0;
    SERIALIZE_STEP(relatedQubits);
    SERIALIZE_STEP(num_gates);
    for (auto& gate: gates) {
        auto g = gate.serialize();
        result.insert(result.end(), g.begin(), g.end());
    }
    return result;
}

GateGroup GateGroup::deserialize(const unsigned char* arr, int& cur) {
    GateGroup gg;
    DESERIALIZE_STEP(gg.relatedQubits);
    decltype(gg.gates.size()) num_gates;
    DESERIALIZE_STEP(num_gates);
    for (int i = 0; i < num_gates; i++) {
        gg.gates.push_back(Gate::deserialize(arr, cur));
    }
    return gg;
}

std::vector<unsigned char> LocalGroup::serialize() const {
    auto num_gg = gateGroups.size();
    std::vector<unsigned char> result;
    result.resize(sizeof(num_gg));
    auto arr = result.data();
    int cur = 0;
    SERIALIZE_STEP(num_gg);
    for (auto& gateGroup: gateGroups) {
        auto gg = gateGroup.serialize();
        result.insert(result.end(), gg.begin(), gg.end());
    }
    return result;
}


LocalGroup LocalGroup::deserialize(const unsigned char* arr, int& cur) {
    LocalGroup s;
    decltype(s.gateGroups.size()) num_gg;
    DESERIALIZE_STEP(num_gg);
    for (int i = 0; i < num_gg; i++) {
        s.gateGroups.push_back(GateGroup::deserialize(arr, cur));
    }
    return s;
}

std::vector<unsigned char> Schedule::serialize() const {
    auto num_lg = localGroups.size();
    std::vector<unsigned char> result;
    result.resize(sizeof(num_lg));
    auto arr = result.data();
    int cur = 0;
    SERIALIZE_STEP(num_lg);
    for (auto& localGroup: localGroups) {
        auto lg = localGroup.serialize();
        result.insert(result.end(), lg.begin(), lg.end());
    }
    return result;
}


Schedule Schedule::deserialize(const unsigned char* arr, int& cur) {
    Schedule s;
    decltype(s.localGroups.size()) num_lg;
    DESERIALIZE_STEP(num_lg);
    for (int i = 0; i < num_lg; i++) {
        s.localGroups.push_back(LocalGroup::deserialize(arr, cur));
    }
    return s;
}