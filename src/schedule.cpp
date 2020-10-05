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
    for (auto& gg: gateGroups) {
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
                    printf(gg.contains(i) ? "+" : "|");
                    for (int j = 1; j < L; j++) printf(" ");
                }
            }
            printf("\n");
        }
        printf("\n");
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
