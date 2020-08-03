#include "compiler.h"
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


Compiler::Compiler(int numQubits, int localSize, std::vector<Gate> inputGates): numQubits(numQubits), localSize(localSize), remainGates(inputGates) { }

Schedule Compiler::run() {
    schedule.gateGroups.clear();
    int cnt = 0;
    while (true) {
        GateGroup gg = getGroup();
        schedule.gateGroups.push_back(gg);
        removeFromSchedule(gg);
        if (remainGates.size() == 0)
            break;
        cnt ++;
        assert(cnt < 100);
    }
#ifdef SHOW_SCHEDULE
    schedule.dump(numQubits);
#endif
    return schedule;
}

GateGroup Compiler::getGroup() {
    GateGroup cur[numQubits];
    bool full[numQubits];
    memset(full, 0, sizeof(full));

    auto canMerge2 = [&](const GateGroup& a, const GateGroup & b) {
        return bitCount(a.relatedQubits | b.relatedQubits) <= localSize;
    };
    auto canMerge3 = [&](const GateGroup& a, const GateGroup &b, const GateGroup &c) {
        return bitCount(a.relatedQubits | b.relatedQubits | c.relatedQubits) <= localSize;
    };

    for (auto& gate: remainGates) {
        if (gate.isC2Gate()) {
            if (!full[gate.controlQubit2] && !full[gate.controlQubit] && !full[gate.targetQubit] && canMerge3(cur[gate.controlQubit2], cur[gate.controlQubit], cur[gate.targetQubit])) {
                GateGroup newGroup = GateGroup::merge(cur[gate.controlQubit], cur[gate.controlQubit2]);
                newGroup = GateGroup::merge(newGroup, cur[gate.targetQubit]);
                newGroup.addGate(gate);
                cur[gate.controlQubit2] = cur[gate.controlQubit] = cur[gate.targetQubit] = newGroup;
            } else {
                full[gate.controlQubit2] = full[gate.controlQubit] = full[gate.targetQubit] = 1;
            }
        } else if (gate.isControlGate()) {
            if (!full[gate.controlQubit] && !full[gate.targetQubit] && canMerge2(cur[gate.controlQubit], cur[gate.targetQubit])) {
                GateGroup newGroup = GateGroup::merge(cur[gate.controlQubit], cur[gate.targetQubit]);
                newGroup.addGate(gate);
                cur[gate.controlQubit] = cur[gate.targetQubit] = newGroup;
            } else {
                full[gate.controlQubit] = full[gate.targetQubit] = 1;
            }
        } else {
            if (!full[gate.targetQubit])
                cur[gate.targetQubit].addGate(gate);
        }
    }

    GateGroup selected;
    selected.relatedQubits = 0;
    while (true) {
        size_t mx = selected.gates.size();
        int qid = -1;
        for (int i = 0; i < numQubits; i++) {
            if (canMerge2(selected, cur[i]) && cur[i].gates.size() > 0) {
                GateGroup newGroup = GateGroup::merge(cur[i], selected);
                if (newGroup.gates.size() > mx) {
                    mx = newGroup.gates.size();
                    qid = i;
                }
            }
        }
        if (mx == selected.gates.size())
            break;
        selected = GateGroup::merge(cur[qid], selected);
    }

    std::vector<int> usedID = selected.toID();
    std::sort(usedID.begin(), usedID.end());
    bool blocked[numQubits];
    memset(blocked, 0, sizeof(blocked));
    for (auto& g: remainGates) {
        if (std::binary_search(usedID.begin(), usedID.end(), g.gateID)) continue;
        if (g.isDiagonal()) {
            // TODO: Diagonal C2 Gate
            if (g.isControlGate()) {
                if (!blocked[g.controlQubit] && !blocked[g.targetQubit]) {
                    selected.gates.push_back(g);
                } else {
                    blocked[g.controlQubit] = blocked[g.targetQubit] = 1;
                }
            } else {
                if (!blocked[g.targetQubit]) {
                    selected.gates.push_back(g);
                }
            }
        } else {
            if (g.isControlGate()) {
                blocked[g.controlQubit] = 1;
            }
            blocked[g.targetQubit] = 1;
        }
    }
    return selected;
}

void Compiler::removeFromSchedule(GateGroup& gg) {
    std::vector<int> usedID = gg.toID();
    std::sort(usedID.begin(), usedID.end());
    auto temp = remainGates;
    remainGates.clear();
    for (auto& g: temp) {
        if (!std::binary_search(usedID.begin(), usedID.end(), g.gateID)) {
            remainGates.push_back(g);
        }
    }
}