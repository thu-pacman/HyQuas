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
        if (gate.isControlGate()) {
            relatedQubits |= qindex(1) << gate.controlQubit;
        }
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

Compiler::Compiler(int numQubits, int localSize, std::vector<Gate> inputGates): numQubits(numQubits), localSize(localSize), remainGates(inputGates) { }

Schedule Compiler::run() {
    schedule.gateGroups.clear();
    int cnt = 0;
    while (true) {
        GateGroup gg = getGroup();
        moveToSchedule(gg);
        if (remainGates.size() == 0)
            break;
        cnt ++;
    }
    schedule.dump(numQubits);
    return schedule;
}

GateGroup Compiler::getGroup() {
    GateGroup cur[numQubits];
    bool full[numQubits];
    memset(full, 0, sizeof(full));
    GateGroup ret;
    auto canMerge = [&](const GateGroup& a, const GateGroup & b) {
        return bitCount(a.relatedQubits | b.relatedQubits) <= localSize;
    };

    for (auto& gate: remainGates) {
        if (gate.isControlGate()) {
            if (!full[gate.controlQubit] && !full[gate.targetQubit] && canMerge(cur[gate.controlQubit], cur[gate.targetQubit])) {
                GateGroup newGroup = GateGroup::merge(cur[gate.controlQubit], cur[gate.targetQubit]);
                newGroup.addGate(gate);
                cur[gate.controlQubit] = cur[gate.targetQubit] = newGroup;
            } else {
                if (!full[gate.controlQubit]) {
                    full[gate.controlQubit] = 1;
                    if (canMerge(ret, cur[gate.controlQubit]))
                        ret = GateGroup::merge(ret, cur[gate.controlQubit]);
                }
                if (!full[gate.targetQubit]) {
                    full[gate.targetQubit] = 1;
                    if (canMerge(ret, cur[gate.targetQubit]))
                        ret = GateGroup::merge(ret, cur[gate.targetQubit]);
                }
            }
        } else {
            if (!full[gate.targetQubit])
                cur[gate.targetQubit].addGate(gate);
        }
    }

    for (int i = 0; i < numQubits; i++) {
        if (!full[i] && canMerge(ret, cur[i]))
            ret = GateGroup::merge(ret, cur[i]);
    }

    // for (int i = 0; i < numQubits; i++)
    //     if (!ret.contains(i)) {
    //         for (auto& g: cur[i].gates) {
    //             if (g.targetQubit != i) continue;
    //             if (g.isDiagonal() && g.targetQubit == i) {
    //                 ret.gates.push_back(g);
    //             } else {
    //                 break;
    //             }
    //         }
    //     }
    return ret;
}

void Compiler::moveToSchedule(GateGroup& gg) {
    std::vector<int> usedID;
    for (auto& g: gg.gates) {
        usedID.push_back(g.gateID);
    }
    std::sort(usedID.begin(), usedID.end());
    std::vector<Gate> temp = remainGates;
    remainGates.clear();
    bool blocked[numQubits];
    memset(blocked, 0, sizeof(blocked));
    for (auto& g: temp) {
        if (std::binary_search(usedID.begin(), usedID.end(), g.gateID)) continue;
        if (g.isDiagonal()) {
            if (g.isControlGate()) {
                if (blocked[g.controlQubit] || blocked[g.targetQubit]) {
                    blocked[g.controlQubit] = blocked[g.targetQubit] = 1;
                    remainGates.push_back(g);
                } else {
                    gg.gates.push_back(g);
                }
            } else {
                if (blocked[g.targetQubit]) {
                    remainGates.push_back(g);
                } else {
                    gg.gates.push_back(g);
                }
            }
        } else {
            if (g.isControlGate()) {
                blocked[g.controlQubit] = 1;
            }
            blocked[g.targetQubit] = 1;
            remainGates.push_back(g);
        }
    }
    schedule.gateGroups.push_back(gg);
}