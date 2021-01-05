#include "compiler.h"

#include <cstring>
#include <algorithm>
#include <assert.h>

Compiler::Compiler(int numQubits, int localSize, int shareSize, std::vector<Gate> inputGates, bool enableGlobal):
    numQubits(numQubits), localSize(localSize), shareSize(shareSize), enableGlobal(enableGlobal), gates(inputGates) {}

Schedule Compiler::run() {
    OneLayerCompiler localCompiler(numQubits, localSize, gates, enableGlobal);
    LocalGroup localGroup = localCompiler.run();
    Schedule schedule;
    for (auto& gg: localGroup.fullGroups) {
        OneLayerCompiler shareCompiler(numQubits, shareSize, gg.gates, enableGlobal);
        schedule.localGroups.push_back(shareCompiler.run());
    }
    return schedule;
}

OneLayerCompiler::OneLayerCompiler(int numQubits, int localSize, std::vector<Gate> inputGates, bool enableGlobal):
    numQubits(numQubits), localSize(localSize), enableGlobal(enableGlobal), remainGates(inputGates) {}

LocalGroup OneLayerCompiler::run() {
    LocalGroup lg;
    lg.relatedQubits = 0;
    int cnt = 0;
    while (true) {
        GateGroup gg = getGroup();
        lg.fullGroups.push_back(gg.copyGates());
        lg.relatedQubits |= gg.relatedQubits;
        remove(gg);
        if (remainGates.size() == 0)
            break;
        cnt ++;
        assert(cnt < 1000);
    }
    return std::move(lg);
}

GateGroup OneLayerCompiler::getGroup() {
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
                newGroup = GateGroup::merge(std::move(newGroup), cur[gate.targetQubit]);
                newGroup.addGate(gate, enableGlobal);
                cur[gate.controlQubit2] = newGroup.copyGates();
                cur[gate.controlQubit] = newGroup.copyGates();
                cur[gate.targetQubit] = newGroup.copyGates();
            } else {
                full[gate.controlQubit2] = full[gate.controlQubit] = full[gate.targetQubit] = 1;
            }
        } else if (gate.isControlGate()) {
            if (!full[gate.controlQubit] && !full[gate.targetQubit] && canMerge2(cur[gate.controlQubit], cur[gate.targetQubit])) {
                GateGroup newGroup = GateGroup::merge(cur[gate.controlQubit], cur[gate.targetQubit]);
                newGroup.addGate(gate, enableGlobal);
                cur[gate.controlQubit] = newGroup.copyGates();
                cur[gate.targetQubit] = newGroup.copyGates();
            } else {
                full[gate.controlQubit] = full[gate.targetQubit] = 1;
            }
        } else {
            if (!full[gate.targetQubit])
                cur[gate.targetQubit].addGate(gate, enableGlobal);
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
        if (g.isDiagonal() && enableGlobal) {
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
    return std::move(selected);
}

void OneLayerCompiler::remove(GateGroup& gg) {
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