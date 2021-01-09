#include "compiler.h"

#include <cstring>
#include <algorithm>
#include <assert.h>

Compiler::Compiler(int numQubits, int localSize, int shareSize, std::vector<Gate> inputGates, bool enableGlobal):
    numQubits(numQubits), localSize(localSize), shareSize(shareSize), enableGlobal(enableGlobal), gates(inputGates) {}


void Compiler::fillLocals(LocalGroup& lg) {
    int numLocalQubits = numQubits - MyGlobalVars::bit;
    for (auto& gg: lg.fullGroups) {
        qindex related = gg.relatedQubits;
        int numRelated = bitCount(related);
        assert(numRelated <= numLocalQubits);
        if (numRelated < numLocalQubits) {
            for (int i = 0;; i++)
                if (!(related >> i & 1)) {
                    related |= ((qindex) 1) << i;
                    numRelated ++;
                    if (numRelated == numLocalQubits)
                        break;
                }
        }
        gg.relatedQubits = related;
    }
}

std::vector<std::vector<Gate>> Compiler::moveToNext(LocalGroup& lg) {
    std::vector<std::vector<Gate>> result;
    result.push_back(std::vector<Gate>());
    for (size_t i = 1; i < lg.fullGroups.size(); i++) {
        std::vector<Gate> gates = lg.fullGroups[i-1].gates;
        std::reverse(gates.begin(), gates.end());
        assert(lg.fullGroups[i-1].relatedQubits != 0);
        OneLayerCompiler backCompiler(numQubits, numQubits - 2 * MyGlobalVars::bit, gates,
                                        enableGlobal, lg.fullGroups[i-1].relatedQubits, lg.fullGroups[i].relatedQubits);
        LocalGroup toRemove = backCompiler.run();
        assert(toRemove.fullGroups.size() == 1);
        std::vector<Gate> toRemoveGates = toRemove.fullGroups[0].gates;
        std::reverse(toRemoveGates.begin(), toRemoveGates.end());
        
        removeGates(lg.fullGroups[i-1].gates, toRemoveGates);
        result.push_back(toRemoveGates);
        lg.fullGroups[i].relatedQubits |= toRemove.relatedQubits;
    }
    return std::move(result);
}

Schedule Compiler::run() {
    OneLayerCompiler localCompiler(numQubits, localSize, gates, enableGlobal);
    LocalGroup localGroup = localCompiler.run();
    auto moveBack = moveToNext(localGroup);
    fillLocals(localGroup);
    Schedule schedule;
    for (size_t i = 0; i < localGroup.fullGroups.size(); i++) {
        auto& gg = localGroup.fullGroups[i];
        OneLayerCompiler shareCompiler(numQubits, shareSize, gg.gates, enableGlobal);
        schedule.localGroups.push_back(shareCompiler.run());
        schedule.localGroups.back().relatedQubits = gg.relatedQubits;
        if (!moveBack[i].empty()) {
            OneLayerCompiler shareCompiler2(numQubits, shareSize, moveBack[i], enableGlobal);
            schedule.localGroups.back().overlapGroups = shareCompiler2.run().fullGroups;
        }
    }
    return schedule;
}

OneLayerCompiler::OneLayerCompiler(int numQubits, int localSize, std::vector<Gate> inputGates, bool enableGlobal, qindex whiteList, qindex required):
    numQubits(numQubits), localSize(localSize), enableGlobal(enableGlobal), whiteList(whiteList), required(required), remainGates(inputGates) {}

LocalGroup OneLayerCompiler::run() {
    LocalGroup lg;
    lg.relatedQubits = 0;
    int cnt = 0;
    while (true) {
        GateGroup gg = getGroup();
        lg.fullGroups.push_back(gg.copyGates());
        lg.relatedQubits |= gg.relatedQubits;
        removeGates(remainGates, gg.gates);
        if (remainGates.size() == 0)
            break;
        if (whiteList != 0)
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
    if (whiteList) {
        for (int i = 0; i < numQubits; i++)
            if (!(whiteList >> i & 1))
                full[i] = 1;
        for (int i = 0; i < numQubits; i++)
            cur[i].relatedQubits = required;
    }

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

    std::vector<int> usedID;
    for (auto& g: selected.gates) usedID.push_back(g.gateID);
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
