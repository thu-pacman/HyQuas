#include "compiler.h"

#include <cstring>
#include <algorithm>
#include <assert.h>
#include "dbg.h"

Compiler::Compiler(int numQubits, std::vector<Gate> inputGates):
    numQubits(numQubits), localSize(numQubits - MyGlobalVars::bit), gates(inputGates) {}


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
        SimpleCompiler backCompiler(numQubits, numQubits - 2 * MyGlobalVars::bit, numQubits - 2 * MyGlobalVars::bit, gates,
                                        true, lg.fullGroups[i-1].relatedQubits, lg.fullGroups[i].relatedQubits);
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
    SimpleCompiler localCompiler(numQubits, localSize, localSize, gates, true);
    LocalGroup localGroup = localCompiler.run();
    auto moveBack = moveToNext(localGroup);
    fillLocals(localGroup);
    Schedule schedule;
    State state;
    for (size_t i = 0; i < localGroup.fullGroups.size(); i++) {
        auto& gg = localGroup.fullGroups[i];
        LocalGroup lg;
        AdvanceCompiler overlapCompiler(numQubits, gg.relatedQubits, moveBack[i]);
        lg.overlapGroups = overlapCompiler.run(state, true, false, LOCAL_QUBIT_SIZE, BLAS_MAT_LIMIT).fullGroups;
        AdvanceCompiler fullCompiler(numQubits, gg.relatedQubits, gg.gates);
        switch (BACKEND) {
            case 1: {
                lg.fullGroups = fullCompiler.run(state, true, false, LOCAL_QUBIT_SIZE, BLAS_MAT_LIMIT).fullGroups;
                break;
            }
            case 3: {
                lg.fullGroups = fullCompiler.run(state, false, true, LOCAL_QUBIT_SIZE, BLAS_MAT_LIMIT).fullGroups;
                break;
            }
            case 4: {
                lg.fullGroups = fullCompiler.run(state, true, true, LOCAL_QUBIT_SIZE, BLAS_MAT_LIMIT).fullGroups;
                break;
            }
            default: {
                lg.fullGroups = fullCompiler.run(state, true, false, LOCAL_QUBIT_SIZE, BLAS_MAT_LIMIT).fullGroups;
                break;
            }
        }
        lg.relatedQubits = gg.relatedQubits;
        schedule.localGroups.push_back(std::move(lg));
    }
    return schedule;
}

OneLayerCompiler::OneLayerCompiler(int numQubits, const std::vector<Gate> &inputGates):
    numQubits(numQubits), remainGates(inputGates) {}

SimpleCompiler::SimpleCompiler(int numQubits, int localSize, qindex localQubits, const std::vector<Gate>& inputGates, bool enableGlobal, qindex whiteList, qindex required):
    OneLayerCompiler(numQubits, inputGates), localSize(localSize), localQubits(localQubits), enableGlobal(enableGlobal), whiteList(whiteList), required(required) {}

AdvanceCompiler::AdvanceCompiler(int numQubits, qindex localQubits, std::vector<Gate> inputGates):
    OneLayerCompiler(numQubits, inputGates), localQubits(localQubits) {}

LocalGroup SimpleCompiler::run() {
    LocalGroup lg;
    lg.relatedQubits = 0;
    int cnt = 0;
    while (remainGates.size() > 0) {
        qindex related[numQubits];
        bool full[numQubits];
        memset(full, 0, sizeof(full));
        memset(related, 0, sizeof(related));
        if (whiteList) {
            for (int i = 0; i < numQubits; i++)
                if (!(whiteList >> i & 1))
                    full[i] = 1;
            for (int i = 0; i < numQubits; i++)
                related[i] = required;
        }

        GateGroup gg = getGroup(full, related, enableGlobal, localSize, localQubits);
        lg.fullGroups.push_back(gg.copyGates());
        lg.relatedQubits |= gg.relatedQubits;
        removeGates(remainGates, gg.gates);
        if (whiteList != 0)
            break;
        cnt ++;
        assert(cnt < 1000);
    }
    return std::move(lg);
}

LocalGroup AdvanceCompiler::run(State& state, bool usePerGate, bool useBLAS, int perGateSize, int blasSize) {
    assert(usePerGate || useBLAS);
    LocalGroup lg;
    lg.relatedQubits = 0;
    int cnt = 0;
    while (remainGates.size() > 0) {
        qindex related[numQubits];
        bool full[numQubits];
        memset(full, 0, sizeof(full));
        memset(related, 0, sizeof(related));
        GateGroup gg;
        if (usePerGate) {
            gg = getGroup(full, related, true, perGateSize, -1ll);
            gg.backend = Backend::PerGate;
        } else {
            gg = getGroup(full, related, false, blasSize, localQubits);
            gg.backend = Backend::BLAS;
        }
        lg.fullGroups.push_back(gg.copyGates());
        lg.relatedQubits |= gg.relatedQubits;
        removeGates(remainGates, gg.gates);
        cnt ++;
        assert(cnt < 1000);
    }
    return std::move(lg);
}

GateGroup OneLayerCompiler::getGroup(bool full[], qindex related[], bool enableGlobal, int localSize, qindex localQubits) {
    GateGroup cur[numQubits];
    for (int i = 0; i < numQubits; i++)
        cur[i].relatedQubits = related[i];
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
                newGroup.addGate(gate, localQubits, enableGlobal);
                cur[gate.controlQubit2] = newGroup.copyGates();
                cur[gate.controlQubit] = newGroup.copyGates();
                cur[gate.targetQubit] = newGroup.copyGates();
            } else {
                full[gate.controlQubit2] = full[gate.controlQubit] = full[gate.targetQubit] = 1;
            }
        } else if (gate.isControlGate()) {
            if (!full[gate.controlQubit] && !full[gate.targetQubit] && canMerge2(cur[gate.controlQubit], cur[gate.targetQubit])) {
                GateGroup newGroup = GateGroup::merge(cur[gate.controlQubit], cur[gate.targetQubit]);
                newGroup.addGate(gate, localQubits, enableGlobal);
                cur[gate.controlQubit] = newGroup.copyGates();
                cur[gate.targetQubit] = newGroup.copyGates();
            } else {
                full[gate.controlQubit] = full[gate.targetQubit] = 1;
            }
        } else {
            if (!full[gate.targetQubit])
                cur[gate.targetQubit].addGate(gate, localQubits, enableGlobal);
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
