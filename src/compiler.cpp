#include "compiler.h"

#include <cstring>
#include <algorithm>
#include <assert.h>
#include "dbg.h"
#include "logger.h"
#include "evaluator.h"

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

std::vector<std::pair<std::vector<Gate>, qindex>> Compiler::moveToNext(LocalGroup& lg) {
    std::vector<std::pair<std::vector<Gate>, qindex>> result;
    result.push_back(make_pair(std::vector<Gate>(), 0));
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
        result.push_back(make_pair(toRemoveGates, toRemove.fullGroups[0].relatedQubits));
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
    State state(numQubits);
    int numLocalQubits = numQubits - MyGlobalVars::bit;
    for (size_t id = 0; id < localGroup.fullGroups.size(); id++) {
        auto& gg = localGroup.fullGroups[id];

        std::vector<int> newGlobals;
        std::vector<int> newLocals;
        for (int i = 0; i < numQubits; i++) {
            if (! (gg.relatedQubits >> i & 1)) {
                newGlobals.push_back(i);
            }
        }
        assert(newGlobals.size() == MyGlobalVars::bit);
        
        auto globalPos = [this, numLocalQubits](const std::vector<int>& layout, int x) {
            auto pos = std::find(layout.data() + numLocalQubits, layout.data() + numQubits, x);
            return std::make_tuple(pos != layout.data() + numQubits, pos - layout.data() - numLocalQubits);
        };

        int overlapGlobals = 0;
        int overlapCnt = 0;
        // put overlapped global qubit into the previous position
        for (size_t i = 0; i < newGlobals.size(); i++) {
            bool isGlobal;
            int p;
            std::tie(isGlobal, p) = globalPos(state.layout, newGlobals[i]);
            if (isGlobal) {
                std::swap(newGlobals[p], newGlobals[i]);
                overlapGlobals |= qindex(1) << p;
                overlapCnt ++;
            }
        }
#ifdef SHOW_SCHEDULE
        printf("globals: "); for (auto x: newGlobals) printf("%d ", x); printf("\n");
#endif

        LocalGroup lg;
        lg.relatedQubits = gg.relatedQubits;
        if (id == 0) {
            state = lg.initFirstGroupState(state, numQubits, newGlobals);
        } else {
            state = lg.initState(state, numQubits, newGlobals, overlapGlobals, moveBack[id].second);
        }

        AdvanceCompiler overlapCompiler(numQubits, gg.relatedQubits, moveBack[id].first);
        lg.overlapGroups = overlapCompiler.run(state, true, false, LOCAL_QUBIT_SIZE, BLAS_MAT_LIMIT, numLocalQubits - MyGlobalVars::bit).fullGroups;
        AdvanceCompiler fullCompiler(numQubits, gg.relatedQubits, gg.gates);
        switch (BACKEND) {
            case 1: {
                lg.fullGroups = fullCompiler.run(state, true, false, LOCAL_QUBIT_SIZE, BLAS_MAT_LIMIT, numLocalQubits).fullGroups;
                break;
            }
            case 3: {
                lg.fullGroups = fullCompiler.run(state, false, true, LOCAL_QUBIT_SIZE, BLAS_MAT_LIMIT, numLocalQubits).fullGroups;
                break;
            }
            case 4: {
                lg.fullGroups = fullCompiler.run(state, true, true, LOCAL_QUBIT_SIZE, BLAS_MAT_LIMIT, numLocalQubits).fullGroups;
                break;
            }
            default: {
                lg.fullGroups = fullCompiler.run(state, true, false, LOCAL_QUBIT_SIZE, BLAS_MAT_LIMIT, numLocalQubits).fullGroups;
                break;
            }
        }
        schedule.localGroups.push_back(std::move(lg));
    }
    schedule.finalState = state;
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

LocalGroup AdvanceCompiler::run(State& state, bool usePerGate, bool useBLAS, int perGateSize, int blasSize, int cuttSize) {
    assert(usePerGate || useBLAS);
    LocalGroup lg;
    lg.relatedQubits = 0;
    int cnt = 0;
    while (remainGates.size() > 0) {
        qindex related[numQubits];
        bool full[numQubits];
        auto fillRelated = [this](qindex related[], const std::vector<int>& layout) {
            for (int i = 0; i < numQubits; i++) {
                related[i] = 0;
                for (int j = 0; j < COALESCE_GLOBAL; j++)
                    related[i] |= ((qindex) 1) << layout[j];
            }
        };
        GateGroup gg;
        if (usePerGate && useBLAS) {
            // get the gate group for pergate backend
            memset(full, 0, sizeof(full));
            fillRelated(related, state.layout);
            GateGroup pg = getGroup(full, related, true, perGateSize, -1ll);
            // get the gate group for blas backend
            memset(full, 0, sizeof(full));
            memset(related, 0, sizeof(related));
            GateGroup blas = getGroup(full, related, false, blasSize, localQubits);

            if (pg.gates.empty()) {
                gg = std::move(blas);
                gg.backend = Backend::BLAS;
            } else if (blas.gates.empty()) {
                gg = std::move(pg);
                gg.backend = Backend::PerGate;
            } else {
                // TODO: select the backend in a cleverer way
                /*if (rand() & 1) {
                    gg = std::move(pg);
                    gg.backend = Backend::PerGate;
                } else {
                    gg = std::move(blas);
                    gg.backend = Backend::BLAS;
                }*/
                /*
                int pg_sz = pg.gates.size();
                int blas_sz = blas.gates.size();
                if(pg_sz * 0.8 > blas_sz) {
                    gg = std::move(pg);
                    gg.backend = Backend::PerGate;
                } else {
                    gg = std::move(blas);
                    gg.backend = Backend::BLAS;
                }*/
                if(Evaluator::getInstance() -> PerGateOrBLAS(&pg, &blas, numQubits, blasSize)) {
                    gg = std::move(pg);
                    gg.backend = Backend::PerGate;
                } else {
                    gg = std::move(blas);
                    gg.backend = Backend::BLAS;    
                }
                //Logger::add("perf pergate : %f,", Evaluator::getInstance() -> perfPerGate(&pg));
            }
            state = gg.initState(state, cuttSize);
        } else if (usePerGate && !useBLAS) {
            fillRelated(related, state.layout);
            memset(full, 0, sizeof(full));
            gg = getGroup(full, related, true, perGateSize, -1ll);
            gg.backend = Backend::PerGate;

            Logger::add("perf pergate : %f,", Evaluator::getInstance() -> perfPerGate(numQubits, &gg));

            state = gg.initState(state, cuttSize);
        } else if (!usePerGate && useBLAS) {
            memset(related, 0, sizeof(related));
            memset(full, 0, sizeof(full));
            gg = getGroup(full, related, false, blasSize, localQubits);
            gg.backend = Backend::BLAS;

            Logger::add("perf BLAS : %f,", Evaluator::getInstance() -> perfBLAS(numQubits, blasSize));

            state = gg.initState(state, cuttSize);
        } else {
            UNREACHABLE();
        }
        lg.relatedQubits |= gg.relatedQubits;
        removeGates(remainGates, gg.gates);
        lg.fullGroups.push_back(std::move(gg));
        cnt ++;
        assert(cnt < 1000);
    }
    //Logger::add("local group cnt : %d", cnt);
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
