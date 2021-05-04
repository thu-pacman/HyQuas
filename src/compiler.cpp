#include "compiler.h"

#include <cstring>
#include <algorithm>
#include <assert.h>
#include <set>
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
#ifndef ENABLE_OVERLAP
    for (size_t i = 0; i < lg.fullGroups.size(); i++) {
        result.push_back(make_pair(std::vector<Gate>(), 0));
    }
    return std::move(result);
#endif
    result.push_back(make_pair(std::vector<Gate>(), 0));
    for (size_t i = 1; i < lg.fullGroups.size(); i++) {
        std::vector<Gate> gates = lg.fullGroups[i-1].gates;
        std::reverse(gates.begin(), gates.end());
        assert(lg.fullGroups[i-1].relatedQubits != 0);
#if BACKEND == 3
        SimpleCompiler backCompiler(numQubits, numQubits - 2 * MyGlobalVars::bit, numQubits - 2 * MyGlobalVars::bit, gates,
                                        false, lg.fullGroups[i-1].relatedQubits, lg.fullGroups[i].relatedQubits);
#else
        SimpleCompiler backCompiler(numQubits, numQubits - 2 * MyGlobalVars::bit, numQubits - 2 * MyGlobalVars::bit, gates,
                                        true, lg.fullGroups[i-1].relatedQubits, lg.fullGroups[i].relatedQubits);
#endif
        LocalGroup toRemove = backCompiler.run();
        if (toRemove.fullGroups.size() == 0) {
            result.push_back(make_pair(std::vector<Gate>(), 0));
            continue;
        }
        assert(toRemove.fullGroups.size() == 1);
        std::vector<Gate> toRemoveGates = toRemove.fullGroups[0].gates;
        std::reverse(toRemoveGates.begin(), toRemoveGates.end());
        
        removeGates(lg.fullGroups[i-1].gates, toRemoveGates); // TODO: can we optimize this remove?
        result.push_back(make_pair(toRemoveGates, toRemove.fullGroups[0].relatedQubits));
        lg.fullGroups[i].relatedQubits |= toRemove.relatedQubits;
    }
    return std::move(result);
}

Schedule Compiler::run() {
    SimpleCompiler localCompiler(numQubits, localSize, (qindex) -1, gates, true, 0, (1 << INPLACE) - 1);
    // ChunkCompiler localCompiler(numQubits, localSize, 21, gates);
    LocalGroup localGroup = localCompiler.run();
    auto moveBack = moveToNext(localGroup);
    fillLocals(localGroup);
    Schedule schedule;
    State state(numQubits);
    int numLocalQubits = numQubits - MyGlobalVars::bit;
    for (size_t id = 0; id < localGroup.fullGroups.size(); id++) {
        auto& gg = localGroup.fullGroups[id];

        std::vector<int> newGlobals;
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

        qindex overlapGlobals = 0;
        int overlapCnt = 0;
        // put overlapped global qubit into the previous position
        bool modified = true;
        while (modified) {
            modified = false;
            overlapGlobals = 0;
            overlapCnt = 0;
            for (size_t i = 0; i < newGlobals.size(); i++) {
                bool isGlobal;
                int p;
                std::tie(isGlobal, p) = globalPos(state.layout, newGlobals[i]);
                if (isGlobal) {
                    std::swap(newGlobals[p], newGlobals[i]);
                    overlapGlobals |= qindex(1) << p;
                    overlapCnt ++;
                    if (p != int(i)) {
                        modified = true;
                    }
                }
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
            if (INPLACE) {
                state = lg.initStateInplace(state, numQubits, newGlobals, overlapGlobals);
            } else {
                state = lg.initState(state, numQubits, newGlobals, overlapGlobals, moveBack[id].second);
            }

        }

        qindex overlapLocals = gg.relatedQubits;
        qindex overlapBlasForbid = 0;
        if (id > 0) {
            overlapLocals &= localGroup.fullGroups[id - 1].relatedQubits;
            overlapBlasForbid = (~localGroup.fullGroups[id - 1].relatedQubits) & gg.relatedQubits;
            // printf("overlapBlasForbid %llx\n", overlapBlasForbid);
        }
        AdvanceCompiler overlapCompiler(numQubits, overlapLocals, overlapBlasForbid, moveBack[id].first);
        AdvanceCompiler fullCompiler(numQubits, gg.relatedQubits, 0, gg.gates);
        switch (BACKEND) {
            case 1: {
                lg.overlapGroups = overlapCompiler.run(state, true, false, LOCAL_QUBIT_SIZE, BLAS_MAT_LIMIT, numLocalQubits - MyGlobalVars::bit).fullGroups;
                lg.fullGroups = fullCompiler.run(state, true, false, LOCAL_QUBIT_SIZE, BLAS_MAT_LIMIT, numLocalQubits).fullGroups;
                break;
            }
            case 3: // no break
            case 5: {
                lg.overlapGroups = overlapCompiler.run(state, false, true, LOCAL_QUBIT_SIZE, BLAS_MAT_LIMIT, numLocalQubits - MyGlobalVars::bit).fullGroups;
                lg.fullGroups = fullCompiler.run(state, false, true, LOCAL_QUBIT_SIZE, BLAS_MAT_LIMIT, numLocalQubits).fullGroups;
                break;
            }
            case 2: // no break;
            case 4: {
                lg.overlapGroups = overlapCompiler.run(state, true, true, LOCAL_QUBIT_SIZE, BLAS_MAT_LIMIT, numLocalQubits - MyGlobalVars::bit).fullGroups;
                lg.fullGroups = fullCompiler.run(state, true, true, LOCAL_QUBIT_SIZE, BLAS_MAT_LIMIT, numLocalQubits).fullGroups;
                break;
            }
            default: {
                UNREACHABLE()
                break;
            }
        }
        schedule.localGroups.push_back(std::move(lg));
    }
    schedule.finalState = state;
    return schedule;
}

template<int MAX_GATES>
OneLayerCompiler<MAX_GATES>::OneLayerCompiler(int numQubits, const std::vector<Gate> &inputGates):
    numQubits(numQubits), remainGates(inputGates) {}

SimpleCompiler::SimpleCompiler(int numQubits, int localSize, qindex localQubits, const std::vector<Gate>& inputGates, bool enableGlobal, qindex whiteList, qindex required):
    OneLayerCompiler<2048>(numQubits, inputGates), localSize(localSize), localQubits(localQubits), enableGlobal(enableGlobal), whiteList(whiteList), required(required) {}

AdvanceCompiler::AdvanceCompiler(int numQubits, qindex localQubits, qindex blasForbid, std::vector<Gate> inputGates):
    OneLayerCompiler<512>(numQubits, inputGates), localQubits(localQubits), blasForbid(blasForbid) {}

LocalGroup SimpleCompiler::run() {
    LocalGroup lg;
    if (localSize == numQubits) {
        GateGroup gg;
        for (auto& g: remainGates)
            gg.addGate(g, localQubits, enableGlobal);
        lg.relatedQubits = gg.relatedQubits;
        lg.fullGroups.push_back(gg.copyGates());
        return std::move(lg);
    }
    lg.relatedQubits = 0;
    remain.clear();
    for (size_t i = 0; i < remainGates.size(); i++)
        remain.insert(i);
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
        }
        for (int i = 0; i < numQubits; i++)
            related[i] = required;

        std::vector<int> idx = getGroupOpt(full, related, enableGlobal, localSize, localQubits);
        GateGroup gg;
        for (auto& x: idx)
            gg.addGate(remainGates[x], localQubits, enableGlobal);
        lg.fullGroups.push_back(gg.copyGates()); // TODO: redundant copy?
        lg.relatedQubits |= gg.relatedQubits;
        removeGatesOpt(idx);
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
    remain.clear();
    for (size_t i = 0; i < remainGates.size(); i++)
        remain.insert(i);
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
        auto fillFull = [this](bool full[], qindex forbid) {
            memset(full, 0, sizeof(bool) * numQubits);
            for (int i = 0; i < numQubits; i++)
                if (forbid >> i & 1)
                    full[i] = true;
        };
        GateGroup gg;
        std::vector<int> ggIdx;
        Backend ggBackend;
        qindex cacheRelated = 0;
        if (usePerGate && useBLAS) {
            // get the gate group for pergate backend
            memset(full, 0, sizeof(full));
            fillRelated(related, state.layout);
            cacheRelated = related[0];
            ggIdx = getGroupOpt(full, related, true, perGateSize, -1ll);
            ggBackend = Backend::PerGate;
            double bestEff;
            if (ggIdx.size() == 0) {
                bestEff = 1e10;
            } else {
                std::vector<GateType> tys;
                for (auto& x: ggIdx) tys.push_back(remainGates[x].type);
                bestEff = Evaluator::getInstance() -> perfPerGate(numQubits - MyGlobalVars::bit, tys) / ggIdx.size();
                // printf("eff-pergate %f %d %f\n", Evaluator::getInstance() -> perfPerGate(numQubits - MyGlobalVars::bit, tys), (int) ggIdx.size(), bestEff);
            }

            for (int matSize = 4; matSize < 8; matSize ++) {
                fillFull(full, blasForbid);
                memset(related, 0, sizeof(related));
                std::vector<int> idx = getGroupOpt(full, related, false, matSize, localQubits | blasForbid);
                if (idx.size() == 0)
                    continue;
                double eff = Evaluator::getInstance() -> perfBLAS(numQubits - MyGlobalVars::bit, matSize) / idx.size();
                // printf("eff-blas(%d) %f %d %f\n", matSize, Evaluator::getInstance() -> perfBLAS(numQubits - MyGlobalVars::bit, matSize), (int) idx.size(), eff);
                if (eff < bestEff) {
                    ggIdx = idx;
                    ggBackend = Backend::BLAS;
                    bestEff = eff;
                }
            }    
            // printf("BACKEND %s\n", bestBackend == Backend::BLAS ? "blas" : "pergate");
        } else if (usePerGate && !useBLAS) {
            fillRelated(related, state.layout);
            memset(full, 0, sizeof(full));
            cacheRelated = related[0];
            ggIdx = getGroupOpt(full, related, true, perGateSize, -1ll);
            ggBackend = Backend::PerGate;
        } else if (!usePerGate && useBLAS) {
            memset(related, 0, sizeof(related));
            fillFull(full, blasForbid);
            ggIdx = getGroupOpt(full, related, false, blasSize, localQubits | blasForbid);
            GateGroup gg;
            ggBackend = Backend::BLAS;
        } else {
            UNREACHABLE();
        }
        if (ggBackend == Backend::PerGate) {
            for (auto& x: ggIdx)
                gg.addGate(remainGates[x], -1ll, true);
#ifdef LOG_EVALUATOR
            Logger::add("perf pergate : %f,", Evaluator::getInstance() -> perfPerGate(numQubits, &gg));
#endif
            gg.relatedQubits |= cacheRelated;
        } else {
            for (auto& x: ggIdx)
                gg.addGate(remainGates[x], localQubits, false);
#ifdef LOG_EVALUATOR
            Logger::add("perf BLAS : %f,", Evaluator::getInstance() -> perfBLAS(numQubits, blasSize));
#endif
        }
        gg.backend = ggBackend;
        state = gg.initState(state, cuttSize);
        removeGatesOpt(ggIdx);
        lg.relatedQubits |= gg.relatedQubits;
        lg.fullGroups.push_back(std::move(gg));
        cnt ++;
        assert(cnt < 1000);
    }
    //Logger::add("local group cnt : %d", cnt);
    return std::move(lg);
}

template<int MAX_GATES>
std::vector<int> OneLayerCompiler<MAX_GATES>::getGroupOpt(bool full[], qindex related[], bool enableGlobal, int localSize, qindex localQubits) {
    std::bitset<MAX_GATES> cur[numQubits], new_cur, selected;
    int gateIDs[MAX_GATES], gate_num;
    
    {
        int id = 0;
        for (auto gateID: remain) {
            gateIDs[id++] = gateID;
            if (id == MAX_GATES) break;
        }
        gate_num = id;
    }
    
    int cnt = 0, x;
    for(int id = 0; id < gate_num; id++) {
        x = gateIDs[id];
        if(id % 100 == 0) {
            bool live = false;
            for(int i = 0; i < numQubits; i++)
                if(!full[i])
                    live = true;
            if(!live)
                break;
        }
        // printf("gate_num %d x=%d\n", gate_num, x);
        auto& gate = remainGates[x];
        if (gate.isControlGate()) {
            if (!full[gate.controlQubit] && !full[gate.targetQubit]) { 
                int c = gate.controlQubit, t = gate.targetQubit;
                qindex newRelated = related[c] | related[t];
                newRelated = GateGroup::newRelated(newRelated, gate, localQubits, enableGlobal);
                if (bitCount(newRelated) <= localSize) {
                    new_cur = cur[c] | cur[t];
                    new_cur[id] = 1;
                    cur[c] = new_cur;
                    cur[t]= new_cur;
                    related[c] = related[t] = newRelated;
                    continue;
                }
            }
            full[gate.controlQubit] = full[gate.targetQubit] = 1;
        } else {
            if (!full[gate.targetQubit]) {
                cur[gate.targetQubit][id] = 1;
                related[gate.targetQubit] = GateGroup::newRelated(related[gate.targetQubit], gate, localQubits, enableGlobal);
            }
        }
    }

    bool blocked[numQubits];
    memset(blocked, 0, sizeof(blocked));
    qindex selectedRelated = 0;
    while (true) {
        int mx = 0, id = -1;
        for (int i = 0; i < numQubits; i++) {
            int count_i = cur[i].count();
            if (!blocked[i] && count_i > mx) {
                if (bitCount(selectedRelated | related[i]) <= localSize) {
                    mx = count_i;
                    id = i;
                } else {
                    blocked[i] = true;
                }
            }
        }
        if (mx == 0)
            break;
        selected |= cur[id];
        selectedRelated |= related[id];
        blocked[id] = true;
        for (int i = 0; i < numQubits; i++)
            if (!blocked[i] && cur[i].any()) {
                if ((related[i] | selectedRelated) == selectedRelated) {
                    selected |= cur[i];
                    blocked[i] = true;
                } else {
                    cur[i] &= ~cur[id];
                }
            }
    }

    if (!enableGlobal) {
        std::vector<int> ret;
        for(int id = 0; id < gate_num; id++) {
            if(selected.test(id))
                ret.push_back(gateIDs[id]);
        }
        return ret;
    }

    memset(blocked, 0, sizeof(blocked));
    cnt = 0;
    for(int id = 0; id < gate_num; id++) {
        x = gateIDs[id];
        cnt ++;
        if (cnt % 100 == 0) {
            bool live = false;
            for (int i = 0; i < numQubits; i++)
                if (!blocked[i])
                    live = true;
            if (!live)
                break;
        }
        if (selected.test(id)) continue;
        auto& g = remainGates[x];
        if (g.isDiagonal() && enableGlobal) {
            if (g.isControlGate()) {
                if (!blocked[g.controlQubit] && !blocked[g.targetQubit]) {
                    selected[id] = 1;
                } else {
                    blocked[g.controlQubit] = blocked[g.targetQubit] = 1;
                }
            } else {
                if (!blocked[g.targetQubit]) {
                    selected[id] = 1;
                }
            }
        } else {
            if (g.isControlGate()) {
                blocked[g.controlQubit] = 1;
            }
            blocked[g.targetQubit] = 1;
        }
    }
    std::vector<int> ret;
    for(int id = 0; id < gate_num; id++) {
        if(selected.test(id))
            ret.push_back(gateIDs[id]);
    }
    return ret;
}

ChunkCompiler::ChunkCompiler(int numQubits, int localSize, int chunkSize, const std::vector<Gate> &inputGates):
    OneLayerCompiler(numQubits, inputGates), localSize(localSize), chunkSize(chunkSize) {}

LocalGroup ChunkCompiler::run() {
    std::set<int> locals;
    for (int i = 0; i < localSize; i++)
        locals.insert(i);
    LocalGroup lg;
    GateGroup cur;
    cur.relatedQubits = 0;
    for (size_t i = 0; i < remainGates.size(); i++) {
        if (remainGates[i].isDiagonal() || locals.find(remainGates[i].targetQubit) != locals.end()) {
            cur.addGate(remainGates[i], -1ll, 1);
            continue;
        }
        qindex newRelated = 0;
        for (auto x: locals)
            newRelated |= ((qindex) 1) << x;
        cur.relatedQubits = newRelated;
        lg.relatedQubits |= newRelated;
        lg.fullGroups.push_back(std::move(cur));
        cur = GateGroup(); cur.relatedQubits = 0;
        cur.addGate(remainGates[i], -1ll, 1);
        std::set<int> cur_locals;
        for (int j = chunkSize + 1; j < numQubits; j++)
            if (locals.find(j) != locals.end())
                cur_locals.insert(j);
        for (size_t j = i + 1; j < remainGates.size() && cur_locals.size() > 1; j++) {
            if (!remainGates[i].isDiagonal())
                cur_locals.erase(remainGates[i].targetQubit);
        }
        int to_move = *cur_locals.begin();
        locals.erase(to_move);
        locals.insert(remainGates[i].targetQubit);
    }
    qindex newRelated = 0;
        for (auto x: locals)
            newRelated |= ((qindex) 1) << x;
    cur.relatedQubits = newRelated;
    lg.relatedQubits |= cur.relatedQubits;
    lg.fullGroups.push_back(std::move(cur));
    return std::move(lg);
}

template<int MAX_GATES>
void OneLayerCompiler<MAX_GATES>::removeGatesOpt(const std::vector<int>& remove) {
    for (auto& x: remove)
        remain.erase(x);
    if (remain.empty())
        remainGates.clear();
}