#include "schedule.h"
#include <cstring>
#include <algorithm>
#include <assert.h>
#include <tuple>

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
    // TODO DELETE
    if (gate.controlQubit != -1) {
        relatedQubits |= qindex(1) << gate.controlQubit;
    }
    if (gate.controlQubit2 != -1) {
        relatedQubits |= qindex(1) << gate.controlQubit2;
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
    printf("%d %d\n", (int) cuttPlans.size(), (int) midPos.size());
    for (size_t i = 0; i < localGroups.size(); i++) {
        printf("Global: ");
        for (int j = 0; j < numQubits; j++) {
            if (!(localGroups[i].relatedQubits >> j & 1)) {
                printf("%d ", j);
            }
        }
        printf("\n");
        printf("pos: ");
        for (int j = 0; j < numQubits; j++) {
            printf("[%d: %d] ", j, midPos[i][j]);
        }
        printf("\n");
        printf("layout: ");
        for (int j = 0; j < numQubits; j++) {
            printf("[%d: %d] ", j, midLayout[i][j]);
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
    for (decltype(num_gates) i = 0; i < num_gates; i++) {
        gg.gates.push_back(Gate::deserialize(arr, cur));
    }
    return gg;
}

std::vector<unsigned char> LocalGroup::serialize() const {
    auto num_gg = gateGroups.size();
    std::vector<unsigned char> result;
    result.resize(sizeof(relatedQubits) + sizeof(num_gg));
    auto arr = result.data();
    int cur = 0;
    SERIALIZE_STEP(relatedQubits);
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
    DESERIALIZE_STEP(s.relatedQubits);
    DESERIALIZE_STEP(num_gg);
    for (decltype(num_gg) i = 0; i < num_gg; i++) {
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
    for (decltype(num_lg) i = 0; i < num_lg; i++) {
        s.localGroups.push_back(LocalGroup::deserialize(arr, cur));
    }
    return s;
}

void Schedule::initCuttPlans(int numQubits) {
    auto gen_perm_vector = [](int len) {
        std::vector<int> ret;
        for (int i = 0; i < len; i++)
            ret.push_back(i);
        return ret;
    };
    int numLocalQubits = numQubits - MyGlobalVars::bit;
    std::vector<int> pos = gen_perm_vector(numQubits); // The position of qubit x is pos[x]
    std::vector<int> layout = gen_perm_vector(numQubits); // The qubit locate at x is layout[x]
    std::vector<int> dim(numLocalQubits + 1, 2);
    midPos.clear();
    midLayout.clear();
    cuttPlans.clear(); cuttPlans.resize(MyGlobalVars::numGPUs);
    a2aComm.clear();
    a2aCommSize.clear();
    // printf("len %d\n", dim.size());
    for (size_t lgID = 0; lgID < localGroups.size(); lgID++) {
        auto& localGroup = localGroups[lgID];
        std::vector<int> newGlobals;
        std::vector<int> newLocals;
        for (int i = 0; i < numQubits; i++) {
            if (! (localGroup.relatedQubits >> i & 1)) {
                newGlobals.push_back(i);
            }
        }
        
        auto globalPos = [numQubits, numLocalQubits](const std::vector<int>& layout, int x) {
            auto pos = std::find(layout.data() + numLocalQubits, layout.data() + numQubits, x);
            return std::make_tuple(pos != layout.data() + numQubits, pos - layout.data() - numLocalQubits);
        };

        int overlapGlobals = 0;
        int overlapCnt = 0;
        // put overlapped global qubit into the previous
        for (size_t i = 0; i < newGlobals.size(); i++) {
            bool isGlobal;
            int p;
            std::tie(isGlobal, p) = globalPos(layout, newGlobals[i]);
            if (isGlobal) {
                std::swap(newGlobals[p], newGlobals[i]);
                overlapGlobals |= qindex(1) << p;
                overlapCnt ++;
            }
        }
#ifdef SHOW_SCHEDULE
        printf("globals: "); for (auto x: newGlobals) printf("%d ", x); printf("\n");
#endif
        assert(newGlobals.size() >= MyGlobalVars::bit);
        newGlobals.resize(MyGlobalVars::bit);

        if (lgID == 0) {
            for (size_t i = 0; i < newGlobals.size(); i++) {
                int x = newGlobals[i];
                if (pos[x] >= numLocalQubits)
                    continue;
                for (int y = numLocalQubits; y < numQubits; y++) {
                    if (std::find(newGlobals.begin(), newGlobals.end(), layout[y]) == newGlobals.end()) {
                        std::swap(pos[x], pos[y]);
                        layout[pos[x]] = x; layout[pos[y]] = y;
                        break;
                    }
                }
                
            }
            midPos.push_back(pos);
            midLayout.push_back(layout);
            for (int g = 0; g < MyGlobalVars::numGPUs; g++) {
                cuttPlans[g].push_back(cuttHandle());
            }
            a2aCommSize.push_back(-1);
            a2aComm.emplace_back();
#ifdef SHOW_SCHEDULE
            printf("pos: "); for (auto x: pos) printf("%d ", x); printf("\n");
            printf("layout: "); for (auto x: layout) printf("%d ", x); printf("\n");
            printf("------------------------------------------------------\n");
#endif
            continue;
        }
        std::vector<int> perm = gen_perm_vector(numLocalQubits);
        int c = numLocalQubits - MyGlobalVars::bit + overlapCnt;
        for (int i = 0; i < MyGlobalVars::bit; i++) {
            if (overlapGlobals >> i & 1) continue;
            std::swap(perm[pos[newGlobals[i]]], perm[c]);
            int swappedQid = layout[c];
            pos[swappedQid] = pos[newGlobals[i]];
            pos[newGlobals[i]] = c;
            layout[pos[newGlobals[i]]] = newGlobals[i];
            layout[pos[swappedQid]] = swappedQid;
            c++;
        }
#ifdef SHOW_SCHEDULE
        printf("perm: "); for (auto x: perm) printf("%d ", x); printf("\n");
        printf("pos: "); for (auto x: pos) printf("%d ", x); printf("\n");
        printf("layout: "); for (auto x: layout) printf("%d ", x); printf("\n\n");
#endif
        // complex have two floats
        perm.push_back(0);
        for (int i = perm.size() - 1; i; i--) {
            perm[i] = perm[i-1] + 1;
        }
        perm[0] = 0;

        c = numLocalQubits - MyGlobalVars::bit + overlapCnt;
        for (int i = 0; i < MyGlobalVars::bit; i++) {
            if (overlapGlobals >> i & 1) continue;
            int a = i + numLocalQubits;
            int qa = layout[a], qc = layout[c];
            layout[a] = qc; pos[qc] = a;
            layout[c] = qa; pos[qa] = c;
            c++;
        }
#ifdef SHOW_SCHEDULE
        printf("pos: "); for (auto x: pos) printf("%d ", x); printf("\n");
        printf("layout: "); for (auto x: layout) printf("%d ", x); printf("\n");
        printf("------------------------------------------------------\n");
#endif
        std::vector<std::pair<int, int>> newCommPair;
        for (int i = 0; i < MyGlobalVars::numGPUs; i++) {
            newCommPair.push_back(std::make_pair(i & overlapGlobals, i));
        }
        std::sort(newCommPair.begin(), newCommPair.end());
        std::vector<int> newComm;
        for (auto x: newCommPair) {
            newComm.push_back(x.second);
        }
        a2aComm.push_back(newComm);
        a2aCommSize.push_back(MyGlobalVars::numGPUs >> overlapCnt);

        for (int g = 0; g < MyGlobalVars::numGPUs; g++) {
            cuttHandle plan;
            checkCudaErrors(cudaSetDevice(g));
            checkCuttErrors(cuttPlan(&plan, numLocalQubits + 1, dim.data(), perm.data(), sizeof(qComplex) / 2, MyGlobalVars::streams[g]));
            cuttPlans[g].push_back(plan);
        }
        midPos.push_back(pos);
        midLayout.push_back(layout);
    }
    finalPos = pos;
}