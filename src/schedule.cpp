#include "schedule.h"
#include <cstring>
#include <algorithm>
#include <assert.h>
#include <tuple>
#include <omp.h>

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
    return std::move(ret);
}

void GateGroup::addGate(const Gate& gate, bool enableGlobal) {
    gates.push_back(gate);
    if (enableGlobal) {
        if (!gate.isDiagonal()) {
            relatedQubits |= qindex(1) << gate.targetQubit;
        }
    } else {
        relatedQubits |= qindex(1) << gate.targetQubit;
        if (gate.controlQubit != -1)
            relatedQubits |= qindex(1) << gate.controlQubit;
        if (gate.controlQubit2 != -1)
            relatedQubits |= qindex(1) << gate.controlQubit2;
    }
}

GateGroup GateGroup::copyGates() {
    GateGroup ret;
    ret.gates = this->gates;
    ret.relatedQubits = this->relatedQubits;
    return std::move(ret);
}

void Schedule::dump(int numQubits) {
    int L = 3;
    for (auto& lg: localGroups) {
        for (auto& gg: lg.fullGroups) {
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
#if BACKEND == 1 || BACKEND == 2 || BACKEND == 3
    for (size_t i = 0; i < localGroups.size(); i++) {
        const LocalGroup& lg = localGroups[i];
        printf("Global: ");
        for (int j = 0; j < numQubits; j++) {
            if (!(lg.relatedQubits >> j & 1)) {
                printf("%d ", j);
            }
        }
        printf("\n");
        printf("pos: ");
        for (int j = 0; j < numQubits; j++) {
            printf("[%d: %d] ", j, lg.state.pos[j]);
        }
        printf("\n");
        printf("layout: ");
        for (int j = 0; j < numQubits; j++) {
            printf("[%d: %d] ", j, lg.state.pos[j]);
        }
        printf("\n\n");
    }
#endif
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
    auto num_gg = fullGroups.size();
    std::vector<unsigned char> result;
    result.resize(sizeof(relatedQubits) + sizeof(num_gg));
    auto arr = result.data();
    int cur = 0;
    SERIALIZE_STEP(relatedQubits);
    SERIALIZE_STEP(num_gg);
    for (auto& gateGroup: fullGroups) {
        auto gg = gateGroup.serialize();
        result.insert(result.end(), gg.begin(), gg.end());
    }
    return result;
}


LocalGroup LocalGroup::deserialize(const unsigned char* arr, int& cur) {
    LocalGroup s;
    decltype(s.fullGroups.size()) num_gg;
    DESERIALIZE_STEP(s.relatedQubits);
    DESERIALIZE_STEP(num_gg);
    for (decltype(num_gg) i = 0; i < num_gg; i++) {
        s.fullGroups.push_back(GateGroup::deserialize(arr, cur));
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

std::vector<int> gen_perm_vector(int len) {
    std::vector<int> ret;
    for (int i = 0; i < len; i++)
        ret.push_back(i);
    return ret;
}

State GateGroup::initPerGateState(const State& oldState) {
    state = oldState;
    cuttPlans.clear();
    return state;
}

State GateGroup::initBlasState(const State& oldState, int numQubits) {
    int numLocalQubits = numQubits - MyGlobalVars::bit;
    std::vector<int> pos = oldState.pos;
    std::vector<int> layout = oldState.layout;
    std::vector<int> dim(numLocalQubits + 1, 2);

    std::vector<int> toGlobal; // qubit id
    std::vector<int> toLocal; // qubit id
    int numMatQubits = bitCount(relatedQubits);
    for (int i = 0; i < numMatQubits; i++) {
        int q = layout[i];
        if (!(relatedQubits >> q & 1))
            toGlobal.push_back(q);
    }
    for (int i = numMatQubits; i < numLocalQubits; i++) {
        int q = layout[i];
        if (relatedQubits >> q & 1)
            toLocal.push_back(q);
    }
    assert(toGlobal.size() == toLocal.size());
    std::vector<int> perm = gen_perm_vector(numLocalQubits);
    for (size_t i = 0; i < toGlobal.size(); i++) {
        int x = toGlobal[i], y = toLocal[i];
        std::swap(perm[pos[x]], perm[pos[y]]);
        std::swap(pos[x], pos[y]);
        layout[pos[x]] = x; layout[pos[y]] = y;
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
    cuttPlans.clear();
    for (int g = 0; g < MyGlobalVars::numGPUs; g++) {
        cuttHandle plan;
        checkCudaErrors(cudaSetDevice(g));
        checkCuttErrors(cuttPlan(&plan, numLocalQubits + 1, dim.data(), perm.data(), sizeof(qComplex) / 2, MyGlobalVars::streams[g]));
        cuttPlans.push_back(plan);
    }
    State newState = State(pos, layout);
    this->state = newState;
    return newState;
}


State GateGroup::initState(const State& oldState, int numQubits) {
    if (BACKEND == 1) {
        backend = Backend::PerGate;
    } else if (BACKEND == 3) {
        backend = Backend::BLAS;
    } else {
        UNREACHABLE();
    }

    if (backend == Backend::PerGate) {
        return initPerGateState(oldState);
    }
    if (backend == Backend::BLAS) {
        return initBlasState(oldState, numQubits);
    }
    UNREACHABLE();
}

State LocalGroup::initState(const State& oldState, int numQubits, const std::vector<int>& newGlobals, qindex overlapGlobals) {
    int numLocalQubits = numQubits - MyGlobalVars::bit;
    auto pos = oldState.pos, layout = oldState.layout;
    int overlapCnt = bitCount(overlapGlobals);
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
    a2aComm = newComm;
    a2aCommSize = MyGlobalVars::numGPUs >> overlapCnt;
    cuttPlans.clear();
    std::vector<int> dim(numLocalQubits + 1, 2);
    for (int g = 0; g < MyGlobalVars::numGPUs; g++) {
        cuttHandle plan;
        checkCudaErrors(cudaSetDevice(g));
        checkCuttErrors(cuttPlan(&plan, numLocalQubits + 1, dim.data(), perm.data(), sizeof(qComplex) / 2, MyGlobalVars::streams[g]));
        cuttPlans.push_back(plan);
    }
    auto newState = State(pos, layout);
    this->state = newState;
    for (auto& gg: fullGroups) {
        newState = gg.initState(newState, numQubits);
    }
    return newState;
}

State LocalGroup::initFirstGroupState(const State& oldState, int numQubits, const std::vector<int>& newGlobals) {
    int numLocalQubits = numQubits - MyGlobalVars::bit;
    auto pos = oldState.pos, layout = oldState.layout;
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
    state = State(pos, layout);
    cuttPlans.clear();
    for (int g = 0; g < MyGlobalVars::numGPUs; g++) {
        cuttPlans.push_back(cuttHandle());
    }
    a2aCommSize = -1;
    a2aComm.clear();
#ifdef SHOW_SCHEDULE
    printf("pos: "); for (auto x: pos) printf("%d ", x); printf("\n");
    printf("layout: "); for (auto x: layout) printf("%d ", x); printf("\n");
    printf("------------------------------------------------------\n");
#endif
    auto newState = State(pos, layout);
    this->state = newState;
    for (auto& gg: fullGroups) {
        newState = gg.initState(newState, numQubits);
    }
    return newState;
}

void GateGroup::initCPUMatrix() {
    auto& pos = state.pos;
    int numMatQubits = bitCount(relatedQubits);
    int n = 1 << numMatQubits;
    std::unique_ptr<qComplex[]> mat(new qComplex[n * n]);
    for (int i = 0; i < n * n; i++) mat[i] = make_qComplex(0.0, 0.0);
    for (int i = 0; i < n; i++) {
        mat[i * n + i] = make_qComplex(1.0, 0.0);
    }
    auto insertBit = [](int x, int pos) {
        return (x >> pos << (pos + 1)) | (x & ((qindex(1) << pos) - 1));
    };
    for (auto& gate: gates) {
        if (gate.controlQubit2 != -1) {
            int c2 = pos[gate.controlQubit2];
            int c1 = pos[gate.controlQubit];
            int t = pos[gate.targetQubit];
            assert(c2 < numMatQubits);
            assert(c1 < numMatQubits);
            assert(t < numMatQubits);
            // sort
            int a = std::max(std::max(c1, c2), t);
            int c = std::min(std::min(c1, c2), t);
            int b = c2 + c1 + t - a - c;
            
            #pragma omp parallel for
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < (n >> 3); j++) {
                    int lo = j;
                    lo = insertBit(lo, c);
                    lo = insertBit(lo, b);
                    lo = insertBit(lo, a);
                    lo += i * n;
                    lo |= 1 << c2;
                    lo |= 1 << c1;
                    int hi = lo | 1 << t;
                    qComplex v0 = mat[lo];
                    qComplex v1 = mat[hi];
                    mat[lo] = v0 * qComplex(gate.mat[0][0]) + v1 * qComplex(gate.mat[0][1]);
                    mat[hi] = v0 * qComplex(gate.mat[1][0]) + v1 * qComplex(gate.mat[1][1]);
                }
            }
        } else if (gate.controlQubit != -1) {
            int c = pos[gate.controlQubit];
            int t = pos[gate.targetQubit];
            assert(c < numMatQubits);
            assert(t < numMatQubits);
            // sort
            int a = std::max(c, t);
            int b = std::min(c, t);

            #pragma omp parallel for
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < (n >> 2); j++) {
                    int lo = j;
                    lo = insertBit(lo, b);
                    lo = insertBit(lo, a);
                    lo += i * n;
                    lo |= 1 << c;
                    int hi = lo | 1 << t;
                    qComplex v0 = mat[lo];
                    qComplex v1 = mat[hi];
                    mat[lo] = v0 * qComplex(gate.mat[0][0]) + v1 * qComplex(gate.mat[0][1]);
                    mat[hi] = v0 * qComplex(gate.mat[1][0]) + v1 * qComplex(gate.mat[1][1]);
                }
            }
        } else {
            int t = pos[gate.targetQubit];
            assert(t < numMatQubits);
            #pragma omp parallel for
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < (n >> 1); j++) {
                    int lo = j;
                    lo = insertBit(lo, t);
                    lo += i * n;
                    int hi = lo | 1 << t;
                    qComplex v0 = mat[lo];
                    qComplex v1 = mat[hi];
                    mat[lo] = v0 * qComplex(gate.mat[0][0]) + v1 * qComplex(gate.mat[0][1]);
                    mat[hi] = v0 * qComplex(gate.mat[1][0]) + v1 * qComplex(gate.mat[1][1]);
                }
            }
        }
    }
    // assert(isUnitary(mat, n));
    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < n; j++)
    //         printf("(%.2f %.2f) ", mat[i * n + j].x, mat[i * n + j].y);
    //     printf("\n");
    // }
    matrix = std::move(mat);
}

void GateGroup::initGPUMatrix() {
    int n = 1 << bitCount(relatedQubits);
    qreal realMat[2 * n][2 * n];
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            qComplex val = matrix[i * n + j];
            realMat[i * 2][j * 2] = val.x;
            realMat[i * 2][j * 2 + 1] = val.y;
            realMat[i * 2 + 1][j * 2] = -val.y;
            realMat[i * 2 + 1][j * 2 + 1] = val.x;
        }
    assert(deviceMats.size() == 0);
    deviceMats.clear();
    for (int g = 0; g < MyGlobalVars::numGPUs; g++) {
        checkCudaErrors(cudaSetDevice(g));
        qreal* mat;
        cudaMalloc(&mat, n * n * 4 * sizeof(qreal));
        cudaMemcpyAsync(mat, realMat, n * n * 4 * sizeof(qreal), cudaMemcpyHostToDevice, MyGlobalVars::streams[g]);
        deviceMats.push_back(mat);
    }
}

void GateGroup::initMatrix() {
    if (backend == Backend::BLAS) {
        initCPUMatrix();
        initGPUMatrix();
    }
}


#if BACKEND == 1 || BACKEND == 3
void Schedule::initCuttPlans(int numQubits) {
    int numLocalQubits = numQubits - MyGlobalVars::bit;
    State state(numQubits);

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
        assert(newGlobals.size() >= MyGlobalVars::bit);
        newGlobals.resize(MyGlobalVars::bit);

        if (lgID == 0) {
            state = localGroup.initFirstGroupState(state, numQubits, newGlobals);
        } else {
            state = localGroup.initState(state, numQubits, newGlobals, overlapGlobals);
        }
    }
    finalState = state;
}

void Schedule::initMatrix() {
    for (auto& lg: localGroups) {
        for (auto& gg: lg.fullGroups) {
            gg.initMatrix();
        }
    }
}

#else
void Schedule::initCuttPlans(int numQubits) {
    UNREACHABLE()
}

void Schedule::initMatrix() {
    UNREACHABLE()
}
#endif