#pragma once
#include <vector>
#include <cutt.h>
#include <memory>
#include <string>
#include "utils.h"
#include "gate.h"

enum class Backend {
    None, PerGate, BLAS
};

std::string to_string(Backend b);

struct State {
    std::vector<int> pos;
    std::vector<int> layout;
    State() = default;
    State(const State&) = default;
    State(const std::vector<int>& p, const std::vector<int>& l): pos(p), layout(l) {};
    State(int numQubits) {
        pos.clear();
        for (int i = 0; i < numQubits; i++) {
            pos.push_back(i);
        }
        layout.clear();
        for (int i = 0; i < numQubits; i++) {
            layout.push_back(i);
        }
    }

    std::vector<unsigned char> serialize() const;
    static State deserialize(const unsigned char* arr, int& cur);
};

struct GateGroup {
    std::vector<Gate> gates;
    qindex relatedQubits;
    State state;
    std::vector<int> cuttPerm;
    int matQubit;
    Backend backend;

    std::vector<cuttHandle> cuttPlans;

    std::vector<std::unique_ptr<qComplex[]>> matrix;
    std::vector<qComplex*> deviceMats;

    GateGroup(GateGroup&&) = default;
    GateGroup& operator = (GateGroup&&) = default;
    GateGroup(): relatedQubits(0) {}
    GateGroup copyGates();

    static GateGroup merge(const GateGroup& a, const GateGroup& b);
    static qindex newRelated(qindex old, const Gate& g, qindex localQubits, bool enableGlobal);
    void addGate(const Gate& g, qindex localQubits, bool enableGlobal);
    
    bool contains(int i) { return (relatedQubits >> i) & 1; }
    
    std::vector<unsigned char> serialize() const;
    static GateGroup deserialize(const unsigned char* arr, int& cur);

    State initState(const State& oldState, int numLocalQubits);
    State initPerGateState(const State& oldState);
    State initBlasState(const State& oldState, int numLocalQubit);
    void initCPUMatrix(int numLocalQubit);
    void initGPUMatrix();
    void initMatrix(int numLocalQubit);
    void getCuttPlanPointers(int numLocalQubits, std::vector<cuttHandle*> &cuttPlanPointers, std::vector<int*> &cuttPermPointers, std::vector<int> &locals);
};

struct LocalGroup {
    State state;
    int a2aCommSize;
    std::vector<int> a2aComm;
    std::vector<int> cuttPerm;

    std::vector<GateGroup> overlapGroups;
    std::vector<GateGroup> fullGroups;
    qindex relatedQubits;

    std::vector<cuttHandle> cuttPlans;
    
    LocalGroup() = default;
    LocalGroup(LocalGroup&&) = default;

    bool contains(int i) { return (relatedQubits >> i) & 1; }
    void getCuttPlanPointers(int numLocalQubits, std::vector<cuttHandle*> &cuttPlanPointers, std::vector<int*> &cuttPermPointers, std::vector<int> &locals, bool isFirstGroup = false);
    State initState(const State& oldState, int numQubits, const std::vector<int>& newGlobals, qindex overlapGlobals, qindex overlapRelated);
    State initFirstGroupState(const State& oldState, int numQubits, const std::vector<int>& newGlobals);
    State initStateInplace(const State& oldState, int numQubits, const std::vector<int>& newGlobals, qindex overlapGlobals);
    std::vector<unsigned char> serialize() const;
    static LocalGroup deserialize(const unsigned char* arr, int& cur);
};

struct Schedule {
    std::vector<LocalGroup> localGroups;
    State finalState;
    
    void dump(int numQubits);
    std::vector<unsigned char> serialize() const;
    static Schedule deserialize(const unsigned char* arr, int& cur);
    void initMatrix(int numQubits);
    void initCuttPlans(int numLocalQubits);
};

void removeGates(std::vector<Gate>& remain, const std::vector<Gate>& remove); // remain := remain - remove        