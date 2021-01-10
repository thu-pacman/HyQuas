#pragma once
#include <vector>
#include <cutt.h>
#include <memory>
#include "utils.h"
#include "gate.h"

enum class Backend {
    None, PerGate, BLAS
};

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
};

struct GateGroup {
    std::vector<Gate> gates;
    qindex relatedQubits;
    State state;
    std::vector<cuttHandle> cuttPlans;

    Backend backend;
    std::vector<std::unique_ptr<qComplex[]>> matrix;
    std::vector<qComplex*> deviceMats;

    GateGroup(GateGroup&&) = default;
    GateGroup& operator = (GateGroup&&) = default;
    GateGroup(): relatedQubits(0) {}
    GateGroup copyGates();

    static GateGroup merge(const GateGroup& a, const GateGroup& b);
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
};

struct LocalGroup {
    State state;
    std::vector<cuttHandle> cuttPlans;
    int a2aCommSize;
    std::vector<int> a2aComm;

    std::vector<GateGroup> overlapGroups;
    std::vector<GateGroup> fullGroups;
    qindex relatedQubits;

    LocalGroup() = default;
    LocalGroup(LocalGroup&&) = default;

    bool contains(int i) { return (relatedQubits >> i) & 1; }
    State initState(const State& oldState, int numQubits, const std::vector<int>& newGlobals, qindex overlapGlobals);
    State initFirstGroupState(const State& oldState, int numQubits, const std::vector<int>& newGlobals);
    std::vector<unsigned char> serialize() const;
    static LocalGroup deserialize(const unsigned char* arr, int& cur);
};

struct Schedule {
    std::vector<LocalGroup> localGroups;
    State finalState;
    
    void dump(int numQubits);
    std::vector<unsigned char> serialize() const;
    static Schedule deserialize(const unsigned char* arr, int& cur);
    void initCuttPlans(int numQubits);
    void initMatrix(int numQubits);
};

void removeGates(std::vector<Gate>& remain, const std::vector<Gate>& remove); // remain := remain - remove        