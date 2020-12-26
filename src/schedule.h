#pragma once
#include <vector>
#include <cutt.h>
#include "utils.h"
#include "gate.h"

struct GateGroup {
    std::vector<Gate> gates;
    qindex relatedQubits;
    GateGroup(const GateGroup&) = default;
    GateGroup(): relatedQubits(0) {}
    static GateGroup merge(const GateGroup& a, const GateGroup& b);
    void addGate(const Gate& g, bool enableGlobal = false);
    bool contains(int i) { return (relatedQubits >> i) & 1; }
    std::vector<int> toID() const;
    std::vector<unsigned char> serialize() const;
    static GateGroup deserialize(const unsigned char* arr, int& cur);\
};

struct LocalGroup {
    std::vector<GateGroup> gateGroups;
    qindex relatedQubits;
    bool contains(int i) { return (relatedQubits >> i) & 1; }
    std::vector<unsigned char> serialize() const;
    static LocalGroup deserialize(const unsigned char* arr, int& cur);
};

struct Schedule {
    std::vector<LocalGroup> localGroups;
    std::vector<std::vector<cuttHandle>> cuttPlans;
    std::vector<std::vector<int>> midPos;
    std::vector<std::vector<int>> midLayout;
    std::vector<int> a2aCommSize;
    std::vector<std::vector<int>> a2aComm;
    std::vector<int> finalPos;
    void dump(int numQubits);
    std::vector<unsigned char> serialize() const;
    static Schedule deserialize(const unsigned char* arr, int& cur);
    void initCuttPlans(int numQubits);
};