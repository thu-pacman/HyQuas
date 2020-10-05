#pragma once
#include <vector>
#include "schedule.h"
#include "utils.h"
#include "gate.h"

struct GateGroup {
    std::vector<Gate> gates;
    qindex relatedQubits;
    GateGroup(const GateGroup&) = default;
    GateGroup(): relatedQubits(0) {}
    static GateGroup merge(const GateGroup& a, const GateGroup& b);
    void addGate(const Gate& g);
    bool contains(int i) { return (relatedQubits >> i) & 1; }
    std::vector<int> toID() const;
};

struct Schedule {
    std::vector<GateGroup> gateGroups;
    void dump(int numQubits);
};