#pragma once
#include "utils.h"

#include <cutt.h>
#include <vector>
#include <map>

#include "schedule.h"

struct State {
    std::vector<int> pos;
    std::vector<int> layout;
};

class Executor {
public:
    Executor(std::vector<qComplex*> deviceStateVec, int numQubits, const Schedule& schedule);
    void run();
private:
    // instructions
    void transpose(std::vector<cuttHandle> plans);
    void all2all(int commSize, std::vector<int> comm);
    void setState(std::vector<int> new_pos, std::vector<int> new_layout);
    void applyGateGroup(const GateGroup& gg);
    void finalize();
    // void Checkpoint();
    // void Restore();

    // utils
    qindex toPhyQubitSet(qindex logicQubitset) const;
    qindex fillRelatedQubits(qindex related) const;
    KernelGate getGate(const Gate& g, int gpu_id, qindex relatedLogicQb, const std::map<int, int>& toID) const;

    // internal
    void prepareBitMap(qindex relatedQubits, qindex& blockHot, qindex& enumerate); // allocate threadBias
    std::map<int, int> getLogicShareMap(int relatedQubits) const; // input: physical, output logic -> share

    State state;

    // constants
    std::vector<qindex*> threadBias;
    std::vector<qComplex*> deviceStateVec;
    std::vector<qComplex*> deviceBuffer;
    int numQubits, numLocalQubits, numElements;

    //schedule
    const Schedule& schedule;
    
};