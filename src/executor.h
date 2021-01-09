#pragma once
#include "utils.h"

#include <cutt.h>
#include <vector>
#include <map>

#include "schedule.h"

class Executor {
public:
    Executor(std::vector<qComplex*> deviceStateVec, int numQubits, const Schedule& schedule);
    void run();
private:
    // instructions
    void transpose(std::vector<cuttHandle> plans);
    void all2all(int commSize, std::vector<int> comm);
    void setState(const State& newState) { state = newState; }
    void applyGateGroup(const GateGroup& gg, int sliceID = -1);
    void applyPerGateGroup(const GateGroup& gg);
    void applyBlasGroup(const GateGroup& gg);
    void applyPerGateGroupSliced(const GateGroup& gg, int sliceID);
    void applyBlasGroupSliced(const GateGroup& gg, int sliceID);
    void finalize();
    void storeState();
    void loadState();
    void sliceBarrier(int sliceID);
    void allBarrier();

    // utils
    qindex toPhyQubitSet(qindex logicQubitset) const;
    qindex fillRelatedQubits(qindex related) const;
    KernelGate getGate(const Gate& gate, int part_id, int numLocalQubits, qindex relatedLogicQb, const std::map<int, int>& toID) const;

    // internal
    void prepareBitMap(qindex relatedQubits, qindex& blockHot, qindex& enumerate, int numLocalQubits); // allocate threadBias
    std::map<int, int> getLogicShareMap(int relatedQubits, int numLocalQubits) const; // input: physical, output logic -> share

    State state;
    State oldState;
    std::vector<cudaEvent_t> commEvents; // commEvents[slice][gpuID]
    std::vector<int> partID; // partID[slice][gpuID]

    // constants
    std::vector<qindex*> threadBias;
    std::vector<qComplex*> deviceStateVec;
    std::vector<qComplex*> deviceBuffer;
    int numQubits;
    int numSlice, numSliceBit;

    //schedule
    const Schedule& schedule;
    
};