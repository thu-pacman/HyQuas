#pragma once
#include "utils.h"

#include <cutt.h>
#include <vector>
#include <map>

#include "schedule.h"

class Executor {
public:
    Executor(std::vector<qComplex*> deviceStateVec, int numQubits, Schedule& schedule);
    void run();
private:
    // instructions
    void transpose(std::vector<cuttHandle> plans);
    void inplaceAll2All(int commSize, std::vector<int> comm, const State& newState);
    void all2all(int commSize, std::vector<int> comm);
    void setState(const State& newState) { state = newState; }
    void applyGateGroup(GateGroup& gg, int sliceID = -1);
    void applyPerGateGroup(GateGroup& gg);
    void applyBlasGroup(GateGroup& gg);
    void applyPerGateGroupSliced(GateGroup& gg, int sliceID);
    void applyBlasGroupSliced(GateGroup& gg, int sliceID);
    void finalize();
    void storeState();
    void loadState();
    void sliceBarrier(int sliceID);
    void eventBarrier();
    void eventBarrierAll();
    void allBarrier();

    // utils
    qindex toPhyQubitSet(qindex logicQubitset) const;
    qindex fillRelatedQubits(qindex related) const;
    KernelGate getGate(const Gate& gate, int part_id, int numLocalQubits, qindex relatedLogicQb, const std::map<int, int>& toID) const;

    // internal
    void prepareBitMap(qindex relatedQubits, unsigned int& blockHot, unsigned int& threadBias, int numLocalQubits); // allocate threadBias
    std::map<int, int> getLogicShareMap(qindex relatedQubits, int numLocalQubits) const; // input: physical, output logic -> share

    State state;
    State oldState;
    std::vector<cudaEvent_t> commEvents; // commEvents[slice][gpuID]
    std::vector<int> partID; // partID[slice][gpuID]
    std::vector<int> peer; // peer[slice][gpuID]

    // constants
    std::vector<unsigned int*> threadBias;
    std::vector<qComplex*> deviceStateVec;
    std::vector<qComplex*> deviceBuffer;
    int numQubits;
    int numSlice, numSliceBit;

    //schedule
    Schedule& schedule;
    
};