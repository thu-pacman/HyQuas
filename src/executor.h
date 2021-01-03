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
    Executor(std::vector<qComplex*> deviceStateVec, int numQubits);
    void transpose(std::vector<cuttHandle> plans);
    void all2all(int commSize, std::vector<int> comm);
    void setState(std::vector<int> new_pos, std::vector<int> new_layout);
    void applyGateGroup(const GateGroup& gg);
    void finalize();
    // void Checkpoint();
    // void Restore();
private:
    // utils
    qindex toPhyQubitSet(qindex logicQubitset);
    qindex fillRelatedQubits(qindex related);

    // internal
    void prepareBitMap(qindex relatedQubits, qindex& blockHot, qindex& enumerate);
    std::map<int, int> getLogicShareMap(int relatedQubits); // input: physical, output logic -> share

    State state;

    // constants
    std::vector<qindex*> threadBias;
    std::vector<qComplex*> deviceStateVec;
    std::vector<qComplex*> deviceBuffer;
    int numQubits, numLocalQubits, numElements;
    
};