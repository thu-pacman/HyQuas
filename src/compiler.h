#pragma once
#include <vector>
#include <set>
#include <bitset>
#include "schedule.h"
#include "utils.h"
#include "gate.h"

class Compiler {
public:
    Compiler(int numQubits, std::vector<Gate> inputGates);
    Schedule run();
private:
    void fillLocals(LocalGroup& lg);
    std::vector<std::pair<std::vector<Gate>, qindex>> moveToNext(LocalGroup& lg);
    int numQubits;
    int localSize;
    int shareSize;
    bool enableGlobal;
    std::vector<Gate> gates;
};

template<int MAX_GATES>
class OneLayerCompiler {
public:
    OneLayerCompiler(int numQubits, const std::vector<Gate>& inputGates);
protected:
    int numQubits;
    std::vector<Gate> remainGates;
    std::vector<int> getGroupOpt(qindex full, qindex related[], bool enableGlobal, int localSize, qindex localQubits);
    void removeGatesOpt(const std::vector<int>& remove);
    std::set<int> remain;
};

class SimpleCompiler: public OneLayerCompiler<2048> {
public:
    SimpleCompiler(int numQubits, int localSize, qindex localQubits, const std::vector<Gate>& inputGates, bool enableGlobal, qindex whiteList = 0, qindex required = 0);
    LocalGroup run();
private:
    int localSize;
    qindex localQubits;
    bool enableGlobal;
    qindex whiteList;
    qindex required;
};

class AdvanceCompiler: public OneLayerCompiler<512> {
public:
    AdvanceCompiler(int numQubits, qindex localQubits, qindex blasForbid, std::vector<Gate> inputGates);
    LocalGroup run(State &state, bool usePerGate, bool useBLAS, int preGateSize, int blasSize, int cuttSize);
private:
    qindex localQubits;
    qindex blasForbid;
};

class ChunkCompiler: public OneLayerCompiler<512> {
public:
    ChunkCompiler(int numQubits, int localSize, int chunkSize, const std::vector<Gate> &inputGates);
    LocalGroup run();
private:
    int localSize, chunkSize;
};