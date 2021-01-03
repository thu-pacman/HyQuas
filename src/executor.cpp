#include "executor.h"

#include <cuda_runtime.h>

#include "utils.h"
#include "assert.h"
#include "kernel.h"
#include "dbg.h"

Executor::Executor(std::vector<qComplex*> deviceStateVec, int numQubits):
    deviceStateVec(deviceStateVec),
    numQubits(numQubits),
    numLocalQubits(numQubits - MyGlobalVars::bit) {
    threadBias.resize(MyGlobalVars::numGPUs);
    for (int g = 0; g < MyGlobalVars::numGPUs; g++) {
        checkCudaErrors(cudaSetDevice(g));
        checkCudaErrors(cudaMalloc(&threadBias[g], sizeof(qindex) << THREAD_DEP));
    }
    std::vector<qreal> ret;
    int numLocalQubits = numQubits - MyGlobalVars::bit;
    numElements = 1 << numLocalQubits;
    deviceBuffer.resize(MyGlobalVars::numGPUs);
    for (int g = 0; g < MyGlobalVars::numGPUs; g++) {
        deviceBuffer[g] = deviceStateVec[g] + numElements;        
    }
    // TODO
    // initialize pos
}

void Executor::transpose(std::vector<cuttHandle> plans) {
    for (int g = 0; g < MyGlobalVars::numGPUs; g++) {
        cudaSetDevice(g);
        checkCuttErrors(cuttExecute(plans[g], deviceStateVec[g], deviceBuffer[g]));
    }
}

void Executor::all2all(int commSize, std::vector<int> comm) {
    for (int g = 0; g < MyGlobalVars::numGPUs; g++) {
        checkCudaErrors(cudaStreamSynchronize(MyGlobalVars::streams[g]));
    }
    int partSize = numElements / commSize;
    for (int xr = 0; xr < commSize; xr++) {
        for (int a = 0; a < MyGlobalVars::numGPUs; a++) {
            int b = a ^ xr;
            checkCudaErrors(cudaMemcpyAsync(
                deviceStateVec[comm[a]] + b % commSize * partSize,
                deviceBuffer[comm[b]] + a % commSize * partSize,
                partSize * sizeof(qComplex),
                cudaMemcpyDeviceToDevice,
                MyGlobalVars::streams[comm[b]]
            ));
        }
    }
    for (int g = 0; g < MyGlobalVars::numGPUs; g++) {
        checkCudaErrors(cudaStreamSynchronize(MyGlobalVars::streams[g]));
    }
}

void Executor::setState(std::vector<int> new_pos, std::vector<int> new_layout) {
    state.pos = new_pos;
    state.layout = new_layout;
}

#define SET_GATE_TO_ID(g, i) { \
    Complex mat[2][2] = {1, 0, 0, 1}; \
    hostGates[g * gates.size() + i] = KernelGate(GateType::ID, 0, 0, mat); \
}

void Executor::applyGateGroup(const GateGroup& gg) {
    auto& gates = gg.gates;
    // initialize blockHot, enumerate, threadBias
    qindex relatedLogicQb = gg.relatedQubits;
    if (bitCount(relatedLogicQb) < LOCAL_QUBIT_SIZE) {
        relatedLogicQb = fillRelatedQubits(relatedLogicQb);
    }
    qindex relatedQubits = toPhyQubitSet(relatedLogicQb);
    
    qindex blockHot, enumerate;
    prepareBitMap(relatedQubits, blockHot, enumerate);

    // initialize gates
    std::map<int, int> toID = getLogicShareMap(relatedQubits);
    
    auto isShareQubit = [relatedLogicQb] (int logicIdx) {
        return relatedLogicQb >> logicIdx & 1;
    };
    auto isLocalQubit = [this] (int logicIdx) {
        return state.pos[logicIdx] < numLocalQubits;
    };
    KernelGate hostGates[MyGlobalVars::numGPUs * gates.size()];
    assert(gates.size() < MAX_GATE);
    #pragma omp parallel for num_threads(MyGlobalVars::numGPUs)
    for (int g = 0; g < MyGlobalVars::numGPUs; g++) {
        auto isHiGPU = [this](int gpu_id, int logicIdx) {
            return gpu_id >> (state.pos[logicIdx] - numLocalQubits) & 1;
        };
        for (size_t i = 0; i < gates.size(); i++) {
            if (gates[i].controlQubit2 != -1) {
                // Assume no CC-Diagonal
                int c1 = gates[i].controlQubit;
                int c2 = gates[i].controlQubit2;
                if (isLocalQubit(c2) && !isLocalQubit(c1)) {
                    int c = c1; c1 = c2; c2 = c;
                }
                if (isLocalQubit(c1) && isLocalQubit(c2)) { // CCU(c1, c2, t)
                    if (isShareQubit(c2) && !isShareQubit(c1)) {
                        int c = c1; c1 = c2; c2 = c;
                    }
                    hostGates[g * gates.size() + i] = KernelGate(
                        gates[i].type,
                        toID[c2], 1 - isShareQubit(c2),
                        toID[c1], 1 - isShareQubit(c1),
                        toID[gates[i].targetQubit], 1 - isShareQubit(gates[i].targetQubit),
                        gates[i].mat
                    );
                } else if (isLocalQubit(c1) && !isLocalQubit(c2)) {
                    if (isHiGPU(g, c2)) { // CU(c1, t)
                        hostGates[g * gates.size() + i] = KernelGate(
                            Gate::toCU(gates[i].type),
                            toID[c1], 1 - isShareQubit(c1),
                            toID[gates[i].targetQubit], 1 - isShareQubit(gates[i].targetQubit),
                            gates[i].mat
                        );
                    } else { // ID(t)
                        SET_GATE_TO_ID(g, i)
                    }
                } else { // !isLocalQubit(c1) && !isLocalQubit(c2)
                    if (isHiGPU(g, c1) && isHiGPU(g, c2)) { // U(t)
                        hostGates[g * gates.size() + i] = KernelGate(
                            Gate::toU(gates[i].type),
                            toID[gates[i].targetQubit], 1 - isShareQubit(gates[i].targetQubit),
                            gates[i].mat
                        );
                    } else { // ID(t)
                        SET_GATE_TO_ID(g, i)
                    }
                }
            } else if (gates[i].controlQubit != -1) {
                int c = gates[i].controlQubit, t = gates[i].targetQubit;
                if (isLocalQubit(c) && isLocalQubit(t)) { // CU(c, t)
                    hostGates[g * gates.size() + i] = KernelGate(
                        gates[i].type,
                        toID[c], 1 - isShareQubit(c),
                        toID[t], 1 - isShareQubit(t),
                        gates[i].mat
                    );
                } else if (isLocalQubit(c) && !isLocalQubit(t)) { // U(c)
                    switch (gates[i].type) {
                        case GateType::CZ: {
                            if (isHiGPU(g, t)) {
                                hostGates[g * gates.size() + i] = KernelGate(
                                    GateType::Z,
                                    toID[c], 1 - isShareQubit(c),
                                    gates[i].mat
                                );
                            } else {
                                SET_GATE_TO_ID(g, i)
                            }
                            break;
                        }
                        case GateType::CRZ: { // GOC(c)
                            Complex mat[2][2] = {1, 0, 0, isHiGPU(g, t) ? gates[i].mat[1][1]: gates[i].mat[0][0]};
                            hostGates[g * gates.size() + i] = KernelGate(
                                GateType::GOC,
                                toID[c], 1 - isShareQubit(c),
                                mat
                            );
                            break;
                        }
                        default: {
                            UNREACHABLE()
                        }
                    }
                } else if (!isLocalQubit(c) && isLocalQubit(t)) {
                    if (isHiGPU(g, c)) { // U(t)
                        hostGates[g * gates.size() + i] = KernelGate(
                            Gate::toU(gates[i].type),
                            toID[t], 1 - isShareQubit(t),
                            gates[i].mat
                        );
                    } else {
                        SET_GATE_TO_ID(g, i)
                    }
                } else { // !isLocalQubit(c) && !isLocalQubit(t)
                    assert(gates[i].isDiagonal());
                    if (isHiGPU(g, c)) {
                        switch (gates[i].type) {
                            case GateType::CZ: {
                                if (isHiGPU(g, t)) {
                                    Complex mat[2][2] = {-1, 0, 0, -1};
                                    hostGates[g * gates.size() + i] = KernelGate(
                                        GateType::GZZ,
                                        0, 0,
                                        mat
                                    );
                                } else {
                                    SET_GATE_TO_ID(g, i)
                                }
                                break;
                            }
                            case GateType::CRZ: {
                                Complex val = isHiGPU(g, t) ? gates[i].mat[1][1]: gates[i].mat[0][0];
                                Complex mat[2][2] = {val, 0, 0, val};
                                hostGates[g * gates.size() + i] = KernelGate(
                                    GateType::GCC,
                                    0, 0,
                                    mat
                                );
                                break;
                            }
                            default: {
                                UNREACHABLE()
                            }
                        }
                    } else {
                        SET_GATE_TO_ID(g, i);
                    }
                }
            } else {
                int t = gates[i].targetQubit;
                if (!isLocalQubit(t)) { // GCC(t)
                    switch (gates[i].type) {
                        case GateType::U1: {
                            if (isHiGPU(g, t)) {
                                Complex val = gates[i].mat[1][1];
                                Complex mat[2][2] = {val, 0, 0, val};
                                hostGates[g * gates.size() + i] = KernelGate(
                                    GateType::GCC,
                                    0, 0,
                                    mat
                                );
                            } else {
                                SET_GATE_TO_ID(g, i)
                            }
                            break;
                        }
                        case GateType::Z: {
                            if (isHiGPU(g, t)) {
                                Complex mat[2][2] = {-1, 0, 0, -1};
                                hostGates[g * gates.size() + i] = KernelGate(
                                    GateType::GZZ,
                                    0, 0,
                                    mat
                                );
                            } else {
                                SET_GATE_TO_ID(g, i)
                            }
                            break;
                        }
                        case GateType::S: {
                            if (isHiGPU(g, t)) {
                                Complex val(0, 1);
                                Complex mat[2][2] = {val, 0, 0, val};
                                hostGates[g * gates.size() + i] = KernelGate(
                                    GateType::GII,
                                    0, 0,
                                    mat
                                );
                            } else {
                                SET_GATE_TO_ID(g, i)
                            }
                            break;
                        }
                        case GateType::T: {
                            if (isHiGPU(g, t)) {
                                Complex val = gates[i].mat[1][1];
                                Complex mat[2][2] = {val, 0, 0, val};
                                hostGates[g * gates.size() + i] = KernelGate(
                                    GateType::GCC,
                                    0, 0,
                                    mat
                                );
                            } else {
                                SET_GATE_TO_ID(g, i)
                            }
                            break;
                        }
                        case GateType::RZ: {
                            Complex val = isHiGPU(g, t) ? gates[i].mat[1][1]: gates[i].mat[0][0];
                            Complex mat[2][2] = {val, 0, 0, val};
                            hostGates[g * gates.size() + i] = KernelGate(
                                GateType::GCC,
                                0, 0,
                                mat
                            );
                            break;
                        }
                        default: {
                            UNREACHABLE()
                        }
                    }
                } else { // isLocalQubit(t) -> U(t)
                    hostGates[g * gates.size() + i] = KernelGate(
                        gates[i].type,
                        toID[t], 1 - isShareQubit(t),
                        gates[i].mat
                    );
                }
            }
        }
    }

    copyGatesToSymbol(hostGates, gates.size());
    qindex gridDim = (1 << numLocalQubits) >> LOCAL_QUBIT_SIZE;
    launchExecutor(gridDim, deviceStateVec, threadBias, numLocalQubits, gates.size(), blockHot, enumerate);
}

qindex Executor::toPhyQubitSet(qindex logicQubitset) {
     qindex ret = 0;
    for (int i = 0; i < numQubits; i++)
        if (logicQubitset >> i & 1)
            ret |= qindex(1) << state.pos[i];
    return ret;
}

qindex Executor::fillRelatedQubits(qindex relatedLogicQb) {
    int cnt = bitCount(relatedLogicQb);
    for (int i = 0; i < LOCAL_QUBIT_SIZE; i++) {
        if (!(relatedLogicQb & (1 << state.layout[i]))) {
            cnt++;
            relatedLogicQb |= (1 << state.layout[i]);
            if (cnt == LOCAL_QUBIT_SIZE)
                break;
        }
    }
    return relatedLogicQb;
}

void Executor::prepareBitMap(qindex relatedQubits, qindex& blockHot, qindex& enumerate) {
    blockHot = (qindex(1) << numLocalQubits) - 1 - relatedQubits;
    enumerate = relatedQubits;
    qindex threadHot = 0;
    for (int i = 0; i < THREAD_DEP; i++) {
        qindex x = enumerate & (-enumerate);
        threadHot += x;
        enumerate -= x;
    }
    qindex hostThreadBias[1 << THREAD_DEP];
    assert((threadHot | enumerate) == relatedQubits);
    for (int i = (1 << THREAD_DEP) - 1, j = threadHot; i >= 0; i--, j = threadHot & (j - 1)) {
        hostThreadBias[i] = j;
    }
    for (int g = 0; g < MyGlobalVars::numGPUs; g++) {
        checkCudaErrors(cudaMemcpyAsync(threadBias[g], hostThreadBias, sizeof(hostThreadBias), cudaMemcpyHostToDevice, MyGlobalVars::streams[g]));
    }
    // printf("related %x blockHot %x enumerate %x hostThreadBias[5] %x\n", relatedQubits, blockHot, enumerate, hostThreadBias[5]);
}

std::map<int, int> Executor::getLogicShareMap(int relatedQubits) {
    int shareCnt = 0;
    int localCnt = 0;
    int globalCnt = 0;
    std::map<int, int> toID; 
    for (int i = 0; i < numLocalQubits; i++) {
        if (relatedQubits & (qindex(1) << i)) {
            toID[state.layout[i]] = shareCnt++;
        } else {
            toID[state.layout[i]] = localCnt++;
        }
    }
    for (int i = numLocalQubits; i < numQubits; i++)
        toID[state.layout[i]] = globalCnt++;
    return toID;
}

void Executor::finalize() {
    for (int g = 0; g < MyGlobalVars::numGPUs; g++) {
        checkCudaErrors(cudaStreamSynchronize(MyGlobalVars::streams[g]));
        checkCudaErrors(cudaFree(threadBias[g]));
    }
}