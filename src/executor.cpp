#include "executor.h"

#include <cuda_runtime.h>

#include "utils.h"
#include "assert.h"
#include "kernel.h"
#include "dbg.h"

Executor::Executor(std::vector<qComplex*> deviceStateVec, int numQubits, const Schedule& schedule):
    deviceStateVec(deviceStateVec),
    numQubits(numQubits),
    numLocalQubits(numQubits - MyGlobalVars::bit),
    schedule(schedule) {
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

void Executor::run() {
    for (size_t lgID = 0; lgID < schedule.localGroups.size(); lgID ++) {
        auto& localGroup = schedule.localGroups[lgID];
        if (lgID > 0) {
            this->transpose(localGroup.cuttPlans);
            this->all2all(localGroup.a2aCommSize, localGroup.a2aComm);
        }
        this->setState(localGroup.state);

        auto& fullGroups = schedule.localGroups[lgID].fullGroups;        
        for (size_t gg = 0; gg < fullGroups.size(); gg++) {
#ifdef MEASURE_STAGE
            // TODO multistream
            cudaEvent_t start, stop;
            checkCudaErrors(cudaEventCreate(&start));
            checkCudaErrors(cudaEventCreate(&stop));
            checkCudaErrors(cudaEventRecord(start, 0));
#endif
            this->applyGateGroup(fullGroups[gg]);
#ifdef MEASURE_STAGE
            // TODO multistream support
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float time;
            cudaEventElapsedTime(&time, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            printf("[Group %d] time for %x: %f\n", int(g), relatedQubits, time);
#endif
            // printf("Group End\n");
        }
    }
    this->finalize();
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

#define SET_GATE_TO_ID(g, i) { \
    qComplex mat[2][2] = {1, 0, 0, 1}; \
    hostGates[g * gates.size() + i] = KernelGate(GateType::ID, 0, 0, mat); \
}

#define IS_SHARE_QUBIT(logicIdx) ((relatedLogicQb >> logicIdx & 1) > 0)
#define IS_LOCAL_QUBIT(logicIdx) (state.pos[logicIdx] < numLocalQubits)
#define IS_HIGH_GPU(gpu_id, logicIdx) ((gpu_id >> (state.pos[logicIdx] - numLocalQubits) & 1) > 0)

KernelGate Executor::getGate(const Gate& gate, int gpu_id, qindex relatedLogicQb, const std::map<int, int>& toID) const {
    if (gate.controlQubit2 != -1) {
        // Assume no CC-Diagonal
        int c1 = gate.controlQubit;
        int c2 = gate.controlQubit2;
        if (IS_LOCAL_QUBIT(c2) && !IS_LOCAL_QUBIT(c1)) {
            int c = c1; c1 = c2; c2 = c;
        }
        if (IS_LOCAL_QUBIT(c1) && IS_LOCAL_QUBIT(c2)) { // CCU(c1, c2, t)
            if (IS_SHARE_QUBIT(c2) && !IS_SHARE_QUBIT(c1)) {
                int c = c1; c1 = c2; c2 = c;
            }
            return KernelGate(
                gate.type,
                toID.at(c2), 1 - IS_SHARE_QUBIT(c2),
                toID.at(c1), 1 - IS_SHARE_QUBIT(c1),
                toID.at(gate.targetQubit), 1 - IS_SHARE_QUBIT(gate.targetQubit),
                gate.mat
            );
        } else if (IS_LOCAL_QUBIT(c1) && !IS_LOCAL_QUBIT(c2)) {
            if (IS_HIGH_GPU(gpu_id, c2)) { // CU(c1, t)
                return KernelGate(
                    Gate::toCU(gate.type),
                    toID.at(c1), 1 - IS_SHARE_QUBIT(c1),
                    toID.at(gate.targetQubit), 1 - IS_SHARE_QUBIT(gate.targetQubit),
                    gate.mat
                );
            } else { // ID(t)
                return KernelGate::ID();
            }
        } else { // !IS_LOCAL_QUBIT(c1) && !IS_LOCAL_QUBIT(c2)
            if (IS_HIGH_GPU(gpu_id, c1) && IS_HIGH_GPU(gpu_id, c2)) { // U(t)
                return KernelGate(
                    Gate::toU(gate.type),
                    toID.at(gate.targetQubit), 1 - IS_SHARE_QUBIT(gate.targetQubit),
                    gate.mat
                );
            } else { // ID(t)
                return KernelGate::ID();
            }
        }
    } else if (gate.controlQubit != -1) {
        int c = gate.controlQubit, t = gate.targetQubit;
        if (IS_LOCAL_QUBIT(c) && IS_LOCAL_QUBIT(t)) { // CU(c, t)
            return KernelGate(
                gate.type,
                toID.at(c), 1 - IS_SHARE_QUBIT(c),
                toID.at(t), 1 - IS_SHARE_QUBIT(t),
                gate.mat
            );
        } else if (IS_LOCAL_QUBIT(c) && !IS_LOCAL_QUBIT(t)) { // U(c)
            switch (gate.type) {
                case GateType::CZ: {
                    if (IS_HIGH_GPU(gpu_id, t)) {
                        return KernelGate(
                            GateType::Z,
                            toID.at(c), 1 - IS_SHARE_QUBIT(c),
                            gate.mat
                        );
                    } else {
                        return KernelGate::ID();
                    }
                }
                case GateType::CRZ: { // GOC(c)
                    qComplex mat[2][2] = {make_qComplex(1), make_qComplex(0), make_qComplex(0), IS_HIGH_GPU(gpu_id, t) ? gate.mat[1][1]: gate.mat[0][0]};
                    return KernelGate(
                        GateType::GOC,
                        toID.at(c), 1 - IS_SHARE_QUBIT(c),
                        mat
                    );
                }
                default: {
                    UNREACHABLE()
                }
            }
        } else if (!IS_LOCAL_QUBIT(c) && IS_LOCAL_QUBIT(t)) {
            if (IS_HIGH_GPU(gpu_id, c)) { // U(t)
                return KernelGate(
                    Gate::toU(gate.type),
                    toID.at(t), 1 - IS_SHARE_QUBIT(t),
                    gate.mat
                );
            } else {
                return KernelGate::ID();
            }
        } else { // !IS_LOCAL_QUBIT(c) && !IS_LOCAL_QUBIT(t)
            assert(gate.isDiagonal());
            if (IS_HIGH_GPU(gpu_id, c)) {
                switch (gate.type) {
                    case GateType::CZ: {
                        if (IS_HIGH_GPU(gpu_id, t)) {
                            qComplex mat[2][2] = {make_qComplex(-1), make_qComplex(0), make_qComplex(0), make_qComplex(-1)};
                            return KernelGate(
                                GateType::GZZ,
                                0, 0,
                                mat
                            );
                        } else {
                            return KernelGate::ID();
                        }
                    }
                    case GateType::CRZ: {
                        qComplex val = IS_HIGH_GPU(gpu_id, t) ? gate.mat[1][1]: gate.mat[0][0];
                        qComplex mat[2][2] = {val, make_qComplex(0), make_qComplex(0), val};
                        return KernelGate(
                            GateType::GCC,
                            0, 0,
                            mat
                        );
                    }
                    default: {
                        UNREACHABLE()
                    }
                }
            } else {
                return KernelGate::ID();
            }
        }
    } else {
        int t = gate.targetQubit;
        if (!IS_LOCAL_QUBIT(t)) { // GCC(t)
            switch (gate.type) {
                case GateType::U1: {
                    if (IS_HIGH_GPU(gpu_id, t)) {
                        qComplex val = gate.mat[1][1];
                        qComplex mat[2][2] = {val, make_qComplex(0), make_qComplex(0), val};
                        return KernelGate(GateType::GCC, 0, 0, mat);
                    } else {
                        return KernelGate::ID();
                    }
                }
                case GateType::Z: {
                    if (IS_HIGH_GPU(gpu_id, t)) {
                        qComplex mat[2][2] = {make_qComplex(-1), make_qComplex(0), make_qComplex(0), make_qComplex(-1)};
                        return KernelGate(GateType::GZZ, 0, 0, mat);
                    } else {
                        return KernelGate::ID();
                    }
                }
                case GateType::S: {
                    if (IS_HIGH_GPU(gpu_id, t)) {
                        qComplex val = make_qComplex(0, 1);
                        qComplex mat[2][2] = {val, make_qComplex(0), make_qComplex(0), val};
                        return KernelGate(GateType::GII, 0, 0, mat);
                    } else {
                        return KernelGate::ID();
                    }
                }
                case GateType::T: {
                    if (IS_HIGH_GPU(gpu_id, t)) {
                        qComplex val = gate.mat[1][1];
                        qComplex mat[2][2] = {val, make_qComplex(0), make_qComplex(0), val};
                        return KernelGate(GateType::GCC, 0, 0, mat);
                    } else {
                        return KernelGate::ID();
                    }
                }
                case GateType::RZ: {
                    qComplex val = IS_HIGH_GPU(gpu_id, t) ? gate.mat[1][1]: gate.mat[0][0];
                    qComplex mat[2][2] = {val, make_qComplex(0), make_qComplex(0), val};
                    return KernelGate(GateType::GCC, 0, 0, mat);
                }
                default: {
                    UNREACHABLE()
                }
            }
        } else { // IS_LOCAL_QUBIT(t) -> U(t)
            return KernelGate(gate.type, toID.at(t), 1 - IS_SHARE_QUBIT(t), gate.mat);
        }
    }
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
    
    KernelGate hostGates[MyGlobalVars::numGPUs * gates.size()];
    assert(gates.size() < MAX_GATE);
    #pragma omp parallel for num_threads(MyGlobalVars::numGPUs)
    for (int g = 0; g < MyGlobalVars::numGPUs; g++) {
        for (size_t i = 0; i < gates.size(); i++) {
           hostGates[g * gates.size() + i] = getGate(gates[i], g, relatedLogicQb, toID);
        }
    }
    copyGatesToSymbol(hostGates, gates.size());
    qindex gridDim = (1 << numLocalQubits) >> LOCAL_QUBIT_SIZE;
    launchExecutor(gridDim, deviceStateVec, threadBias, numLocalQubits, gates.size(), blockHot, enumerate);
}

qindex Executor::toPhyQubitSet(qindex logicQubitset) const {
     qindex ret = 0;
    for (int i = 0; i < numQubits; i++)
        if (logicQubitset >> i & 1)
            ret |= qindex(1) << state.pos[i];
    return ret;
}

qindex Executor::fillRelatedQubits(qindex relatedLogicQb) const {
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

std::map<int, int> Executor::getLogicShareMap(int relatedQubits) const{
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