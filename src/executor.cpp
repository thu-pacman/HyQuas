#include "executor.h"

#include <cuda_runtime.h>
#include <algorithm>

#include "utils.h"
#include "assert.h"
#include "kernel.h"
#include "dbg.h"

Executor::Executor(std::vector<qComplex*> deviceStateVec, int numQubits, Schedule& schedule):
    deviceStateVec(deviceStateVec),
    numQubits(numQubits),
    schedule(schedule) {
    threadBias.resize(MyGlobalVars::localGPUs);
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaSetDevice(g));
        checkCudaErrors(cudaMalloc(&threadBias[g], sizeof(qindex) << THREAD_DEP));
    }
    std::vector<qreal> ret;
    int numLocalQubits = numQubits - MyGlobalVars::bit;
    qindex numElements = qindex(1) << numLocalQubits;
    deviceBuffer.resize(MyGlobalVars::localGPUs);
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        deviceBuffer[g] = deviceStateVec[g] + numElements;        
    }
    numSlice = MyGlobalVars::numGPUs;
    numSliceBit = MyGlobalVars::bit;
    // TODO
    // initialize pos
}

void Executor::run() {
    for (size_t lgID = 0; lgID < schedule.localGroups.size(); lgID ++) {
        auto& localGroup = schedule.localGroups[lgID];
        if (lgID > 0) {
            if (INPLACE) {
                this->inplaceAll2All(localGroup.a2aCommSize, localGroup.a2aComm, localGroup.state);
            } else {
                this->transpose(localGroup.cuttPlans);
                this->all2all(localGroup.a2aCommSize, localGroup.a2aComm);
            }
            this->setState(localGroup.state);
#ifdef ENABLE_OVERLAP
            this->storeState();
            for (int s = 0; s < numSlice; s++) {
                this->loadState();
                this->sliceBarrier(s);
                for (auto& gg: schedule.localGroups[lgID].overlapGroups) {
                    this->applyGateGroup(gg, s);
                }
            }
#endif
        } else {
            this->setState(localGroup.state);
            assert(localGroup.overlapGroups.size() == 0);
        }
        for (auto& gg: schedule.localGroups[lgID].fullGroups) {
            this->applyGateGroup(gg, -1);
        }
    }
    this->finalize();
}

void Executor::transpose(std::vector<cuttHandle> plans) {
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        cudaSetDevice(g);
        checkCuttErrors(cuttExecute(plans[g], deviceStateVec[g], deviceBuffer[g]));
    }
}

void Executor::inplaceAll2All(int commSize, std::vector<int> comm, const State& newState) {
    int numLocalQubits = numQubits - MyGlobalVars::bit;
    qindex oldGlobals = 0;
    for (int i = numLocalQubits; i < numQubits; i++)
        oldGlobals |= 1ll << state.layout[i];
    qindex newGlobals = 0;
    for (int i = numLocalQubits; i < numQubits; i++)
        newGlobals |= 1ll << newState.layout[i];
    
    qindex globalMask = 0;
    qindex localMasks[commSize];
    qindex localMask = 0;
    for (int i = numLocalQubits; i < numQubits; i++)
        if (newState.layout[i] != state.layout[i]) {
            int x = state.layout[i];
            globalMask |= 1ll << i;
            localMask |= 1ll << newState.pos[x];
        }

    for (qindex i = commSize-1, msk = localMask; i >= 0; i--, msk = localMask & (msk - 1)) {
        localMasks[i] = msk;
    }

    int sliceSize = INPLACE;
    while (sliceSize < MAX_SLICE && !(localMask >> sliceSize & 1))
        sliceSize ++;

    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaSetDevice(g));
        checkCudaErrors(cudaEventRecord(MyGlobalVars::events[g], MyGlobalVars::streams[g]));
        checkCudaErrors(cudaStreamWaitEvent(MyGlobalVars::streams_comm[g], MyGlobalVars::events[g], 0));
    }

    qComplex* tmpBuffer[MyGlobalVars::localGPUs];
    size_t tmpStart = 1ll << numLocalQubits;
    if (BACKEND == 3 || BACKEND == 4)
        tmpStart <<= 1;
    for (int i = 0; i < MyGlobalVars::localGPUs; i++)
        tmpBuffer[i] = deviceStateVec[i] + tmpStart;

    for (qindex iter = 0; iter < (1ll << numLocalQubits); iter += (1 << sliceSize)) {
        if (iter & localMask) continue;
        for (int xr = 1; xr < commSize; xr++) {
            // copy from src to tmp_buffer
            for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
                checkCudaErrors(cudaSetDevice(g));
                checkCudaErrors(cudaEventRecord(MyGlobalVars::events[g], MyGlobalVars::streams_comm[g]));
            }
#if USE_MPI
            checkNCCLErrors(ncclGroupStart());
#endif
            for (int a = 0; a < MyGlobalVars::numGPUs; a++) {
                // the (a%commSize)-th GPU in the a/commSize comm_world (comm[a]) ->
                // the (a%commSize)^xr -th GPU in the same comm_world comm[a^xr]
                int b = a ^ xr;
                if (comm[a] / MyGlobalVars::localGPUs != MyMPI::rank)
                    continue;
                qindex srcBias = iter + localMasks[b % commSize];
#if USE_MPI
                int comm_a = comm[a] %  MyGlobalVars::localGPUs;
                if (a < b) {
                    checkNCCLErrors(ncclSend(
                        deviceStateVec[comm_a] + srcBias,
                        1 << (sliceSize + 1), // use double rather than complex
                        NCCL_FLOAT_TYPE,
                        comm[b],
                        MyGlobalVars::ncclComms[comm_a],
                        MyGlobalVars::streams_comm[comm_a]
                    ));
                    checkNCCLErrors(ncclRecv(
                        tmpBuffer[comm_a],
                        1 << (sliceSize + 1), // use double rather than complex
                        NCCL_FLOAT_TYPE,
                        comm[b],
                        MyGlobalVars::ncclComms[comm_a],
                        MyGlobalVars::streams_comm[comm_a]
                    ));
                } else {
                    checkNCCLErrors(ncclRecv(
                        tmpBuffer[comm_a],
                        1 << (sliceSize + 1), // use double rather than complex
                        NCCL_FLOAT_TYPE,
                        comm[b],
                        MyGlobalVars::ncclComms[comm_a],
                        MyGlobalVars::streams_comm[comm_a]
                    ));
                    checkNCCLErrors(ncclSend(
                        deviceStateVec[comm_a] + srcBias,
                        1 << (sliceSize + 1), // use double rather than complex
                        NCCL_FLOAT_TYPE,
                        comm[b],
                        MyGlobalVars::ncclComms[comm_a],
                        MyGlobalVars::streams_comm[comm_a]
                    ));
                }
#else
                checkCudaErrors(cudaSetDevice(comm[b]));
                checkCudaErrors(cudaStreamWaitEvent(MyGlobalVars::streams_comm[comm[b]], MyGlobalVars::events[comm[a]], 0));
                checkCudaErrors(cudaMemcpyAsync(
                    tmpBuffer[comm[b]],
                    deviceStateVec[comm[a]] + srcBias,
                    (sizeof(qComplex) << sliceSize),
                    cudaMemcpyDeviceToDevice,
                    MyGlobalVars::streams_comm[comm[b]]
                ));
#endif
            }
#if USE_MPI
            checkNCCLErrors(ncclGroupEnd());
#else
            for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
                checkCudaErrors(cudaSetDevice(g));
                checkCudaErrors(cudaEventRecord(MyGlobalVars::events[g], MyGlobalVars::streams_comm[g]));
            }
#endif
            // copy from tmp_buffer to dst
            for (int a = 0; a < MyGlobalVars::numGPUs; a++) {
                int b = a ^ xr;
                if (comm[b] / MyGlobalVars::localGPUs != MyMPI::rank)
                    continue;
                qindex dstBias = iter + localMasks[a % commSize];
                int comm_b = comm[b] % MyGlobalVars::localGPUs;
                checkCudaErrors(cudaSetDevice(comm_b));
#if not USE_MPI
                // no need to sync nccl calls, as nccl calls are synchronized.
                checkCudaErrors(cudaStreamWaitEvent(MyGlobalVars::streams_comm[comm_b], MyGlobalVars::events[comm[a]], 0));
#endif
                checkCudaErrors(cudaMemcpyAsync(
                    deviceStateVec[comm_b] + dstBias,
                    tmpBuffer[comm_b],
                    (sizeof(qComplex) << sliceSize),
                    cudaMemcpyDeviceToDevice,
                    MyGlobalVars::streams_comm[comm_b]
                ));
            }
        }
    }
    this->eventBarrier();
}

void Executor::all2all(int commSize, std::vector<int> comm) {
    int numLocalQubit = numQubits - MyGlobalVars::bit;
    qindex numElements = 1ll << numLocalQubit;
    int numPart = numSlice / commSize;
    qindex partSize = numElements / numSlice;
    commEvents.resize(numSlice * MyGlobalVars::localGPUs);
    partID.resize(numSlice * MyGlobalVars::localGPUs);
    peer.resize(numSlice * MyGlobalVars::localGPUs);
    int sliceID = 0;
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaSetDevice(g));
        checkCudaErrors(cudaEventCreate(&MyGlobalVars::events[g]));
        checkCudaErrors(cudaEventRecord(MyGlobalVars::events[g], MyGlobalVars::streams[g]));
    }
    for (int xr = 0; xr < commSize; xr++) {
        for (int p = 0; p < numPart; p++) {
#if USE_MPI
            checkNCCLErrors(ncclGroupStart());
#endif
            for (int a = 0; a < MyGlobalVars::numGPUs; a++) {
                int b = a ^ xr;
                if (comm[a] / MyGlobalVars::localGPUs != MyMPI::rank)
                    continue;
                int comm_a = comm[a] % MyGlobalVars::localGPUs;
                int srcPart = a % commSize * numPart + p;
                int dstPart = b % commSize * numPart + p;
#if USE_MPI
                if (p == 0) {
                    checkCudaErrors(cudaStreamWaitEvent(
                        MyGlobalVars::streams_comm[comm_a],
                        MyGlobalVars::events[comm_a], 0)
                    );
                }
                checkCudaErrors(cudaSetDevice(comm_a));
                if (a == b) {
                    checkCudaErrors(cudaMemcpyAsync(
                        deviceStateVec[comm_a] + dstPart * partSize,
                        deviceBuffer[comm_a] + srcPart * partSize,
                        partSize * sizeof(qComplex),
                        cudaMemcpyDeviceToDevice,
                        MyGlobalVars::streams_comm[comm_a]
                    ));
                } else if (a < b) {
                    checkNCCLErrors(ncclSend(
                        deviceBuffer[comm_a] + dstPart * partSize,
                        partSize * 2, // use double rather than complex
                        NCCL_FLOAT_TYPE,
                        comm[b],
                        MyGlobalVars::ncclComms[comm_a],
                        MyGlobalVars::streams_comm[comm_a]
                    ));
                    checkNCCLErrors(ncclRecv(
                        deviceStateVec[comm_a] + dstPart * partSize,
                        partSize * 2, // use double rather than complex
                        NCCL_FLOAT_TYPE,
                        comm[b],
                        MyGlobalVars::ncclComms[comm_a],
                        MyGlobalVars::streams_comm[comm_a]
                    ));
                } else {
                    checkNCCLErrors(ncclRecv(
                        deviceStateVec[comm_a] + dstPart * partSize,
                        partSize * 2, // use double rather than complex
                        NCCL_FLOAT_TYPE,
                        comm[b],
                        MyGlobalVars::ncclComms[comm_a],
                        MyGlobalVars::streams_comm[comm_a]
                    ));
                    checkNCCLErrors(ncclSend(
                        deviceBuffer[comm_a] + dstPart * partSize,
                        partSize * 2, // use double rather than complex
                        NCCL_FLOAT_TYPE,
                        comm[b],
                        MyGlobalVars::ncclComms[comm_a],
                        MyGlobalVars::streams_comm[comm_a]
                    ));
                }
#else
                if (p == 0) {
                    checkCudaErrors(cudaStreamWaitEvent(
                        MyGlobalVars::streams_comm[comm[a]],
                        MyGlobalVars::events[comm[b]], 0)
                    );
                }
                checkCudaErrors(cudaMemcpyAsync(
                    deviceStateVec[comm[a]] + dstPart * partSize,
                    deviceBuffer[comm[b]] + srcPart * partSize,
                    partSize * sizeof(qComplex),
                    cudaMemcpyDeviceToDevice,
                    MyGlobalVars::streams_comm[comm[a]]
                ));
#endif
                partID[sliceID * MyGlobalVars::localGPUs + comm_a] = dstPart;
                peer[sliceID * MyGlobalVars::localGPUs + comm_a] = comm[b];
            }
#if USE_MPI
            checkNCCLErrors(ncclGroupEnd());
#endif
            // events should be recorded after ncclGroupEnd
#ifdef ENABLE_OVERLAP
            for (int a = 0; a < MyGlobalVars::numGPUs; a++) {
                if (USE_MPI && comm[a] / MyGlobalVars::localGPUs != MyMPI::rank)
                    continue;
                int comm_a = comm[a] % MyGlobalVars::localGPUs;
                cudaEvent_t event;
                checkCudaErrors(cudaSetDevice(comm_a));
                checkCudaErrors(cudaEventCreate(&event));
                checkCudaErrors(cudaEventRecord(event, MyGlobalVars::streams_comm[comm_a]));
                commEvents[sliceID * MyGlobalVars::localGPUs + comm_a] = event;
            }
#endif
            sliceID++;
        }
    }
#ifndef ENABLE_OVERLAP
    this->eventBarrierAll();
#endif
}

#define SET_GATE_TO_ID(g, i) { \
    qComplex mat[2][2] = {1, 0, 0, 1}; \
    hostGates[g * gates.size() + i] = KernelGate(GateType::ID, 0, 0, mat); \
}

#define IS_SHARE_QUBIT(logicIdx) ((relatedLogicQb >> logicIdx & 1) > 0)
#define IS_LOCAL_QUBIT(logicIdx) (state.pos[logicIdx] < numLocalQubits)
#define IS_HIGH_PART(part_id, logicIdx) ((part_id >> (state.pos[logicIdx] - numLocalQubits) & 1) > 0)

KernelGate Executor::getGate(const Gate& gate, int part_id, int numLocalQubits, qindex relatedLogicQb, const std::map<int, int>& toID) const {
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
            if (IS_HIGH_PART(part_id, c2)) { // CU(c1, t)
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
            if (IS_HIGH_PART(part_id, c1) && IS_HIGH_PART(part_id, c2)) { // U(t)
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
                    if (IS_HIGH_PART(part_id, t)) {
                        return KernelGate(
                            GateType::Z,
                            toID.at(c), 1 - IS_SHARE_QUBIT(c),
                            gate.mat
                        );
                    } else {
                        return KernelGate::ID();
                    }
                }
                case GateType::CU1: {
                    if (IS_HIGH_PART(part_id, t)) {
                        return KernelGate(
                            GateType::U1,
                            toID.at(c), 1 - IS_SHARE_QUBIT(c),
                            gate.mat
                        );
                    } else {
                        return KernelGate::ID();
                    }
                }
                case GateType::CRZ: { // GOC(c)
                    qComplex mat[2][2] = {make_qComplex(1), make_qComplex(0), make_qComplex(0), IS_HIGH_PART(part_id, t) ? gate.mat[1][1]: gate.mat[0][0]};
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
            if (IS_HIGH_PART(part_id, c)) { // U(t)
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
            if (IS_HIGH_PART(part_id, c)) {
                switch (gate.type) {
                    case GateType::CZ: {
                        if (IS_HIGH_PART(part_id, t)) {
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
                    case GateType::CU1: {
                        if (IS_HIGH_PART(part_id, t)) {
                            qComplex mat[2][2] = {gate.mat[1][1], make_qComplex(0), make_qComplex(0), gate.mat[1][1]};
                            return KernelGate(
                                GateType::GCC,
                                0, 0,
                                mat
                            );
                        }
                    }
                    case GateType::CRZ: {
                        qComplex val = IS_HIGH_PART(part_id, t) ? gate.mat[1][1]: gate.mat[0][0];
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
                    if (IS_HIGH_PART(part_id, t)) {
                        qComplex val = gate.mat[1][1];
                        qComplex mat[2][2] = {val, make_qComplex(0), make_qComplex(0), val};
                        return KernelGate(GateType::GCC, 0, 0, mat);
                    } else {
                        return KernelGate::ID();
                    }
                }
                case GateType::Z: {
                    if (IS_HIGH_PART(part_id, t)) {
                        qComplex mat[2][2] = {make_qComplex(-1), make_qComplex(0), make_qComplex(0), make_qComplex(-1)};
                        return KernelGate(GateType::GZZ, 0, 0, mat);
                    } else {
                        return KernelGate::ID();
                    }
                }
                case GateType::S: {
                    if (IS_HIGH_PART(part_id, t)) {
                        qComplex val = make_qComplex(0, 1);
                        qComplex mat[2][2] = {val, make_qComplex(0), make_qComplex(0), val};
                        return KernelGate(GateType::GII, 0, 0, mat);
                    } else {
                        return KernelGate::ID();
                    }
                }
                case GateType::SDG: {
                    // FIXME
                    if (IS_HIGH_PART(part_id, t)) {
                        qComplex val = make_qComplex(0, -1);
                        qComplex mat[2][2] = {val, make_qComplex(0), make_qComplex(0), val};
                        return KernelGate(GateType::GCC, 0, 0, mat);
                    } else {
                        return KernelGate::ID();
                    }
                }
                case GateType::T: {
                    if (IS_HIGH_PART(part_id, t)) {
                        qComplex val = gate.mat[1][1];
                        qComplex mat[2][2] = {val, make_qComplex(0), make_qComplex(0), val};
                        return KernelGate(GateType::GCC, 0, 0, mat);
                    } else {
                        return KernelGate::ID();
                    }
                }
                case GateType::TDG: {
                    if (IS_HIGH_PART(part_id, t)) {
                        qComplex val = gate.mat[1][1];
                        qComplex mat[2][2] = {val, make_qComplex(0), make_qComplex(0), val};
                        return KernelGate(GateType::GCC, 0, 0, mat);
                    } else {
                        return KernelGate::ID();
                    }
                }
                case GateType::RZ: {
                    qComplex val = IS_HIGH_PART(part_id, t) ? gate.mat[1][1]: gate.mat[0][0];
                    qComplex mat[2][2] = {val, make_qComplex(0), make_qComplex(0), val};
                    return KernelGate(GateType::GCC, 0, 0, mat);
                }
                case GateType::ID: {
                    return KernelGate::ID();
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

void Executor::applyGateGroup(GateGroup& gg, int sliceID) {
#ifdef MEASURE_STAGE
    cudaEvent_t start[MyGlobalVars::localGPUs], stop[MyGlobalVars::localGPUs];
    for (int i = 0; i < MyGlobalVars::localGPUs; i++) {
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaEventCreate(&start[i]));
        checkCudaErrors(cudaEventCreate(&stop[i]));    
    }
    for (int i = 0; i < MyGlobalVars::localGPUs; i++) {
        checkCudaErrors(cudaEventRecord(start[i], MyGlobalVars::streams[i]));
    }
#endif
    switch (gg.backend) {
        case Backend::PerGate: {
            if (sliceID == -1) {
                applyPerGateGroup(gg);
            } else {
                applyPerGateGroupSliced(gg, sliceID);
            }
            break;
        }
        case Backend::BLAS: {
            if (sliceID == -1) {
                applyBlasGroup(gg);
            } else {
                applyBlasGroupSliced(gg, sliceID);
            }
            break;
        }
        default:
            UNREACHABLE()
    }
    setState(gg.state);
#ifdef MEASURE_STAGE
    // TODO multistream support
    for (int i = 0; i < MyGlobalVars::localGPUs; i++) {
        checkCudaErrors(cudaEventRecord(stop[i], MyGlobalVars::streams[i]));
    }
    float min_time = 1e100, max_time = 0, sum_time = 0;
    for (int i = 0; i < MyGlobalVars::localGPUs; i++) {
        float time;
        cudaEventSynchronize(stop[i]);
        cudaEventElapsedTime(&time, start[i], stop[i]);
        min_time = std::min(min_time, time);
        max_time = std::max(max_time, time);
        sum_time += time;
        cudaEventDestroy(start[i]);
        cudaEventDestroy(stop[i]);
    }

    printf("[ApplyGateGroup] time for %x %d %s %s: [min=%f, max=%f, avg=%f]\n",
            gg.relatedQubits, (int)gg.gates.size(), to_string(gg.backend).c_str(), sliceID == -1 ? "full" : "slice", 
            min_time, max_time, sum_time / MyGlobalVars::localGPUs);
#endif
    // printf("Group End\n");
}

void Executor::applyPerGateGroup(GateGroup& gg) {
    auto& gates = gg.gates;
    int numLocalQubits = numQubits - MyGlobalVars::bit;
    // initialize blockHot, enumerate, threadBias
    qindex relatedLogicQb = gg.relatedQubits;
    if (bitCount(relatedLogicQb) < LOCAL_QUBIT_SIZE) {
        relatedLogicQb = fillRelatedQubits(relatedLogicQb);
    }
    qindex relatedQubits = toPhyQubitSet(relatedLogicQb);
    
    unsigned int blockHot, enumerate;
    prepareBitMap(relatedQubits, blockHot, enumerate, numLocalQubits);

    // initialize gates
    std::map<int, int> toID = getLogicShareMap(relatedQubits, numLocalQubits);
    
    KernelGate hostGates[MyGlobalVars::localGPUs * gates.size()];
    assert(gates.size() < MAX_GATE);
    #pragma omp parallel for num_threads(MyGlobalVars::localGPUs)
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        int globalGPUID = MyMPI::rank * MyGlobalVars::localGPUs + g;
        for (size_t i = 0; i < gates.size(); i++) {
           hostGates[g * gates.size() + i] = getGate(gates[i], globalGPUID, numLocalQubits, relatedLogicQb, toID);
        }
    }
    qindex gridDim = (qindex(1) << numLocalQubits) >> LOCAL_QUBIT_SIZE;
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaSetDevice(g));
        copyGatesToSymbol(hostGates, gates.size(), MyGlobalVars::streams[g], g);
        launchExecutor(gridDim, deviceStateVec[g], threadBias[g], numLocalQubits, gates.size(), blockHot, enumerate, MyGlobalVars::streams[g], g);
    }
}

void Executor::applyPerGateGroupSliced(GateGroup& gg, int sliceID) {
    auto& gates = gg.gates;
    int numLocalQubits = numQubits - 2 * MyGlobalVars::bit;
    // initialize blockHot, enumerate, threadBias
    qindex relatedLogicQb = gg.relatedQubits;
    if (bitCount(relatedLogicQb) < LOCAL_QUBIT_SIZE) {
        relatedLogicQb = fillRelatedQubits(relatedLogicQb);
    }
    qindex relatedQubits = toPhyQubitSet(relatedLogicQb);
    
    unsigned int blockHot, enumerate;
    prepareBitMap(relatedQubits, blockHot, enumerate, numLocalQubits);

    // initialize gates
    std::map<int, int> toID = getLogicShareMap(relatedQubits, numLocalQubits);
    
    KernelGate hostGates[MyGlobalVars::localGPUs * gates.size()];
    assert(gates.size() < MAX_GATE);
    
    qindex partSize = qindex(1) << numLocalQubits;
    int numSlice = MyGlobalVars::numGPUs;
    #pragma omp parallel for num_threads(MyGlobalVars::localGPUs)
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        int pID = partID[sliceID * MyGlobalVars::localGPUs + g];
        int globalGPUID = MyMPI::rank * MyGlobalVars::localGPUs + g;
        for (size_t i = 0; i < gates.size(); i++) {
            hostGates[g * gates.size() + i] = getGate(gates[i], globalGPUID * numSlice + pID, numLocalQubits, relatedLogicQb, toID);
        }
    }
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaSetDevice(g));
        copyGatesToSymbol(hostGates, gates.size(), MyGlobalVars::streams[g], g);
        qindex gridDim = (qindex(1) << numLocalQubits) >> LOCAL_QUBIT_SIZE;
        int pID = partID[sliceID * MyGlobalVars::localGPUs + g];
        launchExecutor(gridDim, deviceStateVec[g] + pID * partSize, threadBias[g], numLocalQubits, gates.size(), blockHot, enumerate, MyGlobalVars::streams[g], g);
    }
}

void Executor::applyBlasGroup(GateGroup& gg) {
    int numLocalQubits = numQubits - MyGlobalVars::bit;
#ifdef OVERLAP_MAT
    gg.initMatrix(numLocalQubits);
#endif
    qindex numElements = qindex(1) << numLocalQubits;
    qComplex alpha = make_qComplex(1.0, 0.0), beta = make_qComplex(0.0, 0.0);
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaSetDevice(g));
        checkCuttErrors(cuttExecute(gg.cuttPlans[g], deviceStateVec[g], deviceBuffer[g]));
        int K = 1 << gg.matQubit;
        checkBlasErrors(cublasGEMM(MyGlobalVars::blasHandles[g], CUBLAS_OP_N, CUBLAS_OP_N,
            K, numElements / K, K, // M, N, K
            &alpha, gg.deviceMats[g], K, // alpha, a, lda
            deviceBuffer[g], K, // b, ldb
            &beta, deviceStateVec[g], K // beta, c, ldc
        ));
    }
}

void Executor::applyBlasGroupSliced(GateGroup& gg, int sliceID) {
    int numLocalQubits = numQubits - 2 * MyGlobalVars::bit;
    // qubits at position [n - 2 bit, n - bit) should be excluded by the compiler
#ifdef OVERLAP_MAT
    if(sliceID == 0)
        gg.initMatrix(numQubits - MyGlobalVars::bit);
#endif
    qindex numElements = qindex(1) << numLocalQubits;
    qComplex alpha = make_qComplex(1.0, 0.0), beta = make_qComplex(0.0, 0.0);
    qindex partSize = qindex(1) << numLocalQubits;
    int K = 1 << gg.matQubit;
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaSetDevice(g));
        int pID = partID[sliceID * MyGlobalVars::localGPUs + g];
        checkCuttErrors(cuttExecute(gg.cuttPlans[g], deviceStateVec[g] + partSize * pID, deviceBuffer[g] + partSize * pID));
        checkBlasErrors(cublasGEMM(MyGlobalVars::blasHandles[g], CUBLAS_OP_N, CUBLAS_OP_N,
            K, numElements / K, K, // M, N, K
            &alpha, gg.deviceMats[g], K, // alpha, a, lda
            deviceBuffer[g] + partSize * pID, K, // b, ldb
            &beta, deviceStateVec[g] + partSize * pID, K // beta, c, ldc
        ));
    }
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
        if (!(relatedLogicQb & (1ll << state.layout[i]))) {
            cnt++;
            relatedLogicQb |= (1ll << state.layout[i]);
            if (cnt == LOCAL_QUBIT_SIZE)
                break;
        }
    }
    return relatedLogicQb;
}

void Executor::prepareBitMap(qindex relatedQubits, unsigned int& blockHot, unsigned int& enumerate, int numLocalQubits) {
    blockHot = (qindex(1) << numLocalQubits) - 1 - relatedQubits;
    enumerate = relatedQubits;
    qindex threadHot = 0;
    for (int i = 0; i < THREAD_DEP; i++) {
        qindex x = enumerate & (-enumerate);
        threadHot += x;
        enumerate -= x;
    }
    unsigned int hostThreadBias[1 << THREAD_DEP];
    assert((threadHot | enumerate) == relatedQubits);
    for (qindex i = (1 << THREAD_DEP) - 1, j = threadHot; i >= 0; i--, j = threadHot & (j - 1)) {
        hostThreadBias[i] = j;
    }
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaMemcpyAsync(threadBias[g], hostThreadBias, sizeof(hostThreadBias), cudaMemcpyHostToDevice, MyGlobalVars::streams[g]));
    }
}

std::map<int, int> Executor::getLogicShareMap(qindex relatedQubits, int numLocalQubits) const{
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
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaStreamSynchronize(MyGlobalVars::streams[g]));
        checkCudaErrors(cudaFree(threadBias[g]));
    }
    schedule.finalState = state;
}

void Executor::storeState() {
    oldState = state;
}

void Executor::loadState() {
    state = oldState;
}

void Executor::sliceBarrier(int sliceID) {
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaSetDevice(g));
        checkCudaErrors(cudaStreamWaitEvent(MyGlobalVars::streams[g], commEvents[sliceID * MyGlobalVars::localGPUs + g], 0));
#if !USE_MPI
        int peerID = peer[sliceID * MyGlobalVars::localGPUs + g];
        checkCudaErrors(cudaStreamWaitEvent(MyGlobalVars::streams[g], commEvents[sliceID * MyGlobalVars::localGPUs + peerID], 0));
#endif
    }
}

void Executor::allBarrier() {
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaStreamSynchronize(MyGlobalVars::streams[g]));
        checkCudaErrors(cudaStreamSynchronize(MyGlobalVars::streams_comm[g]));
    }
#if USE_MPI
    checkMPIErrors(MPI_Barrier(MPI_COMM_WORLD));
#endif
}

void Executor::eventBarrierAll() {
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaSetDevice(g));
        checkCudaErrors(cudaEventRecord(MyGlobalVars::events[g], MyGlobalVars::streams_comm[g]));
    }
    for (int gg = 0; gg < MyGlobalVars::localGPUs; gg++) {
        checkCudaErrors(cudaSetDevice(gg));
        for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
            checkCudaErrors(cudaStreamWaitEvent(MyGlobalVars::streams[gg], MyGlobalVars::events[g], 0));
        }
    }
}

void Executor::eventBarrier() {
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaSetDevice(g));
        checkCudaErrors(cudaEventRecord(MyGlobalVars::events[g], MyGlobalVars::streams_comm[g]));
        checkCudaErrors(cudaStreamWaitEvent(MyGlobalVars::streams[g], MyGlobalVars::events[g], 0));
    }
}