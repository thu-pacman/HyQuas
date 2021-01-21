#include "evaluator.h"

Evaluator* Evaluator::instance_ptr = nullptr;

double Evaluator::perfPerGate(const GateGroup* gg, Devices device) {
    double tim_pred = pergate_group_overhead;
    for(auto gate : (gg -> gates)) {
        switch(gate.type) {
            case GateType::CCX : 
                tim_pred += pergate_ctr_perf[int(device)][int(GateType::CNOT)][0][2]; break;
            case GateType::CNOT : 
                tim_pred += pergate_ctr_perf[int(device)][int(GateType::CNOT)][0][2]; break;
            case GateType::CY : 
                tim_pred += pergate_ctr_perf[int(device)][int(GateType::CY)][0][2]; break;
            case GateType::CZ : 
                tim_pred += pergate_ctr_perf[int(device)][int(GateType::CZ)][0][2]; break;
            case GateType::CRX : 
                tim_pred += pergate_ctr_perf[int(device)][int(GateType::CRX)][0][2]; break;
            case GateType::CRY : 
                tim_pred += pergate_ctr_perf[int(device)][int(GateType::CRY)][0][2]; break;
            case GateType::CRZ : 
                tim_pred += pergate_ctr_perf[int(device)][int(GateType::CRZ)][0][2]; break;
            case GateType::U1 : 
                tim_pred += pergate_single_perf[int(device)][int(GateType::U1)][1]; break;
            case GateType::U2 : 
                tim_pred += pergate_single_perf[int(device)][int(GateType::U2)][1]; break;
            case GateType::U3 : 
                tim_pred += pergate_single_perf[int(device)][int(GateType::U3)][1]; break;
            case GateType::H : 
                tim_pred += pergate_single_perf[int(device)][int(GateType::H)][1]; break;
            case GateType::X : 
                tim_pred += pergate_single_perf[int(device)][int(GateType::X)][1]; break;
            case GateType::Y : 
                tim_pred += pergate_single_perf[int(device)][int(GateType::Y)][1]; break;
            case GateType::Z : 
                tim_pred += pergate_single_perf[int(device)][int(GateType::Z)][1]; break;
            case GateType::S : 
                tim_pred += pergate_single_perf[int(device)][int(GateType::S)][1]; break;
            case GateType::T : 
                tim_pred += pergate_single_perf[int(device)][int(GateType::T)][1]; break;
            case GateType::RX : 
                tim_pred += pergate_single_perf[int(device)][int(GateType::RX)][1]; break;
            case GateType::RY : 
                tim_pred += pergate_single_perf[int(device)][int(GateType::RY)][1]; break;
            case GateType::RZ : 
                tim_pred += pergate_single_perf[int(device)][int(GateType::RZ)][1]; break;
            default:
                printf("meet wrong gate : %s\n", Gate::get_name(gate.type).c_str());
                UNREACHABLE()
        }
    }
    return tim_pred / 512 + 2.0;
}

double Evaluator::perfBLAS(int numQubits, int blasSize, Devices device) {
    double bias = (numQubits < 28) ? ((qindex)1 << (28 - numQubits)) : (1.0 / ((qindex)1 << (numQubits - 28)));
    //return 35.0 * bias;
    return 35.0;
}

bool Evaluator::PerGateOrBLAS(const GateGroup* gg_pergate, const GateGroup* gg_blas, int numQubits, int blasSize, Devices device) {
    double pergate = perfPerGate(gg_pergate, device);
    double blas = perfBLAS(numQubits, blasSize, device);
    return pergate / (gg_pergate -> gates).size() < blas / (gg_blas -> gates).size();
}