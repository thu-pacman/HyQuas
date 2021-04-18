#pragma once
#include "schedule.h"
#include "utils.h"
#include "gate.h"

#define GATE_NUM 24
#define MAX_QBITS 40

#define CALC_ALL_PARAM 0
#define CALC_PARTIAL_PARAM 1

/*
* build performance model to choose between BLAS and perGate backend
* Is a singleton class
**/
class Evaluator {
private:
    const double V100_U1[LOCAL_QUBIT_SIZE] = {235,225,225,225,225,225,224,225,225,225};
    const double V100_U2[LOCAL_QUBIT_SIZE] = {470,469,469,469,469,469,469,470,469,469};
    const double V100_U3[LOCAL_QUBIT_SIZE] = {469,469,469,469,469,469,469,469,469,469};
    const double V100_H[LOCAL_QUBIT_SIZE]  = {352,352,352,352,352,352,352,352,352,352};
    const double V100_X[LOCAL_QUBIT_SIZE]  = {350,350,350,350,350,350,350,350,350,350};
    const double V100_Y[LOCAL_QUBIT_SIZE]  = {350,350,350,350,350,349,349,350,350,350};
    const double V100_Z[LOCAL_QUBIT_SIZE]  = {194,194,194,194,194,194,194,194,194,194};
    const double V100_S[LOCAL_QUBIT_SIZE]  = {209,209,209,209,209,209,209,209,209,209};
    const double V100_SDG[LOCAL_QUBIT_SIZE]  = {209,209,209,209,209,209,209,209,209,209}; // TODO
    const double V100_T[LOCAL_QUBIT_SIZE]  = {216,216,216,216,216,216,217,216,216,216};
    const double V100_TDG[LOCAL_QUBIT_SIZE]  = {216,216,216,216,216,216,217,216,216,216}; // TODO
    const double V100_RX[LOCAL_QUBIT_SIZE] = {370,370,370,370,370,370,370,370,370,370};
    const double V100_RY[LOCAL_QUBIT_SIZE] = {367,367,367,367,367,367,367,367,367,367};
    const double V100_RZ[LOCAL_QUBIT_SIZE] = {369,369,369,369,369,369,369,369,369,369};

    const double V100_CN[LOCAL_QUBIT_SIZE][LOCAL_QUBIT_SIZE] = {
        0,213,195,345,193,193,193,193,193,193,
        193,0,193,193,345,193,193,193,193,193,
        193,193,0,193,193,345,193,193,193,193,
        345,193,193,0,193,193,193,193,193,193,
        193,345,193,193,0,193,193,193,193,193,
        193,193,345,193,193,0,193,193,193,193,
        193,193,193,193,193,193,0,193,193,193,
        193,193,193,193,193,193,193,0,193,193,
        193,193,193,193,193,193,193,193,0,193,
        193,193,193,193,193,193,193,193,193,0,
    };
    const double V100_CY[LOCAL_QUBIT_SIZE][LOCAL_QUBIT_SIZE] = {
        0,193,193,346,193,193,193,193,193,193,
        193,0,193,193,346,193,193,193,193,193,
        193,193,0,193,193,345,193,193,193,193,
        346,193,193,0,193,193,192,193,193,193,
        193,345,193,193,0,193,193,193,193,193,
        193,193,345,193,193,0,193,193,193,193,
        193,193,193,193,193,193,0,193,192,193,
        193,193,193,193,193,193,193,0,193,193,
        193,193,193,193,193,192,193,193,0,193,
        193,193,192,193,193,192,193,193,193,0,
    };
    const double V100_CZ[LOCAL_QUBIT_SIZE][LOCAL_QUBIT_SIZE] = {
        0,137,137,191,137,137,137,137,137,137,
        137,0,137,137,190,137,137,137,137,137,
        137,137,0,137,137,191,137,137,137,137,
        190,137,137,0,137,137,137,137,137,137,
        137,190,137,137,0,137,137,137,137,137,
        137,137,191,137,137,0,137,137,137,137,
        137,137,137,137,137,137,0,137,137,137,
        137,137,137,137,137,137,137,0,137,137,
        137,137,137,137,137,137,137,137,0,137,
        137,137,137,137,137,137,137,137,137,0,
    };
    const double V100_CRX[LOCAL_QUBIT_SIZE][LOCAL_QUBIT_SIZE] = {
        0,224,224,358,224,224,223,224,224,224,
        224,0,224,224,358,224,224,224,223,224,
        224,224,0,224,224,358,223,224,224,223,
        358,224,223,0,224,223,224,223,223,224,
        223,358,224,224,0,224,224,223,223,224,
        223,223,358,224,224,0,223,224,224,224,
        224,223,223,223,224,224,0,224,224,224,
        224,224,224,224,224,224,223,0,224,223,
        224,224,224,224,224,224,224,223,0,224,
        224,224,224,224,224,224,224,224,223,0,
    };
    const double V100_CRY[LOCAL_QUBIT_SIZE][LOCAL_QUBIT_SIZE] = {
        0,225,225,356,225,225,225,225,225,225,
        225,0,225,225,356,225,224,225,225,225,
        225,225,0,225,224,356,225,225,225,225,
        356,225,225,0,225,225,225,225,224,225,
        225,356,225,225,0,225,224,225,225,225,
        225,225,356,225,225,0,225,225,225,225,
        225,225,225,225,225,224,0,225,225,225,
        225,225,225,225,224,225,225,0,225,225,
        225,225,225,225,225,225,225,225,0,225,
        225,225,225,225,225,225,225,225,225,0,
    };
    const double V100_CU1[LOCAL_QUBIT_SIZE][LOCAL_QUBIT_SIZE] = {
        // FIXME
        0,225,225,356,225,225,225,225,225,225,
        225,0,225,225,356,225,224,225,225,225,
        225,225,0,225,224,356,225,225,225,225,
        356,225,225,0,225,225,225,225,224,225,
        225,356,225,225,0,225,224,225,225,225,
        225,225,356,225,225,0,225,225,225,225,
        225,225,225,225,225,224,0,225,225,225,
        225,225,225,225,224,225,225,0,225,225,
        225,225,225,225,225,225,225,225,0,225,
        225,225,225,225,225,225,225,225,225,0,
    };
    const double V100_CRZ[LOCAL_QUBIT_SIZE][LOCAL_QUBIT_SIZE] = {
        0,224,224,359,224,224,224,224,224,224,
        224,0,224,224,359,224,224,224,224,224,
        224,224,0,224,224,359,224,224,224,224,
        359,224,224,0,224,224,224,224,224,224,
        224,359,224,224,0,224,224,224,224,224,
        224,224,359,224,224,0,224,224,224,224,
        224,224,224,224,224,224,0,224,224,224,
        224,224,224,224,224,224,224,0,224,224,
        224,224,224,224,224,224,224,224,0,224,
        224,224,224,224,224,224,224,224,224,0,
    };

    // pergate single gate performance for 512 runs with 28 qbits
    double pergate_single_perf[MAX_QBITS + 1][GATE_NUM][LOCAL_QUBIT_SIZE];
    // pergate control gate performance for 512 runs with 28 qbits
    double pergate_ctr_perf[MAX_QBITS + 1][GATE_NUM][LOCAL_QUBIT_SIZE][LOCAL_QUBIT_SIZE];
    // overhead of one pergate group
    double BLAS_perf[MAX_QBITS + 1][MAX_QBITS + 1];
    double cutt_cost[MAX_QBITS + 1];
    bool num_qbits_loaded_param[MAX_QBITS + 1];
    const double pergate_group_overhead = 1.0 / (1 << 27);

    int param_type;

    Evaluator();
    
    static Evaluator* instance_ptr;
public:
    static Evaluator* getInstance() {
        if(instance_ptr == nullptr) {
            instance_ptr = new Evaluator;
        }
        return instance_ptr;
    }
    void loadPergateSingle(int numQubits, FILE* qbit_param, GateType gate_type);
    void loadPergateCtr(int numQubits, FILE* qbit_param, GateType gate_type);
    void loadParam(int numQubits);
    double perfPerGate(int numQubits, const GateGroup* gg);
    double perfPerGate(int numQubits, const std::vector<GateType>& types);
    double perfBLAS(int numQubits, int blasSize);
    // return True if choose pergate over BLAS
    bool PerGateOrBLAS(const GateGroup* gg_pergate, const GateGroup* gg_blas, int numQubits, int blasSize);
};
