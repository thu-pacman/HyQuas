#pragma once

#include <string>
#include <vector>
#include "utils.h"
#include "gate.h"
#include "schedule.h"

struct ResultItem {
    ResultItem() = default;
    ResultItem(const qindex& idx, const qComplex& amp): idx(idx), amp(amp) {}
    qindex idx;
    qComplex amp;
    void print() {
        printf("%lld %.12f: %.12f %.12f\n", idx, amp.x * amp.x + amp.y * amp.y, zero_wrapper(amp.x), zero_wrapper(amp.y));
    }
    bool operator < (const ResultItem& b) { return idx < b.idx; }
};

class Circuit {
public:
    Circuit(int numQubits);
    ~Circuit();

    void addGate(const Gate& gate);
    void printState();
    qComplex ampAtGPU(qindex idx);
    const int numQubits;
    qreal measure(int qb);

#ifdef IMPERATIVE
private:
    void applyGates();
#endif
    ResultItem ampAt(qindex idx);
    void dumpGates();
    bool localAmpAt(qindex idx, ResultItem& item);
    void compile();
    int run(bool copy_back = true, bool destroy = true);

private:
    qindex toPhysicalID(qindex idx);
    qindex toLogicID(qindex idx);
    void masterCompile();
#if USE_MPI
    void gatherAndPrint(const std::vector<ResultItem>& results);
#endif
    std::vector<Gate> gates;
    std::vector<qComplex*> deviceStateVec;
    std::vector<std::vector<qComplex*>> deviceMats;
    Schedule schedule;
    std::vector<qComplex> result;
    enum class State {dirty, empty, measured} state;
    std::vector<qreal> measureResults;
    bool destroyed;
};