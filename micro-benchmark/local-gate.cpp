#include <assert.h>
#include <fstream>
#include <cstring>
#include <regex>
#include <cmath>
#include "circuit.h"
#include "logger.h"
using namespace std;

int main(int argc, char* argv[]) {
    int n = 28;
    int num_gates = 512;
    for (int i = 0; i < int(GateType::TOTAL); i++) {
        Circuit c(n);
        for (int j = 0; j < num_gates; j++) {
            c.addGate(Gate::random(0, LOCAL_QUBIT_SIZE, GateType(i)));
        }
        c.compile();
        int time = c.run(false);
        printf("%s: %d ms\n", Gate::get_name(GateType(i)).c_str(), time);
        fflush(stdout);
    }
    return 0;
}