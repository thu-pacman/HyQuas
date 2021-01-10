#include <assert.h>
#include <fstream>
#include <cstring>
#include <regex>
#include <cmath>
#include "circuit.h"
#include "logger.h"
using namespace std;

int main(int argc, char* argv[]) {
    MyGlobalVars::init();
    int n = 28;
    int num_gates = 512;
    for (int i = int(GateType::U1); i < int(GateType::TOTAL); i++) {
        printf("%s: ", Gate::get_name(GateType(i)).c_str());
        for (int j = 0; j < LOCAL_QUBIT_SIZE; j++) {
            Circuit c(n);
            for (int k = 0; k < num_gates; k++) {
                c.addGate(Gate::random(j, j + 1, GateType(i)));
            }
            c.compile();
            int time = c.run(false);
            printf("%d ", time);
            fflush(stdout);
        }
        printf("\n");
    }
    return 0;
}