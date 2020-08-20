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
    for (int i = 0; i < LOCAL_QUBIT_SIZE; i++) {
        for (int j = 0; j < LOCAL_QUBIT_SIZE; j++) {
            if (i == j) { printf("    "); continue; }
            Circuit c(n);
            for (int k = 0; k < num_gates; k++) {
                c.addGate(Gate::CNOT(i, j));
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