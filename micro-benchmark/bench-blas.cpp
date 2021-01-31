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
    printf("MATSIZE %d ", BLAS_MAT_LIMIT);
    for (int tt = 0; tt < 5; tt++) {
        Circuit c(n);
        for (int i = 0; i < 10 * BLAS_MAT_LIMIT; i++) {
            c.addGate(Gate::H(i % (BLAS_MAT_LIMIT * 2)));
        }
        c.compile();
        int time = c.run(false);
        printf("%d ", time);
    }
    printf("\n");
    return 0;
}