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
    for (int n = 24; n <= 30; n++) {
        printf("Measure Time %d:\n", n);
        for (int tt = 0; tt < 5; tt++) {
            Circuit c(n);
            c.addGate(Gate::H(0));
            c.compile();
            c.run(false, false);
            c.measure(0);
            Logger::print();
        }
        printf("\n");
    }
    return 0;
}