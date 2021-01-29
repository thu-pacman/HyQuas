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
    for (int i = 12; i < 150; i += 12) {
        printf("%d:", i);
        for (int tt = 0; tt < 5; tt++) {
            Circuit c(28);
            for (int j = 0; j < i; j++)
                c.addGate(Gate::H(j % 12));
            c.compile();
            int time = c.run(false);
            printf("%d ", time);
        }
        printf("\n");
    }
    return 0;
}