#include "circuit.h"
#include <assert.h>
#include <fstream>
#include <cstring>
using namespace std;
const int BUFFER_SIZE = 1000;
char buffer[BUFFER_SIZE];

std::vector<int> parse_qid(char buf[]) {
    std::vector<int> ret;
    int l = strlen(buf);
    for (int i = 0; i < l; i++) {
        if (buf[i] >= '0' && buf[i] <= '9') {
            int j = i, x = 0;
            while (buf[j] >= '0' && buf[j] <= '9') {
                x = x * 10 + (int)(buf[j] - '0');
                j++;
            }
            i = j - 1;
            ret.push_back(x);
        }
    }
    return ret;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("./parser qasmfile\n");
        exit(1);
    }
    printf("%s\n", argv[1]);
    FILE* f = fopen(argv[1], "r");
    int n = -1;
    Circuit* c;
    while (fscanf(f, "%s", buffer) != EOF) {
        if (strcmp(buffer, "//") == 0 || strcmp(buffer, "OPENQASM") == 0 || strcmp(buffer, "include") == 0) {
        } else if (strcmp(buffer, "qreg") == 0) {
            fscanf(f, "%*c%*c%*c%d", &n);
            printf("%d qubits\n", n);
            c = new Circuit(n);
        } else if (strcmp(buffer, "cx") == 0) {
            fscanf(f, "%s", buffer);
            auto qid = parse_qid(buffer);
            assert(qid.size() == 2);
            printf("cx %d %d\n", qid[0], qid[1]);
        } else if (strcmp(buffer, "ccx") == 0) {
            fscanf(f, "%s", buffer);
            auto qid = parse_qid(buffer);
            assert(qid.size() == 3);
            printf("ccx %d %d %d\n", qid[0], qid[1], qid[2]);
        } else {
            printf("unrecognized token %s\n", buffer);
            exit(1);
        }
        fgets(buffer, BUFFER_SIZE, f);
    }
    // c.compile();
    // c.run();
    return 0;
}