#include <assert.h>
#include <fstream>
#include <cstring>
#include <regex>
#include <cmath>
#include "circuit.h"
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

std::pair<std::string, std::vector<qreal>> parse_gate(char buf[]) {
    qreal pi = acos(-1);
    int l = strlen(buf);
    std::string name;
    int i = 0;
    while (i < l) {
        if (buf[i] != '(')
            name += buf[i];
        else
            break;
        i++;
    }
    std::vector<qreal> params;
    while (i < l) {
        i++;
        std::string st;
        while (buf[i] != ',' && buf[i] != ')') {
            st += buf[i];
            i++;
        }
        qreal param = 1;
        if (st[0] == 'p' && st[1] == 'i' && st[2] == '*') {
            param = pi;
            st = st.erase(0, 3);
        }
        param *= std::stod(st);
        params.push_back(param);
        if (buf[i] == ')')
            break;
    }
    return std::make_pair(name, params);
}

void show(Circuit* c, qindex idx) {
    Complex x = c->ampAt(idx);
    printf("%d %.12f: %.12f %.12f\n", idx, x.real * x.real + x.imag * x.imag, x.real, x.imag);
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
            printf("%d qubits\n", n); fflush(stdout);
            c = new Circuit(n);
        } else if (strcmp(buffer, "cx") == 0) {
            fscanf(f, "%s", buffer);
            auto qid = parse_qid(buffer);
            assert(qid.size() == 2);
            c->addGate(Gate::CNOT(qid[0], qid[1]));
            // printf("cx %d %d\n", qid[0], qid[1]);
        } else if (strcmp(buffer, "ccx") == 0) {
            fscanf(f, "%s", buffer);
            auto qid = parse_qid(buffer);
            assert(qid.size() == 3);
            c->addGate(Gate::CNOT(qid[1], qid[2]));
            printf("warning: ccx -> cx\n");
            // printf("ccx %d %d %d\n", qid[0], qid[1], qid[2]);
        } else if (strcmp(buffer, "cz") == 0) {
            fscanf(f, "%s", buffer);
            auto qid = parse_qid(buffer);
            assert(qid.size() == 2);
            c->addGate(Gate::CZ(qid[0], qid[1]));
            // printf("cz %d %d\n", qid[0], qid[1]);
        } else if (strcmp(buffer, "h") == 0) {
            fscanf(f, "%s", buffer);
            auto qid = parse_qid(buffer);
            assert(qid.size() == 1);
            c->addGate(Gate::H(qid[0]));
            // printf("h %d\n", qid[0]);
        } else if (strcmp(buffer, "x") == 0) {
            fscanf(f, "%s", buffer);
            auto qid = parse_qid(buffer);
            assert(qid.size() == 1);
            c->addGate(Gate::X(qid[0]));
            // pritnf("x %d\n", qid[0]);
        } else {
            auto gate = parse_gate(buffer);
            if (gate.first == "u1") {
                assert(gate.second.size() == 1);
                fscanf(f, "%s", buffer);
                auto qid = parse_qid(buffer);
                assert(qid.size() == 1);
                c->addGate(Gate::U1(qid[0], gate.second[0]));
                // printf("rx %d %f\n", qid[0], gate.second[0]);
            } else if (gate.first == "u3") {
                assert(gate.second.size() == 3);
                fscanf(f, "%s", buffer);
                auto qid = parse_qid(buffer);
                assert(qid.size() == 1);
                c->addGate(Gate::U3(qid[0], gate.second[0], gate.second[1], gate.second[2]));
                // printf("u3 %d %f %f %f\n", qid[0], gate.second[0], gate.second[1], gate.second[2]);
            } else if (gate.first == "rx") {
                assert(gate.second.size() == 1);
                fscanf(f, "%s", buffer);
                auto qid = parse_qid(buffer);
                assert(qid.size() == 1);
                c->addGate(Gate::RX(qid[0], gate.second[0]));
                // printf("rx %d %f\n", qid[0], gate.second[0]);
            } else if (gate.first == "ry") {
                assert(gate.second.size() == 1);
                fscanf(f, "%s", buffer);
                auto qid = parse_qid(buffer);
                assert(qid.size() == 1);
                c->addGate(Gate::RY(qid[0], gate.second[0]));
                // printf("ry %d %f\n", qid[0], gate.second[0]);
            } else if (gate.first == "rz") {
                assert(gate.second.size() == 1);
                fscanf(f, "%s", buffer);
                auto qid = parse_qid(buffer);
                assert(qid.size() == 1);
                c->addGate(Gate::RZ(qid[0], gate.second[0]));
                // printf("rz %d %f\n", qid[0], gate.second[0]);
            } else {
                printf("unrecognized token %s\n", buffer);
                exit(1);
            }
        }
        fgets(buffer, BUFFER_SIZE, f);
    }
    c->compile();
    c->run();
    for (int i = 0; i < 128; i++) {
        show(c, i);
    }
    return 0;
}