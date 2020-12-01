#include <assert.h>
#include <fstream>
#include <cstring>
#include <regex>
#include <cmath>
#include "circuit.h"
#include "logger.h"
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

qreal zero_wrapper(qreal x) {
    const qreal eps = 1e-14;
    if (x > -eps && x < eps) {
        return 0;
    } else {
        return x;
    }
}

void show(std::unique_ptr<Circuit>& c, qindex idx) {
    Complex x = c->ampAt(idx);
    printf("%d %.12f: %.12f %.12f\n", idx, x.real * x.real + x.imag * x.imag, zero_wrapper(x.real), zero_wrapper(x.imag));
}

void conditionShow(std::unique_ptr<Circuit>& c, qindex idx) {
    Complex x = c->ampAt(idx);
    if (x.len() > 0.001) 
        printf("%d %.12f: %.12f %.12f\n", idx, x.real * x.real + x.imag * x.imag, zero_wrapper(x.real), zero_wrapper(x.imag));
}

std::unique_ptr<Circuit> parse_circuit(const std::string &filename) {
    FILE* f = nullptr;
    if ((f = fopen(filename.c_str(), "r")) == NULL) {
        printf("fail to open %s\n", filename.c_str());
        exit(1);
    }
    int n = -1;
    std::unique_ptr<Circuit> c = nullptr;
    while (fscanf(f, "%s", buffer) != EOF) {
        if (strcmp(buffer, "//") == 0 || strcmp(buffer, "OPENQASM") == 0 || strcmp(buffer, "include") == 0) {
        } else if (strcmp(buffer, "qreg") == 0) {
            fscanf(f, "%*c%*c%*c%d", &n);
            c = std::make_unique<Circuit>(n);
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
            c->addGate(Gate::CCX(qid[0], qid[1], qid[2]));
            // printf("ccx %d %d %d\n", qid[0], qid[1], qid[2]);
        } else if (strcmp(buffer, "cy") == 0) {
            fscanf(f, "%s", buffer);
            auto qid = parse_qid(buffer);
            assert(qid.size() == 2);
            c->addGate(Gate::CY(qid[0], qid[1]));
            // printf("cy %d %d\n", qid[0], qid[1]);
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
            // printf("x %d\n", qid[0]);
        } else if (strcmp(buffer, "y") == 0) {
            fscanf(f, "%s", buffer);
            auto qid = parse_qid(buffer);
            assert(qid.size() == 1);
            c->addGate(Gate::Y(qid[0]));
            // printf("y %d\n", qid[0]);
        } else if (strcmp(buffer, "z") == 0) {
            fscanf(f, "%s", buffer);
            auto qid = parse_qid(buffer);
            assert(qid.size() == 1);
            c->addGate(Gate::Z(qid[0]));
            // printf("z %d\n", qid[0]);
        } else if (strcmp(buffer, "s") == 0) {
            fscanf(f, "%s", buffer);
            auto qid = parse_qid(buffer);
            assert(qid.size() == 1);
            c->addGate(Gate::S(qid[0]));
            // printf("s %d\n", qid[0]);
        } else if (strcmp(buffer, "t") == 0) {
            fscanf(f, "%s", buffer);
            auto qid = parse_qid(buffer);
            assert(qid.size() == 1);
            c->addGate(Gate::T(qid[0]));
            // printf("t %d\n", qid[0]);
        } else {
            auto gate = parse_gate(buffer);
            if (gate.first == "crx") {
                assert(gate.second.size() == 1);
                fscanf(f, "%s", buffer);
                auto qid = parse_qid(buffer);
                assert(qid.size() == 2);
                c->addGate(Gate::CRX(qid[0], qid[1], gate.second[0]));
                // printf("crx %d %d %f\n", qid[0], qid[1], gate.second[0]);
            } else if (gate.first == "cry") {
                assert(gate.second.size() == 1);
                fscanf(f, "%s", buffer);
                auto qid = parse_qid(buffer);
                assert(qid.size() == 2);
                c->addGate(Gate::CRY(qid[0], qid[1], gate.second[0]));
                // printf("cry %d %d %f\n", qid[0], qid[1], gate.second[0]);
            } else if (gate.first == "crz") {
                assert(gate.second.size() == 1);
                fscanf(f, "%s", buffer);
                auto qid = parse_qid(buffer);
                assert(qid.size() == 2);
                c->addGate(Gate::CRZ(qid[0], qid[1], gate.second[0]));
                // printf("crz %d %d %f\n", qid[0], qid[1], gate.second[0]);
            } else if (gate.first == "u1") {
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
    fclose(f);
    if (c == nullptr) {
        printf("fail to load circuit\n");
        exit(1);
    }
    return std::move(c);
}

int main(int argc, char* argv[]) {
    MyMPI::init();
    std::unique_ptr<Circuit> c;
    if (argc != 2) {
        printf("./parser qasmfile\n");
        exit(1);
    }
    c = parse_circuit(std::string(argv[1]));
    c->compile();
    // c->run();
    // if (MyMPI::rank == 0) {
    //     for (int i = 0; i < 128; i++) {
    //         show(c, i);
    //     }
    //     for (int i = 128; i < (1 << n); i++) {
    //         conditionShow(c, i);
    //     }
    //     Logger::print();
    // }
    MPI_Finalize();
    return 0;
}