#include "QuEST.h"
#include <cmath>

void controlledNot(Qureg& q, int controlQubit, int targetQubit) {
    Gate g;
    g.type = GateCNot;
    g.mat[0][0] = 0; g.mat[0][1] = 1;
    g.mat[1][0] = 1; g.mat[1][1] = 0;
    g.name = "CN";
    g.targetQubit = targetQubit;
    g.controlQubit = controlQubit;
    q.addGate(g);
}

void controlledPauliY(Qureg& q, int controlQubit, int targetQubit) {
    assert(controlQubit != targetQubit);
    assert(false);
}

void controlledRotateX(Qureg& q, int controlQubit, int targetQubit, qreal angle) {
    assert(controlQubit != targetQubit);
    assert(false);
}

void controlledRotateY(Qureg& q, int controlQubit, int targetQubit, qreal angle) {
    assert(controlQubit != targetQubit);
    assert(false);
}

void controlledRotateZ(Qureg& q, int controlQubit, int targetQubit, qreal angle) {
    assert(controlQubit != targetQubit);
    Gate g;
    g.type = GateCRotateZ;
    g.mat[0][0] = Complex(cos(angle/2), -sin(angle/2)); g.mat[0][1] = 0;
    g.mat[1][0] = 0; g.mat[1][1] = Complex(cos(angle/2), sin(angle/2));
    g.name = "CN";
    g.targetQubit = targetQubit;
    g.controlQubit = controlQubit;
    q.addGate(g);
}

void hadamard(Qureg& q, int targetQubit) {
    Gate g;
    g.type = GateHadamard;
    g.mat[0][0] = 1/sqrt(2); g.mat[0][1] = 1/sqrt(2);
    g.mat[1][0] = 1/sqrt(2); g.mat[1][1] = -1/sqrt(2);
    g.name = "H";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    q.addGate(g);
}

void pauliX(Qureg& q, int targetQubit) {
    assert(false);
}

void pauliY(Qureg& q, int targetQubit) {
    assert(false);
}

void pauliZ(Qureg& q, int targetQubit) {
    assert(false);
}

void rotateX(Qureg& q, int targetQubit, qreal angle) {
    assert(false);
}

void rotateY(Qureg& q, int targetQubit, qreal angle) {
    assert(false);
}

void rotateZ(Qureg& q, int targetQubit, qreal angle) {
    assert(false);
}

void sGate(Qureg& q, int targetQubit) {
    assert(false);    
}

void tGate(Qureg& q, int targetQubit) {
    assert(false);
}
