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
    Gate g;
    g.type = GateCPauliY;
    g.mat[0][0] = 0; g.mat[0][1] = Complex(0, -1);
    g.mat[1][0] = Complex(0, 1); g.mat[1][1] = Complex(0, 0);
    g.name = "CPY";
    g.targetQubit = targetQubit;
    g.controlQubit = controlQubit;
    q.addGate(g);
}

void controlledRotateX(Qureg& q, int controlQubit, int targetQubit, qreal angle) {
    assert(controlQubit != targetQubit);
    Gate g;
    g.type = GateCRotateX;
    g.mat[0][0] = cos(angle/2.0); g.mat[0][1] = Complex(0, sin(angle/2.0));
    g.mat[1][0] = Complex(0, -sin(angle/2.0)); g.mat[1][1] = cos(angle/2.0);
    g.name = "CX";
    g.targetQubit = targetQubit;
    g.controlQubit = controlQubit;
    q.addGate(g);
}

void controlledRotateY(Qureg& q, int controlQubit, int targetQubit, qreal angle) {
    assert(controlQubit != targetQubit);
    Gate g;
    g.type = GateCRotateY;
    g.mat[0][0] = cos(angle/2.0); g.mat[0][1] = -sin(angle/2.0);
    g.mat[1][0] = sin(angle/2.0); g.mat[1][1] = cos(angle/2.0);
    g.name = "CY";
    g.targetQubit = targetQubit;
    g.controlQubit = controlQubit;
    q.addGate(g);
}

void controlledRotateZ(Qureg& q, int controlQubit, int targetQubit, qreal angle) {
    assert(controlQubit != targetQubit);
    Gate g;
    g.type = GateCRotateZ;
    g.mat[0][0] = Complex(cos(angle/2), -sin(angle/2)); g.mat[0][1] = 0;
    g.mat[1][0] = 0; g.mat[1][1] = Complex(cos(angle/2), sin(angle/2));
    g.name = "CZ";
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
    Gate g;
    g.type = GatePauliX;
    g.mat[0][0] = 0; g.mat[0][1] = 1;
    g.mat[1][0] = 1; g.mat[1][1] = 0;
    g.name = "PX";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    q.addGate(g);
}

void pauliY(Qureg& q, int targetQubit) {
    Gate g;
    g.type = GatePauliY;
    g.mat[0][0] = 0; g.mat[0][1] = Complex(0, -1);
    g.mat[1][0] = Complex(0, 1); g.mat[1][1] = Complex(0, 0);
    g.name = "PY";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    q.addGate(g);
}

void pauliZ(Qureg& q, int targetQubit) {
    Gate g;
    g.type = GatePauliZ;
    g.mat[0][0] = 1; g.mat[0][1] = 0;
    g.mat[1][0] = 0; g.mat[1][1] = -1;
    g.name = "PZ";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    q.addGate(g);
}

void rotateX(Qureg& q, int targetQubit, qreal angle) {
    Gate g;
    g.type = GateRotateX;
    g.mat[0][0] = cos(angle/2.0); g.mat[0][1] = Complex(0, sin(angle/2.0));
    g.mat[1][0] = Complex(0, -sin(angle/2.0)); g.mat[1][1] = cos(angle/2.0);
    g.name = "X";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    q.addGate(g);
}

void rotateY(Qureg& q, int targetQubit, qreal angle) {
    Gate g;
    g.type = GateRotateY;
    g.mat[0][0] = cos(angle/2.0); g.mat[0][1] = -sin(angle/2.0);
    g.mat[1][0] = sin(angle/2.0); g.mat[1][1] = cos(angle/2.0);
    g.name = "Y";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    q.addGate(g);
}

void rotateZ(Qureg& q, int targetQubit, qreal angle) {
    Gate g;
    g.type = GateRotateZ;
    g.mat[0][0] = Complex(cos(angle/2), -sin(angle/2)); g.mat[0][1] = 0;
    g.mat[1][0] = 0; g.mat[1][1] = Complex(cos(angle/2), sin(angle/2));
    g.name = "Z";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    q.addGate(g);
}

void sGate(Qureg& q, int targetQubit) {
    Gate g;
    g.type = GateS;
    g.mat[0][0] = 1; g.mat[0][1] = 0;
    g.mat[1][0] = 0; g.mat[1][1] = Complex(0, 1);
    g.name = "S";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    q.addGate(g); 
}

void tGate(Qureg& q, int targetQubit) {
    Gate g;
    g.type = GateT;
    g.mat[0][0] = 1; g.mat[0][1] = 0;
    g.mat[1][0] = 0; g.mat[1][1] = Complex(1/sqrt(2), 1/sqrt(2));
    g.name = "T";
    g.targetQubit = targetQubit;
    g.controlQubit = -1;
    q.addGate(g); 
}
