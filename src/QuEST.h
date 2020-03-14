#pragma once
#include <assert.h>

#include "QuESTEnv.h"
#include "Qureg.h"
#include "utils.h"

// in QuESTEnv.cpp
QuESTEnv createQuESTEnv();
void destroyQuESTEnv(QuESTEnv& env);

// in Qureg.cpp
Qureg createQureg(int numQubits, QuESTEnv& env);
void destroyQureg(Qureg& q, QuESTEnv& env);
qreal calcProbOfOutcome(Qureg& q, int measureQubit, int outcome);
Complex getAmp(Qureg& q, long long int index);

// in gates.cpp
void controlledNot(Qureg& q, int controlQubit, int targetQubit);
void controlledPauliY(Qureg& q, int controlQubit, int targetQubit);
void controlledRotateX(Qureg& q, int controlQubit, int targetQubit, qreal angle);
void controlledRotateY(Qureg& q, int controlQubit, int targetQubit, qreal angle);
void controlledRotateZ(Qureg& q, int controlQubit, int targetQubit, qreal angle);
void hadamard(Qureg& q, int targetQubit);
void pauliX(Qureg& q, int targetQubit);
void pauliY(Qureg& q, int targetQubit);
void pauliZ(Qureg& q, int targetQubit);
void rotateX(Qureg& q, int targetQubit, qreal angle);
void rotateY(Qureg& q, int targetQubit, qreal angle);
void rotateZ(Qureg& q, int targetQubit, qreal angle);
void sGate(Qureg& q, int targetQubit);
void tGate(Qureg& q, int targetQubit);