// quest:multiControlledUnitary
// quest:multiControlledPhaseShift
// quest:multiControlledPhaseFlip

statuc Gate Unitary(int targetQubit, qComplex a0, qComplex a1, qComplex b0, qComplex b1); // quest:unitary
statuc Gate CompactUnitary(int targetQubit, qComplex alpha, qComplex beta); // quest:compactUnitary
statuc Gate ControlledUnitary(int controlQubit, int targetQubit, qComplex a0, qComplex a1, qComplex b0, qComplex b1); // quest:controlledUnitary
statuc Gate ControlledCompactUnitary(int controlQubit, int targetQubit, qComplex alpha, qComplex beta); // quest:controlledCompactUnitary


static Gate CCX(int c1, int c2, int targetQubit);
static Gate CNOT(int controlQubit, int targetQubit); // qiskit:CXGate quest:controlledNot
static Gate CY(int controlQubit, int targetQubit); // qiskit:CYGate quest:controlledPauliY
static Gate CZ(int controlQubit, int targetQubit); // qiskit:CZGate quest:controlledPhaseFlip
static Gate CRX(int controlQubit, int targetQubit, qreal angle); // qiskit:CRXGate quest:controlledRotateX
static Gate CRY(int controlQubit, int targetQubit, qreal angle); // qiskit:CRYGate quest:controlledRotateY
static Gate CU1(int controlQubit, int targetQubit, qreal lambda); // qiskit:CPhaseGate quest:controlledPhaseShift
static Gate CRZ(int controlQubit, int targetQubit, qreal angle); // qiskit:CRZGate quest:controlledRotateZ
static Gate U1(int targetQubit, qreal lambda); // qiskit:PhaseGate quest:phaseShift
static Gate U2(int targetQubit, qreal phi, qreal lambda);
static Gate U3(int targetQubit, qreal theta, qreal phi, qreal lambda);
static Gate H(int targetQubit); // qiskit:HGate quest:hadamard
static Gate X(int targetQubit); // qiskit:XGate quest:pauliX
static Gate Y(int targetQubit); // qiskit:YGate quest:pauliY
static Gate Z(int targetQubit); // qiskit:ZGate quest:pauliZ
static Gate S(int targetQubit); // qiskit:SGate quest:sGate
static Gate SDG(int targetQubit); 
static Gate T(int targetQubit); // qiskit:TGate quest:tGate
static Gate TDG(int targetQubit);
static Gate RX(int targetQubit, qreal angle); // qiskit:RXGate quest:rotateX
static Gate RY(int targetQubit, qreal angle); // qiskit:RYGate quest:rotateY
static Gate RZ(int targetQubit, qreal angle); // qiskit:RZGate quest:rotateZ
static Gate ID(int targetQubit);
static Gate GII(int targetQubit);
static Gate GTT(int targetQubit);
static Gate GZZ(int targetQubit);
static Gate GOC(int targetQubit, qreal real, qreal imag);
static Gate GCC(int targetQubit, qreal real, qreal imag);