#pragma once

typedef double qreal;
typedef int qindex;
const int LOCAL_QUBIT_SIZE = 10; // TODO find the best value

struct Complex {
    qreal real;
    qreal imag;

    Complex() = default;
    Complex(qreal real, qreal imag): real(real), imag(imag) {}
    Complex(const Complex&) = default;
    Complex& operator = (qreal x) {
        real = x;
        imag = 0;
        return *this;
    }
};

struct ComplexArray {
    qreal *real;
    qreal *imag;
};