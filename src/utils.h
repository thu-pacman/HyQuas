#pragma once

#include <cstdio>

typedef double qreal;
typedef int qindex;
const int LOCAL_QUBIT_SIZE = 10; // TODO find the best value

struct Complex {
    qreal real;
    qreal imag;

    Complex() = default;
    Complex(qreal x): real(x), imag(0) {}
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

template<typename T>
int bitCount(T x) {
    int ret = 0;
    for (T i = x; i; i -= i & (-i)) {
        ret++;
    }
    return ret;
}