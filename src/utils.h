#pragma once

typedef double qreal;

struct Complex {
    qreal real;
    qreal imag;

    Complex(qreal real, qreal imag): real(real), imag(imag) {}
    Complex(const Complex&) = default;
};
