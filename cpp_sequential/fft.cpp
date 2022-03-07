#include "fft.h"

using cf = std::complex<float>;
using cfv = std::vector<cf>;

void dft(cfv &x) {
    std::vector<std::complex<float>> res;
    const std::complex<float> I(0, 1);
    size_t N = x.size();
    for (size_t k = 0; k < N; k++) {
        cf sum = 0;
        for (int n = 0; n < N; n++) {
            sum += x[n] * std::exp(cf(0, 1) * static_cast<float>(2 * M_PI * k * n / N));
        }
        res.push_back(sum);
    }
    x = res;
}

void fft(cfv &x) {
    size_t n = x.size();
    if (n == 1) {
        return;
    }

    cfv x0(n / 2), x1(n / 2);
    for (size_t i = 0; 2 * i < n; i++) {
        x0[i] = x[2 * i];
        x1[i] = x[2 * i + 1];
    }
    fft(x0);
    fft(x1);

    float ang = 2.0f * M_PI / n;
    cf w(1), wn(std::cos(ang), std::sin(ang));
    for (size_t i = 0; 2 * i < n; i++) {
        x[i] = x0[i] + w * x1[i];
        x[i + n / 2] = x0[i] - w * x1[i];
        w *= wn;
    }
}