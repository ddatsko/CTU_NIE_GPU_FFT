#include "fft.h"
#include "utils.hpp"
#include <iostream>

using cf = std::complex<float>;
using vcf = std::vector<cf>;

void dft(vcf &x) {
    std::vector<std::complex<float>> res;
    const size_t N = x.size();
    for (size_t k = 0; k < N; k++) {
        cf sum = 0;
        for (int n = 0; n < N; n++) {
            sum += x[n] * std::exp(cf(0, 1) * static_cast<float>(2 * M_PI * k * n / N));
        }
        res.push_back(sum);
    }
    x = res;
}

void fft(vcf &x) {
    const size_t n = x.size();
    if (n == 1) {
        return;
    }

    vcf x0(n / 2), x1(n / 2);
    for (size_t i = 0; 2 * i < n; i++) {
        x0[i] = x[2 * i];
        x1[i] = x[2 * i + 1];
    }
    fft(x0);
    fft(x1);

    const float ang = 2.0f * static_cast<float>(M_PI) / static_cast<float>(n);
    cf w(1), wn(std::cos(ang), std::sin(ang));
    for (size_t i = 0; 2 * i < n; i++) {
        x[i] = x0[i] + w * x1[i];
        x[i + n / 2] = x0[i] - w * x1[i];
        w *= wn;
    }
}


uint32_t reverseBits(uint32_t i) {
    uint32_t mask = 0x55555555; // 0101...
    i = ((i & mask) << 1) | ((i >> 1) & mask);
    mask = 0x33333333; // 0011...
    i = ((i & mask) << 2) | ((i >> 2) & mask);
    mask = 0x0f0f0f0f; // 00001111...
    i = ((i & mask) << 4) | ((i >> 4) & mask);
    mask = 0x00ff00ff; // 0000000011111111...
    i = ((i & mask) << 8) | ((i >> 8) & mask);
    // 00000000000000001111111111111111 no need for mask
    i = (i << 16) | (i >> 16);
    return i;
}

void fft_iter(vcf &a) {
    int n = a.size();

    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;

        if (i < j)
            swap(a[i], a[j]);
    }

    for (int len = 2; len <= n; len <<= 1) {
        float ang = 2 * M_PI / len;
        cf wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            cf w(1);
            for (int j = 0; j < len / 2; j++) {
                cf u = a[i+j], v = a[i+j+len/2] * w;
                a[i+j] = u + v;
                a[i+j+len/2] = u - v;
                w *= wlen;
            }
        }
    }
}