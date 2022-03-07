#include "fft.h"

std::vector<std::complex<float>> dft(const std::vector<float> &x) {
    std::vector<std::complex<float>> res;
    const std::complex<float> I(0, 1);
    size_t N = x.size();
    for (size_t k = 0; k < N; k++) {
        std::complex<float> sum = 0;
        for (int n = 0; n < N; n++) {
            sum += x[n] * std::exp(std::complex<float>(0, 1) * static_cast<float>(2 * M_PI * k * n / N));
        }
        res.push_back(sum);
    }
    return res;
}

std::vector<std::complex<float>> fft(const std::vector<float> &x)  {

}
