#ifndef CPP_SEQUENTIAL_FFT_H
#define CPP_SEQUENTIAL_FFT_H

#include <vector>
#include <complex>

//! Direct computation of Fourier transform for results check.
//! Result is placed in the same vector x
//! \param x vector of data in time/space domain that will be transformed
void dft(std::vector<std::complex<float>> &x);

//! Fast Fourier Transform
//! Result is placed in the same vector x
//! \param x vector of data in time/space domain that will be transformed
void fft(std::vector<std::complex<float>> &x);


void fft_iter(std::vector<std::complex<float>> &x);


#endif //CPP_SEQUENTIAL_FFT_H
