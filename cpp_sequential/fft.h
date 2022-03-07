//
// Created by mrs on 07.03.22.
//

#ifndef CPP_SEQUENTIAL_FFT_H
#define CPP_SEQUENTIAL_FFT_H

#include <vector>
#include <complex>

//! Direct computation of Fourier transform for results check
//! \param x vector of data in time/space domain
//! \return Vector of complex numbers in the frequency domain
std::vector<std::complex<float>> dft(const std::vector<float> &x);

//! Fast Fourier Transform algorithm
//! \param x vector of data in time/space domain
//! \return Vector of complex numbers in the frequency domain
std::vector<std::complex<float>> fft(const std::vector<float> &x);


#endif //CPP_SEQUENTIAL_FFT_H
