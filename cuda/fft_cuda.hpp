//
// Created by mrs on 27.03.22.
//

#ifndef FFT_CUDA_FFT_CUDA_HPP
#define FFT_CUDA_FFT_CUDA_HPP

#include <cuComplex.h>

using cf = cuFloatComplex;

/*!
 * Compute FFT using CUDA
 * @param arr Array to compute FFT on. The result will be stored in it too
 * @param N Number of elements in the array
 */
void fft_cuda(cf* arr, int N);

#endif //FFT_CUDA_FFT_CUDA_HPP
