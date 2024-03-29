#include <iostream>
#include "utils.hpp"
#include <complex>
#include "fft_cpu.h"
#include "fft_cuda.hpp"
#include <cuComplex.h>
#include <fftw3.h>

const double MAX_ABS_ERROR = 1e1;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Wrong number of arguments\nUsage: $ ./fft <input_file>" << std::endl;
        return -1;
    }
    auto points = points_to_complex(read_sequence_from_file(argv[1]));
    if (points.empty()) {
        std::cerr << "Error while reading data from file. File is empty or could not be opened" << std::endl;
        return -1;
    }
    if ((points.size() & (points.size() - 1)) != 0) {
        std::cout << "Number of input number is not a power of 2. Exiting" << std::endl;
        return -2;
    }

    auto points_dft = points;
    auto points_fft = points;
    auto points_fft_iter = points;
    std::vector<std::complex<float>> fftw_res(points.size());
    std::vector<cuFloatComplex> cuda_fft_iter;
    for (const auto &point: points) {
        cuda_fft_iter.push_back({point.real(), point.imag()});
    }


    {
        const size_t N = points.size();
        fftw_complex *in, *out;
        // TODO: check for nullptr here
        in = static_cast<fftw_complex *>(fftw_malloc(sizeof(fftw_complex) * N));
        out = static_cast<fftw_complex *>(fftw_malloc(sizeof(fftw_complex) * N));

        for (size_t i = 0; i < N; ++i) {
            in[i][0] = points[i].real();
            in[i][1] = points[i].imag();
        }

        auto start_time_dft = get_time();

        fftw_plan plan = fftw_plan_dft_1d(static_cast<int>(N), in, out, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(plan);

        std::cout << "FFTW CPU time: " << time_elapsed(start_time_dft) << std::endl;

        for (size_t i = 0; i < N; ++i) {
            fftw_res[i] = {static_cast<float>(out[i][0]), static_cast<float>(out[i][1])};
        }

        fftw_destroy_plan(plan);
        fftw_free(in);
        fftw_free(out);

    }

    {
        auto start_time_iter = get_time();
        fft_iter(points_fft_iter);
        std::cout << "Iterative CPU time: " << time_elapsed(start_time_iter) << std::endl;
    }

    {
        auto start_time_recursive = get_time();
        fft(points_fft);
        std::cout << "Recursive CPU time: " << time_elapsed(start_time_recursive) << std::endl;
    }

    {
        auto start_time_cuda = get_time();
        fft_cuda(cuda_fft_iter.data(), static_cast<int>(cuda_fft_iter.size()));
        std::cout << "Cuda time: " << time_elapsed(start_time_cuda) << std::endl;
    }

   for (size_t i = 0; i < points.size(); i++) {
//       std::cerr << "FFTW CPU: " << fftw_res[i] << "; CUDA: " << cuda_fft_iter[i].x << ", " << cuda_fft_iter[i].y << std::endl;

       if (std::abs(fftw_res[i].real() - cuda_fft_iter[i].x) > MAX_ABS_ERROR ||
          std::abs(fftw_res[i].imag() - cuda_fft_iter[i].y) > MAX_ABS_ERROR) {
           std::cerr << "Results do not match at index " << i << ": " << std::endl;
           std::cout << std::abs(fftw_res[i].real() - cuda_fft_iter[i].x) << " " << std::abs(fftw_res[i].imag() - cuda_fft_iter[i].y) << std::endl;
           std::cerr << "FFTW CPU: " << fftw_res[i] << "; CUDA: " << cuda_fft_iter[i].x << ", " << cuda_fft_iter[i].y << std::endl;
           return -1;
       }
   }
}
