#include <iostream>
#include "utils.hpp"
#include <complex>
#include "fft_cpu.h"
#include "fft_cuda.hpp"
#include <cuComplex.h>

const double MAX_ABS_ERROR = 1e-1;

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
    std::vector<cuFloatComplex> cuda_fft_iter;
    for (const auto &point: points) {
        cuda_fft_iter.push_back({point.real(), point.imag()});
    }

    {
        auto start_time_iter = get_time();
        fft_iter(points_fft_iter);
        std::cout << "Iterative CPU time: " << time_elapsed(start_time_iter) << std::endl;
    }

    {
        auto start_time_dft = get_time();
        dft(points_dft);
        std::cout << "Direct transform CPU time: " << time_elapsed(start_time_dft) << std::endl;
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
       if (std::abs(points_dft[i].real() - cuda_fft_iter[i].x) > MAX_ABS_ERROR ||
          std::abs(points_dft[i].imag() - cuda_fft_iter[i].y) > MAX_ABS_ERROR) {
           std::cerr << "Results does not match at index " << i << ": " << std::endl;
           std::cerr << "DFT: " << points_dft[i] << "; CUDA: " << cuda_fft_iter[i].x << ", " << cuda_fft_iter[i].y << std::endl;
           return -1;
       }
   }


}
