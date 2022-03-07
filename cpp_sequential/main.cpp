#include <iostream>
#include "utils.hpp"
#include <complex>
#include "fft.h"


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
    auto &points_fft = points;
    dft(points_dft);
    fft(points_fft);

   for (size_t i = 0; i < points.size(); i++) {
       std::cout << points_fft[i] << " " << points_dft[i] << std::endl;
   }


}
