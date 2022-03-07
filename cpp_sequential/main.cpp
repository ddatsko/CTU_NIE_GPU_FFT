#include <iostream>
#include "utils.hpp"
#include <complex>
#include "fft.h"


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Wrong number of arguments\nUsage: $ ./fft <input_file>" << std::endl;
        return -1;
    }
    auto original = read_sequence_from_file(argv[1]);
    if (original.empty()) {
        std::cerr << "Error while reading data from file. File is empty or could not be opened" << std::endl;
        return -1;
    }
    if ((original.size() & (original.size() - 1)) != 0) {
        std::cout << "Number of input number is not a power of 2. Exiting" << std::endl;
        return -2;
    }

    auto transformed = dft(original);
    for (const auto &p: transformed) {
        std::cout << p << " ";
    }


}
