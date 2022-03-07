//
// Created by mrs on 07.03.22.
//

#ifndef CPP_SEQUENTIAL_UTILS_HPP
#define CPP_SEQUENTIAL_UTILS_HPP

#include <vector>
#include <string>
#include <complex>

std::vector<float> read_sequence_from_file(const std::string &filename);

std::vector<std::complex<float>> points_to_complex(const std::vector<float> &points);

#endif //CPP_SEQUENTIAL_UTILS_HPP
