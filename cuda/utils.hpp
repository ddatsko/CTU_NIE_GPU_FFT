//
// Created by mrs on 07.03.22.
//

#ifndef CPP_SEQUENTIAL_UTILS_HPP
#define CPP_SEQUENTIAL_UTILS_HPP

#include <vector>
#include <string>
#include <complex>
#include <chrono>

using time_point_t = std::chrono::high_resolution_clock::time_point;

std::vector<float> read_sequence_from_file(const std::string &filename);

std::vector<std::complex<float>> points_to_complex(const std::vector<float> &points);

time_point_t get_time();

long long time_elapsed(time_point_t start_time);

#endif //CPP_SEQUENTIAL_UTILS_HPP
