#include <iostream>
#include "utils.hpp"
#include "fstream"

std::vector<float> read_sequence_from_file(const std::string &filename) {
    std::ifstream is{filename};
    if (!is) {
        std::cerr << "Could not open file";
        return {};
    }

    std::vector<float> res;
    float tmp;
    while (is >> tmp) {
        res.push_back(tmp);
    }
    return res;
}

std::vector<std::complex<float>> points_to_complex(const std::vector<float> &points) {
    std::vector<std::complex<float>> res;
    res.reserve(points.size());
    for (const auto &p: points) {
        res.emplace_back(p, 0);
    }
    return res;
}