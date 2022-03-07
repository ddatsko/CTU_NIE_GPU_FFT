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