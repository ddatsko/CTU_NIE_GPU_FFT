#include <iostream>
#include "utils.hpp"
#include "fstream"
#include <atomic>
#include <chrono>


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

time_point_t get_time() {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    auto resTime = std::chrono::high_resolution_clock::now();
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return resTime;
}

long long time_elapsed(time_point_t start_time) {
    auto finish_time = get_time();

    auto int_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(finish_time - start_time);
    std::chrono::duration<long long, std::nano> long_sec = int_ns;
    return long_sec.count();
}