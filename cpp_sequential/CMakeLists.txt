cmake_minimum_required(VERSION 3.20)
project(cpp_sequential)

set(CMAKE_CXX_STANDARD 17)

add_executable(cpp_sequential main.cpp fft.cpp fft.h utils.hpp utils.cpp)
