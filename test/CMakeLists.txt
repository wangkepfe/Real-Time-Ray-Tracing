cmake_minimum_required(VERSION 3.10)
project(unittest LANGUAGES CUDA CXX)

function(addTest exe)
    target_include_directories(${exe} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../src")

    # c++
    set_target_properties(${exe} PROPERTIES CXX_STANDARD 20)
    add_definitions(-DNOMINMAX)

    # cuda
    target_include_directories(${exe} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../ext/cuda-samples/Common)
    target_link_libraries(${exe} cudadevrt.lib)
    set_target_properties(${exe} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endfunction(addTest)

file(GLOB scanTestSrc
    "testCommon.h"
    "../src/scan.cuh"
    "scan/main.cu")

add_executable(scanTest ${scanTestSrc})
addTest(scanTest)