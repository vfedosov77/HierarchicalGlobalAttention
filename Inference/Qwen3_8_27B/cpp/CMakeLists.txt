cmake_minimum_required(VERSION 3.16)
project(hga_qwen38 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(hga STATIC src/hga.cpp)
target_include_directories(hga PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_compile_options(hga PRIVATE -O3 -fno-strict-aliasing
        -mavx2 -mfma -mf16c)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx512f" HGA_HAS_AVX512)
    if (HGA_HAS_AVX512)
        target_compile_options(hga PRIVATE -mavx512f -mavx512bw -mavx512vl -mavx512dq)
    endif()
endif()

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(hga PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(hga-bench src/hga_bench.cpp)
target_link_libraries(hga-bench PRIVATE hga)

add_executable(hga-test tests/test_hga.cpp)
target_link_libraries(hga-test PRIVATE hga)

enable_testing()
add_test(NAME hga-test COMMAND hga-test)
