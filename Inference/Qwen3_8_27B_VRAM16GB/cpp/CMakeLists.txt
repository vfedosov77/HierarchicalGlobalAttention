cmake_minimum_required(VERSION 3.16)
project(hga_qwen38 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(hga STATIC src/hga.cpp src/hga_l2.cpp)
target_include_directories(hga PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
find_package(Threads)
if (Threads_FOUND)
    target_link_libraries(hga PUBLIC Threads::Threads)
endif()

if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_compile_options(hga PRIVATE -O3 -fno-strict-aliasing
        -mavx2 -mfma -mf16c)
    # Compiler support is not enough: gcc accepts -mavx512f on a Broadwell
    # host and the binary then SIGILLs in ggml/HGA init. Match setup.sh:
    # only emit AVX-512 when this CPU reports avx512f.
    set(HGA_CPU_AVX512 0)
    if (EXISTS "/proc/cpuinfo")
        file(READ "/proc/cpuinfo" HGA_CPUINFO)
        if (HGA_CPUINFO MATCHES " avx512f")
            set(HGA_CPU_AVX512 1)
        endif()
    endif()
    if (HGA_CPU_AVX512)
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("-mavx512f" HGA_HAS_AVX512)
        if (HGA_HAS_AVX512)
            target_compile_options(hga PRIVATE
                -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512cd)
        endif()
    endif()
endif()

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(hga PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(hga-bench src/hga_bench.cpp)
target_link_libraries(hga-bench PRIVATE hga)

add_executable(hga-route-bench src/hga_route_bench.cpp)
target_link_libraries(hga-route-bench PRIVATE hga)

add_executable(hga-test tests/test_hga.cpp)
target_link_libraries(hga-test PRIVATE hga)

enable_testing()
add_test(NAME hga-test COMMAND hga-test)
