cmake_minimum_required(VERSION 3.20)

project(GemmBench LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------------
# MIPPv2 Dependency Resolution
# -----------------------------------------------------------------------------
set(MIPPV2_INCLUDE_DIR "" CACHE PATH "Path to MIPPv2 include directory (containing mipp.hpp)")

if(NOT MIPPV2_INCLUDE_DIR)
    find_path(MIPPV2_FOUND_DIR
        NAMES mipp.hpp
        PATHS
            ENV MIPPV2_DIR
            ENV CPATH
            ${CMAKE_CURRENT_SOURCE_DIR}/libs/include
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/MIPPv2/include
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/mippv2/include
            /usr/local/include
            /usr/include
    )
    if(MIPPV2_FOUND_DIR)
        set(MIPPV2_INCLUDE_DIR "${MIPPV2_FOUND_DIR}")
    endif()
endif()

if(NOT MIPPV2_INCLUDE_DIR)
    message(FATAL_ERROR
        "MIPPv2 headers not found! Please specify -DMIPPV2_INCLUDE_DIR=/path/to/mippv2/include or install MIPPv2 (see README.md)."
    )
endif()

message(STATUS "MIPPv2 include directory: ${MIPPV2_INCLUDE_DIR}")

# -----------------------------------------------------------------------------
# Core GemmBench Library
# -----------------------------------------------------------------------------
add_library(gemmbench INTERFACE)

target_include_directories(gemmbench
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/microkernels
        ${MIPPV2_INCLUDE_DIR}
)

option(GEMMBENCH_ENABLE_EXPLO "Build and enable exploratory GEMM kernels" OFF)
if(GEMMBENCH_ENABLE_EXPLO)
    target_compile_definitions(gemmbench INTERFACE GEMMBENCH_ENABLE_EXPLO)
endif()

# -----------------------------------------------------------------------------
# LIKWID Marker API (Optional, x86_64 validation)
# -----------------------------------------------------------------------------
option(GEMMBENCH_ENABLE_LIKWID "Build with LIKWID Marker API support" OFF)
set(LIKWID_FOUND FALSE)
if(GEMMBENCH_ENABLE_LIKWID)
    find_path(LIKWID_INCLUDE_DIR NAMES likwid-marker.h PATHS /usr/local/include /usr/include)
    find_library(LIKWID_LIBRARY NAMES likwid PATHS /usr/local/lib /usr/lib /usr/lib64 /usr/local/lib64)
    if(LIKWID_INCLUDE_DIR AND LIKWID_LIBRARY)
        message(STATUS "Found LIKWID: ${LIKWID_LIBRARY} (headers: ${LIKWID_INCLUDE_DIR})")
        set(LIKWID_FOUND TRUE)
    else()
        message(WARNING "LIKWID not found! Building GemmBench without LIKWID support.")
    endif()
endif()

# -----------------------------------------------------------------------------
# Benchmarks
# -----------------------------------------------------------------------------
add_executable(GemmBench
    src/main.cpp
)

target_link_libraries(GemmBench
    PRIVATE
        gemmbench
)

target_compile_options(GemmBench
    PRIVATE
        -O3
        -ffast-math
        -finline-functions
        -funroll-loops
)

if(LIKWID_FOUND)
    target_compile_definitions(GemmBench PRIVATE GEMMBENCH_USE_LIKWID LIKWID_PERFMON)
    target_include_directories(GemmBench PRIVATE ${LIKWID_INCLUDE_DIR})
    target_link_libraries(GemmBench PRIVATE ${LIKWID_LIBRARY})
endif()

# Optional microarchitectural experiment benchmark
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/x100_uarch_exp/broadcast_bench.cpp")
    add_executable(broadcast_bench
        x100_uarch_exp/broadcast_bench.cpp
    )
    target_link_libraries(broadcast_bench
        PRIVATE
            gemmbench
    )
    target_compile_options(broadcast_bench
        PRIVATE
            -O3
            -ffast-math
            -finline-functions
            -funroll-loops
    )
endif()



# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
option(GEMMBENCH_BUILD_TESTS "Build GEMMBench tests" OFF)

if(GEMMBENCH_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
