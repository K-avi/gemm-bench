cmake_minimum_required(VERSION 3.20)

project(GemmBench LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Synchronize C compilation flags with CXX flags for SIMD and architecture intrinsics
if(CMAKE_CXX_FLAGS)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_CXX_FLAGS}")
endif()
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

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
# Math Optimization Flags (Fast-Math vs Strict IEEE-754)
# -----------------------------------------------------------------------------
option(GEMMBENCH_ENABLE_FASTMATH "Enable -ffast-math optimizations (relax IEEE-754)" ON)
if(GEMMBENCH_ENABLE_FASTMATH)
    set(GEMMBENCH_MATH_FLAGS "-ffast-math")
else()
    set(GEMMBENCH_MATH_FLAGS "-fno-fast-math" "-ffp-contract=fast")
    message(STATUS "Strict IEEE-754 mode enabled (-fno-fast-math -ffp-contract=fast)")
endif()

# -----------------------------------------------------------------------------
# Native BLIS Reference Micro-Kernels
# -----------------------------------------------------------------------------
option(GEMMBENCH_ENABLE_BLIS "Build and enable reference native BLIS micro-kernels" ON)

set(GEMMBENCH_BLIS_TARGET "AUTO" CACHE STRING "Target BLIS reference kernel (AUTO, SKX, HASWELL, ARMV8A, X60)")

if(GEMMBENCH_ENABLE_BLIS)
    set(BLIS_KERNEL_SRC "")
    set(BLIS_ARCH_TAG "")

    string(TOUPPER "${GEMMBENCH_BLIS_TARGET}" BLIS_TARGET_UPPER)

    if("${BLIS_TARGET_UPPER}" STREQUAL "SKX")
        set(BLIS_KERNEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/libs/blis_reference/src/skx_d16x14.c")
        set(BLIS_ARCH_TAG "GEMMBENCH_BLIS_SKX")
    elseif("${BLIS_TARGET_UPPER}" STREQUAL "HASWELL")
        set(BLIS_KERNEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/libs/blis_reference/src/haswell_d6x8.c")
        set(BLIS_ARCH_TAG "GEMMBENCH_BLIS_HASWELL")
    elseif("${BLIS_TARGET_UPPER}" STREQUAL "ARMV8A")
        set(BLIS_KERNEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/libs/blis_reference/src/armv8a_d6x8.c")
        set(BLIS_ARCH_TAG "GEMMBENCH_BLIS_ARMV8A")
    elseif("${BLIS_TARGET_UPPER}" STREQUAL "X60")
        set(BLIS_KERNEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/libs/blis_reference/src/dgemm_x60_2vx14.c")
        set(BLIS_ARCH_TAG "GEMMBENCH_BLIS_X60")
    else()
        # AUTO detection
        if("${CMAKE_CXX_FLAGS} ${CMAKE_C_FLAGS}" MATCHES "-mavx512|-march=znver4|-march=znver5")
            set(BLIS_KERNEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/libs/blis_reference/src/skx_d16x14.c")
            set(BLIS_ARCH_TAG "GEMMBENCH_BLIS_SKX")
        elseif("${CMAKE_CXX_FLAGS} ${CMAKE_C_FLAGS}" MATCHES "-mavx2")
            set(BLIS_KERNEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/libs/blis_reference/src/haswell_d6x8.c")
            set(BLIS_ARCH_TAG "GEMMBENCH_BLIS_HASWELL")
        elseif("${CMAKE_CXX_FLAGS} ${CMAKE_C_FLAGS}" MATCHES "-march=armv8|-march=armv9" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
            set(BLIS_KERNEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/libs/blis_reference/src/armv8a_d6x8.c")
            set(BLIS_ARCH_TAG "GEMMBENCH_BLIS_ARMV8A")
        elseif("${CMAKE_CXX_FLAGS} ${CMAKE_C_FLAGS}" MATCHES "rv64gcv" OR CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64")
            set(BLIS_KERNEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/libs/blis_reference/src/dgemm_x60_2vx14.c")
            set(BLIS_ARCH_TAG "GEMMBENCH_BLIS_X60")
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
            set(BLIS_KERNEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/libs/blis_reference/src/haswell_d6x8.c")
            set(BLIS_ARCH_TAG "GEMMBENCH_BLIS_HASWELL")
        endif()
    endif()

    if(BLIS_KERNEL_SRC AND EXISTS "${BLIS_KERNEL_SRC}")
        message(STATUS "BLIS Reference Kernel enabled: ${BLIS_KERNEL_SRC} (${BLIS_ARCH_TAG})")
        add_library(blis_reference STATIC ${BLIS_KERNEL_SRC})
        target_include_directories(blis_reference PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/libs/blis_reference/include)
        target_compile_definitions(blis_reference PUBLIC GEMMBENCH_HAS_BLIS ${BLIS_ARCH_TAG})
        if("${BLIS_ARCH_TAG}" STREQUAL "GEMMBENCH_BLIS_X60")
            target_compile_options(blis_reference PRIVATE "-march=rv64gcv_zvl256b_zicbop")
        endif()
        target_link_libraries(gemmbench INTERFACE blis_reference)
    endif()
endif()

# -----------------------------------------------------------------------------
# LIKWID Marker API (Optional, x86_64 validation)
# -----------------------------------------------------------------------------
option(GEMMBENCH_ENABLE_LIKWID "Build with LIKWID Marker API support" OFF)
set(LIKWID_FOUND FALSE)
if(GEMMBENCH_ENABLE_LIKWID)
    find_path(LIKWID_INCLUDE_DIR NAMES likwid-marker.h
        PATHS
            ENV LIKWID_PREFIX
            ENV LIKWID_DIR
            /usr/local
            /usr
        PATH_SUFFIXES include
    )
    find_library(LIKWID_LIBRARY NAMES likwid
        PATHS
            ENV LIKWID_PREFIX
            ENV LIKWID_DIR
            /usr/local
            /usr
        PATH_SUFFIXES lib lib64
    )
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
        ${GEMMBENCH_MATH_FLAGS}
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
    add_executable(broadcast_bench EXCLUDE_FROM_ALL
        x100_uarch_exp/broadcast_bench.cpp
    )
    target_link_libraries(broadcast_bench
        PRIVATE
            gemmbench
    )
    target_compile_options(broadcast_bench
        PRIVATE
            -O3
            ${GEMMBENCH_MATH_FLAGS}
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
