#!/usr/bin/env bash

set -euo pipefail


################################################################################
# Configuration
################################################################################

ROOT=$(pwd)


#
# Avoid accidental inherited flags
#
unset CFLAGS || true
unset CXXFLAGS || true



################################################################################
# Compiler
################################################################################

COMPILER=clang++

################################################################################
# Common optimization flags
################################################################################

COMMON_FLAGS="
-O3
-ffast-math
-finline-functions
-funroll-loops
-fno-semantic-interposition
"



################################################################################
# Architecture flags
################################################################################

declare -A ARCH_FLAGS



#
# Scalar reference
#
ARCH_FLAGS[scalar]="
-march=rv64gcv_zvl256b
-mrvv-vector-bits=zvl
-fno-tree-vectorize
-fno-tree-slp-vectorize
"



#
# SpacemiT X100 RVV
#
# RV64 + RVV 1.0
# VLEN = 256 bits
#
ARCH_FLAGS[rvv]="
-march=rv64gcv_zvl256b
-mrvv-vector-bits=zvl
"



################################################################################
# Build versions
################################################################################

VERSIONS=(
    scalar
    rvv
)



################################################################################
# Kernels
################################################################################

SCALAR_KERNELS=(
    ijk
    ikj

    blocked
    blocked_register_blocked
)



MIPP_KERNELS=(
    mippv2

    mippv2_lmul2
    mippv2_lmul4
    mippv2_lmul8

    mippv2_blocked
    mippv2_blocked_lmul2
    mippv2_blocked_lmul4

    mippv2_blocked_unroll2
    mippv2_blocked_unroll_jam4

    mippv2_register_blocked

    mippv2_lmul_register_blocked
    mippv2_lmul2_register_blocked
    mippv2_lmul4_register_blocked

    mippv2_panel
    mippv2_panel_lmul
    mippv2_panel_lmul2
    mippv2_panel_lmul4

    mippv2_dot
    
    mippv2_x100_lmul1
    mippv2_x100_lmul2
    mippv2_x100_lmul4
    mippv2_x100_lmul8

    mippv2_x100_register_blocked_lmul1
    mippv2_x100_register_blocked_lmul2
    mippv2_x100_register_blocked_lmul4
    mippv2_x100_register_blocked_lmul8

    mippv2_x100_register_blocked_apack4

    mippv2_panel_x100_lmul1
    mippv2_panel_x100_lmul2
    mippv2_panel_x100_lmul4
    mippv2_panel_x100_lmul8
)



ALL_KERNELS=(
    "${SCALAR_KERNELS[@]}"
    "${MIPP_KERNELS[@]}"
)



################################################################################
# Matrix sizes
################################################################################

#
# M ITERATIONS WARMUP
#
SIZES=(
    "32 100000 10000"
    "64 10000 1000"
    "128 100 100"
    "256 10 10"
    "512 10 10"
)



################################################################################
# Output
################################################################################

mkdir -p results

CSV=results/gemm_results.csv

echo "Build,Kernel,M,N,K,Time_s,GFLOPS,PaddingA" > "$CSV"



################################################################################
# Build
################################################################################

for VERSION in "${VERSIONS[@]}"
do

    BUILD_DIR="build_${VERSION}"

    echo
    echo "======================================="
    echo "Building ${VERSION}"
    echo "======================================="


    rm -rf "$BUILD_DIR"



    #
    # Merge flags and remove newlines
    #
    COMPILE_FLAGS=$(
        printf "%s\n%s\n" \
            "${COMMON_FLAGS}" \
            "${ARCH_FLAGS[$VERSION]}" |
        tr '\n' ' '
    )


    echo "Flags:"
    echo "$COMPILE_FLAGS"



    cmake \
        -S . \
        -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS="$COMPILE_FLAGS"\
        -DCMAKE_CXX_COMPILER="$COMPILER"\



    cmake --build "$BUILD_DIR" -j

done



################################################################################
# Run benchmarks
################################################################################

for VERSION in "${VERSIONS[@]}"
do

    BIN="build_${VERSION}/GemmBench"


    echo
    echo "======================================="
    echo "Benchmarking ${VERSION}"
    echo "======================================="



    if [[ "$VERSION" == "scalar" ]]
    then
        KERNEL_LIST=(
            "${SCALAR_KERNELS[@]}"
        )
    else
        KERNEL_LIST=(
            "${ALL_KERNELS[@]}"
        )
    fi



    for KERNEL in "${KERNEL_LIST[@]}"
    do

        for ENTRY in "${SIZES[@]}"
        do

            read M ITER WARM <<< "$ENTRY"



            LINE=$(
                "$BIN" \
                    --kernel "$KERNEL" \
                    --m "$M" \
                    --n "$M" \
                    --k "$M" \
                    --iterations "$ITER" \
                    --warmup "$WARM" \
                    --csv
            )



            echo "${VERSION},${LINE}" >> "$CSV"


            echo "${VERSION} ${KERNEL} ${M}x${M}"

        done

    done

done



echo
echo "Results written to:"
echo "  $CSV"
