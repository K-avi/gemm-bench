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
# Compilers & Options
################################################################################

RUN_ALL=false
FORCE_REBUILD=false
ALPHA=1.0
BETA=0.0
REQUESTED_COMPILERS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-all)
            RUN_ALL=true
            shift
            ;;
        --rebuild)
            FORCE_REBUILD=true
            shift
            ;;
        -a|--alpha)
            ALPHA="$2"
            shift 2
            ;;
        -b|--beta)
            BETA="$2"
            shift 2
            ;;
        *)
            REQUESTED_COMPILERS+=("$1")
            shift
            ;;
    esac
done

# Default to running both g++ and clang++, or accept specific compiler(s) as args
if [[ ${#REQUESTED_COMPILERS[@]} -eq 0 ]]; then
    REQUESTED_COMPILERS=("g++" "clang++")
fi

################################################################################
# Common optimization flags
################################################################################

COMMON_FLAGS="
-O3
-ffast-math
-finline-functions
-funroll-loops
-fno-semantic-interposition
-falign-functions=64
-falign-loops=32
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
# Build versions (lightweight by default: rvv only, scalar if --run-all)
################################################################################

if [[ "$RUN_ALL" == "true" ]]; then
    VERSIONS=(
        scalar
        rvv
    )
else
    VERSIONS=(
        rvv
    )
fi



################################################################################
# Kernels
################################################################################

DEFAULT_KERNELS=(
    ijk
    blocked_register_blocked
    mippv2_skylake_register_blocked
    mippv2_skylake_lmul4_register_blocked
    mippv2_x100_register_blocked_lmul2
    mippv2_x100_register_blocked_apack4
)

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
    mippv2_blocked_lmul8

    mippv2_blocked_unroll2
    mippv2_blocked_unroll4
    mippv2_blocked_unroll_jam4

    mippv2_skylake_register_blocked

    mippv2_skylake_lmul_register_blocked
    mippv2_skylake_lmul2_register_blocked
    mippv2_skylake_lmul4_register_blocked
    mippv2_skylake_lmul8_register_blocked

    mippv2_skylake_panel
    mippv2_skylake_panel_lmul
    mippv2_skylake_panel_lmul2
    mippv2_skylake_panel_lmul4
    mippv2_skylake_panel_lmul8

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
if [[ "$RUN_ALL" == "true" ]]; then
    SIZES=(
        "32 100000 10000"
        "64 10000 1000"
        "128 100 100"
        "256 10 10"
        "512 10 10"
    )
else
    SIZES=(
        "32 100000 10000"
        "64 10000 1000"
    )
fi



################################################################################
# Output directory
################################################################################

mkdir -p results



################################################################################
# Benchmark Loop across compilers
################################################################################

for COMPILER_REQ in "${REQUESTED_COMPILERS[@]}"
do
    if [[ "$COMPILER_REQ" == "gcc" || "$COMPILER_REQ" == "g++" ]]; then
        COMPILER="g++"
        COMPILER_TAG="gcc"
    elif [[ "$COMPILER_REQ" == "clang" || "$COMPILER_REQ" == "clang++" ]]; then
        COMPILER="clang++"
        COMPILER_TAG="clang"
    else
        COMPILER="$COMPILER_REQ"
        COMPILER_TAG="$COMPILER_REQ"
    fi

    echo
    echo "################################################################################"
    echo "# Benchmarking with ${COMPILER} (${COMPILER_TAG})"
    echo "################################################################################"

    CSV="results/gemm_results_${COMPILER_TAG}.csv"
    echo "Build,Kernel,M,N,K,Alpha,Beta,Time_s,GFLOPS" > "$CSV"

    # Build
    for VERSION in "${VERSIONS[@]}"
    do
        if [[ "$RUN_ALL" == "true" ]]; then
            BUILD_DIR="build_${COMPILER_TAG}_${VERSION}_all"
            EXPLO_FLAG="-DGEMMBENCH_ENABLE_EXPLO=ON"
        else
            BUILD_DIR="build_${COMPILER_TAG}_${VERSION}"
            EXPLO_FLAG="-DGEMMBENCH_ENABLE_EXPLO=OFF"
        fi
        BIN="${BUILD_DIR}/GemmBench"

        if [[ -f "$BIN" && "$FORCE_REBUILD" != "true" ]]; then
            echo
            echo "======================================="
            echo "Binary ${BIN} already exists, skipping build (use --rebuild to force)."
            echo "======================================="
            continue
        fi

        echo
        echo "======================================="
        echo "Building ${VERSION} (${COMPILER}) in ${BUILD_DIR}"
        echo "======================================="

        rm -rf "$BUILD_DIR"

        COMPILE_FLAGS=$(
            printf "%s\n%s\n" \
                "${COMMON_FLAGS}" \
                "${ARCH_FLAGS[$VERSION]}" |
            tr '\n' ' '
        )

        echo "Flags: $COMPILE_FLAGS"

        cmake \
            -S . \
            -B "$BUILD_DIR" \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CXX_FLAGS="$COMPILE_FLAGS" \
            -DCMAKE_CXX_COMPILER="$COMPILER" \
            "$EXPLO_FLAG"

        cmake --build "$BUILD_DIR" -j
    done

    # Run benchmarks
    for VERSION in "${VERSIONS[@]}"
    do
        if [[ "$RUN_ALL" == "true" ]]; then
            BUILD_DIR="build_${COMPILER_TAG}_${VERSION}_all"
        else
            BUILD_DIR="build_${COMPILER_TAG}_${VERSION}"
        fi
        BIN="${BUILD_DIR}/GemmBench"

        echo
        echo "======================================="
        echo "Benchmarking ${VERSION} (${COMPILER})"
        echo "======================================="

        if [[ "$RUN_ALL" == "true" ]]; then
            if [[ "$VERSION" == "scalar" ]]; then
                KERNEL_LIST=("${SCALAR_KERNELS[@]}")
            else
                KERNEL_LIST=("${ALL_KERNELS[@]}")
            fi
        else
            KERNEL_LIST=("${DEFAULT_KERNELS[@]}")
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
                        --alpha "$ALPHA" \
                        --beta "$BETA" \
                        --iterations "$ITER" \
                        --warmup "$WARM" \
                        --csv
                )

                echo "${VERSION},${LINE}" >> "$CSV"
                echo "[${COMPILER_TAG}] ${VERSION} ${KERNEL} ${M}x${M}"
            done
        done
    done

    # Maintain backward compatibility for scripts expecting gemm_results.csv (preserve gcc as canonical)
    if [[ "$COMPILER_TAG" == "gcc" || ! -f "results/gemm_results.csv" ]]; then
        cp "$CSV" results/gemm_results.csv
    fi

    echo
    echo "Results for ${COMPILER} written to:"
    echo "  $CSV"

done
