#!/usr/bin/env bash

set -euo pipefail

################################################################################
# Configuration & Hardware Context: SpacemiT K3 (A100 AI-CPU)
#
# Hardware Architecture (from SpacemiT K3 technical documentation):
# - 8x X100 general-purpose cores (CPU 0-7): RVA23, VLEN=256 bits, 2.4 GHz
# - 8x A100 AI-CPU cores (CPU 8-15): RVA23, VLEN=1024 bits, 2.1 GHz
#
# Schedulability & Core Affinity:
# Under Linux SMP on K3, CPU 0-7 represent the default general-purpose domain,
# while CPU 8-15 form the AI-exclusive compute domain. Threads executing
# 1024-bit vector code must be pinned to A100 cores (CPU 8-15) via taskset.
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
PIN_CORE="8" # Default to first A100 AI core (CPU 8)
REQUESTED_COMPILERS=()
CUSTOM_KERNELS=()
CUSTOM_SIZES=()

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
        -c|--core|--pin-core)
            PIN_CORE="$2"
            shift 2
            ;;
        --no-pin)
            PIN_CORE=""
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
        -k|--kernel)
            CUSTOM_KERNELS+=("$2")
            shift 2
            ;;
        -s|--size)
            CUSTOM_SIZES+=("$2")
            shift 2
            ;;
        --sizes)
            IFS=' ' read -r -a CUSTOM_SIZES <<< "$2"
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

# SpacemiT K3 Kernel Gatekeeper (/proc/set_ai_thread)
# By default, Linux on K3 prohibits scheduling on A100 cores (8-15) until
# the process is registered as an AI thread. Writing PID to /proc/set_ai_thread unlocks them.
if [[ -e /proc/set_ai_thread ]]; then
    if echo $$ > /proc/set_ai_thread 2>/dev/null; then
        echo "================================================================================"
        echo "SpacemiT K3: Successfully unlocked A100 cores via /proc/set_ai_thread (PID $$)"
        echo "================================================================================"
    elif sudo sh -c "echo $$ > /proc/set_ai_thread" 2>/dev/null; then
        echo "================================================================================"
        echo "SpacemiT K3: Unlocked A100 cores via sudo /proc/set_ai_thread (PID $$)"
        echo "================================================================================"
    else
        echo "================================================================================"
        echo "Notice: /proc/set_ai_thread exists. If taskset fails, please run:"
        echo "        echo \$\$ | sudo tee /proc/set_ai_thread"
        echo "================================================================================"
    fi
fi

if [[ -n "$PIN_CORE" ]]; then
    if command -v taskset &>/dev/null; then
        if taskset -c "${PIN_CORE}" true 2>/dev/null; then
            TASKSET_CMD="taskset -c ${PIN_CORE}"
            echo "================================================================================"
            echo "A100 Core Affinity: Benchmarks will be pinned to Core #${PIN_CORE} via taskset"
            echo "================================================================================"
        else
            TASKSET_CMD=""
            echo "================================================================================"
            echo "Warning: CPU core #${PIN_CORE} is not accessible via taskset."
            echo "         Tip: On K3, run 'echo \$\$ | sudo tee /proc/set_ai_thread' or use 'ai'."
            echo "         Falling back to unpinned execution."
            echo "================================================================================"
        fi
    else
        TASKSET_CMD=""
        echo "Warning: 'taskset' utility not found. Running without core pinning."
    fi
else
    TASKSET_CMD=""
    echo "Core pinning disabled (--no-pin specified)."
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
# Architecture flags: SpacemiT A100 (VLEN = 1024 bits)
################################################################################

declare -A ARCH_FLAGS

#
# Scalar reference
#
ARCH_FLAGS[scalar]="
-march=rv64gcv_zvl1024b
-mrvv-vector-bits=zvl
-fno-tree-vectorize
-fno-tree-slp-vectorize
"

#
# SpacemiT A100 RVV 1.0 (VLEN = 1024 bits, ELEN = 64)
#
ARCH_FLAGS[rvv]="
-march=rv64gcv_zvl1024b
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
    mippv2_meteorlake_mr4_nr3
    mippv2_a76_mr6_nr3
    mippv2_firestorm_mr4_nr4_fmaddi
    mippv2_firestorm_mr6_nr4_fmaddi
    mippv2_zen4_mr4_nr4
    mippv2_zen4_mr4_nr4_fmaddi
    mippv2_x100_register_blocked_lmul2
    mippv2_x100_register_blocked_apack4
    mippv2_a100_mr7_nr4_pipe
    mippv2_a100_mr7_nr2_lmul2_pipe
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
# Note: For VLEN=1024, VL=16 (LMUL=1), VL=32 (LMUL=2), VL=64 (LMUL=4).
# A default size of 128 enables full register tiling coverage up to LMUL=4.
#
if [[ ${#CUSTOM_SIZES[@]} -gt 0 ]]; then
    SIZES=()
    for S in "${CUSTOM_SIZES[@]}"; do
        if [[ "$S" -le 32 ]]; then
            SIZES+=("$S 100000 10000")
        elif [[ "$S" -le 64 ]]; then
            SIZES+=("$S 10000 1000")
        else
            SIZES+=("$S 1000 200")
        fi
    done
elif [[ "$RUN_ALL" == "true" ]]; then
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
        "128 10000 5000"
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
    echo "# Benchmarking SpacemiT A100 with ${COMPILER} (${COMPILER_TAG})"
    echo "################################################################################"

    CSV="results/gemm_results_a100_${COMPILER_TAG}.csv"
    echo "Build,Kernel,M,N,K,Alpha,Beta,Time_s,GFLOPS" > "$CSV"

    # Build
    for VERSION in "${VERSIONS[@]}"
    do
        if [[ "$RUN_ALL" == "true" ]]; then
            BUILD_DIR="build_a100_${COMPILER_TAG}_${VERSION}_all"
            EXPLO_FLAG="-DGEMMBENCH_ENABLE_EXPLO=ON"
        else
            BUILD_DIR="build_a100_${COMPILER_TAG}_${VERSION}"
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
        echo "Building ${VERSION} for SpacemiT A100 (${COMPILER}) in ${BUILD_DIR}"
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
            BUILD_DIR="build_a100_${COMPILER_TAG}_${VERSION}_all"
        else
            BUILD_DIR="build_a100_${COMPILER_TAG}_${VERSION}"
        fi
        BIN="${BUILD_DIR}/GemmBench"

        echo
        echo "======================================="
        echo "Benchmarking ${VERSION} on SpacemiT A100 (${COMPILER})"
        echo "======================================="

        if [[ ${#CUSTOM_KERNELS[@]} -gt 0 ]]; then
            KERNEL_LIST=("${CUSTOM_KERNELS[@]}")
        elif [[ "$RUN_ALL" == "true" ]]; then
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
                    $TASKSET_CMD "$BIN" \
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

    # Maintain canonical results file for A100 (preserve gcc as default canonical)
    if [[ "$COMPILER_TAG" == "gcc" || ! -f "results/gemm_results_a100.csv" ]]; then
        cp "$CSV" results/gemm_results_a100.csv
    fi

    echo
    echo "Results for ${COMPILER} written to:"
    echo "  $CSV"

done
