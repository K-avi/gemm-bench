#!/usr/bin/env bash

set -euo pipefail

################################################################################
# Usage and Help
################################################################################

show_help() {
    cat << 'EOF'
Usage: ./run_benchmarks.sh <platform> [options] [compilers...]

Platforms:
  avx2       x86_64 AVX2 / FMA
  avx512     x86_64 AVX-512
  neon       ARMv8-A NEON
  rvv_x100   SpacemiT X100 (RVV 256-bit)
  rvv_a100   SpacemiT A100 (RVV 1024-bit)

Options:
  --run-all               Run all versions (including scalar) and exploratory kernels/sizes
  --explo                 Enable exploratory kernels (-DGEMMBENCH_ENABLE_EXPLO=ON)
  --rebuild               Force clean re-compilation even if binary exists
  -c, --core <N>          Pin execution to specific CPU core via taskset
  --no-pin                Disable CPU pinning
  -k, --kernel <name>     Run only specified kernel(s) (can be repeated)
  -s, --size <N>          Run only specified size(s) (can be repeated)
  --sizes "N1 N2 ..."     Run specified list of sizes
  -o, --output <prefix>   Prefix for result CSV filename (e.g. --output zen4 -> zen4_gemm_results_gcc.csv)
  -a, --alpha <val>       Alpha parameter (default: 1.0)
  -b, --beta <val>        Beta parameter (default: 0.0)
  -h, --help              Show this help message

Compilers:
  gcc, clang, or specific binary names (default: g++ and clang++)

Examples:
  ./run_benchmarks.sh avx2
  ./run_benchmarks.sh avx2 --run-all gcc
  ./run_benchmarks.sh neon -c 4 -k mippv2_firestorm_mr4_nr4_fmaddi
  ./run_benchmarks.sh rvv_a100 --rebuild -s 64 -s 128
EOF
}

if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

################################################################################
# Platform Selection & Configuration
################################################################################

PLATFORM="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/bench_configs/${PLATFORM}.sh"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Unknown platform '${PLATFORM}'."
    echo "Available platforms:"
    for cfg in "${SCRIPT_DIR}"/bench_configs/*.sh; do
        if [[ -f "$cfg" ]]; then
            basename "$cfg" .sh | sed 's/^/  - /'
        fi
    done
    exit 1
fi

# Avoid accidental inherited flags
unset CFLAGS || true
unset CXXFLAGS || true

# Platform defaults
PLATFORM_NAME="${PLATFORM}"
BUILD_DIR_PREFIX="build"
CSV_PREFIX="gemm_results"
DEFAULT_VERSION="native"
DEFAULT_PIN_CORE=""
COMMON_FLAGS=""

declare -A ARCH_FLAGS

# Source platform configuration
source "$CONFIG_FILE"

################################################################################
# Options & CLI Argument Parsing
################################################################################

RUN_ALL=false
ENABLE_EXPLO_FLAG=false
RUN_TESTS=false
TESTS_ONLY=false
FORCE_REBUILD=false
ALPHA=1.0
BETA=0.0
PIN_CORE="${DEFAULT_PIN_CORE}"
AUTO_PIN=true
if [[ -n "${DEFAULT_PIN_CORE}" ]]; then
    AUTO_PIN=false
fi
REQUESTED_COMPILERS=()
CUSTOM_KERNELS=()
CUSTOM_SIZES=()
OUTPUT_PREFIX=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tests)
            RUN_TESTS=true
            shift
            ;;
        --tests-only)
            RUN_TESTS=true
            TESTS_ONLY=true
            shift
            ;;
        --run-all)
            RUN_ALL=true
            shift
            ;;
        --explo)
            ENABLE_EXPLO_FLAG=true
            shift
            ;;
        --rebuild)
            FORCE_REBUILD=true
            shift
            ;;
        -c|--core|--pin-core)
            PIN_CORE="$2"
            AUTO_PIN=false
            shift 2
            ;;
        --no-pin)
            AUTO_PIN=false
            PIN_CORE=""
            shift
            ;;
        -o|--output|--prefix)
            OUTPUT_PREFIX="$2"
            shift 2
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
        -h|--help)
            show_help
            exit 0
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

# Run platform setup hook if defined (e.g. P-core auto-detection, AI-thread unlock)
if declare -f platform_setup > /dev/null; then
    platform_setup
fi

# Taskset / CPU pinning configuration
TASKSET_PREFIX=()
if [[ -n "$PIN_CORE" ]]; then
    if command -v taskset &>/dev/null; then
        if taskset -c "${PIN_CORE}" true 2>/dev/null; then
            TASKSET_PREFIX=(taskset -c "${PIN_CORE}")
            echo "================================================================================"
            echo "Core Affinity: Benchmarks will be pinned to Core #${PIN_CORE} via taskset"
            echo "================================================================================"
        else
            echo "================================================================================"
            echo "Warning: CPU core #${PIN_CORE} is not accessible via taskset."
            echo "         Falling back to unpinned execution."
            echo "================================================================================"
        fi
    else
        echo "Warning: 'taskset' utility not found. Running without core pinning."
    fi
else
    echo "Core pinning disabled or unconfigured."
fi

################################################################################
# Build Versions
################################################################################

if [[ "$RUN_ALL" == "true" ]]; then
    VERSIONS=(
        scalar
        "${DEFAULT_VERSION}"
    )
else
    VERSIONS=(
        "${DEFAULT_VERSION}"
    )
fi

################################################################################
# Kernels
################################################################################

UNIVERSAL_KERNELS=(
    ijk
    blocked_register_blocked
    mippv2_skylake_register_blocked
    mippv2_skylake_lmul_register_blocked
    mippv2_skylake_lmul2_register_blocked
    mippv2_skylake_lmul4_register_blocked
    mippv2_skylake_panel
    mippv2_skylake_panel_lmul
    mippv2_skylake_panel_lmul2
    mippv2_skylake_panel_lmul4
    mippv2_meteorlake_mr4_nr3
    mippv2_a76_mr6_nr3
    mippv2_firestorm_mr4_nr4
    mippv2_firestorm_mr4_nr4_fmaddi
    mippv2_firestorm_mr6_nr4
    mippv2_firestorm_mr6_nr4_fmaddi
    mippv2_zen4_mr4_nr4
    mippv2_zen4_mr4_nr4_fmaddi
    mippv2_x100_register_blocked_lmul1
    mippv2_x100_register_blocked_lmul2
    mippv2_x100_register_blocked_lmul4
    mippv2_x100_register_blocked_apack4
    mippv2_panel_x100_lmul1
    mippv2_panel_x100_lmul2
    mippv2_panel_x100_lmul4
    mippv2_x60_mr6_nr4_fmaddi
    mippv2_a100_mr7_nr4_pipe
    mippv2_a100_mr7_nr2_lmul2_pipe
)

SCALAR_KERNELS=(
    ijk
    ikj
    blocked
    blocked_register_blocked
)

MIPP_EXPLO_KERNELS=(
    mippv2
    mippv2_lmul2
    mippv2_lmul4
    mippv2_lmul8
    mippv2_blocked
    mippv2_blocked_lmul2
    mippv2_blocked_lmul4
    mippv2_blocked_lmul8
    mippv2_panel
    mippv2_panel_lmul2
    mippv2_panel_lmul4
    mippv2_panel_lmul8
)

################################################################################
# Matrix Sizes (N, iterations, warmup)
################################################################################

if [[ ${#CUSTOM_SIZES[@]} -gt 0 ]]; then
    SIZES=()
    for s in "${CUSTOM_SIZES[@]}"; do
        if (( s <= 32 )); then
            SIZES+=("$s 100000 10000")
        elif (( s <= 96 )); then
            SIZES+=("$s 10000 1000")
        elif (( s <= 128 )); then
            SIZES+=("$s 10000 5000")
        else
            SIZES+=("$s 100 100")
        fi
    done
elif [[ "$RUN_ALL" == "true" ]]; then
    SIZES=(
        "32 100000 10000"
        "64 10000 1000"
        "96 10000 1000"
        "128 100 100"
        "256 10 10"
        "512 10 10"
    )
else
    SIZES=(
        "32 100000 10000"
        "64 10000 1000"
        "96 10000 1000"
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

for COMPILER_REQ in "${REQUESTED_COMPILERS[@]}"; do
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

    EXTRA_COMPILER_FLAGS=""
    if declare -f platform_compiler_flags > /dev/null; then
        EXTRA_COMPILER_FLAGS="$(platform_compiler_flags "$COMPILER_TAG")"
    fi

    echo
    echo "################################################################################"
    echo "# Benchmarking ${PLATFORM_NAME} with ${COMPILER} (${COMPILER_TAG})"
    echo "################################################################################"

    if [[ -n "${OUTPUT_PREFIX}" ]]; then
        local_prefix="${OUTPUT_PREFIX%_}"
        if [[ "${local_prefix}" == gemm_results_* ]]; then
            CSV="results/${local_prefix}_${COMPILER_TAG}.csv"
        else
            CSV="results/gemm_results_${local_prefix}_${COMPILER_TAG}.csv"
        fi
    else
        CSV="results/${CSV_PREFIX}_${COMPILER_TAG}.csv"
    fi
    mkdir -p "$(dirname "$CSV")"
    echo "Build,Kernel,M,N,K,Alpha,Beta,Time_s,GFLOPS" > "$CSV"

    # Determine whether exploratory kernels are required
    NEED_EXPLO=false
    if [[ "$RUN_ALL" == "true" || "$ENABLE_EXPLO_FLAG" == "true" ]]; then
        NEED_EXPLO=true
    elif [[ ${#CUSTOM_KERNELS[@]} -gt 0 ]]; then
        for ck in "${CUSTOM_KERNELS[@]}"; do
            case "$ck" in
                mippv2_meteorlake_mr4_nr3|mippv2_a76_mr6_nr3|mippv2_zen4_mr4_nr4_fmaddi|\
                mippv2_firestorm_mr4_nr4_fmaddi|mippv2_firestorm_mr6_nr4_fmaddi|\
                mippv2_x60_mr6_nr4_fmaddi|mippv2_x100_register_blocked_apack4|\
                mippv2_a100_mr7_nr4_pipe|mippv2_a100_mr7_nr2_lmul2_pipe|ijk)
                    ;;
                *)
                    NEED_EXPLO=true
                    break
                    ;;
            esac
        done
    fi

    # Build phase
    for VERSION in "${VERSIONS[@]}"; do
        if [[ "$NEED_EXPLO" == "true" ]]; then
            BUILD_DIR="${BUILD_DIR_PREFIX}_${COMPILER_TAG}_${VERSION}_all"
            EXPLO_FLAG="-DGEMMBENCH_ENABLE_EXPLO=ON"
        else
            BUILD_DIR="${BUILD_DIR_PREFIX}_${COMPILER_TAG}_${VERSION}"
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
        echo "Building ${VERSION} for ${PLATFORM_NAME} (${COMPILER}) in ${BUILD_DIR}"
        echo "======================================="

        FULL_FLAGS="${COMMON_FLAGS} ${ARCH_FLAGS[${VERSION}]} ${EXTRA_COMPILER_FLAGS}"
        FORMATTED_FLAGS=$(echo "$FULL_FLAGS" | tr '\n' ' ' | tr -s ' ')

        echo "Flags: ${FORMATTED_FLAGS}"

        TEST_BUILD_FLAG=""
        if [[ "$RUN_TESTS" == "true" ]]; then
            TEST_BUILD_FLAG="-DGEMMBENCH_BUILD_TESTS=ON"
        fi

        LOCAL_CATCH2="$HOME/.local/catch2/$(uname -m)"
        CATCH2_PREFIX_FLAG=""
        if [[ -d "${LOCAL_CATCH2}" ]]; then
            CATCH2_PREFIX_FLAG="-DCMAKE_PREFIX_PATH=${LOCAL_CATCH2}"
        fi

        cmake -B "${BUILD_DIR}" -S . \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CXX_COMPILER="${COMPILER}" \
            -DCMAKE_CXX_FLAGS="${FORMATTED_FLAGS}" \
            ${EXPLO_FLAG} \
            ${TEST_BUILD_FLAG} \
            ${CATCH2_PREFIX_FLAG}

        BUILD_TARGET_FLAG=""
        if [[ "$TESTS_ONLY" == "true" ]]; then
            BUILD_TARGET_FLAG="--target GemmBenchTests"
        fi

        cmake --build "${BUILD_DIR}" ${BUILD_TARGET_FLAG} --parallel

        if [[ "$RUN_TESTS" == "true" ]]; then
            echo
            echo "======================================="
            echo "Running Catch2 Tests for ${VERSION} (${COMPILER})"
            echo "======================================="
            if ! "${TASKSET_PREFIX[@]}" "${BUILD_DIR}/tests/GemmBenchTests"; then
                echo "Error: Tests failed for ${VERSION} (${COMPILER})!"
                exit 1
            fi
            echo "All tests passed for ${VERSION} (${COMPILER})!"
        fi
    done

    if [[ "$TESTS_ONLY" == "true" ]]; then
        continue
    fi

    # Run phase
    for VERSION in "${VERSIONS[@]}"; do
        if [[ "$NEED_EXPLO" == "true" ]]; then
            BUILD_DIR="${BUILD_DIR_PREFIX}_${COMPILER_TAG}_${VERSION}_all"
        else
            BUILD_DIR="${BUILD_DIR_PREFIX}_${COMPILER_TAG}_${VERSION}"
        fi
        BIN="${BUILD_DIR}/GemmBench"

        if [[ ! -f "$BIN" ]]; then
            echo "Error: Binary ${BIN} does not exist. Skipping benchmarks for ${VERSION}."
            continue
        fi

        echo
        echo "======================================="
        echo "Running benchmarks: ${VERSION} (${COMPILER_TAG})"
        echo "======================================="

        if [[ ${#CUSTOM_KERNELS[@]} -gt 0 ]]; then
            RUN_KERNELS=("${CUSTOM_KERNELS[@]}")
        elif [[ "$VERSION" == "scalar" ]]; then
            RUN_KERNELS=("${SCALAR_KERNELS[@]}")
        elif [[ "$RUN_ALL" == "true" ]]; then
            if [[ -n "${ALL_KERNELS+x}" && ${#ALL_KERNELS[@]} -gt 0 ]]; then
                RUN_KERNELS=("${ALL_KERNELS[@]}")
            else
                RUN_KERNELS=("${UNIVERSAL_KERNELS[@]}" "${MIPP_EXPLO_KERNELS[@]}")
            fi
        else
            if [[ -n "${DEFAULT_KERNELS+x}" && ${#DEFAULT_KERNELS[@]} -gt 0 ]]; then
                RUN_KERNELS=("${DEFAULT_KERNELS[@]}")
            else
                RUN_KERNELS=("${UNIVERSAL_KERNELS[@]}")
            fi
        fi

        for KERNEL in "${RUN_KERNELS[@]}"; do
            for SIZE_ENTRY in "${SIZES[@]}"; do
                read -r SIZE ITERS WARMUP <<< "${SIZE_ENTRY}"

                if LINE=$("${TASKSET_PREFIX[@]}" "${BIN}" \
                    --kernel "${KERNEL}" \
                    --m "${SIZE}" \
                    --n "${SIZE}" \
                    --k "${SIZE}" \
                    --alpha "${ALPHA}" \
                    --beta "${BETA}" \
                    --iterations "${ITERS}" \
                    --warmup "${WARMUP}" \
                    --csv); then
                    echo "${VERSION},${LINE}" >> "${CSV}"
                    echo "[${COMPILER_TAG}] ${VERSION} ${KERNEL} ${SIZE}x${SIZE}"
                else
                    echo "[${COMPILER_TAG}] ${VERSION} ${KERNEL} ${SIZE}x${SIZE} FAILED" >&2
                fi
            done
        done
    done
done

echo
echo "================================================================================"
echo "Benchmarking complete."
echo "Results saved in results/"
echo "================================================================================"
