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
  rvv_x60    SpacemiT X60  (RVV 256-bit)
  rvv_a100   SpacemiT A100 (RVV 1024-bit)

Options:
  --run-all               Run all versions (including scalar) and exploratory kernels/sizes
  --crr, --crr-only       Run only CRR (col-major A) kernels with BLIS-like K sweep
  --explo                 Enable exploratory kernels (-DGEMMBENCH_ENABLE_EXPLO=ON)
  --rebuild               Force clean re-compilation even if binary exists
  -c, --core <N>          Pin execution to specific CPU core via taskset
  --no-pin                Disable CPU pinning
  -k, --kernel <name>     Run only specified kernel(s) (can be repeated)
  -s, --size <N>          Run only specified size(s) (can be repeated)
  --sizes "N1 N2 ..."     Run specified list of sizes
  -r, --repetitions <N>   Number of independent repetitions per measurement (default: 15)
  --cooldown <sec>        Cooldown pause in seconds after large sizes (default: 2.0)
  --validate              Run numerical correctness validation against scalar reference before benchmark
  --likwid                Enable LIKWID Marker API instrumentation (-DGEMMBENCH_ENABLE_LIKWID=ON)
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
  ./run_benchmarks.sh avx512 --crr -o zen4_avx512 gcc clang
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
VALIDATE_FLAG=false
ALPHA=1.0
BETA=0.0
PIN_CORE="${DEFAULT_PIN_CORE:-}"
REQUESTED_COMPILERS=()
CUSTOM_KERNELS=()
CUSTOM_SIZES=()
OUTPUT_PREFIX=""
CRR_ONLY=false
RRR_ONLY=false
NUM_RUNS=3
REPETITIONS=15
COOLDOWN_SEC=2.0
ENABLE_LIKWID_FLAG=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --crr|--crr-only)
            CRR_ONLY=true
            RRR_ONLY=false
            shift
            ;;
        --rrr|--rrr-only)
            RRR_ONLY=true
            CRR_ONLY=false
            shift
            ;;
        --validate)
            VALIDATE_FLAG=true
            shift
            ;;
        --likwid)
            ENABLE_LIKWID_FLAG=true
            shift
            ;;
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
            shift 2
            ;;
        --no-pin)
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
        --runs|--repeats)
            NUM_RUNS="$2"
            shift 2
            ;;
        -r|--repetitions)
            REPETITIONS="$2"
            shift 2
            ;;
        --cooldown)
            COOLDOWN_SEC="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        -*)
            echo "Error: Unrecognized option '$1'" >&2
            show_help
            exit 1
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

# Passive inspection of CPU frequency and governor (no sudo required)
CPU_GOV="unknown"
if [[ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]]; then
    CPU_GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unknown")
elif [[ -f /sys/devices/system/cpu/cpufreq/scaling_governor ]]; then
    CPU_GOV=$(cat /sys/devices/system/cpu/cpufreq/scaling_governor 2>/dev/null || echo "unknown")
fi
if [[ "$CPU_GOV" != "performance" && "$CPU_GOV" != "unknown" ]]; then
    echo "Notice: CPU governor is '${CPU_GOV}'. For minimal timing jitter, 'performance' is recommended."
fi

CPU_FREQ_STR=""
if [[ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq ]]; then
    CPU_FREQ_KHZ=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo "0")
    if (( CPU_FREQ_KHZ > 0 )); then
        CPU_FREQ_STR=" (${CPU_GOV} @ $(( CPU_FREQ_KHZ / 1000 )) MHz)"
    fi
fi
if [[ -n "$CPU_FREQ_STR" ]]; then
    echo "CPU Status: ${CPU_FREQ_STR}"
fi

# Helper: read current CPU frequency in kHz for the pinned core (or cpu0 fallback)
read_cpu_freq_khz() {
    local core="${PIN_CORE:-0}"
    local fpath="/sys/devices/system/cpu/cpu${core}/cpufreq/scaling_cur_freq"
    if [[ -f "$fpath" ]]; then
        cat "$fpath" 2>/dev/null || echo "0"
    else
        echo "0"
    fi
}

INITIAL_FREQ_KHZ=$(read_cpu_freq_khz)
if (( INITIAL_FREQ_KHZ > 0 )); then
    echo "Initial CPU frequency (core ${PIN_CORE:-0}): $(( INITIAL_FREQ_KHZ / 1000 )) MHz"
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
    mippv2_meteorlake_mr4_nr3_crr
    mippv2_a76_mr6_nr3
    mippv2_a76_mr6_nr4_fmaddi
    mippv2_a76_mr6_nr4_fmaddi_crr
    mippv2_a76_mr6_nr3_crr
    mippv2_firestorm_mr4_nr4
    mippv2_firestorm_mr4_nr4_fmaddi
    mippv2_firestorm_mr4_nr4_fmaddi_crr
    mippv2_firestorm_mr6_nr4
    mippv2_firestorm_mr6_nr4_fmaddi
    mippv2_firestorm_mr6_nr4_fmaddi_crr
    mippv2_zen4_mr4_nr4
    mippv2_zen4_mr4_nr4_fmaddi
    mippv2_zen4_mr4_nr4_fmaddi_crr
    mippv2_x100_register_blocked_lmul1
    mippv2_x100_register_blocked_lmul2
    mippv2_x100_register_blocked_lmul4
    mippv2_x100_register_blocked_apack4
    mippv2_x100_register_blocked_apack4_crr
    mippv2_panel_x100_lmul1
    mippv2_panel_x100_lmul2
    mippv2_panel_x100_lmul4
    mippv2_x60_mr6_nr4_fmaddi
    mippv2_x60_mr6_nr4_fmaddi_crr
    mippv2_a100_mr7_nr4_pipe
    mippv2_a100_mr7_nr4_pipe_crr
    mippv2_a100_mr7_nr2_lmul2_pipe
    mippv2_a100_mr7_nr2_lmul2_pipe_crr
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

if [[ "$CRR_ONLY" == "true" && ${#CUSTOM_SIZES[@]} -eq 0 ]]; then
    # In CRR BLIS mode: sweep K with iterations scaled inverse-proportionally to K (~0.5s per measurement)
    SIZES=(
        "32 5000000 500000"
        "64 2500000 250000"
        "128 1250000 125000"
        "256 600000 60000"
        "512 300000 30000"
        "1024 150000 15000"
    )
elif [[ ${#CUSTOM_SIZES[@]} -gt 0 ]]; then
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

    file_suffix=""
    if [[ "$CRR_ONLY" == "true" ]]; then
        file_suffix="_crr"
    elif [[ "$RRR_ONLY" == "true" ]]; then
        file_suffix="_rrr"
    fi

    target_dir="${RESULTS_DIR:-results}"
    if [[ -n "${OUTPUT_PREFIX}" ]]; then
        local_prefix="${OUTPUT_PREFIX%_}"
        if [[ "${local_prefix}" == gemm_results_* ]]; then
            CSV="${target_dir}/${local_prefix}${file_suffix}_${COMPILER_TAG}.csv"
        else
            CSV="${target_dir}/gemm_results_${local_prefix}${file_suffix}_${COMPILER_TAG}.csv"
        fi
    else
        CSV="${target_dir}/${CSV_PREFIX}${file_suffix}_${COMPILER_TAG}.csv"
    fi
    mkdir -p "$(dirname "$CSV")"
    echo "Build,Kernel,M,N,K,Alpha,Beta,Time_median_s,Time_min_s,Time_max_s,Time_stddev_s,Time_cv_pct,GFLOPS_median,GFLOPS_peak,Cycles_median,Cycles_min,Cycles_max,Cycles_stddev,Cycles_cv_pct,Eff_GHz,FLOP_per_cycle_median,FLOP_per_cycle_peak,Repetitions,InterRun_CV_pct" > "$CSV"

    # Log provenance metadata
    GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    COMPILER_VER=$("$COMPILER" --version 2>/dev/null | head -1 || echo "unknown")
    echo "Metadata: commit=${GIT_COMMIT} | compiler=${COMPILER_VER} | core=${PIN_CORE:-none}"

    # Determine whether exploratory kernels are required
    NEED_EXPLO=false
    if [[ "$RUN_ALL" == "true" || "$ENABLE_EXPLO_FLAG" == "true" ]]; then
        NEED_EXPLO=true
    elif [[ ${#CUSTOM_KERNELS[@]} -gt 0 ]]; then
        for ck in "${CUSTOM_KERNELS[@]}"; do
            case "$ck" in
                mippv2_meteorlake_mr4_nr3|mippv2_meteorlake_mr4_nr3_crr|\
                mippv2_a76_mr6_nr3|mippv2_a76_mr6_nr4_fmaddi|mippv2_a76_mr6_nr4_fmaddi_crr|\
                mippv2_zen4_mr4_nr4_fmaddi|mippv2_zen4_mr4_nr4_fmaddi_crr|\
                mippv2_firestorm_mr4_nr4_fmaddi|mippv2_firestorm_mr4_nr4_fmaddi_crr|\
                mippv2_firestorm_mr6_nr4_fmaddi|mippv2_firestorm_mr6_nr4_fmaddi_crr|\
                mippv2_x60_mr6_nr4_fmaddi|mippv2_x60_mr6_nr4_fmaddi_crr|\
                mippv2_x100_register_blocked_apack4|mippv2_x100_register_blocked_apack4_crr|\
                mippv2_a100_mr7_nr4_pipe|mippv2_a100_mr7_nr4_pipe_crr|\
                mippv2_a100_mr7_nr2_lmul2_pipe|mippv2_a100_mr7_nr2_lmul2_pipe_crr|ijk|blis_native)
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
        TEST_BIN="${BUILD_DIR}/tests/GemmBenchTests"

        FULL_FLAGS="${COMMON_FLAGS} ${ARCH_FLAGS[${VERSION}]} ${EXTRA_COMPILER_FLAGS}"
        FORMATTED_FLAGS=$(echo "$FULL_FLAGS" | tr '\n' ' ' | tr -s ' ')

        TEST_BUILD_FLAG="-DGEMMBENCH_BUILD_TESTS=OFF"
        if [[ "$RUN_TESTS" == "true" || "$TESTS_ONLY" == "true" ]]; then
            TEST_BUILD_FLAG="-DGEMMBENCH_BUILD_TESTS=ON"
        fi

        LOCAL_CATCH2="$HOME/.local/catch2/$(uname -m)"
        CATCH2_PREFIX_FLAG=""
        if [[ -d "${LOCAL_CATCH2}" ]]; then
            CATCH2_PREFIX_FLAG="-DCMAKE_PREFIX_PATH=${LOCAL_CATCH2}"
        fi

        if [[ "${PLATFORM_TAG:-}" == "x60" && "$COMPILER_TAG" == "clang" && -f "$BIN" ]]; then
            echo
            echo "======================================="
            echo "Notice: Using precompiled clang binary for x60 from NFS ($BIN)"
            echo "======================================="
        else
            # Configure if CMakeCache is missing or full rebuild requested
            if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" || "$FORCE_REBUILD" == "true" ]]; then
                echo
                echo "======================================="
                echo "Configuring ${VERSION} for ${PLATFORM_NAME} (${COMPILER}) in ${BUILD_DIR}"
                echo "======================================="
                echo "Flags: ${FORMATTED_FLAGS}"

                LIKWID_FLAG="-DGEMMBENCH_ENABLE_LIKWID=OFF"
                if [[ "$ENABLE_LIKWID_FLAG" == "true" ]]; then
                    LIKWID_FLAG="-DGEMMBENCH_ENABLE_LIKWID=ON"
                fi

                cmake -B "${BUILD_DIR}" -S . \
                    -DCMAKE_BUILD_TYPE=Release \
                    -DCMAKE_CXX_COMPILER="${COMPILER}" \
                    -DCMAKE_CXX_FLAGS="${FORMATTED_FLAGS}" \
                    ${EXPLO_FLAG} \
                    ${TEST_BUILD_FLAG} \
                    ${LIKWID_FLAG} \
                    ${CATCH2_PREFIX_FLAG}
            fi

            BUILD_TARGET_FLAG="--target GemmBench"
            if [[ "$TESTS_ONLY" == "true" ]]; then
                BUILD_TARGET_FLAG="--target GemmBenchTests"
            elif [[ "$RUN_TESTS" == "true" ]]; then
                BUILD_TARGET_FLAG=""
            fi

            BUILD_OPTS=()
            if [[ "$FORCE_REBUILD" == "true" ]]; then
                BUILD_OPTS+=(--clean-first)
            fi

            echo
            echo "======================================="
            echo "Building ${VERSION} (${COMPILER}) in ${BUILD_DIR}..."
            echo "======================================="
            cmake --build "${BUILD_DIR}" ${BUILD_TARGET_FLAG} "${BUILD_OPTS[@]}" --parallel
        fi

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

        # Filter for CRR_ONLY or RRR_ONLY if requested
        if [[ "$CRR_ONLY" == "true" ]]; then
            FILTERED_KERNELS=()
            for k in "${RUN_KERNELS[@]}"; do
                if [[ "$k" == *_crr* || "$k" == *panel* || "$k" == "blis_native" ]]; then
                    FILTERED_KERNELS+=("$k")
                fi
            done
            RUN_KERNELS=("${FILTERED_KERNELS[@]}")
        elif [[ "$RRR_ONLY" == "true" ]]; then
            FILTERED_KERNELS=()
            for k in "${RUN_KERNELS[@]}"; do
                if [[ "$k" != *_crr* && "$k" != *panel* && "$k" != "blis_native" ]]; then
                    FILTERED_KERNELS+=("$k")
                fi
            done
            RUN_KERNELS=("${FILTERED_KERNELS[@]}")
        fi

        for KERNEL in "${RUN_KERNELS[@]}"; do
            for SIZE_ENTRY in "${SIZES[@]}"; do
                read -r SIZE ITERS WARMUP <<< "${SIZE_ENTRY}"

                BENCH_CMD=()
                EXTRA_BENCH_ARGS=()
                if [[ "$VALIDATE_FLAG" == "true" ]]; then
                    EXTRA_BENCH_ARGS+=(--validate)
                fi

                if [[ "$KERNEL" == *_crr* || "$KERNEL" == "blis_native" ]]; then
                    # In micro-tile (CRR / BLIS) mode: size is K, while M and N are auto-detected native MR x NR
                    BENCH_CMD=("${TASKSET_PREFIX[@]}" "${BIN}" \
                        --kernel "${KERNEL}" \
                        --k "${SIZE}" \
                        --alpha "${ALPHA}" \
                        --beta "${BETA}" \
                        --iterations "${ITERS}" \
                        --repetitions "${REPETITIONS}" \
                        --warmup "${WARMUP}" \
                        "${EXTRA_BENCH_ARGS[@]}" \
                        --csv)
                    DESC_STR="K=${SIZE}"
                else
                    # In RRR mode: legacy full square matrix
                    BENCH_CMD=("${TASKSET_PREFIX[@]}" "${BIN}" \
                        --kernel "${KERNEL}" \
                        --m "${SIZE}" \
                        --n "${SIZE}" \
                        --k "${SIZE}" \
                        --alpha "${ALPHA}" \
                        --beta "${BETA}" \
                        --iterations "${ITERS}" \
                        --repetitions "${REPETITIONS}" \
                        --warmup "${WARMUP}" \
                        "${EXTRA_BENCH_ARGS[@]}" \
                        --legacy \
                        --csv)
                    DESC_STR="${SIZE}x${SIZE}"
                fi

                BEST_LINE=""
                BEST_TIME=""
                ALL_TIMES=()
                for (( run_idx=1; run_idx<=NUM_RUNS; run_idx++ )); do
                    if CUR_LINE=$("${BENCH_CMD[@]}"); then
                        CUR_TIME=$(echo "$CUR_LINE" | awk -F',' '{print $7}')
                        ALL_TIMES+=("$CUR_TIME")
                        if [[ -z "$BEST_TIME" ]] || awk "BEGIN {exit !($CUR_TIME < $BEST_TIME)}"; then
                            BEST_TIME="$CUR_TIME"
                            BEST_LINE="$CUR_LINE"
                        fi
                    fi
                done

                # Compute inter-run CV (sample coefficient of variation across NUM_RUNS medians)
                INTER_CV=""
                if [[ ${#ALL_TIMES[@]} -ge 2 ]]; then
                    INTER_CV=$(printf '%s\n' "${ALL_TIMES[@]}" | awk '{
                        x[NR]=$1; s+=$1; s2+=$1*$1
                    } END {
                        if (NR>=2) {
                            m=s/NR; v=(s2 - (s*s/NR))/(NR-1);
                            if (v<0) v=0;
                            printf "%.2f", (sqrt(v)/m)*100
                        } else { print "0.00" }
                    }')
                else
                    INTER_CV="0.00"
                fi

                if [[ -n "$BEST_LINE" ]]; then
                    echo "${VERSION},${BEST_LINE},${INTER_CV}" >> "${CSV}"
                    INTER_CV_MSG=""
                    if awk "BEGIN {exit !(${INTER_CV} > 5.0)}" 2>/dev/null; then
                        INTER_CV_MSG=" ⚠ inter-run CV=${INTER_CV}%"
                    fi
                    echo "[${COMPILER_TAG}] ${VERSION} ${KERNEL} (${DESC_STR})${INTER_CV_MSG}"
                else
                    echo "[${COMPILER_TAG}] ${VERSION} ${KERNEL} (${DESC_STR}) FAILED" >&2
                fi

                # Post-measurement frequency check (throttle detection)
                if (( INITIAL_FREQ_KHZ > 0 )); then
                    POST_FREQ_KHZ=$(read_cpu_freq_khz)
                    if (( POST_FREQ_KHZ > 0 )); then
                        FREQ_DROP_PCT=$(awk "BEGIN {printf \"%.1f\", (1.0 - ${POST_FREQ_KHZ}/${INITIAL_FREQ_KHZ})*100}")
                        if awk "BEGIN {exit !(${FREQ_DROP_PCT} > 5.0)}" 2>/dev/null; then
                            echo "  ⚠ CPU freq dropped ${FREQ_DROP_PCT}% ($(( INITIAL_FREQ_KHZ/1000 )) → $(( POST_FREQ_KHZ/1000 )) MHz) — possible thermal throttling" >&2
                        fi
                    fi
                fi

                if (( SIZE >= 256 )) && awk "BEGIN {exit !(${COOLDOWN_SEC} > 0)}" 2>/dev/null; then
                    sleep "${COOLDOWN_SEC}"
                fi
            done
        done
    done
done

echo

# Final frequency check
if (( INITIAL_FREQ_KHZ > 0 )); then
    FINAL_FREQ_KHZ=$(read_cpu_freq_khz)
    if (( FINAL_FREQ_KHZ > 0 )); then
        echo "CPU frequency: $(( INITIAL_FREQ_KHZ / 1000 )) MHz (start) → $(( FINAL_FREQ_KHZ / 1000 )) MHz (end)"
        FINAL_DROP=$(awk "BEGIN {printf \"%.1f\", (1.0 - ${FINAL_FREQ_KHZ}/${INITIAL_FREQ_KHZ})*100}")
        if awk "BEGIN {exit !(${FINAL_DROP} > 5.0)}" 2>/dev/null; then
            echo "⚠ WARNING: CPU frequency dropped ${FINAL_DROP}% during benchmarks — results may be affected by thermal throttling"
        fi
    fi
fi

echo "================================================================================"
echo "Benchmarking complete."
echo "Results saved in ${RESULTS_DIR:-results}/"
echo "================================================================================"
