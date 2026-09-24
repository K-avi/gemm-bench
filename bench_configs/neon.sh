# Platform: ARMv8-A NEON
if [[ "$(hostname)" == *"m1"* || "${PLATFORM_TAG:-}" == "m1" ]]; then
    PLATFORM_NAME="Apple M1 (NEON)"
    BUILD_DIR_PREFIX="build_neon_m1"
    CSV_PREFIX="gemm_results_neon_m1"
    DEFAULT_PIN_CORE="3"
    DEFAULT_KERNELS=(
        mippv2_x60_mr6_nr4_fmaddi
        mippv2_x60_mr6_nr4_fmaddi_crr
    )
elif [[ "$(hostname)" == *"rpi"* || "${PLATFORM_TAG:-}" == "rpi5" ]]; then
    PLATFORM_NAME="Raspberry Pi 5 (NEON)"
    BUILD_DIR_PREFIX="build_neon_rpi5"
    CSV_PREFIX="gemm_results_neon_rpi5"
    DEFAULT_KERNELS=(
        mippv2_a76_mr6_nr3
        mippv2_a76_mr6_nr3_crr
    )
else
    PLATFORM_NAME="ARMv8-A (NEON)"
    BUILD_DIR_PREFIX="build_neon"
    CSV_PREFIX="gemm_results_neon"
    DEFAULT_KERNELS=(
        mippv2_a76_mr6_nr3
        mippv2_a76_mr6_nr3_crr
        mippv2_x60_mr6_nr4_fmaddi
        mippv2_x60_mr6_nr4_fmaddi_crr
    )
fi

ALL_KERNELS=(
    mippv2_a76_mr6_nr3
    mippv2_a76_mr6_nr3_crr
    mippv2_x60_mr6_nr4_fmaddi
    mippv2_x60_mr6_nr4_fmaddi_crr
    mippv2_firestorm_mr4_nr4_fmaddi
    mippv2_firestorm_mr4_nr4_fmaddi_crr
    mippv2_firestorm_mr6_nr4_fmaddi
    mippv2_firestorm_mr6_nr4_fmaddi_crr
    mippv2_firestorm_mr4_nr4
    mippv2_firestorm_mr6_nr4
)

COMMON_FLAGS="
-O3
-ffast-math
-finline-functions
-funroll-loops
-fno-semantic-interposition
"

ARCH_FLAGS[scalar]="
-fno-tree-vectorize
-fno-tree-slp-vectorize
"

ARCH_FLAGS[native]="
-march=native
"

# Auto-detect P-core on big.LITTLE architectures
platform_setup() {
    if [[ "$AUTO_PIN" == "true" && -z "$PIN_CORE" ]]; then
        if command -v taskset &>/dev/null; then
            local max_f=0
            local best_core=""
            for f in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/cpuinfo_max_freq; do
                if [[ -f "$f" ]]; then
                    local freq
                    freq=$(cat "$f" 2>/dev/null || echo 0)
                    local core
                    core=$(echo "$f" | sed -n 's|.*/cpu\([0-9]\+\)/.*|\1|p')
                    if (( freq > max_f )); then
                        max_f=$freq
                        best_core=$core
                    fi
                fi
            done
            if [[ -n "$best_core" && "$max_f" -gt 0 ]]; then
                PIN_CORE="$best_core"
                echo "================================================================================"
                echo "Auto-detected Performance Core (P-core): Core #${PIN_CORE} (${max_f} kHz)"
                echo "================================================================================"
            fi
        fi
    fi
}

# Compiler-specific flags hook (e.g. clang sysroot / M1)
platform_compiler_flags() {
    local compiler_tag="$1"
    if [[ "$compiler_tag" == "clang" ]]; then
        local extra=""
        if [[ -d "/usr/lib/gcc/aarch64-linux-gnu/15" ]]; then
            extra="--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/15"
        elif [[ -d "/usr/lib/gcc/aarch64-linux-gnu" ]]; then
            local detected_gcc
            detected_gcc=$(find /usr/lib/gcc/aarch64-linux-gnu/ -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -V | tail -n 1)
            if [[ -n "$detected_gcc" && -d "$detected_gcc" ]]; then
                extra="--gcc-install-dir=${detected_gcc}"
            fi
        fi
        if grep -q -E "0x61|Firestorm|Apple" /proc/cpuinfo 2>/dev/null; then
            extra="${extra} -mcpu=apple-m1"
        fi
        echo "$extra"
    fi
}
