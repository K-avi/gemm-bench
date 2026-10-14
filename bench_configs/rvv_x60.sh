# Platform: SpacemiT X60 (RVV 1.0, VLEN = 256 bits)
PLATFORM_NAME="SpacemiT X60 (RVV 256-bit)"
BUILD_DIR_PREFIX="build_rvv_x60"
CSV_PREFIX="gemm_results_rvv_x60"
DEFAULT_VERSION="rvv"
DEFAULT_PIN_CORE="1"

# NOTE: For FP64, the A100 mr7_nr4_pipe kernel is the empirical champion on X60
# despite being designed for the A100 core (both share VLEN=256).
# For FP32, the 7×4 tile (7×32 floats) causes severe vector register pressure (28 accumulators / 32),
# so the X60-native 6×4 tile (24 accumulators, leaving 8 regs for loads/staging) is the FP32 champion.
# No Clang results: no LLVM version with functional RVV support on this platform.
DEFAULT_KERNELS=(
    mippv2_a100_mr7_nr4_pipe
    mippv2_a100_mr7_nr4_pipe_crr
)

platform_setup() {
    if [[ "${FP32_FLAG:-false}" == "true" ]]; then
        DEFAULT_KERNELS=(
            mippv2_x60_mr6_nr4_fmaddi
            mippv2_x60_mr6_nr4_fmaddi_crr
        )
    fi
}

ALL_KERNELS=(
    mippv2_a100_mr7_nr4_pipe
    mippv2_a100_mr7_nr4_pipe_crr
    mippv2_x60_mr6_nr4_fmaddi
    mippv2_x60_mr6_nr4_fmaddi_crr
    mippv2_x100_register_blocked_apack4
    mippv2_x100_register_blocked_apack4_crr
)

COMMON_FLAGS="
-O3
-ffast-math
-finline-functions
-funroll-loops
-fno-semantic-interposition
-falign-functions=64
-falign-loops=32
"

ARCH_FLAGS[scalar]="
-march=rv64gcv_zvl256b_zicbop
-mrvv-vector-bits=zvl
-fno-tree-vectorize
-fno-tree-slp-vectorize
"

ARCH_FLAGS[rvv]="
-march=rv64gcv_zvl256b_zicbop
-mrvv-vector-bits=zvl
"
