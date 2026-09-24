# Platform: SpacemiT X100 (RVV 1.0, VLEN = 256 bits)
PLATFORM_NAME="SpacemiT X100 (RVV 256-bit)"
BUILD_DIR_PREFIX="build_rvv_x100"
CSV_PREFIX="gemm_results_rvv_x100"
DEFAULT_VERSION="rvv"
DEFAULT_PIN_CORE="0"

DEFAULT_KERNELS=(
    mippv2_x100_register_blocked_apack4
    mippv2_x100_register_blocked_apack4_crr
)

ALL_KERNELS=(
    mippv2_x100_register_blocked_apack4
    mippv2_x100_register_blocked_apack4_crr
    mippv2_x100_register_blocked_lmul1
    mippv2_x100_register_blocked_lmul2
    mippv2_x100_register_blocked_lmul4
    mippv2_x100_register_blocked_lmul8
    mippv2_panel_x100_lmul1
    mippv2_panel_x100_lmul2
    mippv2_panel_x100_lmul4
    mippv2_panel_x100_lmul8
    mippv2_x100_lmul1
    mippv2_x100_lmul2
    mippv2_x100_lmul4
    mippv2_x100_lmul8
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
-march=rv64gcv_zvl256b
-mrvv-vector-bits=zvl
-fno-tree-vectorize
-fno-tree-slp-vectorize
"

ARCH_FLAGS[rvv]="
-march=rv64gcv_zvl256b
-mrvv-vector-bits=zvl
"
