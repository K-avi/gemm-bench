# Platform: SpacemiT X60 (RVV 1.0, VLEN = 256 bits)
PLATFORM_NAME="SpacemiT X60 (RVV 256-bit)"
BUILD_DIR_PREFIX="build_rvv_x60"
CSV_PREFIX="gemm_results_rvv_x60"
DEFAULT_VERSION="rvv"

DEFAULT_KERNELS=(
    mippv2_a100_mr7_nr4_pipe
)

ALL_KERNELS=(
    mippv2_a100_mr7_nr4_pipe
    mippv2_x60_mr6_nr4_fmaddi
    mippv2_x100_register_blocked_apack4
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
