# Platform: x86_64 AVX2 / FMA
PLATFORM_NAME="x86_64 (AVX2)"
BUILD_DIR_PREFIX="build_avx2"
CSV_PREFIX="gemm_results_avx2"
DEFAULT_PIN_CORE="0"

DEFAULT_KERNELS=(
    mippv2_meteorlake_mr4_nr3
)

ALL_KERNELS=(
    mippv2_meteorlake_mr4_nr3
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
-mavx2
-mfma
"
