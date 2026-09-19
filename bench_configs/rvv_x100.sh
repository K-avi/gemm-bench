# Platform: SpacemiT X100 (RVV 1.0, VLEN = 256 bits)
PLATFORM_NAME="SpacemiT X100 (RVV 256-bit)"
BUILD_DIR_PREFIX="build"
CSV_PREFIX="gemm_results"
DEFAULT_VERSION="rvv"

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
