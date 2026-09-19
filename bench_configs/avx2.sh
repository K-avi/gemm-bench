# Platform: x86_64 AVX2 / FMA
PLATFORM_NAME="x86_64 (AVX2)"
BUILD_DIR_PREFIX="build"
CSV_PREFIX="gemm_results"

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
