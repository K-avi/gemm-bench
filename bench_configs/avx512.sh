# Platform: x86_64 AVX-512
PLATFORM_NAME="x86_64 (AVX-512)"
BUILD_DIR_PREFIX="build_avx512"
CSV_PREFIX="gemm_results_avx512"

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
-mavx512f
-mavx512dq
-mavx512vl
-mavx512bw
"
