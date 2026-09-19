# Platform: SpacemiT A100 (RVV 1.0, VLEN = 1024 bits)
PLATFORM_NAME="SpacemiT A100 (RVV 1024-bit)"
BUILD_DIR_PREFIX="build_a100"
CSV_PREFIX="gemm_results_a100"
DEFAULT_VERSION="rvv"
DEFAULT_PIN_CORE="8"

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
-march=rv64gcv_zvl1024b
-mrvv-vector-bits=zvl
-fno-tree-vectorize
-fno-tree-slp-vectorize
"

ARCH_FLAGS[rvv]="
-march=rv64gcv_zvl1024b
-mrvv-vector-bits=zvl
"

# SpacemiT K3 Kernel Gatekeeper (/proc/set_ai_thread)
platform_setup() {
    if [[ -e /proc/set_ai_thread ]]; then
        if echo $$ > /proc/set_ai_thread 2>/dev/null; then
            echo "================================================================================"
            echo "SpacemiT K3: Successfully unlocked A100 cores via /proc/set_ai_thread (PID $$)"
            echo "================================================================================"
        elif sudo sh -c "echo $$ > /proc/set_ai_thread" 2>/dev/null; then
            echo "================================================================================"
            echo "SpacemiT K3: Unlocked A100 cores via sudo /proc/set_ai_thread (PID $$)"
            echo "================================================================================"
        else
            echo "================================================================================"
            echo "Notice: /proc/set_ai_thread exists. If taskset fails, please run:"
            echo "        echo \$\$ | sudo tee /proc/set_ai_thread"
            echo "================================================================================"
        fi
    fi
}
