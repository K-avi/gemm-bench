#!/usr/bin/env bash

set -euo pipefail

REMOTE_DIR="${DALEK_DIR:-$HOME/Files/gemm-bench}"
cd "${REMOTE_DIR}"

mkdir -p logs

echo "=== Submitting GEMM Robust CRR Benchmarks to Slurm ==="

# 1. RPi5 (Cortex-A76)
JOB_RPI5=$(sbatch --parsable << 'EOF'
#!/usr/bin/env bash
#SBATCH -p mono
#SBATCH -w mono-rpi-5b
#SBATCH --exclusive
#SBATCH -J gemm_rpi5
#SBATCH -o logs/slurm_rpi5_%j.out
#SBATCH -e logs/slurm_rpi5_%j.err

cd "$HOME/Files/gemm-bench"
module load catch2 2>/dev/null || true
echo "=== Running RPi5 (Cortex-A76) CRR Benchmarks on $(hostname) ==="
PLATFORM_TAG=rpi5 ./run_benchmarks.sh neon -o rpi5_neon --crr-only --rebuild -k mippv2_a76_mr6_nr4_fmaddi_crr gcc clang
EOF
)
echo "✓ Submitted RPi5 (Cortex-A76) -> Slurm Job ID: ${JOB_RPI5}"

# 2. K3: Precompile X60 + Run X100 + Run A100
JOB_K3=$(sbatch --parsable << 'EOF'
#!/usr/bin/env bash
#SBATCH -p mono
#SBATCH -w mono-sip-k3
#SBATCH -c 8
#SBATCH -J gemm_k3_rvv
#SBATCH -o logs/slurm_k3_%j.out
#SBATCH -e logs/slurm_k3_%j.err

cd "$HOME/Files/gemm-bench"

echo "=== [1/3] Precompiling X60 Clang on mono-sip-k3 ==="
cmake -B build_rvv_x60_clang_rvv -S . -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_FLAGS="-O3 -ffast-math -finline-functions -funroll-loops -fno-semantic-interposition -falign-functions=64 -falign-loops=32 -march=rv64gcv_zvl256b -mrvv-vector-bits=zvl" \
  -DGEMMBENCH_ENABLE_EXPLO=OFF && cmake --build build_rvv_x60_clang_rvv --target GemmBench --parallel

echo "=== [2/3] Running X100 (256-bit RVV) CRR Benchmarks ==="
PLATFORM_TAG=x100 ./run_benchmarks.sh rvv_x100 -o x100_rvv --crr-only --rebuild -k mippv2_x100_register_blocked_apack4_crr gcc clang

echo "=== [3/3] Running A100 (1024-bit RVV) CRR Benchmarks ==="
PLATFORM_TAG=a100 ./run_benchmarks.sh rvv_a100 -o a100_rvv --crr-only --rebuild -k mippv2_a100_mr7_nr2_lmul2_pipe_crr gcc clang
EOF
)
echo "✓ Submitted K3 (X100 + A100) -> Slurm Job ID: ${JOB_K3}"

# 3. X60 (BPi-F3) - depends on K3 precompilation
JOB_X60=$(sbatch --parsable --dependency=afterok:"${JOB_K3}" << 'EOF'
#!/usr/bin/env bash
#SBATCH -p mono
#SBATCH -w mono-bpi-f3
#SBATCH --exclusive
#SBATCH -J gemm_x60_rvv
#SBATCH -o logs/slurm_x60_%j.out
#SBATCH -e logs/slurm_x60_%j.err

cd "$HOME/Files/gemm-bench"
module load gcc 2>/dev/null || true
echo "=== Running X60 (256-bit RVV) CRR Benchmarks on $(hostname) ==="
PLATFORM_TAG=x60 ./run_benchmarks.sh rvv_x60 -o x60_rvv --crr-only -k mippv2_a100_mr7_nr4_pipe_crr gcc clang
EOF
)
echo "✓ Submitted X60 (BPi-F3, after K3) -> Slurm Job ID: ${JOB_X60}"

echo
echo "=== Current Slurm Queue for $USER ==="
squeue -u "$USER"
