#!/usr/bin/env bash

set -euo pipefail

REMOTE_DIR="${DALEK_DIR:-$HOME/Files/gemm-bench}"
cd "${REMOTE_DIR}"

mkdir -p logs results/cpufp

echo "=== Submitting CPUFP Empirical Peak FP64 Jobs to Slurm ==="

# 1. RPi5 (Cortex-A76)
JOB_RPI5=$(sbatch --parsable << 'EOF'
#!/usr/bin/env bash
#SBATCH -p mono
#SBATCH -w mono-rpi-5b
#SBATCH --exclusive
#SBATCH -J cpufp_rpi5
#SBATCH -o logs/slurm_cpufp_rpi5_%j.out
#SBATCH -e logs/slurm_cpufp_rpi5_%j.err

cd "$HOME/Files/gemm-bench"
echo "=== Running CPUFP on RPi5 (Cortex-A76) ==="
./tools/measure_peak_cpufp.sh rpi5
EOF
)
echo "✓ Submitted CPUFP RPi5 -> Slurm Job ID: ${JOB_RPI5}"

# 2. K3: X100 + A100 (RVV 256b & 1024b)
JOB_K3=$(sbatch --parsable << 'EOF'
#!/usr/bin/env bash
#SBATCH -p mono
#SBATCH -w mono-sip-k3
#SBATCH -c 8
#SBATCH -J cpufp_k3
#SBATCH -o logs/slurm_cpufp_k3_%j.out
#SBATCH -e logs/slurm_cpufp_k3_%j.err

cd "$HOME/Files/gemm-bench"
echo "=== Running CPUFP on X100 & A100 ==="
./tools/measure_peak_cpufp.sh x100 a100
EOF
)
echo "✓ Submitted CPUFP K3 (X100 + A100) -> Slurm Job ID: ${JOB_K3}"

# 3. X60: BPi-F3 (RVV 256b)
JOB_X60=$(sbatch --parsable << 'EOF'
#!/usr/bin/env bash
#SBATCH -p mono
#SBATCH -w mono-bpi-f3
#SBATCH --exclusive
#SBATCH -J cpufp_x60
#SBATCH -o logs/slurm_cpufp_x60_%j.out
#SBATCH -e logs/slurm_cpufp_x60_%j.err

cd "$HOME/Files/gemm-bench"
echo "=== Running CPUFP on X60 (BPi-F3) ==="
./tools/measure_peak_cpufp.sh x60
EOF
)
echo "✓ Submitted CPUFP X60 (BPi-F3) -> Slurm Job ID: ${JOB_X60}"

echo
echo "=== Current Slurm Queue for $USER ==="
squeue -u "$USER"
