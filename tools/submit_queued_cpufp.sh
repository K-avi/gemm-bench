#!/usr/bin/env bash

set -euo pipefail

REMOTE_DIR="${DALEK_DIR:-$HOME/Files/gemm-bench}"
cd "${REMOTE_DIR}"

mkdir -p logs results/cpufp

FP32_FLAG=""
PREC_LABEL="FP64"
if [[ "${1:-}" == "--fp32" ]]; then
    FP32_FLAG="--fp32"
    PREC_LABEL="FP32"
fi

echo "=== Submitting CPUFP Empirical Peak ${PREC_LABEL} Jobs to Slurm ==="

# 1. RPi5 (Cortex-A76)
JOB_RPI5=$(sbatch --parsable << EOF
#!/usr/bin/env bash
#SBATCH -p mono
#SBATCH -w mono-rpi-5b
#SBATCH --exclusive
#SBATCH -J cpufp_rpi5_${PREC_LABEL,,}
#SBATCH -o logs/slurm_cpufp_rpi5_%j.out
#SBATCH -e logs/slurm_cpufp_rpi5_%j.err

cd "$HOME/Files/gemm-bench"
echo "=== Running CPUFP on RPi5 (Cortex-A76) [${PREC_LABEL}] ==="
./tools/measure_peak_cpufp.sh ${FP32_FLAG} rpi5
EOF
)
echo "✓ Submitted CPUFP RPi5 -> Slurm Job ID: ${JOB_RPI5}"

# 2. K3: X100 + A100 (RVV 256b & 1024b)
JOB_K3=$(sbatch --parsable << EOF
#!/usr/bin/env bash
#SBATCH -p mono
#SBATCH -w mono-sip-k3
#SBATCH -c 8
#SBATCH -J cpufp_k3_${PREC_LABEL,,}
#SBATCH -o logs/slurm_cpufp_k3_%j.out
#SBATCH -e logs/slurm_cpufp_k3_%j.err

cd "$HOME/Files/gemm-bench"
echo "=== Running CPUFP on X100 & A100 [${PREC_LABEL}] ==="
./tools/measure_peak_cpufp.sh ${FP32_FLAG} x100 a100
EOF
)
echo "✓ Submitted CPUFP K3 (X100 + A100) -> Slurm Job ID: ${JOB_K3}"

# 3. X60: BPi-F3 (RVV 256b)
JOB_X60=$(sbatch --parsable << EOF
#!/usr/bin/env bash
#SBATCH -p mono
#SBATCH -w mono-bpi-f3
#SBATCH --exclusive
#SBATCH -J cpufp_x60_${PREC_LABEL,,}
#SBATCH -o logs/slurm_cpufp_x60_%j.out
#SBATCH -e logs/slurm_cpufp_x60_%j.err

cd "$HOME/Files/gemm-bench"
echo "=== Running CPUFP on X60 (BPi-F3) [${PREC_LABEL}] ==="
./tools/measure_peak_cpufp.sh ${FP32_FLAG} x60
EOF
)
echo "✓ Submitted CPUFP X60 (BPi-F3) -> Slurm Job ID: ${JOB_X60}"

# 4. Zen 4 (Ryzen 9 7900X)
JOB_ZEN4=$(sbatch --parsable << EOF
#!/usr/bin/env bash
#SBATCH -p az4-a7900
#SBATCH -w az4-a7900-3
#SBATCH --exclusive
#SBATCH -J cpufp_zen4_${PREC_LABEL,,}
#SBATCH -o logs/slurm_cpufp_zen4_%j.out
#SBATCH -e logs/slurm_cpufp_zen4_%j.err

cd "$HOME/Files/gemm-bench"
echo "=== Running CPUFP on Zen 4 [${PREC_LABEL}] ==="
./tools/measure_peak_cpufp.sh ${FP32_FLAG} zen4
EOF
)
echo "✓ Submitted CPUFP Zen 4 -> Slurm Job ID: ${JOB_ZEN4}"

# 5. Zen 5 (Ryzen AI 9 HX 370)
JOB_ZEN5=$(sbatch --parsable << EOF
#!/usr/bin/env bash
#SBATCH -p az5-a890m
#SBATCH -w az5-a890m-0
#SBATCH --exclusive
#SBATCH -J cpufp_zen5_${PREC_LABEL,,}
#SBATCH -o logs/slurm_cpufp_zen5_%j.out
#SBATCH -e logs/slurm_cpufp_zen5_%j.err

cd "$HOME/Files/gemm-bench"
echo "=== Running CPUFP on Zen 5 [${PREC_LABEL}] ==="
./tools/measure_peak_cpufp.sh ${FP32_FLAG} zen5
EOF
)
echo "✓ Submitted CPUFP Zen 5 -> Slurm Job ID: ${JOB_ZEN5}"

# 6. Meteor Lake (Core Ultra)
JOB_IML=$(sbatch --parsable << EOF
#!/usr/bin/env bash
#SBATCH -p iml-ia770
#SBATCH --exclusive
#SBATCH -J cpufp_meteorlake_${PREC_LABEL,,}
#SBATCH -o logs/slurm_cpufp_meteorlake_%j.out
#SBATCH -e logs/slurm_cpufp_meteorlake_%j.err

cd "$HOME/Files/gemm-bench"
echo "=== Running CPUFP on Meteor Lake [${PREC_LABEL}] ==="
./tools/measure_peak_cpufp.sh ${FP32_FLAG} meteorlake
EOF
)
echo "✓ Submitted CPUFP Meteor Lake -> Slurm Job ID: ${JOB_IML}"

# 7. Apple M1 (Firestorm)
JOB_M1=$(sbatch --parsable << EOF
#!/usr/bin/env bash
#SBATCH -p special
#SBATCH -w m1u
#SBATCH --exclusive
#SBATCH -J cpufp_m1_${PREC_LABEL,,}
#SBATCH -o logs/slurm_cpufp_m1_%j.out
#SBATCH -e logs/slurm_cpufp_m1_%j.err

cd "$HOME/Files/gemm-bench"
echo "=== Running CPUFP on Apple M1 [${PREC_LABEL}] ==="
./tools/measure_peak_cpufp.sh ${FP32_FLAG} m1
EOF
)
echo "✓ Submitted CPUFP Apple M1 -> Slurm Job ID: ${JOB_M1}"

echo
echo "=== Current Slurm Queue for $USER ==="
squeue -u "$USER"
