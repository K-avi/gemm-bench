#!/usr/bin/env bash

set -euo pipefail

################################################################################
# CPUFP Automated Peak Double-Precision (FP64) Floating-Point Benchmark Tool
#
# Builds and executes cpufp (https://github.com/pigirons/cpufp) across
# local and Dalek cluster targets, measures empirical peak DP GFLOP/s,
# calculates FLOP/cycle, and formats a comparative microarchitectural report.
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
FRONT_HOST="${DALEK_FRONT:-front.dalek.lip6}"
REMOTE_DIR="${DALEK_DIR:-$HOME/Files/gemm-bench}"
CPUFP_REMOTE_SRC="${CPUFP_REMOTE_SRC:-$HOME/cpufp-fix}"
OUTPUT_DIR="${ROOT_DIR}/results/cpufp"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

mkdir -p "${OUTPUT_DIR}"

show_help() {
    echo -e "${BOLD}Usage:${NC} $0 [options] [cibles...]"
    echo
    echo -e "${BOLD}Cibles disponibles :${NC}"
    echo "  skylake / local Intel Core i5-6200U (AVX2, Core 0, local)"
    echo "  zen4            AMD Ryzen 9 7900X (az4-a7900-3, AVX-512, Core 0)"
    echo "  zen5            AMD Ryzen AI 9 HX 370 (az5-a890m-0, AVX-512, Core 0)"
    echo "  meteorlake      Intel Ultra 7 Redwood Cove (iml-ia770-3, AVX2, Core 0)"
    echo "  m1              Apple M1 Firestorm (m1u, NEON asimd, Core 3 P-core)"
    echo "  rpi5            Raspberry Pi 5 Cortex-A76 (mono-rpi-5b, NEON asimd, Core 0)"
    echo "  x100            SpacemiT X100 (mono-sip-k3, RVV 256b, Core 0)"
    echo "  a100            SpacemiT A100 (mono-sip-k3, RVV 1024b via ai, Core 0)"
    echo "  x60             SpacemiT X60 (mono-bpi-f3, RVV 256b, Core 0)"
    echo "  all             Toutes les cibles ci-dessus"
    echo
    echo -e "${BOLD}Options :${NC}"
    echo "  --rebuild       Forcer la recompilation complète de cpufp"
    echo "  --output-dir    Répertoire de destination des résultats (défaut: results/cpufp)"
    echo "  -h, --help      Afficher cette aide"
}

FORCE_REBUILD=false
SELECTED_TARGETS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rebuild)
            FORCE_REBUILD=true
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            mkdir -p "${OUTPUT_DIR}"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        all)
            SELECTED_TARGETS=(skylake zen4 zen5 meteorlake m1 rpi5 x100 a100 x60)
            shift
            ;;
        skylake|local|zen4|zen5|meteorlake|m1|rpi5|x100|a100|x60)
            SELECTED_TARGETS+=("$1")
            shift
            ;;
        *)
            echo -e "${RED}Erreur : Cible inconnue '$1'${NC}"
            show_help
            exit 1
            ;;
    esac
done

if [[ ${#SELECTED_TARGETS[@]} -eq 0 ]]; then
    SELECTED_TARGETS=(skylake zen4 zen5 meteorlake m1)
fi

echo -e "${BOLD}${CYAN}================================================================================${NC}"
echo -e "${BOLD}${CYAN}   CPUFP — Empirical Peak Double-Precision (FP64) Benchmark Runner             ${NC}"
echo -e "${BOLD}${CYAN}================================================================================${NC}"
echo -e "Cibles sélectionnées : ${BOLD}${SELECTED_TARGETS[*]}${NC}"
echo -e "Dossier de sortie    : ${BOLD}${OUTPUT_DIR}${NC}\n"

# -----------------------------------------------------------------------------
# Exécution par cible
# -----------------------------------------------------------------------------

run_local_skylake() {
    local target="skylake"
    local raw_out="${OUTPUT_DIR}/${target}_cpufp.txt"
    echo -e "${BLUE}→ Exécution locale sur Skylake (Core 0)...${NC}"

    local cpufp_dir="${ROOT_DIR}/third_party/cpufp"
    if [[ ! -d "${cpufp_dir}" ]]; then
        echo -e "${RED}✗ Répertoire ${cpufp_dir} introuvable.${NC}"
        return 1
    fi

    if [[ "$FORCE_REBUILD" == "true" || ! -f "${cpufp_dir}/cpufp" ]]; then
        echo -e "  [+] Compilation locale cpufp (x64)..."
        (cd "${cpufp_dir}" && ./build_x64.sh >/dev/null 2>&1)
    fi

    "${cpufp_dir}/cpufp" --thread_pool=[0] > "${raw_out}" 2>&1
    echo -e "${GREEN}✓ Résultat Skylake enregistré dans ${raw_out}${NC}"
}

run_cluster_target() {
    local target="$1"
    local partition=""
    local node=""
    local arch=""
    local core=0
    local wrapper=""
    local srun_extra=""

    case "$target" in
        zen4)
            partition="az4-a7900"
            node="az4-a7900-3"
            arch="x64"
            core=0
            srun_extra="--exclusive"
            ;;
        zen5)
            partition="az5-a890m"
            node="az5-a890m-0"
            arch="x64"
            core=0
            srun_extra="--exclusive"
            ;;
        meteorlake)
            partition="iml-ia770"
            node=""
            arch="x64"
            core=0
            srun_extra="--exclusive"
            ;;
        m1)
            partition="special"
            node="m1u"
            arch="arm64"
            core=3
            srun_extra="--exclusive"
            ;;
        rpi5)
            partition="mono"
            node="mono-rpi-5b"
            arch="arm64"
            core=0
            srun_extra="--exclusive"
            ;;
        x100)
            partition="mono"
            node="mono-sip-k3"
            arch="riscv64"
            core=0
            srun_extra="-c 8"
            ;;
        a100)
            partition="mono"
            node="mono-sip-k3"
            arch="riscv64"
            wrapper="ai "
            core=0
            srun_extra="-c 8"
            ;;
        x60)
            partition="mono"
            node="mono-bpi-f3"
            arch="riscv64"
            core=0
            srun_extra="--exclusive"
            ;;
    esac

    local raw_out="${OUTPUT_DIR}/${target}_cpufp.txt"
    echo -e "${BLUE}→ Exécution sur Dalek : ${BOLD}${target}${NC} (nœud: ${node}, core: ${core})...${NC}"

    local remote_script="
set -euo pipefail
REMOTE_BASE=\"\${DALEK_DIR:-\$HOME/Files/gemm-bench}\"
CPUFP_SRC=\"\${CPUFP_SRC:-\$HOME/cpufp-fix}\"
BUILD_DIR=\"\${REMOTE_BASE}/build_cpufp_${target}\"
if [[ \"${FORCE_REBUILD}\" == \"true\" || ! -f \"\${BUILD_DIR}/cpufp\" ]]; then
    mkdir -p \"\${BUILD_DIR}\"
    rsync -a --exclude='.git' --exclude='build_dir' \"\${CPUFP_SRC}/\" \"\${BUILD_DIR}/\"
    cd \"\${BUILD_DIR}\"
    ./build_${arch}.sh >/dev/null 2>&1
fi
cd \"\${BUILD_DIR}\"
${wrapper}./cpufp --thread_pool=[${core}]
"

    local node_opt=""
    [[ -n "${node}" ]] && node_opt="-w ${node}"
    if ssh "$FRONT_HOST" "srun -p ${partition} ${node_opt} ${srun_extra} bash -l -c '${remote_script}'" > "${raw_out}" 2>&1; then
        echo -e "${GREEN}✓ Résultat ${target} enregistré dans ${raw_out}${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ Échec ou nœud occupé pour ${target} (voir ${raw_out})${NC}"
        return 1
    fi
}

for target in "${SELECTED_TARGETS[@]}"; do
    if [[ "$target" == "skylake" || "$target" == "local" ]]; then
        run_local_skylake || true
    else
        run_cluster_target "$target" || true
    fi
done

# -----------------------------------------------------------------------------
# Analyse des sorties & Génération du rapport comparatif
# -----------------------------------------------------------------------------

echo
echo -e "${CYAN}→ Analyse et extraction des débits FP64 crête...${NC}"

python3 - << 'EOF'
import json
import re
from pathlib import Path

results_dir = Path("results/cpufp")
config_path = Path("uarch_config.json")

with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

# Flatten config
specs = {}
for family, uarchs in config.items():
    for uid, spec in uarchs.items():
        spec["uarch_id"] = uid
        spec["simd"] = family
        specs[uid] = spec

# Map target names to uarch ids
target_map = {
    "skylake": "skylake",
    "local": "skylake",
    "zen4": "zen4",
    "zen5": "zen5",
    "meteorlake": "meteorlake",
    "m1": "m1",
    "rpi5": "rpi5",
    "x100": "x100",
    "a100": "a100",
    "x60": "x60",
}

records = []

for log_path in sorted(results_dir.glob("*_cpufp.txt")):
    target = log_path.stem.replace("_cpufp", "")
    uid = target_map.get(target, target)
    spec = specs.get(uid, {})

    content = log_path.read_text(encoding="utf-8", errors="ignore")
    lines = content.splitlines()

    best_dp_gflops = 0.0
    best_row_info = None

    for line in lines:
        if "|" not in line or "Instruction Set" in line or "---" in line:
            continue
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) >= 3:
            # Check if this row is FP64 / f64
            # Formats:
            # 4 parts: [ISA, Vector Length, Computation, Performance]
            # 3 parts: [ISA, Computation, Performance]
            comp_str = parts[-2]
            perf_str = parts[-1]
            if "f64" in comp_str.lower() or "fp64" in comp_str.lower():
                # Extract GFLOPS number
                m = re.search(r"([\d\.]+)\s*GFLOPS", perf_str, re.IGNORECASE)
                if m:
                    gflops = float(m.group(1))
                    if gflops > best_dp_gflops:
                        best_dp_gflops = gflops
                        isa = parts[0]
                        vlen = parts[1] if len(parts) == 4 else ""
                        best_row_info = {
                            "isa": isa,
                            "vlen": vlen,
                            "computation": comp_str,
                            "gflops": gflops
                        }

    if best_row_info:
        freq = float(spec.get("frequency_ghz", 1.0))
        theo_flop_cyc = float(spec.get("peak_flop_per_cycle", 16.0))
        theo_peak_gflops = freq * theo_flop_cyc
        measured_flop_cyc = best_dp_gflops / freq if freq > 0 else 0.0
        eff_pct = (best_dp_gflops / theo_peak_gflops) * 100.0 if theo_peak_gflops > 0 else 0.0

        records.append({
            "target": target,
            "uarch_name": spec.get("name", target),
            "cpu_model": spec.get("cpu_model", ""),
            "freq_ghz": freq,
            "isa": best_row_info["isa"],
            "vlen": best_row_info["vlen"],
            "computation": best_row_info["computation"],
            "peak_dp_gflops": best_dp_gflops,
            "measured_flop_per_cycle": measured_flop_cyc,
            "theo_flop_per_cycle": theo_flop_cyc,
            "efficiency_pct": eff_pct
        })

if records:
    # Save CSV
    csv_path = results_dir / "cpufp_peak_dp_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Target,UArchName,CpuModel,FreqGHz,ISA,VectorLength,Computation,Peak_DP_GFLOPS,Measured_FLOP_per_cycle,Theo_FLOP_per_cycle,Efficiency_pct\n")
        for r in records:
            f.write(f"{r['target']},{r['uarch_name']},{r['cpu_model']},{r['freq_ghz']},{r['isa']},{r['vlen']},{r['computation']},{r['peak_dp_gflops']:.3f},{r['measured_flop_per_cycle']:.2f},{r['theo_flop_per_cycle']:.1f},{r['efficiency_pct']:.1f}\n")
    print(f"✓ Synthèse CSV générée dans : {csv_path}\n")

    # Print summary table
    print("=" * 135)
    print("  CPUFP — Empirical Peak Double-Precision (FP64) Floating-Point Measurements")
    print("=" * 135)
    header = f"{'Target':<12} | {'Microarchitecture':<24} | {'ISA / Vector':<18} | {'Core Computation':<23} | {'Peak GFLOP/s':<13} | {'FLOP/cyc':<9} | {'% Theo Peak':<10}"
    print(header)
    print("-" * 135)
    for r in records:
        isa_str = f"{r['isa']} ({r['vlen']})" if r['vlen'] else r['isa']
        row = (
            f"{r['target']:<12} | "
            f"{r['uarch_name']:<24} | "
            f"{isa_str:<18} | "
            f"{r['computation']:<23} | "
            f"{r['peak_dp_gflops']:<13.2f} | "
            f"{r['measured_flop_per_cycle']:<9.2f} | "
            f"{r['efficiency_pct']:<9.1f}%"
        )
        print(row)
    print("=" * 135 + "\n")
else:
    print("Aucune mesure DP valide trouvée dans les logs.")
EOF
