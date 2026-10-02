#!/usr/bin/env bash

set -euo pipefail

################################################################################
# Orchestrateur Multi-Plateformes Dalek (LIP6)
#
# Pilote les builds, tests Catch2 et benchmarks GEMM à distance via Slurm
# depuis la frontale front.dalek.lip6 sur le NFS partagé ~/Files/gemm-bench.
################################################################################

FRONT_HOST="${DALEK_FRONT:-front.dalek.lip6}"
REMOTE_DIR="${DALEK_DIR:-~/Files/gemm-bench}"

# Couleurs d'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

declare -A TARGET_CHAMPIONS=(
    [x100]="mippv2_x100_register_blocked_apack4 mippv2_x100_register_blocked_apack4_crr"
    [a100]="mippv2_a100_mr7_nr2_lmul2_pipe mippv2_a100_mr7_nr2_lmul2_pipe_crr"
    [x60]="mippv2_a100_mr7_nr4_pipe mippv2_a100_mr7_nr4_pipe_crr"
    [rpi5]="mippv2_a76_mr6_nr3 mippv2_a76_mr6_nr4_fmaddi_crr"
    [m1]="mippv2_x60_mr6_nr4_fmaddi mippv2_x60_mr6_nr4_fmaddi_crr"
    [zen4]="mippv2_firestorm_mr4_nr4_fmaddi mippv2_firestorm_mr4_nr4_fmaddi_crr"
    [zen5]="mippv2_firestorm_mr4_nr4_fmaddi mippv2_firestorm_mr4_nr4_fmaddi_crr"
    [meteorlake]="mippv2_meteorlake_mr4_nr3 mippv2_meteorlake_mr4_nr3_crr"
)

get_target_champions() {
    local target="$1"
    local raw="${TARGET_CHAMPIONS[$target]:-}"
    if [[ "$CRR_ONLY" == "true" ]]; then
        local filtered=""
        for k in $raw; do
            if [[ "$k" == *_crr ]]; then
                filtered+="$k "
            fi
        done
        echo "${filtered% }"
    elif [[ "$RRR_ONLY" == "true" ]]; then
        local filtered=""
        for k in $raw; do
            if [[ "$k" != *_crr ]]; then
                filtered+="$k "
            fi
        done
        echo "${filtered% }"
    else
        echo "$raw"
    fi
}

show_help() {
    echo -e "${BOLD}Usage:${NC} $0 [options] [cibles...]

${BOLD}Cibles disponibles :${NC}
  x100         SpacemiT X100 (mono-sip-k3, RVV 256-bit) -> mippv2_x100_register_blocked_apack4 (+ _crr)
  a100         SpacemiT A100 (mono-sip-k3, RVV 1024-bit, via ai) -> mippv2_a100_mr7_nr2_lmul2_pipe (+ _crr)
  x60          SpacemiT X60 (mono-bpi-f3, RVV 256-bit) -> mippv2_a100_mr7_nr4_pipe (+ _crr)
  rpi5         Raspberry Pi 5 (mono-rpi-5b, Cortex-A76 NEON) -> mippv2_a76_mr6_nr3 (+ _crr)
  m1           Apple M1 (m1u, Firestorm NEON, core 3) -> mippv2_x60_mr6_nr4_fmaddi (+ _crr)
  zen4         AMD Zen 4 (az4-a7900-3, AVX-512) -> mippv2_firestorm_mr4_nr4_fmaddi (+ _crr)
  zen5         AMD Zen 5 (az5-a890m-0, AVX-512) -> mippv2_firestorm_mr4_nr4_fmaddi (+ _crr)
  meteorlake   Intel Meteor Lake (iml-ia770-3, AVX2, core 0) -> mippv2_meteorlake_mr4_nr3 (+ _crr)
  all          Toutes les cibles ci-dessus (défaut)

${BOLD}Options :${NC}
  --no-pull         Ne pas exécuter 'git pull' sur la frontale avant les jobs
  --tests-only      Exécuter uniquement la suite de tests Catch2 (pas de benchmarks)
  --bench-only      Exécuter uniquement les benchmarks (pas de tests préliminaires)
  --sequential      Exécuter les cibles séquentiellement (par défaut : parallèle)
  --rebuild         Forcer la recompilation complète sur les nœuds
  --crr-only        Exécuter uniquement les variantes CRR des champions
  --rrr-only        Exécuter uniquement les variantes RRR des champions
  -r, --repetitions <N> Nombre de répétitions par mesure (défaut: 15)
  --cooldown <sec>  Temps de pause thermique après grandes tailles (défaut: 0.5)
  -k, --kernel <n>  Exécuter un kernel spécifique (répétable, remplace le champion par défaut)
  --run-all         Exécuter tous les kernels (champions + exploration) et la version scalaire
  -s, --sizes \"...\" Tailles de matrices pour les benchmarks (ex: \"32 64 96 128\")
  -h, --help        Afficher cette aide

${BOLD}Exemples :${NC}
  $0                                # Pull + tests + benchs champions sur toutes les machines en parallèle
  $0 all --tests-only               # Tests Catch2 sur tous les nœuds simultanément
  $0 x100 a100                      # Uniquement K3 en parallèle (champions x100 & a100)
  $0 rpi5 zen5                      # Benchmarks RPi5 et Zen5
  $0 zen4 meteorlake --tests-only   # Valider les tests sur x86 en parallèle
  $0 x100 --run-all                 # Exécution complète (scalaire + tous kernels) sur X100"
}

DO_PULL=true
TESTS_ONLY=false
BENCH_ONLY=false
SEQUENTIAL=false
RUN_ALL_FLAG=false
CRR_ONLY=false
RRR_ONLY=false
REBUILD_FLAG=""
CUSTOM_SIZES=""
CUSTOM_REPETITIONS=""
CUSTOM_COOLDOWN=""
CUSTOM_KERNELS=()
SELECTED_TARGETS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-pull)
            DO_PULL=false
            shift
            ;;
        --tests-only)
            TESTS_ONLY=true
            shift
            ;;
        --bench-only)
            BENCH_ONLY=true
            shift
            ;;
        --sequential)
            SEQUENTIAL=true
            shift
            ;;
        --parallel|--parallel-k3)
            # Conservé pour compatibilité, le parallélisme étant le comportement par défaut
            shift
            ;;
        --rebuild)
            REBUILD_FLAG="--rebuild"
            shift
            ;;
        --run-all)
            RUN_ALL_FLAG=true
            shift
            ;;
        --crr-only)
            CRR_ONLY=true
            shift
            ;;
        --rrr-only)
            RRR_ONLY=true
            shift
            ;;
        -r|--repetitions)
            CUSTOM_REPETITIONS="$2"
            shift 2
            ;;
        --cooldown)
            CUSTOM_COOLDOWN="$2"
            shift 2
            ;;
        -k|--kernel)
            CUSTOM_KERNELS+=("$2")
            shift 2
            ;;
        -s|--sizes)
            CUSTOM_SIZES="--sizes \"$2\""
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        all)
            SELECTED_TARGETS=(x100 a100 x60 rpi5 m1 zen4 zen5 meteorlake)
            shift
            ;;
        x100|a100|x60|rpi5|m1|zen4|zen5|meteorlake)
            SELECTED_TARGETS+=("$1")
            shift
            ;;
        *)
            echo -e "${RED}Erreur : Cible ou option inconnue '$1'${NC}"
            show_help
            exit 1
            ;;
    esac
done

if [[ ${#SELECTED_TARGETS[@]} -eq 0 ]]; then
    SELECTED_TARGETS=(x100 a100 x60 rpi5 m1 zen4 zen5 meteorlake)
fi

echo -e "${BOLD}${CYAN}================================================================================${NC}"
echo -e "${BOLD}${CYAN}   GEMMBench — Orchestrateur Multi-Plateformes Dalek (LIP6)                    ${NC}"
echo -e "${BOLD}${CYAN}================================================================================${NC}"
echo -e "Frontale : ${BOLD}${FRONT_HOST}${NC}"
echo -e "Dépôt    : ${BOLD}${REMOTE_DIR}${NC}"
echo -e "Cibles   : ${BOLD}${SELECTED_TARGETS[*]}${NC}"
if [[ "$RUN_ALL_FLAG" == "true" ]]; then
    echo -e "Mode     : ${YELLOW}Complet (--run-all : scalaires + champions + exploration)${NC}"
elif [[ ${#CUSTOM_KERNELS[@]} -gt 0 ]]; then
    echo -e "Mode     : ${YELLOW}Personnalisé (${CUSTOM_KERNELS[*]})${NC}"
elif [[ "$CRR_ONLY" == "true" ]]; then
    echo -e "Mode     : ${GREEN}Champions CRR uniquement (--crr-only)${NC}"
elif [[ "$RRR_ONLY" == "true" ]]; then
    echo -e "Mode     : ${GREEN}Champions RRR uniquement (--rrr-only)${NC}"
else
    echo -e "Mode     : ${GREEN}Champions RRR + CRR (baseline scalaire exclue)${NC}"
fi
echo

# -----------------------------------------------------------------------------
# 1. Étape de Synchronisation unique sur le NFS via la frontale
# -----------------------------------------------------------------------------
if [[ "$DO_PULL" == "true" ]]; then
    local_branch="$(git rev-parse --abbrev-ref HEAD)"
    echo -e "${BLUE}[1/3] Synchronisation du dépôt NFS via ${FRONT_HOST} (branche: ${local_branch})...${NC}"
    if ssh "$FRONT_HOST" "cd ${REMOTE_DIR} && git fetch origin && git checkout ${local_branch} && git pull origin ${local_branch}"; then
        echo -e "${GREEN}✓ Dépôt synchronisé sur la branche '${local_branch}' sur le NFS partagé.${NC}\n"
    else
        echo -e "${RED}✗ Échec de la synchronisation de la branche '${local_branch}' sur ${FRONT_HOST}.${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}[1/3] Synchronisation git ignorée (--no-pull).${NC}\n"
fi

# -----------------------------------------------------------------------------
# 1bis. Archivage des résultats antérieurs sur la frontale
# -----------------------------------------------------------------------------
echo -e "${BLUE}[+] Archivage des résultats antérieurs sur ${FRONT_HOST}...${NC}"
ssh "$FRONT_HOST" "bash -l -c 'cd ${REMOTE_DIR} && if [ -d results ] && [ \"\$(ls -A results 2>/dev/null)\" ]; then ARCHIVE_DIR=\"results_archive_\$(date +%Y%m%d_%H%M%S)\"; mkdir -p \"\$ARCHIVE_DIR\" && cp -r results/* \"\$ARCHIVE_DIR/\" 2>/dev/null || true; echo \"✓ Résultats archivés dans \$ARCHIVE_DIR\"; fi'"

# -----------------------------------------------------------------------------
# 1ter. Pré-compilation croisée X60 (Clang) sur le nœud X100 (mono-sip-k3)
# -----------------------------------------------------------------------------
has_x60=false
for t in "${SELECTED_TARGETS[@]}"; do
    [[ "$t" == "x60" ]] && has_x60=true
done
if [[ "$has_x60" == "true" ]]; then
    echo -e "${CYAN}→ Pré-compilation du binaire X60 (Clang) sur le nœud X100 (mono-sip-k3)...${NC}"
    ssh "$FRONT_HOST" "srun -p mono -w mono-sip-k3 bash -l -c 'cd ${REMOTE_DIR} && cmake -B build_rvv_x60_clang_rvv -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS=\"-O3 -ffast-math -finline-functions -funroll-loops -fno-semantic-interposition -falign-functions=64 -falign-loops=32 -march=rv64gcv_zvl256b -mrvv-vector-bits=zvl\" -DGEMMBENCH_ENABLE_EXPLO=OFF && cmake --build build_rvv_x60_clang_rvv --target GemmBench --parallel'"
fi

# -----------------------------------------------------------------------------
# 2. Définition des fonctions d'exécution par cible
# -----------------------------------------------------------------------------

declare -A RESULTS_STATUS
declare -A RESULTS_BENCH

run_target() {
    local target="$1"
    local partition=""
    local node=""
    local platform=""
    local pre_cmd=""
    local wrapper=""
    local srun_extra=""
    local compilers="gcc clang"

    local out_prefix=""

    case "$target" in
        x100)
            partition="mono"
            node="mono-sip-k3"
            platform="rvv_x100"
            out_prefix="x100_rvv"
            srun_extra="-c 8"
            ;;
        a100)
            partition="mono"
            node="mono-sip-k3"
            platform="rvv_a100"
            out_prefix="a100_rvv"
            wrapper="ai "
            srun_extra="-c 8"
            ;;
        x60)
            partition="mono"
            node="mono-bpi-f3"
            platform="rvv_x60"
            out_prefix="x60_rvv"
            pre_cmd="module load gcc 2>/dev/null || true;"
            srun_extra="--exclusive"
            compilers="gcc clang"
            ;;
        rpi5)
            partition="mono"
            node="mono-rpi-5b"
            platform="neon"
            out_prefix="rpi5_neon"
            srun_extra="--exclusive"
            ;;
        m1)
            partition="special"
            node="m1u"
            platform="neon"
            out_prefix="m1_neon"
            srun_extra="--exclusive"
            ;;
        zen4)
            partition="az4-a7900"
            node="az4-a7900-3"
            platform="avx512"
            out_prefix="zen4_avx512"
            pre_cmd="module load catch2 2>/dev/null || true;"
            srun_extra="--exclusive"
            ;;
        zen5)
            partition="az5-a890m"
            node="az5-a890m-0"
            platform="avx512"
            out_prefix="zen5_avx512"
            pre_cmd="module load catch2 2>/dev/null || true;"
            srun_extra="--exclusive"
            ;;
        meteorlake)
            partition="iml-ia770"
            node="iml-ia770-0"
            platform="avx2"
            out_prefix="meteorlake_avx2"
            pre_cmd="module load catch2 2>/dev/null || true;"
            srun_extra="--exclusive"
            ;;
    esac


    local target_champion="$(get_target_champions "$target")"

    local bench_flags=""
    [[ -n "$out_prefix" ]] && bench_flags+="-o ${out_prefix} "
    [[ -n "$REBUILD_FLAG" ]] && bench_flags+="${REBUILD_FLAG} "
    [[ -n "$CUSTOM_SIZES" ]] && bench_flags+="${CUSTOM_SIZES} "
    [[ -n "$CUSTOM_REPETITIONS" ]] && bench_flags+="-r ${CUSTOM_REPETITIONS} "
    [[ -n "$CUSTOM_COOLDOWN" ]] && bench_flags+="--cooldown ${CUSTOM_COOLDOWN} "
    [[ "$CRR_ONLY" == "true" ]] && bench_flags+="--crr-only "
    [[ "$RRR_ONLY" == "true" ]] && bench_flags+="--rrr-only "

    if [[ "$BENCH_ONLY" == "true" ]]; then
        :
    elif [[ "$TESTS_ONLY" == "true" ]]; then
        bench_flags+="--tests-only "
    else
        bench_flags+="--tests "
    fi

    if [[ "$RUN_ALL_FLAG" == "true" ]]; then
        bench_flags+="--run-all "
    elif [[ ${#CUSTOM_KERNELS[@]} -gt 0 ]]; then
        for ck in "${CUSTOM_KERNELS[@]}"; do
            bench_flags+="-k ${ck} "
        done
    elif [[ -n "$target_champion" ]]; then
        for ck in $target_champion; do
            bench_flags+="-k ${ck} "
        done
    fi

    local target_bench_flags="$bench_flags"
    if [[ "$target" == "x60" ]]; then
        # Le binaire Clang a été compilé sur X100; éviter qu'un rebuild local n'appelle clang++ défaillant sur x60
        target_bench_flags="${target_bench_flags//--rebuild/}"
        target_bench_flags="${target_bench_flags//--tests /}"
    fi

    local bench_cmd="${pre_cmd} PLATFORM_TAG=${target} ${wrapper}./run_benchmarks.sh ${platform} ${target_bench_flags} ${compilers}"

    if ssh "$FRONT_HOST" "srun -p ${partition} -w ${node} ${srun_extra} bash -l -c 'cd ${REMOTE_DIR} && ${bench_cmd}'"; then
        return 0
    else
        return 1
    fi
}

# -----------------------------------------------------------------------------
# 3. Exécution séquentielle ou parallèle
# -----------------------------------------------------------------------------
echo -e "${BLUE}[2/3] Exécution des jobs sur les nœuds Dalek...${NC}\n"

LOG_DIR="logs/cluster_runs"
mkdir -p "$LOG_DIR"

if [[ "$SEQUENTIAL" == "true" || ${#SELECTED_TARGETS[@]} -eq 1 ]]; then
    for target in "${SELECTED_TARGETS[@]}"; do
        champ_desc="champions: $(get_target_champions "$target")"
        [[ "$RUN_ALL_FLAG" == "true" ]] && champ_desc="all kernels"
        [[ ${#CUSTOM_KERNELS[@]} -gt 0 ]] && champ_desc="${CUSTOM_KERNELS[*]}"
        echo -e "${BOLD}--------------------------------------------------------------------------------${NC}"
        echo -e "${BOLD}Exécution sur ${CYAN}${target}${NC} (${champ_desc})...${NC}"
        echo -e "${BOLD}--------------------------------------------------------------------------------${NC}"
        if run_target "$target"; then
            echo -e "${GREEN}✓ Succès sur ${target}.${NC}"
            [[ "$BENCH_ONLY" != "true" ]] && RESULTS_STATUS["$target"]="PASS"
            [[ "$TESTS_ONLY" != "true" ]] && RESULTS_BENCH["$target"]="DONE"
        else
            echo -e "${RED}✗ Échec sur ${target} !${NC}"
            [[ "$BENCH_ONLY" != "true" ]] && RESULTS_STATUS["$target"]="FAIL"
            [[ "$TESTS_ONLY" != "true" ]] && RESULTS_BENCH["$target"]="ERROR"
        fi
    done
else
    echo -e "${CYAN}→ Lancement simultané en parallèle sur toutes les cibles : ${BOLD}${SELECTED_TARGETS[*]}${NC}"
    echo -e "${CYAN}→ Les sorties détaillées sont enregistrées dans ${BOLD}${LOG_DIR}/<cible>.log${NC}\n"

    declare -A TARGET_PIDS
    declare -A TARGET_START_TIME

    for target in "${SELECTED_TARGETS[@]}"; do
        TARGET_START_TIME["$target"]=$(date +%s)
        log_file="${LOG_DIR}/${target}.log"
        status_file="${LOG_DIR}/${target}.status"
        rm -f "$status_file"

        champ_desc="champions: $(get_target_champions "$target")"
        [[ "$RUN_ALL_FLAG" == "true" ]] && champ_desc="all kernels"
        [[ ${#CUSTOM_KERNELS[@]} -gt 0 ]] && champ_desc="${CUSTOM_KERNELS[*]}"

        (
            if run_target "$target" > "$log_file" 2>&1; then
                echo "0" > "$status_file"
            else
                echo "1" > "$status_file"
            fi
        ) &
        TARGET_PIDS["$target"]=$!
        echo -e "  [+] Job Slurm lancé pour ${BOLD}${target}${NC} (${CYAN}${champ_desc}${NC}, PID background ${TARGET_PIDS[$target]})"
    done

    echo

    # Surveillance de la fin des jobs
    REMAINING_TARGETS=("${SELECTED_TARGETS[@]}")
    while [[ ${#REMAINING_TARGETS[@]} -gt 0 ]]; do
        NEW_REMAINING=()
        for target in "${REMAINING_TARGETS[@]}"; do
            pid="${TARGET_PIDS[$target]}"
            if kill -0 "$pid" 2>/dev/null; then
                NEW_REMAINING+=("$target")
            else
                wait "$pid" || true
                duration=$(( $(date +%s) - TARGET_START_TIME["$target"] ))
                status_file="${LOG_DIR}/${target}.status"
                log_file="${LOG_DIR}/${target}.log"
                exit_code=1
                [[ -f "$status_file" ]] && exit_code=$(cat "$status_file")

                if [[ "$exit_code" -eq 0 ]]; then
                    echo -e "${GREEN}✓ [${target}] Succès en ${duration}s${NC}"
                    [[ "$BENCH_ONLY" != "true" ]] && RESULTS_STATUS["$target"]="PASS"
                    [[ "$TESTS_ONLY" != "true" ]] && RESULTS_BENCH["$target"]="DONE"
                else
                    echo -e "${RED}✗ [${target}] Échec en ${duration}s (voir ${log_file}) !${NC}"
                    [[ "$BENCH_ONLY" != "true" ]] && RESULTS_STATUS["$target"]="FAIL"
                    [[ "$TESTS_ONLY" != "true" ]] && RESULTS_BENCH["$target"]="ERROR"
                    echo -e "${YELLOW}--- Dernières lignes du journal (${target}) ---${NC}"
                    tail -n 15 "$log_file" | sed 's/^/    /' || true
                    echo -e "${YELLOW}---------------------------------------------${NC}"
                fi
            fi
        done
        REMAINING_TARGETS=("${NEW_REMAINING[@]}")
        [[ ${#REMAINING_TARGETS[@]} -gt 0 ]] && sleep 2
    done
fi

# -----------------------------------------------------------------------------
# 4. Synthèse globale des résultats
# -----------------------------------------------------------------------------
echo
echo -e "${BOLD}${CYAN}================================================================================${NC}"
echo -e "${BOLD}${CYAN}   Synthèse Globale de l'Exécution Dalek                                       ${NC}"
echo -e "${BOLD}${CYAN}================================================================================${NC}"
printf "%-15s | %-15s | %-15s\n" "Cible" "Tests Catch2" "Benchmarks"
echo "----------------+-----------------+-----------------"

for target in "${SELECTED_TARGETS[@]}"; do
    status_test="${RESULTS_STATUS[$target]:-UNKNOWN}"
    status_bench="${RESULTS_BENCH[$target]:-SKIPPED}"

    test_colored="$status_test"
    [[ "$status_test" == "PASS" ]] && test_colored="${GREEN}PASS${NC}"
    [[ "$status_test" == "FAIL" ]] && test_colored="${RED}FAIL${NC}"

    bench_colored="$status_bench"
    [[ "$status_bench" == "DONE" ]] && bench_colored="${GREEN}DONE${NC}"
    [[ "$status_bench" == "ERROR" ]] && bench_colored="${RED}ERROR${NC}"

    printf "%-15s | %-24b | %-24b\n" "$target" "$test_colored" "$bench_colored"
done
echo
echo -e "Les fichiers de résultats CSV sont disponibles dans le dossier partagé : ${BOLD}${REMOTE_DIR}/results/${NC}"

# -----------------------------------------------------------------------------
# 5. Rapatriement automatique des CSV vers la machine locale
# -----------------------------------------------------------------------------
echo
echo -e "${BLUE}[3/3] Rapatriement (scp) des résultats CSV vers la machine locale...${NC}"
mkdir -p gemm-bench-results/crr gemm-bench-results/rrr
if scp -q "${FRONT_HOST}:${REMOTE_DIR}/results/*_crr_*.csv" gemm-bench-results/crr/ 2>/dev/null; then
    echo -e "${GREEN}✓ Résultats CRR rapatriés avec succès dans gemm-bench-results/crr/${NC}"
    ls -lh gemm-bench-results/crr/
fi
if scp -q "${FRONT_HOST}:${REMOTE_DIR}/results/*_rrr_*.csv" gemm-bench-results/rrr/ 2>/dev/null; then
    echo -e "${GREEN}✓ Résultats RRR rapatriés avec succès dans gemm-bench-results/rrr/${NC}"
    ls -lh gemm-bench-results/rrr/
fi

