# Journal de progression — Mesures « paper grade » MIPPv2 vs BLIS

Ce document trace l'avancement phase par phase du plan d'implémentation paper-grade (`paper_grade_plan.md`), conformément à la règle impérative 8.

---

## P1 — Provenance des binaires et des résultats

### P1.a — Build-info embarqué dans le binaire (`--build-info`)
- **Fichiers modifiés / créés** :
  - `cmake/build_info.h.in` : template de métadonnées de build dans `GEMMBench::build_info`.
  - `cmake/gen_build_info.cmake` : script CMake exécuté à chaque build via custom target pour inspecter git (`rev-parse HEAD`, `status --porcelain`), les flags effectifs, le target BLIS et fastmath, avec mise à jour atomique `copy_if_different`.
  - [CMakeLists.txt](file:///home/ivan/Files/projets/gemm_bench/CMakeLists.txt) : intégration de la cible `gemmbench_build_info`, inclusion de `${CMAKE_BINARY_DIR}/generated`, dépendance de `GemmBench` et `GemmBenchTests`.
  - [include/BenchConfig.h](file:///home/ivan/Files/projets/gemm_bench/include/BenchConfig.h) : option CLI `--build-info` affichant chaque champ clé=valeur + `compiler_macro_version=__VERSION__`, sortie immédiate `exit(0)`.
  - [tests/CMakeLists.txt](file:///home/ivan/Files/projets/gemm_bench/tests/CMakeLists.txt) : ajout de la dépendance vers `gemmbench_build_info`.
- **Commandes de vérification** :
  - `cmake -B build_check -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build_check -j`
  - `./build_check/GemmBench --build-info` : commit complet, flags, statut dirty, version compilateur affichés correctement.
  - `ctest --test-dir build_check` : 33/33 tests unitaires Catch2 passés.
- **Commit** : `ca44647` (`feat(provenance): embed git/compiler/flags build-info in GemmBench binary (--build-info)`).
- **Écarts** : Aucun.

### P1.b — Manifeste `meta.json` par CSV + garde-fous
- **Fichiers modifiés / créés** :
  - `tools/write_run_manifest.py` : script stdlib générant un JSON conforme au schéma v1 (horodatage UTC ISO 8601, sha256 du binaire, parsed `--build-info`, topologie caches sysfs `/sys/devices/system/cpu/cpu*/cache/index*`, gouverneur, fréquences start/end, uname, modèle CPU, flags ISA, flags compilateur, etc.).
  - [run_benchmarks.sh](file:///home/ivan/Files/projets/gemm_bench/run_benchmarks.sh) :
    - Garde-fou `--allow-dirty` (défaut `false`) : bloque dès le démarrage si l'arbre de travail a des modifications suivies.
    - Garde-fou `--allow-stale-binary` (défaut `false`) : vérifie avant le benchmark que le commit du binaire correspond exactement à `git rev-parse HEAD`.
    - Colonne CSV `Commit` ajoutée en dernière position.
    - Publication atomique conjointe de `${CSV}` et `${CSV%.csv}.meta.json` via `.tmp.$$` et `mv`.
- **Commandes de vérification** :
  - `./run_benchmarks.sh avx2 ...` sans `--allow-dirty` avec working tree modifié : bloqué avec erreur explicite.
  - `./run_benchmarks.sh avx2 --crr -k mippv2_meteorlake_mr4_nr3_crr -k blis_native --sizes "64 128" -r 3 --runs 2 -o selftest --allow-dirty gcc` : production correcte du CSV avec colonne `Commit` et du `meta.json` avec la section `caches` lue depuis sysfs.
- **Commit** : `4cd457c` (`feat(provenance): write per-CSV meta.json manifest, Commit column, dirty/stale-binary guards`).
- **Écarts** : Aucun.

---

## P2 — Échantillons bruts et sélection de run médian

### P2.a — C++ : export des échantillons bruts
- **Fichiers modifiés** :
  - [include/BenchConfig.h](file:///home/ivan/Files/projets/gemm_bench/include/BenchConfig.h) : ajout des champs `raw_out` et `run_id`, parsing des options CLI `--raw-out` et `--run-id`.
  - [src/main.cpp](file:///home/ivan/Files/projets/gemm_bench/src/main.cpp) :
    - `BenchmarkStats` : conservation des vecteurs `samples_s`, `samples_cyc`, `samples_ins` copiés par `compute_stats`.
    - Fonction `exportRawSamples` : écriture en append format `%.12g` des métriques par itération avec en-tête `Kernel,M,N,K,Alpha,Beta,Fastpath,CCold,Precision,RunId,Rep,Time_s,Cycles,Instructions`.

### P2.b — Script : conservation de tous les runs et sélection médiane
- **Fichiers modifiés** :
  - [run_benchmarks.sh](file:///home/ivan/Files/projets/gemm_bench/run_benchmarks.sh) :
    - Passage de `--raw-out "${TMP_RAW}" --run-id "${run_idx}"` à chaque invocation.
    - Publication atomique de `${CSV%.csv}.raw.csv`.
    - Sélection du run à médiane `floor(NUM_RUNS / 2)` au lieu du meilleur temps.
    - Colonnes `Selection` (`median_of_runs`) et `NumRuns` ajoutées après `Commit`.
- **Commandes de vérification** :
  - `./run_benchmarks.sh avx2 --crr -k blis_native --sizes "128" -r 5 --runs 3 -o selftest gcc`
  - `wc -l results/gemm_results_selftest_crr_gcc.raw.csv` : exactement 16 lignes (1 en-tête + 3 runs × 5 répétitions).
  - Vérification arithmétique manuelle : le `Time_median_s` du CSV correspond exactement à la médiane des 5 répétitions du run médian.
  - `ctest` : 33/33 tests passés.
- **Commit** : `7dd3403` (`feat(metrology): export per-repetition raw samples and select median run instead of best-of-N`).
- **Écarts** : Aucun.

---

## P3 — Correctif des itérations en mode CRR + K=384

- **Fichiers modifiés** :
  - [run_benchmarks.sh](file:///home/ivan/Files/projets/gemm_bench/run_benchmarks.sh) :
    - Fonction `crr_iters_for_k` : `iters = max(1000, 160000000 / K)` et `warmup = iters / 10`.
    - Utilisation systématique de cette mise à l'échelle pour `CRR_ONLY` (que les tailles soient spécifiées via `--sizes` ou par défaut).
    - Balayage par défaut CRR étendu à : `32 64 128 256 384 512 1024` (ajout de $K=384$).
    - Préservation stricte des chemins RRR / legacy.
- **Commandes de vérification** :
  - `bash -x ./run_benchmarks.sh avx2 --crr --sizes "256 384" -k blis_native -r 1 --runs 1 -o selftest --allow-dirty gcc 2>&1 | grep -- "--iterations"`
  - Sortie : `625000` (pour K=256) et `416666` (pour K=384).
- **Commit** : `c76925c` (`fix(bench): scale CRR iterations with K for custom sizes and add K=384 to default sweep`).
- **Écarts** : Aucun.

---

## 🔍 Revue R1 — Checkpoint Harness (P1–P3)

### 1. Diff global P1–P3
```
 CMakeLists.txt              |  44 +++++++++
 cmake/build_info.h.in       |  13 +++
 cmake/gen_build_info.cmake  |  86 +++++++++++++++++
 include/BenchConfig.h       |  29 ++++++
 run_benchmarks.sh           | 183 ++++++++++++++++++++++++++++++-----
 src/main.cpp                |  39 ++++++++
 tests/CMakeLists.txt        |   4 +
 tools/write_run_manifest.py | 230 ++++++++++++++++++++++++++++++++++++++++++++
 8 files changed, 606 insertions(+), 22 deletions(-)
```

### 2. Sortie de `./GemmBench --build-info`
```ini
git_commit=c76925ce3cf89b34051cc8605f8f3fc5477d8cd2
git_dirty=0
cxx_compiler_id=GNU
cxx_compiler_version=14.2.1
cxx_flags=-O3 -DNDEBUG
build_type=Release
fastmath=ON
blis_target=HASWELL
explo=OFF
compiler_macro_version=14.2.1 20250405
```

### 3. Extrait représentatif de `*.meta.json`
```json
{
  "schema_version": 1,
  "date_utc_start": "2026-10-22T12:22:45Z",
  "date_utc_end": "2026-10-22T12:24:00Z",
  "git_commit": "c76925ce3cf89b34051cc8605f8f3fc5477d8cd2",
  "git_dirty": 0,
  "binary_build_info": {
    "git_commit": "c76925ce3cf89b34051cc8605f8f3fc5477d8cd2",
    "git_dirty": "0",
    "cxx_compiler_id": "GNU",
    "cxx_compiler_version": "14.2.1",
    "cxx_flags": "-O3 -ffast-math -finline-functions -funroll-loops -fno-semantic-interposition -march=native -mavx2 -mfma -O3 -DNDEBUG",
    "build_type": "Release",
    "fastmath": "ON",
    "blis_target": "HASWELL",
    "explo": "OFF",
    "compiler_macro_version": "14.2.1 20250405"
  },
  "binary_sha256": "d71e37ad86037d62a66c2021bc5b1e1e28f6453520c15e7767ed354f7645bc39",
  "hostname": "sandbox",
  "slurm_job_id": null,
  "slurm_nodelist": null,
  "uname_a": "Linux sandbox 6.18.54_1 #1 SMP PREEMPT_DYNAMIC Fri Sep 25 20:32:01 UTC 2026 x86_64",
  "cpu_model": "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",
  "caches": [
    {
      "level": 1,
      "type": "Data",
      "size": "32K",
      "ways_of_associativity": 8,
      "coherency_line_size": 64
    },
    {
      "level": 2,
      "type": "Unified",
      "size": "256K",
      "ways_of_associativity": 4,
      "coherency_line_size": 64
    },
    {
      "level": 3,
      "type": "Unified",
      "size": "3072K",
      "ways_of_associativity": 12,
      "coherency_line_size": 64
    }
  ],
  "governor": "powersave",
  "pin_core": 1,
  "taskset_used": true,
  "platform": "avx2",
  "compiler_cmd": "g++",
  "compiler_version_full": "g++ (GCC) 14.2.1 20250405",
  "precision": "fp64",
  "crr_only": true,
  "num_runs": 3,
  "repetitions": 5,
  "sizes": [[128, 1250000, 125000]],
  "kernels": ["blis_native"],
  "throttle_warnings": []
}
```

### 4. Extrait représentatif de `*.raw.csv`
```csv
Kernel,M,N,K,Alpha,Beta,Fastpath,CCold,Precision,RunId,Rep,Time_s,Cycles,Instructions
blis_native,6,8,128,1,0,1,0,fp64,1,0,4.443651304e-07,1007.6208192,2841.0019248
blis_native,6,8,128,1,0,1,0,fp64,1,1,6.914763296e-07,1496.6187872,2841.0008648
blis_native,6,8,128,1,0,1,0,fp64,1,2,3.702521712e-07,840.9791064,2841.000548
blis_native,6,8,128,1,0,1,0,fp64,2,0,3.777590064e-07,857.6848952,2841.0018512
blis_native,6,8,128,1,0,1,0,fp64,2,1,3.722569152e-07,845.411552,2841.0005488
blis_native,6,8,128,1,0,1,0,fp64,2,2,3.823231144e-07,867.3735344,2841.00056
blis_native,6,8,128,1,0,1,0,fp64,3,0,3.920096048e-07,889.0514832,2841.0018672
blis_native,6,8,128,1,0,1,0,fp64,3,1,3.803424624e-07,863.567724,2841.0005576
```

### 5. Extrait représentatif de `*.csv` agrégé
```csv
Build,Kernel,M,N,K,Alpha,Beta,Time_median_s,Time_min_s,Time_max_s,Time_stddev_s,Time_cv_pct,GFLOPS_median,GFLOPS_peak,Cycles_median,Cycles_min,Cycles_max,Cycles_stddev,Cycles_cv_pct,Eff_GHz,IPC,FLOP_per_cycle_median,FLOP_per_cycle_peak,Repetitions,InterRun_CV_pct,Commit,Selection,NumRuns
native,blis_native,6,8,128,1,0,3.823231144e-07,3.722569152e-07,6.706005552e-07,1.27169877784e-07,28.40,32.1404,33.0095,867.4,845.4,1522.8,288.9,28.42,2.2687,3.28,14.1669,14.5349,5,11.50,c76925c,median_of_runs,3
```

---

## P4 — Ablations : `--no-fastpath` et mode « C froid »

### P4.a — Câblage `--no-fastpath` dans les scripts
- **Fichiers modifiés** :
  - [run_benchmarks.sh](file:///home/ivan/Files/projets/gemm_bench/run_benchmarks.sh) : option `--no-fastpath` relayée à `EXTRA_BENCH_ARGS` ; insertion du suffixe `_nofp` dans `file_suffix` ; métadonnée `--fastpath` dans l'appel `write_run_manifest.py` ; colonne `Fastpath` (0 ou 1) ajoutée au CSV après `NumRuns`.
  - [tools/run_cluster_benchmarks.sh](file:///home/ivan/Files/projets/gemm_bench/tools/run_cluster_benchmarks.sh) : option `--no-fastpath` relayée dans `bench_flags`.
- **Commandes de vérification** :
  - `./run_benchmarks.sh avx2 --crr -k blis_native --sizes "128" -r 3 --runs 2 --no-fastpath -o test_nofp --allow-dirty gcc`
  - Fichier produit : `results/gemm_results_test_nofp_crr_nofp_gcc.csv`, avec `Fastpath=0`.
- **Commit** : `37c385a` (`feat(bench): expose --no-fastpath in run scripts (suffix _nofp, Fastpath column)`).
- **Écarts** : Aucun.

### P4.b — Mode C froid (`--c-cold`)
- **Fichiers modifiés / créés** :
  - [include/ColdCPool.h](file:///home/ivan/Files/projets/gemm_bench/include/ColdCPool.h) : classe RAII `ColdCPool<T>` allouant un buffer aligné 64 B de $P \times \text{stride\_bytes}$, avec $P = 2^{\lceil \log_2(\dots) \rceil} \ge 2$, stride avec décalage de 64 B pour casser les conflits d'associativité de cache, initialisation conditionnelle (zéros si $\beta=0$ ou recopie de tuile).
  - [include/BenchConfig.h](file:///home/ivan/Files/projets/gemm_bench/include/BenchConfig.h) : options `--c-cold` et `--c-pool-mib <N>` (défaut 256 MiB).
  - [src/main.cpp](file:///home/ivan/Files/projets/gemm_bench/src/main.cpp) :
    - `measureKernel` : paramètre `const std::vector<T*> *c_pool = nullptr`. Boucle de mesure dédoublée via `if (c_pool)` pour garantir que le chemin chaud (`!c_pool`) conserve une génération de code strictement inchangée sans instruction superflue ni branche dans la boucle critique.
    - Export `CCold` (1 ou 0) dans `exportRawSamples`.
    - Contrôle NaN/Inf post-mesure sur la dernière tuile active du pool avec restauration RAII du pointeur d'origine `C.data`.
  - [tests/GemmUKernelTest.cpp](file:///home/ivan/Files/projets/gemm_bench/tests/GemmUKernelTest.cpp) : test unitaire `Cold-C pool produces identical results` validant bit-à-bit l'identité du résultat entre tuile chaude et tuile du pool froid pour `mippv2_x60_mr6_nr4_fmaddi_crr` et `blis_native`.
  - [run_benchmarks.sh](file:///home/ivan/Files/projets/gemm_bench/run_benchmarks.sh) : option `--c-cold`, suffixe `_coldc` (ordre : `_crr[_strict][_nofp][_coldc][_fp32]`), colonne `CCold` (0/1) après `Fastpath`.
  - [tools/run_cluster_benchmarks.sh](file:///home/ivan/Files/projets/gemm_bench/tools/run_cluster_benchmarks.sh) : option `--c-cold` relayée dans `bench_flags`.
- **Commandes de vérification** :
  - `ctest --test-dir build` : 34/34 tests passés avec succès.
  - Test comparatif chaud vs froid :
    - `./build_avx2_gcc_native/GemmBench --kernel blis_native --k 256 --fp32 --iterations 200000` → FLOP/cyc = 24.6750
    - `./build_avx2_gcc_native/GemmBench --kernel blis_native --k 256 --fp32 --iterations 200000 --c-cold` → FLOP/cyc = 24.1434 (mode froid $\le$ mode chaud, baisse de ~2.1% liée aux écritures hors cache).
  - Vérification `objdump` : isolation de la boucle chaude via branchement extérieur `cmpq $0x0, 0x68(%rsp); je 393b3` ; aucune écriture de pointeur dans le corps de boucle chaude.
- **Commit** : `216c7af` (`feat(bench): add cold-C mode rotating output tiles over a >LLC pool (--c-cold)`).
- **Écarts** : Aucun.

---

## 🔍 Revue R2 — Checkpoint Ablations (P4)

### 1. Diff global P4
- Modifications concentrées sur `include/ColdCPool.h`, `include/BenchConfig.h`, `src/main.cpp`, `tests/GemmUKernelTest.cpp`, `run_benchmarks.sh`, `tools/run_cluster_benchmarks.sh`.
- Intégrité stricte des noyaux MIPPv2 préservée (`GemmUKernel_opti_crr.hpp` et `GemmUKernel_opti_rrr.hpp` intacts).
- Suite de tests unitaires 100% passante (34/34).

### 2. Extrait d'en-tête CSV complet post-P4
```csv
Build,Kernel,M,N,K,Alpha,Beta,Time_median_s,Time_min_s,Time_max_s,Time_stddev_s,Time_cv_pct,GFLOPS_median,GFLOPS_peak,Cycles_median,Cycles_min,Cycles_max,Cycles_stddev,Cycles_cv_pct,Eff_GHz,IPC,FLOP_per_cycle_median,FLOP_per_cycle_peak,Repetitions,InterRun_CV_pct,Commit,Selection,NumRuns,Fastpath,CCold
```

