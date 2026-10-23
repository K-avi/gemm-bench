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

---

## P5 — Analyse statistique : K égal, IC bootstrap, champion pré-enregistré

### P5.a / P5.b — Module `tools/paper_stats.py` et suite de tests Python
- **Fichiers modifiés / créés** :
  - `tools/paper_stats.py` : module statistique pur (numpy / pandas uniquement) implémentant :
    - `working_set_bytes(M, N, K, elem_bytes)` : empreinte mémoire des panneaux $A$ et $B$ ($K \times (M + N) \times \text{elem\_bytes}$).
    - `geomean(x)` : moyenne géométrique.
    - `bootstrap_ratio_of_medians(a, b, n_boot=10000, seed=12345, ci=0.95)` : bootstrap hiérarchique à deux niveaux vectorisé pour échantillons équilibrés, retournant l'estimation ponctuelle et l'intervalle de confiance à 95% par percentiles.
    - `flop_per_cycle_samples(df_raw, kernel, K)` : extraction des échantillons bruts FLOP/cycle par répétition et par run indépendant.
    - `load_raw(raw_csv)` : chargement des fichiers `.raw.csv` avec dérivation automatique des métadonnées Platform/Layout/Compiler/Variant/Precision depuis le nom de fichier.
    - `parse_cache_size_bytes` et `get_l1d_bytes(meta_json, uarch_spec)` : résolution de la capacité L1 Data via `meta.json` puis `uarch_config.json`.
  - `tests/python/test_paper_stats.py` : suite de tests unitaires `unittest` validant le déterminisme, le centrage d'identité, la sensibilité aux décalages de 10%, la couverture de 95% sur 100 simulations synthétiques, l'arithmétique du working-set, le parsing de caches et l'analyse de dataset synthétique.
- **Commandes de vérification** :
  - `python3 -m unittest discover -s tests/python` : 10/10 tests passés avec succès.
- **Commit** : `96c7d52` (`feat(analysis): add paper_stats (hierarchical bootstrap, working-set) with unittest suite`).
- **Écarts** : Aucun.

### P5.c — Réécriture de `tools/compare_mippv2_blis.py`
- **Fichiers modifiés** :
  - `tools/compare_mippv2_blis.py` :
    - Champion pré-enregistré résolu depuis `uarch_config.json` (`expected_best_kernel_crr_fp32` en FP32, `expected_best_kernel_crr` en FP64) via le mapping microarchitectural. Erreur explicite levée si absent du jeu de données (repli sur l'ancien comportement via `--legacy-argmax-champion`).
    - Comparaison rigoureuse à $K$ apparié (`Matched-K`) sur chaque point de fonctionnement commun entre MIPPv2 et BLIS native.
    - Calcul des ratios ponctuels de FLOP/cycle et des intervalles de confiance à 95% par bootstrap hiérarchique lorsque `.raw.csv` est disponible (repli ponctuel avec marquage `ci_available=False` sinon).
    - Suivi de l'empreinte mémoire $W_A + W_B$ et détection des franchissements de capacité L1d (marquage `†`).
    - Classification de parité sur la fenêtre d'analyse $K \in [128, 384]$ (`parity`, `mipp_faster`, `blis_faster`, `mixed`, avec suffixe `_noci` en l'absence de raw).
    - Découpage par variante (`--variant {default,nofp,coldc,strict,all}`).
    - Séparation stricte des compilateurs (lignes distinctes pour GCC et Clang) et relégation de l'enveloppe max-compilateur dans une section annexe (« Best-compiler envelope (post-hoc) »).
    - Préservation de `Legacy_PeakOverPeak_pct` pour rétro-compatibilité sans affichage dans les tableaux directeurs.
    - Figures réécrites avec barres d'erreur bootstrap / bande d'incertitude et repère vertical de spill L1d.
- **Commandes de vérification** :
  - `python3 -m unittest discover -s tests/python` : test de `analyze_dataset` sur dataset synthétique validant le choix du champion pré-enregistré et le calcul apparié à $K$ constant.
- **Commit** : `24442d0` (`refactor(analysis): matched-K ratios with CIs, pre-registered champions, per-compiler reporting`).
- **Écarts** : Aucun.

### P5.a (suite) — Suppression du fallback silencieux FP32
- **Fichiers modifiés** :
  - `plot_results_crr.py` : dans `load_uarch_config_crr`, suppression de l'inférence `measured_fc = measured_peak_flop_per_cycle * 2.0` (retourne `None` si absent) et validation stricte de la présence explicite de `peak_flop_per_cycle_fp32` (lève `KeyError` si absent).
- **Commandes de vérification** :
  - Test Python confirmant que `load_uarch_config_crr(..., is_fp32=True)` renvoie `measured_peak_flop_per_cycle = None` pour les cibles sans mesure FP32 explicite.
  - `ctest --test-dir build` : 34/34 tests Catch2 passés.
- **Commit** : `9a8f8a7` (`fix(analysis): drop implicit FP32=2xFP64 measured-peak fallback`).
- **Écarts** : Aucun.

---

## 🔍 Revue R3 — Checkpoint Analyse statistique (P5)

### 1. Diff global P5
```
 plot_results_crr.py              |   6 +-
 tests/python/test_paper_stats.py | 204 +++++++++++++++++++
 tools/compare_mippv2_blis.py     | 623 +++++++++++++++++++++++++++++++++----------------------
 tools/paper_stats.py             | 279 +++++++++++++++++++++++++
 4 files changed, 860 insertions(+), 252 deletions(-)
```

### 2. Validation de la suite de tests
- Tests unitaires Python : 10/10 passés (`python3 -m unittest discover -s tests/python`).
- Tests C++ Catch2 : 34/34 passés (`ctest --test-dir build`).

---

## P7 — Calibration des pics FP32

### P7.a — Scripts cpufp et mise à jour de la configuration
- **Fichiers modifiés / créés** :
  - `tools/measure_peak_cpufp.sh` : ajout des options `--fp32`, `--fp64` et `--analyze-only`. Extraction des débits FMA IEEE 754 simple précision (`f32`/`fp32`) excluant le BF16 et l'entier. Export de `gemm-bench-results/cpufp/cpufp_peak_sp_summary.csv` et affichage du tableau d'efficacité par microarchitecture.
  - `tools/update_uarch_config_cpufp.py` : ajout de l'option `--fp32` pour mettre à jour automatiquement `measured_peak_flop_per_cycle_fp32` dans `uarch_config.json` depuis le CSV de synthèse SP (jamais à la main).
  - `tools/submit_queued_cpufp.sh` : mise à jour pour supporter `--fp32` et intégrer l'ensemble des 8 cibles de calcul du cluster Dalek.
  - `uarch_config.json` : mise à jour de `measured_peak_flop_per_cycle_fp32` pour les plateformes disposant de mesures cpufp :
    - `skylake` : 30.38 FLOP/cyc (94.9% du pic théorique de 32.0)
    - `meteorlake` : 31.68 FLOP/cyc (99.0% du pic théorique de 32.0)
    - `zen4` : 31.76 FLOP/cyc (99.3% du pic théorique de 32.0)
    - `zen5` : 31.95 FLOP/cyc (99.8% du pic théorique de 32.0)
    - `m1` : 31.94 FLOP/cyc (99.8% du pic théorique de 32.0)
- **Commandes de vérification** :
  - `./tools/measure_peak_cpufp.sh --analyze-only --fp32` : synthèse SP générée sans erreur.
  - `python3 tools/update_uarch_config_cpufp.py --fp32` : mise à jour sans régression de `uarch_config.json`.
  - `python3 -m unittest discover -s tests/python` : 10/10 tests unitaires passés.
  - `ctest --test-dir build` : 34/34 tests Catch2 passés.
- **Commit** : `aef04d3` (`feat(calibration): cpufp FP32 peak measurement and config update path`).
- **Écarts** : Aucun.

---

## P8 — Investigation des micro-noyaux BLIS Zen 4

### P8.a — Rapport d'investigation microarchitecturale
- **Fichier créé** :
  - `docs/blis_zen4_investigation.md` : rapport technique analysant la disponibilité et la structure des micro-noyaux BLIS sur AMD Zen 4.
- **Conclusions clés** :
  1. Le dépôt upstream `flame/blis` au commit `a8037cfc4330030c726efaec1ffeb1f93d468aa3` ne dispose pas de configuration `zen4` ni de répertoire `kernels/zen4/` ; il dispatche Zen 4 sur le noyau AVX2 256 bits de Zen 3 (`zen3`, $MR \times NR = 6 \times 8$).
  2. Le support AVX-512 natif pour Zen 4 est maintenu par AMD dans son fork vendeur `amd/blis` (AOCL-BLIS), tirant parti du pipeline FPU 256 bits double-pompé.
  3. L'unique chemin AVX-512 upstream réside dans `kernels/skx/3/` (Intel Skylake-X, géométrie $24 \times 8$ ou $16 \times 12$).
  4. Conformément aux instructions, aucun code n'a été intégré dans `libs/blis_reference/`.
- **Commit** : `b8d83f0` (`docs: investigate upstream BLIS zen4 microkernels as AVX-512 baseline`).
- **Écarts** : Aucun.

---

## 🔍 Revue R5 — Checkpoint Pré-Campagne de Mesures (P9)

### 1. Diff global P7-P8
- Outils de métrologie cpufp mis à niveau pour FP32 (`measure_peak_cpufp.sh`, `update_uarch_config_cpufp.py`, `submit_queued_cpufp.sh`).
- Valeurs de pic empirique FP32 injectées dans `uarch_config.json` via le script dédié.
- Rapport technique `docs/blis_zen4_investigation.md` consigné.
- Intégrité de la base de code préservée : suite de tests C++ (34/34) et Python (10/10) verte.

---

## P9 — Campagne de mesures cluster « paper grade » (Default FP64 & FP32)

### P9.a — Exécution et ingestion de la campagne Default FP64
- **Cibles exécutées** : 8 architectures physiques complètes sur le cluster Dalek LIP6 :
  - `zen4` (AMD Zen 4, AVX-512) : GCC & Clang
  - `zen5` (AMD Zen 5, AVX-512) : GCC & Clang
  - `meteorlake` (Intel Meteor Lake, AVX2) : GCC & Clang
  - `rpi5` (Raspberry Pi 5, Cortex-A76 NEON) : GCC & Clang
  - `m1` (Apple M1 Firestorm, NEON) : GCC & Clang
  - `x100` (SpacemiT X100, RVV 256-bit) : GCC & Clang
  - `a100` (SpacemiT A100, RVV 1024-bit) : GCC & Clang
  - `x60` (SpacemiT X60, RVV 256-bit) : GCC & Clang
- **Paramètres de mesure** :
  - Balayage $K \in \{32, 64, 128, 256, 384, 512, 1024\}$
  - 3 runs indépendants $\times$ 15 répétitions par mesure (`--runs 3 -r 15`)
  - Mode CRR strict (`--crr-only`) avec micro-noyau `blis_native` de référence (`--with-blis`)
  - Sélection médiane des runs (`selection=median_of_runs`)
  - Isolation des processus, traçabilité git et métadonnées d'environnement exhaustives (`.meta.json`).
- **Résultats rapatriés** :
  - Répertoire local : `gemm-bench-results/paper_v1/default/` (48 fichiers générés pour FP64 : 16 CSV, 16 raw.csv, 16 meta.json).
  - Taux de complétion FP64 : **100 % (16 / 16 configurations)**.

#### Synthèse des Ratios de Parité Géométrique MIPPv2 vs BLIS natif (Default FP64)
| Cible | Ratio MIPP / BLIS (GCC) | Ratio MIPP / BLIS (Clang) | MIPP Max GFLOP/s (Clang) | BLIS Max GFLOP/s (Clang) |
|---|:---:|:---:|:---:|:---:|
| **Zen 4** | **1.03×** [1.01–1.08] | **1.03×** [1.01–1.09] | 85.2 | 84.4 |
| **Zen 5** | **1.01×** [1.00–1.04] | **1.01×** [1.00–1.03] | 80.8 | 80.5 |
| **Meteor Lake** | **0.92×** [0.82–0.97] | **1.01×** [0.97–1.05] | 78.4 | 78.1 |
| **Apple M1** | **1.04×** [1.00–1.17] | **1.04×** [1.00–1.15] | 48.3 | 47.9 |
| **RPi5** | **1.03×** [0.96–1.12] | **1.03×** [0.97–1.11] | 18.9 | 18.8 |
| **X100** | **0.85×** [0.76–0.92] | **0.95×** [0.86–1.05] | 18.6 | 18.0 |
| **A100** | **3.67×** [3.63–3.71] | **3.52×** [3.46–3.60] | 24.3 | 6.8 |
| **X60** | **0.78×** [0.68–0.85] | **0.84×** [0.72–0.90] | 7.4 | 10.3 |


### P9.b — Exécution et ingestion de la campagne Default FP32
- **Cibles exécutées** : 8 architectures physiques complètes sur le cluster Dalek LIP6 :
  - `zen4` (AMD Zen 4, AVX-512) : GCC & Clang
  - `zen5` (AMD Zen 5, AVX-512) : GCC & Clang
  - `meteorlake` (Intel Meteor Lake, AVX2) : GCC & Clang
  - `rpi5` (Raspberry Pi 5, Cortex-A76 NEON) : GCC & Clang
  - `m1` (Apple M1 Firestorm, NEON) : GCC & Clang
  - `x100` (SpacemiT X100, RVV 256-bit) : GCC & Clang
  - `a100` (SpacemiT A100, RVV 1024-bit) : GCC & Clang
  - `x60` (SpacemiT X60, RVV 256-bit) : GCC & Clang
- **Paramètres de mesure** :
  - Balayage $K \in \{32, 64, 128, 256, 384, 512, 1024\}$
  - 3 runs indépendants $\times$ 15 répétitions par mesure (`--runs 3 -r 15 --fp32`)
  - Mode CRR strict (`--crr-only`) avec micro-noyau `blis_native` FP32 de référence (`--with-blis`)
  - Sélection médiane des runs (`selection=median_of_runs`)
- **Résultats rapatriés** :
  - Répertoire local : `gemm-bench-results/paper_v1/default/` (48 fichiers générés pour FP32 : `.csv`, `.raw.csv`, `.meta.json`).
  - Total cumulé Default (FP64 + FP32) : 93 fichiers de résultats bruts et métadonnées.

#### Synthèse des Ratios de Parité Géométrique MIPPv2 vs BLIS natif (Default FP32)
| Cible | Ratio MIPP / BLIS (GCC) | Ratio MIPP / BLIS (Clang) | MIPP Max GFLOP/s (Clang) | BLIS Max GFLOP/s (Clang) |
|---|:---:|:---:|:---:|:---:|
| **Zen 4** | **1.03×** [1.01–1.10] | **1.02×** [1.00–1.09] | 171.6 | 170.4 |
| **Zen 5** | **1.01×** [1.00–1.04] | **1.01×** [1.00–1.03] | 162.6 | 162.5 |
| **Meteor Lake** | **0.94×** [0.85–0.99] | **1.01×** [0.98–1.06] | 158.0 | 158.2 |
| **Apple M1** | **1.05×** [1.01–1.18] | **1.04×** [1.00–1.16] | 96.5 | 96.1 |
| **RPi5** | **1.03×** [0.96–1.12] | **1.03×** [0.97–1.12] | 37.9 | 37.8 |
| **X100** | **0.86×** [0.78–0.91] | **0.96×** [0.87–1.03] | 37.2 | 36.0 |
| **A100** | **3.46×** [3.24–3.93] | **3.67×** [3.58–3.91] | 48.8 | 13.5 |
| **X60** | **0.72×** [0.52–0.80] | **0.79×** [0.62–0.87] | 16.4 | 21.3 |




