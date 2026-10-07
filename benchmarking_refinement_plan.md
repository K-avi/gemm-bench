# Plan d'Implémentation : Fiabilisation & Rigueur de la Mesure de FLOP/Cycle

**Contexte :** Consolidation expérimentale pour la soumission de l'article MIPPv2 / DGEMM  
**Portée :** Multi-architecture (x86_64 AVX2/AVX-512, AArch64 NEON, RISC-V RVV 1.0)  
**Décisions validées :**
- **Architecture de mesure hybride :** Driver C++ interne autonome (zéro dépendance) pour toutes les cibles + Validation croisée formelle LIKWID sur x86 (Skylake, Zen 4, Zen 5, Meteor Lake) pour certifier scientifiquement les mesures RVV/ARM.
- **Format CSV (Option B) :** Réorganisation complète et normalisée de l'en-tête CSV avec mise à jour conjointe des scripts Python.
- **Intégrité matérielle stricte :** `NaN` / `0.0` strict en l'absence de PMU (aucun fallback d'extrapolation silencieux dans les graphiques).

---

## 1. Contexte & Diagnostic des Contraintes Matérielles

L'évaluation de la performance crête en micro-benchmarking GEMM repose sur le ratio :
$$\text{FLOP/cycle} = \frac{\text{Opérations arithmétiques flottantes effectives}}{\text{Cycles d'exécution cœur non-haltés}}$$

L'analyse de l'infrastructure a identifié plusieurs limites méthodologiques :
1. **Surcoût d'appel système dans la fenêtre de mesure :** L'usage de `::read()` via `perf_event_open` dans [src/main.cpp](file:///home/ivan/Files/projets/gemm_bench/src/main.cpp) introduisait un surcoût système (contexte Ring 3 $\to$ Ring 0) et un timing imbriqué pollué par l'appel à `clock_gettime`.
2. **Absence de barrière de sérialisation pipeline (OOO) :** Seule une barrière mémoire compilateur (`asm volatile("" ::: "memory")`) était présente, sans garantie contre la spéculation ou la réorganisation hors-ordre matérielle des instructions FMA.
3. **Découplage statistique :** Le compteur de cycles retenu était celui correspondant à la médiane du temps d'exécution plutôt qu'à la distribution des cycles eux-mêmes.
4. **Fallback statique opaque :** En cas d'indisponibilité du PMU, les scripts de post-traitement imputaient $\text{FLOP/cycle} = \text{GFLOPS} / f_{\text{statique}}$, masquant le throttling thermique et les variations de fréquence.
5. **Hétérogénéité des cibles :** LIKWID n'étant pas disponible sur RISC-V Vector (RVV), une stratégie différenciée mais harmonisée est requise.

### Matrice de Disponibilité et Rôle des Outils

| Plateforme | Microarchitecture | SIMD | Rôle Principal (Production) | Outil de Validation Croisée | Justification Technique |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AMD Zen 4** (7900X) | Zen 4 | AVX-512 | Driver C++ (`perf_event_open`) | **LIKWID (`FLOPS_DP`, `L2`)** | Validation formelle : concordance $<0{,}5\%$ entre FMA mesurés et théoriques ($2MNK$). |
| **AMD Zen 5** (HX 370) | Zen 5 (Strix) | AVX-512 | Driver C++ (`perf_event_open`) | **LIKWID (`FLOPS_DP`, `L2`)** | Validation double-pump FMA 512b et vérification du confinement L1/L2. |
| **Intel Meteor Lake** (185H) | Redwood Cove | AVX2 | Driver C++ (`perf_event_open`) | **LIKWID (`FLOPS_DP`, `L2`)** | Suivi du throttling thermique AVX2 et validation de la saturation P-Core. |
| **Intel Skylake** (i5-6200U) | Skylake | AVX2 | Driver C++ (`perf_event_open`) | **LIKWID (`FLOPS_DP`, `L2`)** | Nœud local de référence pour calibrer le banc de test. |
| **Apple M1** (Firestorm) | Firestorm | NEON | Driver C++ (`perf_event_open` / timer) | Audit ponctuel Linux `perf stat` | LIKWID absent sous ARM Apple. Mesure par timer virtuel / PMU natif. |
| **Raspberry Pi 5** (BCM2712) | Cortex-A76 | NEON | Driver C++ (`perf_event_open`) | **Linux `perf stat` (ARM PMU)** | Validation de l'IPC et des événements NEON `0x74` (`vfp_spec`). |
| **SpacemiT X100** | X100 | RVV 256b | **CSR unprivileged `rdcycle`** | Kernel SBI PMU (si dispo) | LIKWID non portable. Mesure à 1 cycle de surcoût via assembleur inline RV64. |
| **SpacemiT X60** | X60 | RVV 256b | **CSR unprivileged `rdcycle`** | Kernel SBI PMU (si dispo) | Zéro dépendance externe, immunisé contre les limitations d'outillage RVV. |
| **SpacemiT A100** | A100 | RVV 1024b | **CSR unprivileged `rdcycle`** | Kernel SBI PMU (si dispo) | Zéro dépendance externe, mesure matérielle directe. |

---

## 2. Architecture de la Solution Déployée

La méthodologie scientifique repose sur une **stratégie de transfert de crédibilité** :

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    DRIVER INTERNE C++ UNIFIÉ (TOUTES CIBLES)                   │
│  - Sérialisation matérielle OOO (lfence / isb / fence)                          │
│  - Lecture cycles ultra-faible latence :                                        │
│      * RISC-V : CSR rdcycle unprivileged (1 cycle, 0 syscall)                    │
│      * x86 : perf_event_open / rdpmc optimisé                                   │
│      * ARM : perf_event_open avec isolation temporelle                          │
│  - Découplage statistique : cycles_median, cycles_min, cycles_stddev, cycles_cv │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────────────┐
│     CAMPAGNE GÉNÉRALE (9 CIBLES)     │  │   VALIDATION CROISÉE LIKWID (x86)    │
│  - Exécution complète benchmarks     │  │  - Nœuds Zen 4, Zen 5, Skylake, MTL │
│  - Export CSV normalisé (Option B)   │  │  - Mesure FP_ARITH_INST_RETIRED     │
│  - Graphiques avec barres d'erreur   │  │  - Preuve formelle : écart < 0.5%   │
│  - Données RVV/ARM fiabilisées       │  │    ==> Valide la rigueur du driver   │
└──────────────────────────────────────┘  └──────────────────────────────────────┘
```

---

## 3. Détail des Implémentations

### 3.1 Primitives de Sérialisation ([include/HardwareFence.h](file:///home/ivan/Files/projets/gemm_bench/include/HardwareFence.h))
Empêche l'exécution spéculative des instructions vectorielles à travers les bornes de mesure :
- `compiler_fence()` : `asm volatile("" ::: "memory")`
- `pipeline_fence()` : `lfence` (x86), `isb` (ARM), `fence i,r` (RISC-V)

### 3.2 Compteur de Cycles Fiabilisé ([src/main.cpp](file:///home/ivan/Files/projets/gemm_bench/src/main.cpp))
- **RISC-V :** CSR unprivileged `rdcycle` (1 cycle, 0 syscall).
- **x86 / ARM :** `perf_event_open` (`PERF_COUNT_HW_CPU_CYCLES`) avec gestion d'erreur stricte.
- **Isolation :** Découplage de la capture temporelle `steady_clock::now()` en dehors des bornes `[cyc0, cyc1]`.

### 3.3 Découplage Statistique et Format CSV Option B
Format normalisé généré :
```csv
Build,Kernel,M,N,K,Alpha,Beta,Time_median_s,Time_min_s,Time_max_s,Time_stddev_s,Time_cv_pct,GFLOPS_median,GFLOPS_peak,Cycles_median,Cycles_min,Cycles_max,Cycles_stddev,Cycles_cv_pct,Eff_GHz,FLOP_per_cycle_median,FLOP_per_cycle_peak,Repetitions,InterRun_CV_pct
```

### 3.4 Support Conditionnel LIKWID Marker API ([CMakeLists.txt](file:///home/ivan/Files/projets/gemm_bench/CMakeLists.txt))
Option `-DGEMMBENCH_ENABLE_LIKWID=ON` détectant `likwid-marker.h` et liant `-llikwid`. Les balises `LIKWID_MARKER_START` / `STOP` encadrent strictement les itérations du micro-noyau.

### 3.5 Normalisation du Pipeline Python ([plot_results_crr.py](file:///home/ivan/Files/projets/gemm_bench/plot_results_crr.py) & [plot_results.py](file:///home/ivan/Files/projets/gemm_bench/plot_results.py))
- Détection et normalisation automatique des en-têtes Option B.
- Suppression du `fillna(df["GFLOPS"] / spec.frequency_ghz)` : conservation stricte de `np.nan` en l'absence de compteurs matériels.
