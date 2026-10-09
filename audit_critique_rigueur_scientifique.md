# Évaluation Critique de l'État du Dépôt, de l'Ingénierie et de la Rigueur Scientifique de GEMMBench

**Date :** 8 Octobre 2026  
**Dépôt :** `gemm_bench`  
**Branche active :** `feat/blis-ukernels` (commit `fd7a81a`)  
**Cadre de travail :** Microkernels DGEMM portables C++ (MIPPv2) vs Microkernels natifs assembleur (BLIS)  
**Microarchitectures cibles :** x86_64 (AVX2, AVX-512), AArch64 (NEON), RISC-V (RVV 1.0)  

---

## 1. Synthèse Exécutive & Scorecard de Rigueur Académique

Le projet `gemm_bench` présente une qualité métrologique matérielle remarquable sur l'isolation des cycles d'exécution : abandon de l'estimation temporelle naïve, usage direct de `perf_event_open` (`PERF_COUNT_HW_CPU_CYCLES`), barrières microarchitecturales strictes (`pipeline_fence()`) et traitement statistique découplé (médiane, min, CV%).

Cependant, une analyse critique et approfondie de la chaîne expérimentale, du code source et des scripts d'orchestration met en évidence **des biais méthodologiques majeurs**, **des comparaisons asymétriques (« straw man »)** et **des vulnérabilités d'ingénierie logicielle** qui fragilisent la crédibilité des résultats face à un comité de lecture académique de premier plan (SC, ASPLOS, PPoPP, IPDPS, IEEE Micro).

### Grille d'Évaluation de la Rigueur (« Paper-Grade Assessment »)

| Dimension Évaluée | Statut Actuel | Gravité | Risque Scientifique / Ingénierie |
|:---|:---:|:---:|:---|
| **1. Équité des Baselines (MIPPv2 vs BLIS)** | ❌ **Défaillant** | **Bloquant** | **Comparaison asymétrique sur RISC-V A100/X100** : Le kernel BLIS compilé pour A100 est le kernel in-order X60 ($8 \times 14$), affichant un faux gain de $+265\%$ pour MIPPv2. |
| **2. Périmètre Sémantique ("DGEMM" vs Microkernel)** | ⚠️ **Ambigüe** | **Majeur** | Confusion entre saturation du pipeline de registres (*In-Cache Register Micro-Tile*) et multiplication matricielle complète (BLAS GotoBLAS/BLIS à 5 boucles). |
| **3. Régime Mémoire & Capacité Cache (Mur L1d)** | ⚠️ **Incomplet** | **Majeur** | Chute brutale à $K \ge 256$ non corrélée aux misses de cache matériels ; absence de distinction formelle entre cœurs In-Order (X60) et Out-of-Order (Zen/M1). |
| **4. Intégrité Arithmétique & Norme IEEE-754** | ⚠️ **Non documenté** | **Modéré** | Forçage de `-ffast-math` sur tous les microkernels C++, en contradiction avec les exigences contractuelles IEEE-754 des bibliothèques BLAS industrielles. |
| **5. Isolation Métrologique & Pollution de Warmup** | ⚠️ **Imparfait** | **Modéré** | Construction du descripteur `CycleCounter` (appel système noyau) intercalée *après* la boucle de chauffe et *avant* la mesure. |
| **6. Alignement Mémoire & Cache-Line Splits** | ⚠️ **Sous-optimal** | **Mineur** | En layout col-major avec $M_R = 6$ (Haswell, A76), le leading dimension de 48 octets provoque des accès vectoriels chevauchant deux lignes de cache (split loads). |
| **7. Architecture & Dette Technique du Dépôt** | ⚠️ **Monolithique** | **Mineur** | Explosion de duplication dans `GemmUKernel_opti_crr.hpp` (2 400 lignes), éparpillement des scripts d'analyse Python. |

---

## 2. Problèmes Scientifiques & Méthodologiques Majeurs (Niveau 1)

### 2.1 Le Piège de la Baseline Asymétrique : Comparaison « Strawman » sur SpacemiT A100 & X100

#### Constat technique factuel
Dans [CMakeLists.txt (lignes 82-84)](file:///home/ivan/Files/projets/gemm_bench/CMakeLists.txt#L82-L84) :
```cmake
elseif("${CMAKE_CXX_FLAGS} ${CMAKE_C_FLAGS}" MATCHES "rv64gcv" OR CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64")
    set(BLIS_KERNEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/libs/blis_reference/src/dgemm_x60_2vx14.c")
    set(BLIS_ARCH_TAG "GEMMBENCH_BLIS_X60")
```
Quelle que soit la cible RISC-V détectée (SpacemiT X60, SpacemiT X100, ou SpacemiT A100), le système de build sélectionne **inconditionnellement** le fichier source [dgemm_x60_2vx14.c](file:///home/ivan/Files/projets/gemm_bench/libs/blis_reference/src/dgemm_x60_2vx14.c).

#### Conséquence scientifique critique
1. **Nature du microkernel X60 :** `dgemm_x60_2vx14.c` est un microkernel écrit en assembleur en ligne, ultra-spécialisé pour le cœur **SpacemiT X60** (processeur in-order dual-issue avec port vectoriel partagé pour `vle` et `vfmacc`, tuile $8 \times 14$, ordonnancement statique généré par algorithme génétique avec NOPs et réordonnancement manuel des charges).
2. **Exécution sur SpacemiT A100 :** Le cœur A100 possède une microarchitecture différente (unités de décodage, pipelines et gestion des aléas différents, géométrie optimale $7 \times 64$ avec $LMUL=2$). Exécuter le kernel X60 sur A100 donne une performance désastreuse de **1,84 FLOP/cycle**.
3. **Revendication trompeuse :** Le rapport [blis_vs_mippv2_report.md](file:///home/ivan/Files/projets/gemm_bench/gemm-bench-results/blis_comparison/blis_vs_mippv2_report.md#L8-L10) et le tableau [summary_metrics.csv](file:///home/ivan/Files/projets/gemm_bench/gemm-bench-results/blis_comparison/summary_metrics.csv#L2-L3) indiquent :
   - MIPPv2 Peak : 13,36 GFLOP/s (6,71 FLOP/cycle)
   - BLIS Peak : 3,66 GFLOP/s (1,84 FLOP/cycle)
   - **Ratio MIPPv2 / BLIS : 365,1 % (+265 % en faveur de MIPPv2)**.

> [!CAUTION]
> **Verdict pour la publication :**
> Il s'agit d'une comparaison asymétrique évidente (*strawman comparison*). MIPPv2 n'a pas « battu BLIS de 3,6x sur A100 » : un microkernel MIPPv2 sur-mesure pour A100 a battu un microkernel BLIS conçu pour un tout autre processeur (X60).
> **Décision méthodologique adoptée :** Conserver la présentation des microkernels MIPPv2 sur A100 et X100, tout en explicitant sans ambiguïté dans les rapports et figures que le kernel BLIS X60 ($8 \times 14$) sert d'unique baseline RVV 1.0 disponible en amont, expliquant l'écart majeur de performance.

---

### 2.2 Ambiguïté Sémantique : « DGEMM » vs Débit de Registre In-Cache

#### Constat
Le projet se nomme `gemm_bench`, et plusieurs figures et sections du [README.md](file:///home/ivan/Files/projets/gemm_bench/README.md) font référence à la *« DGEMM Performance »*.
Le banc teste exclusivement :
$$M = M_R, \quad N = N_R, \quad K \in [32..1024]$$
avec les matrices $A$ et $B$ conservées et réutilisées dans le même espace mémoire à travers des centaines de milliers d'itérations.

#### Écarts méthodologiques face à un vrai DGEMM (BLAS)
Dans un GEMM académique ou industriel (BLIS, OpenBLAS, MKL) :
1. **Macro-ordonnancement à 5 boucles :** Les boucles $J_C, P_C, I_C$ découpent les grandes matrices de DRAM vers les caches L3 ($B \to \tilde{B}$) et L2 ($A \to \tilde{A}$), puis les boucles $J_R, I_R$ alimentent le microkernel.
2. **Coût de packing :** La copie avec transposition/empaquetage de $A$ et $B$ dans des formats contigus consomme entre $5\%$ et $15\%$ du temps total de calcul en DRAM/L3.
3. **Micro-tuiles de bordure (*edge/fringe handling*) :** Lorsque les dimensions $M, N, K$ ne sont pas des multiples exacts de $M_R, N_R, K_C$, les microkernels doivent exécuter des chemins scalaires ou masqués à faible rendement arithmétique.

> [!IMPORTANT]
> **Défense requise pour les relecteurs :**
> Le banc ne mesure pas un DGEMM complet, mais le **débit crête et la saturation du pipeline FMA sous empaquetage idéal en registre** (*In-Register Microkernel Compute Saturation*).
> **Décision sémantique adoptée :** Clarifier systématiquement cette portée dans les titres, légendes et descriptions textuelles afin d'éviter toute ambiguïté sur l'absence du macro-blocking et du packing dynamique.

---

### 2.3 Rupture de Régime Microarchitectural : Le Mur de Capacité L1d ($K \ge 256$)

#### Analyse empirique sur SpacemiT X60
Sur le Banana Pi BPI-F3 (SpacemiT X60 @ 1,60 GHz), le microkernel BLIS ($8 \times 14$) affiche l'évolution suivante :
- $K = 32$ : 5,98 FLOP/cycle
- $K = 64$ : 6,63 FLOP/cycle
- $K = 128$ : **7,00 FLOP/cycle** (95,6 % du pic de Nassyr à 7,32)
- $K = 256$ : **4,31 FLOP/cycle (-38,4 %)**
- $K = 512$ : 4,55 FLOP/cycle
- $K = 1024$ : 4,63 FLOP/cycle

#### Mécanisme physique
L'empreinte mémoire d'une micro-tuile en fonction de $K$ est :
$$\text{Footprint}(K) = (8 \times K + 14 \times K) \times 8 + (8 \times 14 \times 8) = 176 \times K + 896 \text{ octets}$$

- À $K = 128$ : $\text{Footprint} = 23{,}4\text{ KiB} \le 32\text{ KiB}$ (L1 Data Cache) $\implies$ **Résidence L1 totale**, latence de charge de 3 cycles sans pénalité.
- À $K = 256$ : $\text{Footprint} = 45{,}9\text{ KiB} > 32\text{ KiB}$ $\implies$ **Débordement dans le cache L2**.

#### Conséquence microarchitecturale
1. **Processeurs Out-of-Order (Zen 4, Apple M1) :** Le reorder buffer (ROB) et les unités de prélecture matérielle masquent la latence L2 (12 à 18 cycles). Le débit reste invariant à 15,95–16,00 FLOP/cycle jusqu'à $K=1024$.
2. **Processeurs In-Order (SpacemiT X60) :** Tout miss L1 bloque immédiatement le pipeline d'instruction à l'étape issue/dispatch. Comme les instructions `prefetch.r` sont désactivées dans le code de Nassyr ([dgemm_x60_2vx14.c, L191-194](file:///home/ivan/Files/projets/gemm_bench/libs/blis_reference/src/dgemm_x60_2vx14.c#L191-L194)), la pénalité L2 s'exprime à plein, réduisant le débit de calcul de moitié.

> [!TIP]
> **Recommandation d'analyse :**
> Le rapport et l'article doivent impérativement expliciter que la courbe de performance en fonction de $K$ traverse **deux régimes distincts** :
> - Régime $K \le 128$ : Régime de calcul pur (CPU-bound, L1d-resident).
> - Régime $K \ge 256$ : Régime de bande passante/latence L2 (L2-latency-bound), exacerbé sur architectures in-order.

---

### 2.4 Le Diktat de `-ffast-math` vs l'Intégrité IEEE-754

#### Constat dans les configurations
Dans [bench_configs/avx512.sh](file:///home/ivan/Files/projets/gemm_bench/bench_configs/avx512.sh#L34), [bench_configs/rvv_x60.sh](file:///home/ivan/Files/projets/gemm_bench/bench_configs/rvv_x60.sh#L28), et les autres configurations de plateformes :
```bash
COMMON_FLAGS="
-O3
-ffast-math
-finline-functions
-funroll-loops
-fno-semantic-interposition
"
```

#### Risque d'acceptabilité en calcul scientifique
1. **Ce que fait `-ffast-math` :** Il active `-fassociative-math`, `-fno-signed-zeros`, `-fno-trapping-math`, et `-ffinite-math-only`. Le compilateur est autorisé à réassocier arbitrairement les opérations flottantes ($ (a + b) + c \neq a + (b + c) $ en IEEE-754) et suppose que les NaN ou $\pm\infty$ n'existent jamais.
2. **Contrat des bibliothèques BLAS :** Les bibliothèques standard (OpenBLAS, BLIS, LAPACK, MKL) garantissent la conformité IEEE-754 en précision double (FP64). Une bibliothèque qui requiert `-ffast-math` pour saturer le matériel est souvent critiquée en calcul haute performance car elle peut introduire des dérives numériques incontrôlées dans les solveurs linéaires.
3. **Impact asymétrique :** MIPPv2 (code C++ à base de templates) est sensible à `-ffast-math`. BLIS (assembleur pur) ne l'est pas pour sa boucle interne, mais son wrapper l'est.

> [!TIP]
> **Décision de conformité adoptée & implémentée :** Une option de build CMake (`GEMMBENCH_ENABLE_FASTMATH`) et un drapeau CLI (`--no-fast-math` / `--strict-ieee`) ont été implémentés dans [CMakeLists.txt](file:///home/ivan/Files/projets/gemm_bench/CMakeLists.txt) et [run_benchmarks.sh](file:///home/ivan/Files/projets/gemm_bench/run_benchmarks.sh) pour permettre l'évaluation stricte IEEE-754 avec `-fno-fast-math -ffp-contract=fast`.

---

## 3. Faiblesses d'Ingénierie Métrologique & Logicielle (Niveau 2)

### 3.1 Contamination de la Boucle de Chauffe par l'Appel Système `CycleCounter`

#### Analyse du code
Dans [src/main.cpp (lignes 195-210)](file:///home/ivan/Files/projets/gemm_bench/src/main.cpp#L195-L210) :
```cpp
#define BENCH_KERNEL(CALL)                                                     \
  do {                                                                         \
    for (size_t w = 0; w < cfg.warmup; ++w) {                                  \
      CALL;                                                                    \
    }                                                                          \
                                                                               \
    CycleCounter cyc_counter;                                                  \
    const size_t reps = (cfg.repetitions > 0) ? cfg.repetitions : 1;           \
    ...
```
Dans le constructeur de `CycleCounter` :
```cpp
fd_ = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
```

#### Défaut d'ingénierie
1. La boucle de warmup (`cfg.warmup`) a pour but exact de chauffer les caches de données L1, le cache d'instructions L1i, la TLB et l'historique du prédicteur de branchement (BPU).
2. Juste après ce warmup, le code instancie `CycleCounter`, ce qui déclenche un appel système Linux (`sys_perf_event_open`).
3. L'exécution du code noyau (commutation de contexte ring 3 $\to$ ring 0, manipulation de structures noyau, retour ring 0 $\to$ ring 3) pollue partiellement les caches L1 et perturbe le pipeline immédiatement avant la première répétition chronométrée.
4. **Correction immédiate (appliquée) :** `CycleCounter cyc_counter;` a été déplacé en amont de la boucle de warmup dans [src/main.cpp](file:///home/ivan/Files/projets/gemm_bench/src/main.cpp#L197). L'appel système `sys_perf_event_open` est désormais réalisé avant la chauffe des caches, garantissant une mesure 100 % isolée de toute interférence noyau.

---

### 3.2 Alignement Mémoire des Tuiles et Risque de "Cache-Line Splits"

#### Analyse du code
Dans [include/Alloc.h (lignes 252-265)](file:///home/ivan/Files/projets/gemm_bench/include/Alloc.h#L252-L265) (`allocatePackedTile`) :
```cpp
if constexpr (M::order == PLayout::Col)
{
    m.padded_rows = rows;
    m.padded_cols = cols;
    m.ld = rows;
}
```
Pour Haswell ou Cortex-A76, $M_R = 6$.
- $rows = 6 \implies ld = 6$.
- Une colonne de 6 `double` pèse $6 \times 8 = 48\text{ octets}$.
- La taille d'une ligne de cache standard est de **64 octets**.

#### Conséquence microarchitecturale
Comme $48$ n'est pas un multiple de $64$ :
- Colonne 0 : offset $0$ (alignée à 64 octets).
- Colonne 1 : offset $48$ (traverse une frontière de cache de 64 octets à l'élément 2).
- Colonne 2 : offset $96$ (alignée à 32 octets, traverse une frontière à l'élément 4).
- Colonne 3 : offset $144$ (traverse une frontière).

Sur la plupart des cœurs vectoriels (AVX2/AVX-512, RVV), les chargements vectoriels non alignés qui chevauchent deux lignes de cache physiques (*split cache-line loads*) subissent un cycle de latence supplémentaire et monopolisent deux entrées dans les buffers de chargement (*load fill buffers*).

---

### 3.3 Risque de Divergence Numérique et Pièges FPU sous $\beta \neq 0$

#### Analyse du scénario
Dans [src/main.cpp (lignes 213-216)](file:///home/ivan/Files/projets/gemm_bench/src/main.cpp#L213-L216) :
```cpp
for (size_t i = 0; i < cfg.iterations; ++i) {
  CALL;
  asm volatile("" :: "r"(C.data) : "memory");
}
```
- Lorsque `cfg.beta == 0.0`, le kernel écrase $C$ à chaque itération ($C \leftarrow A \cdot B$). Les valeurs restent bornées dans $[-M, M]$.
- Cependant, si un utilisateur teste un kernel complet avec $\beta = 1.0$ (accumulation), chaque itération fait $C \leftarrow A \cdot B + C$.
- Avec `cfg.iterations = 1 250 000`, la valeur de $C$ augmente linéairement de $1{,}25 \times 10^6 \times (A \cdot B)$.
- Les valeurs peuvent diverger vers des ordres de grandeur extrêmes, voire vers $\pm\infty$ ou des dénormaux (subnormals). Or, le traitement des dénormaux ou des exceptions flottantes sur CPU x86 ou RISC-V déclenche des interruptions microcodées (*floating-point assists*) qui dégradent artificiellement le temps d'exécution de 10x à 100x.

---

### 3.4 Absence de Contrôle de Validité Numérique Pendant les Sweeps de Mesure

#### Constat
Dans [run_benchmarks.sh](file:///home/ivan/Files/projets/gemm_bench/run_benchmarks.sh#L660-L662) :
L'option `--validate` est désactivée par défaut lors des balayages massifs de performance afin de ne pas fausser le temps d'exécution.
Si une régression apparaît (ex. mauvais registre clobberé par une mise à jour de compilateur, écrasement de pile ABI), le benchmark exécute une boucle incorrecte, calcule un résultat corrompu (ou des zéros), mais consigne un débit optimal en GFLOP/s basé sur le calcul théorique $2 \cdot M \cdot N \cdot K / \Delta t$.

> [!TIP]
> **Bonne pratique HPC :**
> Ajouter une passe de vérification formelle (ou un checksum de trace $\sum C_{i,j}$) **une seule fois** à la fin des répétitions chronométrées, avant d'enregistrer la ligne CSV.

---

### 3.5 Monoculture Métrologique : Absence d'IPC et de Profiling Événementiel

Le framework ne capture actuellement qu'un seul compteur matériel : `PERF_COUNT_HW_CPU_CYCLES`.
Pour diagnostiquer avec certitude les écarts de performance (par exemple le delta GCC vs Clang sur X60, ou la perte de 40 % à $K \ge 256$), il manque :
1. **`PERF_COUNT_HW_INSTRUCTIONS` :** Indispensable pour calculer l'**IPC** (*Instructions Per Cycle*). Permet de déterminer si une baisse de FLOP/cycle vient d'une augmentation du nombre d'instructions (spills, moves) ou d'un blocage de pipeline (stalls).
2. **`PERF_COUNT_HW_CACHE_L1D:MISS` / `L2_MISS` :** Permet de corréler mathématiquement la chute de performance avec le trafic mémoire.

---

## 4. Dette Technique & Architecture du Dépôt

### 4.1 Fichiers Monolithiques et Duplication Massive
- [include/microkernels/GemmUKernel_opti_crr.hpp](file:///home/ivan/Files/projets/gemm_bench/include/microkernels/GemmUKernel_opti_crr.hpp) fait **90 Ko pour 2 413 lignes**.
- [include/microkernels/GemmUKernel_opti_rrr.hpp](file:///home/ivan/Files/projets/gemm_bench/include/microkernels/GemmUKernel_opti_rrr.hpp) fait **112 Ko**.
Ces deux fichiers contiennent des dizaines d'instanciations manuelles de boucles déroulées (déroulement 2, 4, register blocking).
- [include/microkernels/GemmUKernel_opti.hpp](file:///home/ivan/Files/projets/gemm_bench/include/microkernels/GemmUKernel_opti.hpp) ne contient que 72 octets (fichier vide / vestige).

### 4.2 Fragmentation des Outils de Visualisation
Trois scripts Python distincts coexistent avec du code dupliqué de parsing CSV, de filtrage et de configuration matplotlib :
- `plot_results.py` (33 Ko)
- `plot_results_crr.py` (49 Ko)
- `tools/compare_mippv2_blis.py` (22 Ko)
Une factorisation en un module commun (`tools/gemmbench_viz/`) réduirait la maintenance et harmoniserait les chart styles.

---

## 5. Matrice de Remédiation Priorisée

| Priorité | Domaine | Problème Identifié | Action Corrective & Statut |
|:---:|:---|:---|:---|
| 🔴 **P0** | **Méthodologie** | Baseline BLIS faussée sur RISC-V A100 & X100 (kernel X60 utilisé par défaut) | ✅ **Explicité** : Maintenu dans les graphes mais annoté explicitement comme baseline non-optimisée X60 dans `compare_mippv2_blis.py` et le rapport. |
| 🔴 **P0** | **Sémantique** | Confusion entre DGEMM complet et Microkernel In-Cache | ✅ **Cadré** : Précisé formellement dans la documentation et les rapports (« *In-Register Microkernel Compute Saturation* »). |
| 🟠 **P1** | **Ingénierie** | Pollution de warmup par l'instanciation de `CycleCounter` | ✅ **Résolu** : `CycleCounter cyc_counter;` déplacé en amont de `cfg.warmup` dans [src/main.cpp](file:///home/ivan/Files/projets/gemm_bench/src/main.cpp). |
| 🟠 **P1** | **Méthodologie** | Chute à $K \ge 256$ non expliquée microarchitecturalement | ✅ **Documenté** : Transition L1d $\to$ L2 ($176 \times K$ octets) explicitée (saturation de capacité L1d sur cœur in-order). |
| 🟡 **P2** | **Conformité** | Utilisation inconditionnelle de `-ffast-math` | ✅ **Implémenté** : Option CMake `GEMMBENCH_ENABLE_FASTMATH` et drapeau CLI `--no-fast-math` ajoutés. |
| 🟡 **P2** | **Métrologie** | Compteur unique (cycles seuls) | ⏳ **En cours** : Intégration future de `PERF_COUNT_HW_INSTRUCTIONS` pour l'IPC. |
| 🟢 **P3** | **Architecture** | Fragmentation des scripts Python et fichiers monolithiques | ⏳ **À planifier** : Factorisation des scripts d'analyse et nettoyage des fichiers orphelins. |
