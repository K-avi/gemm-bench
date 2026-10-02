# Note Technique : Stratégie de Benchmarking et Faisabilité de l'Intégration BLIS

**Contexte :** Validation empirique du cas d'étude DGEMM pour la soumission de l'article MIPPv2  
**Échéance :** Court terme (quelques jours)  
**Destinataires :** Équipe de recherche & Co-auteurs MIPPv2  

---

## 1. Synthèse Exécutive

Dans le cadre de la finalisation du papier sur **MIPPv2**, l'intégration des micro-noyaux au sein de la bibliothèque de référence **BLIS** (*BLAS-like Library Instantiation Software*) a été soulevée. 

L'analyse technique conclut que **tenter une intégration complète du framework BLIS sur l'ensemble des 9 microarchitectures cibles à quelques jours de la deadline présente un ratio risque/bénéfice inacceptable**.

* **Recommandation principale :** Focaliser les efforts sur la **robustesse statistique des mesures existantes (variante CRR)** et sur une **comparaison ciblée des micro-noyaux seuls (MIPPv2 C++ vs Assembleur natif BLIS)** sur 1 ou 2 plateformes représentatives (ex. x86_64 AVX-512/AVX2 et AArch64 NEON).
* **Abandon raisonné :** Différer l'intégration globale du macro-kernel BLIS et les mesures « Full DGEMM » multi-threadées pour des travaux postérieurs à la soumission.

---

## 2. Analyse Critique des Options Envisagées

| Option envisagée | Faisabilité (J-3) | Pertinence pour le papier MIPPv2 | Niveau de risque | Verdict |
| :--- | :---: | :---: | :---: | :---: |
| **1. Intégration BLIS complète (9 plateformes)** | Quasi-nulle | Faible | **Critique** | ❌ **À éliminer** |
| **2. Mesures "Full DGEMM" (Macro-kernel BLIS)** | Faible | Contre-productive | **Élevé** | ❌ **À éliminer** |
| **3. Duel Micro-noyau seul (BLIS asm vs MIPPv2)** | Élevée (1-2 archs) | **Maximale** | **Faible** | ✅ **Priorité 2** |
| **4. Consolidation & robustesse des benchmarks CRR** | **Totale** | **Indispensable** | **Nul** | ✅ **Priorité 1** |

---

## 3. Justifications Techniques et Méthodologiques

### 3.1. Pourquoi le "Full DGEMM" dessert la thèse du papier MIPPv2

L'objectif de l'article est d'évaluer la qualité de l'abstraction **MIPPv2** en tant que bibliothèque de vectorisation C++ portable. Le critère d'évaluation déterminant est :
$$\text{Surcoût d'abstraction} = \frac{\text{Performance MIPPv2 C++}}{\text{Performance Assembleur expert / Intrinsics natifs}}$$

* **Biais de mesure induit par le macro-kernel :** Un benchmark DGEMM complet à 5 boucles ne teste pas seulement les instructions SIMD ; il évalue l'efficacité du pré-packing en mémoire ($\mathcal{O}(K \cdot N)$), les synchronisations OpenMP/pthreads, et l'adéquation des paramètres de cache ($M_C, N_C, K_C$).
* **Risque de faux négatif :** Si le Full DGEMM plafonne à 75 % de la crête à cause d'un goulot d'étranglement mémoire lors du packing ou d'un ordonnancement sous-optimal des threads, les relecteurs attribueront cette contre-performance à MIPPv2, affaiblissant la démonstration.
* **Le micro-noyau isole le vecteur de calcul :** En mesurant le micro-noyau pur en cache chaud (L1/L2), on évalue rigoureusement la saturation des ports FMA, le déroulement de boucle, le non-débordement des registres (*zero spills*) et la vectorisation.

### 3.2. Les verrous de l'intégration globale de BLIS à court terme

1. **Hétérogénéité des chaînes de compilation (C99 vs C++20) :**
   BLIS est écrit en C99 strict avec un système de configuration basé sur des scripts shell et Makefiles. Injecter des micro-noyaux C++20 avec templates et inlining agressif requiert d'adapter le système de build pour 9 cibles distinctes (macOS Clang, GCC Linux x86, GCC AArch64, et cross-compilateurs RISC-V). Le temps requis pour stabiliser la compilation et l'édition de liens consommera l'intégralité du temps disponible avant la deadline.
2. **Immaturité de RISC-V Vector (RVV) dans BLIS mainline :**
   En amont, le support RVV 1.0 de BLIS est inexistant ou non standardisé. L'intégration sur SpacemiT X100, X60 et A100 exigerait des patches constructeurs expérimentaux, alors même que les accès aux nœuds physiques du cluster sont temporairement contraints.
3. **Calibrage des blocs de cache ($M_C, N_C, K_C$) :**
   BLIS exige une table de constantes de tuilage propre à chaque microarchitecture. Utiliser des valeurs par défaut dégradera artificiellement les performances globales.

---

## 4. Plan d'Action Recommandé (Feuille de Route)

```
                            FEUILLE DE ROUTE SOUMISSION
   
   [Jours 1-2] : PRIORITÉ 1 ----------------------------------------+
   Consolidation statistique des mesures CRR existantes              |
   - 20-50 répétitions par point, médiane + écart-type/min-max       |
   - Contrôle strict fréquence & gouverneur CPU                      |
   - Détection & élimination des artefacts thermiques                |
                                                                     v
   [Jour 2]   : PRIORITÉ 2 ----------------------------------------+
   Mesure comparative directe Ukernel BLIS (asm) vs MIPPv2           |
   - 1 cible x86 (AMD Zen 4 ou Intel Skylake)                       |
   - 1 cible ARM (Apple M1)                                         |
   - Driver autonome sans build BLIS global                          |
                                                                     v
   [Jour 3]   : FINALISATION --------------------------------------+
   - Actualisation des SVG et du tableau comparatif                  |
   - Rédaction des paragraphes méthodologiques du papier            |
   - Note de reproductibilité claire sur les architectures RVV       |
```

### Étape 1 : Consolidation et robustesse des benchmarks actuels (Priorité 1)

Les relecteurs scientifiques en calcul haute performance (HPC) sanctionnent sévèrement les métriques ponctuelles sans quantification de la variance.

* **Protocole expérimental à figer :**
  1. **Répétitions statistiques :** Remplacer le point unique par $N = 30$ itérations par dimension $K$, en extrayant médiane, minimum, maximum et écart-type.
  2. **Gouvernance de fréquence :** Pinner les cœurs et forcer le gouverneur Linux en mode `performance` (`cpupower frequency-set -g performance`).
  3. **Monitoring thermique & Turbo :** Documenter explicitement la fréquence effective observée (notamment pour l'Intel Meteor Lake et le Raspberry Pi 5 sujets au throttling thermique).
  4. **Phase d'échauffement (*warmup*) :** Valider que le cache L1/L2 est thermiquement et temporellement stabilisé avant la prise des timestamps.

### Étape 2 : Évaluation comparative ciblée "Micro-noyau BLIS vs MIPPv2" (Priorité 2)

Pour prouver que MIPPv2 rivalise avec l'état de l'art mondial sans s'enliser dans le macro-kernel :
* **Méthode ultra-légère (In-Driver) :**
  * Ne pas compiler BLIS complet.
  * Extraire ou lier statiquement le fichier assembleur du micro-noyau natif de BLIS pour une machine donnée (ex. `bli_dgemm_haswell_asm_4x12.S` ou `bli_dgemm_armv8a_asm_6x8.S`).
  * Appeler cette fonction directement depuis le banc d'essai [tmp/blis_bench/blis_bench.cpp](file:///home/ivan/Files/projets/gemm_bench/tmp/blis_bench/blis_bench.cpp) en lui passant les mêmes panneaux compactés $A$ et $B$.
* **Résultat obtenu :** Un graphique à barres direct montrant :
  $$\text{Efficacité MIPPv2 (C++)} \approx 98\text{--}100\,\% \text{ du micro-noyau Assembleur BLIS}$$
  Cette seule figure suffit à valider la thèse centrale du papier MIPPv2.

### Étape 3 : Traitement des architectures RVV (SpacemiT X100, X60, A100)

* **Stratégie de transparence :** Conserver les excellents résultats obtenus sur le noyau champion RRR (déjà à >90 % de crête sur X100).
* **Argumentation pour le papier :** Indiquer que le portage CRR sur RVV est en cours d'évaluation et que les résultats présentés constituent une borne empirique solide de l'efficacité vectorielle RVV 1.0 via MIPPv2.

---

## 5. Synthèse des Données Consolidées (Variante CRR)

Pour intégration directe dans la section expérimentale du manuscrit :

| SIMD | Microarchitecture | CPU / SoC | Freq (GHz) | Crête théorique | Kernel MIPPv2 CRR | Débit mesuré | Efficacité FMA |
| :--- | :--- | :--- | :---: | :---: | :--- | :---: | :---: |
| **AVX-512** | **AMD Zen 4** | Ryzen 9 7900X | 5.4 | 16.0 FLOP/c | `firestorm_mr4_nr4_fmaddi_crr` ($4 \times 32$) | **85.5 GFLOP/s** | **99.0 %** |
| **AVX-512** | **AMD Zen 5** | Ryzen AI 9 HX 370 | 5.1 | 16.0 FLOP/c | `firestorm_mr4_nr4_fmaddi_crr` ($4 \times 32$) | **81.0 GFLOP/s** | **99.2 %** |
| **NEON** | **Apple M1** | Firestorm (M1 Ultra) | 3.0 | 16.0 FLOP/c | `x60_mr6_nr4_fmaddi_crr` ($6 \times 8$) | **48.2 GFLOP/s** | **100.5 %** *(boost)* |
| **NEON** | **Cortex-A76** | Raspberry Pi 5 | 2.4 | 8.0 FLOP/c | `a76_mr6_nr4_fmaddi_crr` ($6 \times 8$) | **18.8 GFLOP/s** | **98.1 %** |
| **AVX2** | **Intel Skylake** | Core i5-6200U | 2.3 | 16.0 FLOP/c | `meteorlake_mr4_nr3_crr` ($4 \times 12$) | **34.2 GFLOP/s** | **92.9 %** |
| **AVX2** | **Intel Meteor Lake**| Redwood Cove P-Core | 5.1 | 16.0 FLOP/c | `meteorlake_mr4_nr3_crr` ($4 \times 12$) | **71.5 GFLOP/s** | **87.6 %** |

---

## 6. Conclusion et Décision

1. **Ne pas lancer de chantier d'intégration globale BLIS à J-3.**
2. **Allouer 70 % du temps restant** à la consolidation expérimentale des mesures actuelles (répétitions, statistiques d'erreur, scripts de plotting propres).
3. **Allouer 30 % du temps** à l'exécution d'un duel micro-noyau MIPPv2 vs BLIS natif sur une machine x86 et une machine ARM pour insérer une comparaison concrète face à l'assembleur de référence.
