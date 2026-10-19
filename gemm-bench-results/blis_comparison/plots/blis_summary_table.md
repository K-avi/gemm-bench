# GEMMBench — Comparative Analysis: MIPPv2 vs BLIS Reference Micro-Kernels (Double-Precision (FP64 / DGEMM))

Empirical comparison of sustained in-register compute density and throughput between MIPPv2 champions
and hand-tuned BLIS assembly micro-kernels across target microarchitectures in Double-Precision (FP64 / DGEMM).

## 1. Symmetric Microarchitectural Comparison (Direct Micro-Kernel Equivalents)

> [!NOTE]
> **Methodological Scope:**
> - **In-Register Scope:** This benchmark strictly measures in-register compute saturation ($M_R \times N_R \times K$) on hot L1 cache panels, isolating FMA pipeline throughput without end-to-end DGEMM cache blocking (GotoBLAS $J_C, P_C, I_C$), dynamic packing overheads, or fringe tiles.
> - **Architectural Symmetry:** Evaluates targets where BLIS provides a native, ISA-matched microkernel (AVX-512, AVX2, NEON, RVV 256b).
> - **Native Epilogue Layout Alignment:** Each microkernel is evaluated against its optimal hardware storage layout: MIPPv2 writes $C$ in row-major (`CRR`), while BLIS writes $C$ in native column-major (`CRC` on SKX, X60, and ARMv8A) to ensure both frameworks leverage contiguous vector stores and avoid scalar scatter/gather epilogue distortion.
> - **Small-$K$ Epilogue & Prologue Specialization ($K=32, 64$):** By default, MIPPv2 microkernels use compile-time `FastPath` specialization ($\alpha=1, \beta=0$) that emits direct vector stores without $\alpha$ scaling or $\beta$ branch checks, whereas BLIS assembly microkernels execute runtime $\alpha/\beta$ epilogue scaling and unconditional prologue $C$-tile prefetches (e.g., 24 `prfm pldl1keep` instructions in `armv8a`). At $K=32$, this fixed prologue/epilogue instruction overhead accounts for the 25–35% throughput gap on ARMv8-A and AVX-512 (use `--no-fastpath` to evaluate MIPPv2 with runtime $\alpha/\beta$ epilogue).
> - **L1d/L2 Working Set Boundary:** While $K \in [32, 128]$ fits entirely within L1d cache, $K \ge 256$ spills the packed panel $B$ ($W_B = K \times N_R \times 4\text{ bytes}$) into L2 cache (e.g. 64–256 KiB on AVX-512 with $N_R=64$), testing sustained L2 streaming throughput rather than pure in-register FMA saturation.

| Microarchitecture | Compiler | Layout | MIPPv2 Champion | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **% of BLIS Peak** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Status |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `m1_neon` | `clang` | `CRR` | `x60_mr6_nr4_fmaddi_crr` | **48.26** | **47.93** | **100.7%** | 15.93 | 15.82 | 🟢 Parity (98–102%) |
| `m1_neon` | `gcc` | `CRR` | `x60_mr6_nr4_fmaddi_crr` | **48.29** | **47.93** | **100.8%** | 15.93 | 15.81 | 🟢 Parity (98–102%) |
| `meteorlake_avx2` | `clang` | `CRR` | `meteorlake_mr4_nr3_crr` | **74.11** | **74.10** | **100.0%** | 15.54 | 15.44 | 🟢 Parity (98–102%) |
| `meteorlake_avx2` | `gcc` | `CRR` | `meteorlake_mr4_nr3_crr` | **72.49** | **74.20** | **97.7%** | 15.14 | 15.43 | 🟢 Near-Parity (≥95%) |
| `rpi5_neon` | `clang` | `CRR` | `a76_mr6_nr4_fmaddi_crr` | **18.90** | **18.77** | **100.7%** | 7.90 | 7.85 | 🟢 Parity (98–102%) |
| `rpi5_neon` | `gcc` | `CRR` | `a76_mr6_nr4_fmaddi_crr` | **18.92** | **18.77** | **100.8%** | 7.91 | 7.85 | 🟢 Parity (98–102%) |
| `x100_rvv` | `clang` | `CRR` | `x100_register_blocked_apack4_crr` | **18.51** | **18.01*** | **102.7%** | 7.72 | 7.51 | 🔵 MIPPv2 (+2.7%) |
| `x100_rvv` | `gcc` | `CRR` | `x100_register_blocked_apack4_crr` | **16.20** | **18.05*** | **89.8%** | 6.76 | 7.53 | 🔴 Delta (<90%) |
| `x60_rvv` | `clang` | `CRR` | `a100_mr7_nr4_pipe_crr` | **7.43** | **10.27** | **72.3%** | 4.65 | 6.43 | 🔴 Delta (<90%) |
| `x60_rvv` | `gcc` | `CRR` | `a100_mr7_nr4_pipe_crr` | **7.29** | **10.27** | **71.0%** | 4.57 | 6.43 | 🔴 Delta (<90%) |
| `zen4_avx512` | `clang` | `CRR` | `firestorm_mr4_nr4_fmaddi_crr` | **85.25** | **84.54** | **100.8%** | 15.99 | 15.95 | 🟢 Parity (98–102%) |
| `zen4_avx512` | `gcc` | `CRR` | `firestorm_mr4_nr4_fmaddi_crr` | **85.99** | **84.66** | **101.6%** | 15.95 | 15.95 | 🟢 Parity (98–102%) |
| `zen5_avx512` | `clang` | `CRR` | `firestorm_mr4_nr4_fmaddi_crr` | **81.20** | **80.98** | **100.3%** | 16.00 | 15.98 | 🟢 Parity (98–102%) |
| `zen5_avx512` | `gcc` | `CRR` | `firestorm_mr4_nr4_fmaddi_crr` | **82.11** | **81.13** | **101.2%** | 16.00 | 15.98 | 🟢 Parity (98–102%) |

*\*Note: On SpacemiT X100 (RVV 256b, OOO dual-issue), BLIS runs the in-order X60 microkernel (`dgemm_x60_2vx14.c`).*

---

## 2. VLEN Agility & Cross-Vector Portability Study (SpacemiT A100 VLEN=1024b)

> [!IMPORTANT]
> **Asymmetry Disclosure & Portability Framing:**
> - Upstream BLIS provides no native 1024-bit vector microkernel; executing BLIS on A100 defaults to the 256-bit in-order X60 assembly kernel (`dgemm_x60_2vx14.c, 8×14`), utilizing only 25% of vector capacity per iteration.
> - MIPPv2's parameterized template vectorization automatically adapts to the 1024-bit VLEN across precision targets.
> - This comparison is presented not as a peer algorithmic match, but as an empirical demonstration of C++ template agility across variable vector lengths versus static assembly fragility.

| Microarchitecture | Compiler | MIPPv2 Champion | MIPPv2 Peak (GFLOP/s) | MIPPv2 (FLOP/cyc) | Hardware Peak (FLOP/cyc) | % Machine Peak | BLIS Fallback (X60 asm GFLOP/s) | Relative Speedup |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `a100_rvv` | `clang` | `a100_mr7_nr2_lmul2_pipe_crr` | **12.80** | 6.43 | 8.0 | **80.3%** | 3.69 (X60) | **3.47×** |
| `a100_rvv` | `gcc` | `a100_mr7_nr2_lmul2_pipe_crr` | **13.35** | 6.70 | 8.0 | **83.8%** | 3.66 (X60) | **3.65×** |

---

## 3. Visual Comparative Figures

- **Peak Performance Ratio:** [plots/blis_vs_mippv2_ratio_comparison.svg](plots/blis_vs_mippv2_ratio_comparison.svg)
- **FLOP / Cycle Compute Density:** [plots/blis_vs_mippv2_flop_per_cycle_comparison.svg](plots/blis_vs_mippv2_flop_per_cycle_comparison.svg)
- **Scaling Grid Across K ∈ [32..1024]:** [plots/blis_vs_mippv2_k_scaling_grid.svg](plots/blis_vs_mippv2_k_scaling_grid.svg)


---

## 4. Per-Size Scaling & Efficiency Analysis ($K \in [32, 1024]$)

### Platform: `a100_rvv` (clang, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32.0 | 11.96 | 3.45 | **347.1%** | 6.00 | 1.73 | MIPPv2 (+247.1%) |
| K=64.0 | 12.40 | 3.57 | **347.8%** | 6.22 | 1.79 | MIPPv2 (+247.8%) |
| K=128.0 | 12.64 | 3.63 | **348.3%** | 6.34 | 1.82 | MIPPv2 (+248.3%) |
| K=256.0 | 12.75 | 3.65 | **349.1%** | 6.40 | 1.83 | MIPPv2 (+249.1%) |
| K=512.0 | 12.80 | 3.67 | **349.3%** | 6.43 | 1.84 | MIPPv2 (+249.3%) |
| K=1024.0 | 12.74 | 3.69 | **345.3%** | 6.40 | 1.85 | MIPPv2 (+245.3%) |

### Platform: `a100_rvv` (gcc, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32.0 | 12.73 | 3.45 | **368.9%** | 6.38 | 1.73 | MIPPv2 (+268.9%) |
| K=64.0 | 13.12 | 3.57 | **367.4%** | 6.58 | 1.79 | MIPPv2 (+267.4%) |
| K=128.0 | 13.22 | 3.63 | **364.1%** | 6.63 | 1.82 | MIPPv2 (+264.1%) |
| K=256.0 | 13.31 | 3.66 | **364.0%** | 6.68 | 1.83 | MIPPv2 (+264.0%) |
| K=512.0 | 13.35 | 3.60 | **370.7%** | 6.70 | 1.81 | MIPPv2 (+270.7%) |
| K=1024.0 | 13.30 | 3.59 | **370.4%** | 6.68 | 1.80 | MIPPv2 (+270.4%) |

### Platform: `m1_neon` (clang, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32.0 | 44.98 | 36.57 | **123.0%** | 14.84 | 12.06 | MIPPv2 (+23.0%) |
| K=64.0 | 46.68 | 41.69 | **112.0%** | 15.40 | 13.75 | MIPPv2 (+12.0%) |
| K=128.0 | 46.79 | 44.86 | **104.3%** | 15.44 | 14.80 | MIPPv2 (+4.3%) |
| K=256.0 | 47.62 | 46.60 | **102.2%** | 15.71 | 15.37 | MIPPv2 (+2.2%) |
| K=512.0 | 48.06 | 47.40 | **101.4%** | 15.85 | 15.64 | MIPPv2 (+1.4%) |
| K=1024.0 | 48.26 | 47.93 | **100.7%** | 15.93 | 15.82 | MIPPv2 (+0.7%) |

### Platform: `m1_neon` (gcc, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32.0 | 45.53 | 36.34 | **125.3%** | 15.02 | 11.99 | MIPPv2 (+25.3%) |
| K=64.0 | 46.97 | 41.55 | **113.0%** | 15.49 | 13.71 | MIPPv2 (+13.0%) |
| K=128.0 | 46.97 | 44.76 | **104.9%** | 15.49 | 14.76 | MIPPv2 (+4.9%) |
| K=256.0 | 47.72 | 46.55 | **102.5%** | 15.74 | 15.36 | MIPPv2 (+2.5%) |
| K=512.0 | 48.11 | 47.38 | **101.5%** | 15.87 | 15.63 | MIPPv2 (+1.5%) |
| K=1024.0 | 48.29 | 47.93 | **100.8%** | 15.93 | 15.81 | MIPPv2 (+0.8%) |

### Platform: `meteorlake_avx2` (clang, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32.0 | 71.41 | 67.26 | **106.2%** | 14.93 | 14.01 | MIPPv2 (+6.2%) |
| K=64.0 | 73.10 | 71.78 | **101.8%** | 15.31 | 14.98 | MIPPv2 (+1.8%) |
| K=128.0 | 73.33 | 72.90 | **100.6%** | 15.34 | 15.18 | MIPPv2 (+0.6%) |
| K=256.0 | 74.08 | 74.10 | **100.0%** | 15.51 | 15.43 | Equivalent |
| K=512.0 | 73.46 | 72.82 | **100.9%** | 15.44 | 15.20 | MIPPv2 (+0.9%) |
| K=1024.0 | 74.11 | 74.04 | **100.1%** | 15.54 | 15.44 | Equivalent |

### Platform: `meteorlake_avx2` (gcc, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32.0 | 65.68 | 67.32 | **97.6%** | 13.51 | 14.01 | BLIS (+2.4%) |
| K=64.0 | 67.41 | 71.37 | **94.4%** | 14.01 | 14.87 | BLIS (+5.6%) |
| K=128.0 | 69.96 | 72.64 | **96.3%** | 14.61 | 15.10 | BLIS (+3.7%) |
| K=256.0 | 72.49 | 73.87 | **98.1%** | 15.14 | 15.33 | BLIS (+1.9%) |
| K=512.0 | 62.44 | 73.25 | **85.2%** | 12.94 | 15.23 | BLIS (+14.8%) |
| K=1024.0 | 61.38 | 74.20 | **82.7%** | 12.73 | 15.43 | BLIS (+17.3%) |

### Platform: `rpi5_neon` (clang, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32.0 | 17.53 | 12.45 | **140.8%** | 7.33 | 5.20 | MIPPv2 (+40.8%) |
| K=64.0 | 18.26 | 15.07 | **121.2%** | 7.64 | 6.30 | MIPPv2 (+21.2%) |
| K=128.0 | 18.65 | 16.86 | **110.6%** | 7.80 | 7.05 | MIPPv2 (+10.6%) |
| K=256.0 | 18.89 | 17.92 | **105.4%** | 7.90 | 7.49 | MIPPv2 (+5.4%) |
| K=512.0 | 18.90 | 18.50 | **102.1%** | 7.90 | 7.74 | MIPPv2 (+2.1%) |
| K=1024.0 | 18.51 | 18.77 | **98.6%** | 7.74 | 7.85 | BLIS (+1.4%) |

### Platform: `rpi5_neon` (gcc, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32.0 | 17.57 | 12.56 | **139.8%** | 7.35 | 5.25 | MIPPv2 (+39.8%) |
| K=64.0 | 18.32 | 15.14 | **121.0%** | 7.66 | 6.33 | MIPPv2 (+21.0%) |
| K=128.0 | 18.71 | 16.89 | **110.8%** | 7.83 | 7.06 | MIPPv2 (+10.8%) |
| K=256.0 | 18.92 | 17.94 | **105.4%** | 7.91 | 7.50 | MIPPv2 (+5.4%) |
| K=512.0 | 18.90 | 18.51 | **102.1%** | 7.90 | 7.74 | MIPPv2 (+2.1%) |
| K=1024.0 | 18.12 | 18.77 | **96.5%** | 7.58 | 7.85 | BLIS (+3.5%) |

### Platform: `x100_rvv` (clang, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32.0 | 14.94 | 16.31 | **91.6%** | 6.23 | 6.80 | BLIS (+8.4%) |
| K=64.0 | 16.78 | 17.21 | **97.5%** | 7.00 | 7.18 | BLIS (+2.5%) |
| K=128.0 | 17.88 | 17.69 | **101.1%** | 7.46 | 7.38 | MIPPv2 (+1.1%) |
| K=256.0 | 18.51 | 17.89 | **103.4%** | 7.72 | 7.46 | MIPPv2 (+3.4%) |
| K=512.0 | 15.66 | 17.86 | **87.7%** | 6.54 | 7.45 | BLIS (+12.3%) |
| K=1024.0 | 15.63 | 18.01 | **86.8%** | 6.52 | 7.51 | BLIS (+13.2%) |

### Platform: `x100_rvv` (gcc, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32.0 | 13.98 | 16.35 | **85.5%** | 5.83 | 6.82 | BLIS (+14.5%) |
| K=64.0 | 15.15 | 17.20 | **88.1%** | 6.32 | 7.17 | BLIS (+11.9%) |
| K=128.0 | 15.83 | 17.70 | **89.4%** | 6.61 | 7.38 | BLIS (+10.6%) |
| K=256.0 | 16.20 | 17.91 | **90.5%** | 6.76 | 7.47 | BLIS (+9.5%) |
| K=512.0 | 13.69 | 17.94 | **76.3%** | 5.71 | 7.49 | BLIS (+23.7%) |
| K=1024.0 | 14.14 | 18.05 | **78.3%** | 5.90 | 7.53 | BLIS (+21.7%) |

### Platform: `x60_rvv` (clang, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32.0 | 6.46 | 9.00 | **71.8%** | 4.04 | 5.63 | BLIS (+28.2%) |
| K=64.0 | 7.07 | 9.82 | **72.0%** | 4.43 | 6.15 | BLIS (+28.0%) |
| K=128.0 | 7.43 | 10.27 | **72.3%** | 4.65 | 6.43 | BLIS (+27.7%) |
| K=256.0 | 6.14 | 6.84 | **89.8%** | 3.84 | 4.28 | BLIS (+10.2%) |
| K=512.0 | 6.43 | 7.17 | **89.8%** | 4.03 | 4.49 | BLIS (+10.2%) |
| K=1024.0 | 6.51 | 7.30 | **89.3%** | 4.08 | 4.57 | BLIS (+10.7%) |

### Platform: `x60_rvv` (gcc, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32.0 | 6.57 | 8.89 | **74.0%** | 4.11 | 5.56 | BLIS (+26.0%) |
| K=64.0 | 7.04 | 9.81 | **71.7%** | 4.41 | 6.14 | BLIS (+28.3%) |
| K=128.0 | 7.29 | 10.27 | **71.0%** | 4.57 | 6.43 | BLIS (+29.0%) |
| K=256.0 | 5.88 | 7.01 | **83.9%** | 3.68 | 4.39 | BLIS (+16.1%) |
| K=512.0 | 6.16 | 7.28 | **84.6%** | 3.85 | 4.55 | BLIS (+15.4%) |
| K=1024.0 | 6.28 | 7.36 | **85.3%** | 3.93 | 4.61 | BLIS (+14.7%) |

### Platform: `zen4_avx512` (clang, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32.0 | 85.25 | 78.66 | **108.4%** | 15.99 | 14.77 | MIPPv2 (+8.4%) |
| K=64.0 | 85.14 | 81.32 | **104.7%** | 15.99 | 15.33 | MIPPv2 (+4.7%) |
| K=128.0 | 85.01 | 82.75 | **102.7%** | 15.97 | 15.61 | MIPPv2 (+2.7%) |
| K=256.0 | 85.02 | 83.89 | **101.3%** | 15.97 | 15.82 | MIPPv2 (+1.3%) |
| K=512.0 | 84.98 | 84.29 | **100.8%** | 15.97 | 15.92 | MIPPv2 (+0.8%) |
| K=1024.0 | 85.01 | 84.54 | **100.6%** | 15.97 | 15.95 | MIPPv2 (+0.6%) |

### Platform: `zen4_avx512` (gcc, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32.0 | 85.99 | 78.15 | **110.0%** | 15.92 | 14.62 | MIPPv2 (+10.0%) |
| K=64.0 | 85.61 | 81.42 | **105.1%** | 15.95 | 15.30 | MIPPv2 (+5.1%) |
| K=128.0 | 85.43 | 83.00 | **102.9%** | 15.92 | 15.63 | MIPPv2 (+2.9%) |
| K=256.0 | 85.37 | 84.04 | **101.6%** | 15.95 | 15.84 | MIPPv2 (+1.6%) |
| K=512.0 | 85.37 | 84.43 | **101.1%** | 15.94 | 15.92 | MIPPv2 (+1.1%) |
| K=1024.0 | 85.30 | 84.66 | **100.8%** | 15.94 | 15.95 | MIPPv2 (+0.8%) |

### Platform: `zen5_avx512` (clang, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32.0 | 81.20 | 78.72 | **103.1%** | 16.00 | 15.51 | MIPPv2 (+3.1%) |
| K=64.0 | 81.19 | 79.93 | **101.6%** | 16.00 | 15.75 | MIPPv2 (+1.6%) |
| K=128.0 | 81.17 | 80.54 | **100.8%** | 16.00 | 15.87 | MIPPv2 (+0.8%) |
| K=256.0 | 80.94 | 80.73 | **100.3%** | 16.00 | 15.93 | Equivalent |
| K=512.0 | 81.02 | 80.89 | **100.2%** | 16.00 | 15.96 | Equivalent |
| K=1024.0 | 81.03 | 80.98 | **100.1%** | 16.00 | 15.98 | Equivalent |

### Platform: `zen5_avx512` (gcc, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32.0 | 82.11 | 79.08 | **103.8%** | 15.99 | 15.51 | MIPPv2 (+3.8%) |
| K=64.0 | 81.85 | 80.19 | **102.1%** | 16.00 | 15.75 | MIPPv2 (+2.1%) |
| K=128.0 | 81.68 | 80.75 | **101.2%** | 16.00 | 15.87 | MIPPv2 (+1.2%) |
| K=256.0 | 81.36 | 80.89 | **100.6%** | 16.00 | 15.93 | MIPPv2 (+0.6%) |
| K=512.0 | 81.40 | 81.07 | **100.4%** | 16.00 | 15.96 | Equivalent |
| K=1024.0 | 81.40 | 81.13 | **100.3%** | 16.00 | 15.98 | Equivalent |


---

## 5. Reference Assembly Kernels Provenance & Acknowledgements

The hand-tuned assembly microkernels used as baseline references in this benchmark originate from:

- **RISC-V Vector 1.0 (SpacemiT X60):** Authored and tuned by **Stepan Nassyr** via genetic loop exploration (`asmgen-gemm`), available in the [`stepannassyr/blis`](https://github.com/stepannassyr/blis/tree/x60/kernels) repository (`x60` branch, commit [`6ffc1d3`](https://github.com/stepannassyr/blis/commit/6ffc1d35fcafe74788b2aeaf00d5792d2618e0b9)).
- **x86-64 (AVX2, AVX-512) & AArch64 (NEON):** Sourced from the official upstream **FLAME BLIS** project ([`flame/blis`](https://github.com/flame/blis)), maintained by The University of Texas at Austin and contributors (commit [`a8037cf`](https://github.com/flame/blis/commit/a8037cfc4330030c726efaec1ffeb1f93d468aa3), based on BLIS 2.1).
