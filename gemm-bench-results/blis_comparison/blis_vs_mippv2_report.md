# GEMMBench — Comparative Analysis: MIPPv2 vs BLIS Reference Micro-Kernels

Empirical comparison of sustained in-register compute density and throughput between MIPPv2 champions
and hand-tuned BLIS assembly micro-kernels across target microarchitectures.

## 1. Executive Summary Table (% of BLIS Peak Reached)

> [!NOTE]
> **Methodological Scope & RISC-V Baseline Distinction:**
> - **In-Register Scope:** This benchmark strictly measures in-register compute saturation ($M_R \times N_R \times K$) on hot L1 cache panels, isolating FMA pipeline efficiency without end-to-end DGEMM cache blocking (GotoBLAS $J_C, P_C, I_C$), dynamic packing overheads, or fringe tiles.
> - **RISC-V Baselines (`a100_rvv` & `x100_rvv`):** Upstream BLIS only provides an RVV microkernel for SpacemiT X60 (`dgemm_x60_2vx14.c`, $8 \times 14$). On A100 and X100, this X60 kernel is used as an untuned cross-RVV baseline. The large relative advantage of MIPPv2 on A100 reflects the absence of a native A100-tailored assembly kernel in BLIS.

| Microarchitecture | Compiler | Layout | MIPPv2 Champion | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **% of BLIS Peak** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Status |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `a100_rvv` | `clang` | `CRR` | `a100_mr7_nr2_lmul2_pipe_crr` | **12.81** | **3.66*** | **350.0%** | 6.43 | 1.84 | 🟢 Parity (≥99%) |
| `a100_rvv` | `gcc` | `CRR` | `a100_mr7_nr2_lmul2_pipe_crr` | **13.36** | **3.66*** | **365.1%** | 6.71 | 1.84 | 🟢 Parity (≥99%) |
| `m1_neon` | `clang` | `CRR` | `x60_mr6_nr4_fmaddi_crr` | **48.27** | **47.93** | **100.7%** | 15.93 | 15.81 | 🟢 Parity (≥99%) |
| `m1_neon` | `gcc` | `CRR` | `x60_mr6_nr4_fmaddi_crr` | **48.30** | **47.94** | **100.8%** | 15.93 | 15.81 | 🟢 Parity (≥99%) |
| `meteorlake_avx2` | `clang` | `CRR` | `meteorlake_mr4_nr3_crr` | **72.05** | **72.22** | **99.8%** | 15.55 | 15.42 | 🟢 Parity (≥99%) |
| `meteorlake_avx2` | `gcc` | `CRR` | `meteorlake_mr4_nr3_crr` | **69.97** | **72.81** | **96.1%** | 15.10 | 15.47 | 🟢 Near-Parity (≥95%) |
| `rpi5_neon` | `gcc` | `CRR` | `a76_mr6_nr4_fmaddi_crr` | **18.92** | **18.78** | **100.8%** | 7.91 | 7.85 | 🟢 Parity (≥99%) |
| `x100_rvv` | `clang` | `CRR` | `x100_register_blocked_apack4_crr` | **18.49** | **18.02*** | **102.6%** | 7.72 | 7.52 | 🟢 Parity (≥99%) |
| `x100_rvv` | `gcc` | `CRR` | `x100_register_blocked_apack4_crr` | **16.22** | **18.06*** | **89.8%** | 6.77 | 7.54 | 🔴 Delta (<90%) |
| `x60_rvv` | `clang` | `CRR` | `a100_mr7_nr4_pipe_crr` | **7.43** | **10.28** | **72.2%** | 4.65 | 6.44 | 🔴 Delta (<90%) |
| `x60_rvv` | `gcc` | `CRR` | `a100_mr7_nr4_pipe_crr` | **7.30** | **11.19** | **65.2%** | 4.57 | 7.00 | 🔴 Delta (<90%) |
| `zen4_avx512` | `clang` | `CRR` | `firestorm_mr4_nr4_fmaddi_crr` | **83.82** | **83.12** | **100.8%** | 16.00 | 15.95 | 🟢 Parity (≥99%) |
| `zen4_avx512` | `gcc` | `CRR` | `firestorm_mr4_nr4_fmaddi_crr` | **84.05** | **83.24** | **101.0%** | 15.99 | 15.94 | 🟢 Parity (≥99%) |
| `zen5_avx512` | `clang` | `CRR` | `firestorm_mr4_nr4_fmaddi_crr` | **79.88** | **79.81** | **100.1%** | 16.00 | 15.98 | 🟢 Parity (≥99%) |
| `zen5_avx512` | `gcc` | `CRR` | `firestorm_mr4_nr4_fmaddi_crr` | **81.06** | **79.79** | **101.6%** | 16.00 | 15.98 | 🟢 Parity (≥99%) |

*\*Note: On A100 and X100, BLIS uses the X60 kernel fallback as an untuned cross-RVV baseline.*

---

## 2. Visual Comparative Figures

- **Peak Performance Ratio:** [plots/blis_vs_mippv2_ratio_comparison.svg](plots/blis_vs_mippv2_ratio_comparison.svg)
- **FLOP / Cycle Compute Density:** [plots/blis_vs_mippv2_flop_per_cycle_comparison.svg](plots/blis_vs_mippv2_flop_per_cycle_comparison.svg)
- **Scaling Grid Across K ∈ [32..1024]:** [plots/blis_vs_mippv2_k_scaling_grid.svg](plots/blis_vs_mippv2_k_scaling_grid.svg)


---

## 3. Per-Size Scaling & Efficiency Analysis ($K \in [32, 1024]$)

### Platform: `a100_rvv` (clang, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32.0 | 11.95 | 3.45 | **346.4%** | 6.00 | 1.73 | MIPPv2 (+246.4%) |
| K=64.0 | 12.40 | 3.57 | **347.2%** | 6.22 | 1.79 | MIPPv2 (+247.2%) |
| K=128.0 | 12.64 | 3.63 | **347.9%** | 6.34 | 1.82 | MIPPv2 (+247.9%) |
| K=256.0 | 12.76 | 3.66 | **348.5%** | 6.40 | 1.84 | MIPPv2 (+248.5%) |
| K=512.0 | 12.81 | 3.60 | **355.4%** | 6.43 | 1.81 | MIPPv2 (+255.4%) |
| K=1024.0 | 12.78 | 3.59 | **355.7%** | 6.42 | 1.80 | MIPPv2 (+255.7%) |

### Platform: `a100_rvv` (gcc, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32.0 | 12.73 | 3.45 | **369.1%** | 6.38 | 1.73 | MIPPv2 (+269.1%) |
| K=64.0 | 13.09 | 3.57 | **366.9%** | 6.57 | 1.79 | MIPPv2 (+266.9%) |
| K=128.0 | 13.22 | 3.63 | **364.2%** | 6.63 | 1.82 | MIPPv2 (+264.2%) |
| K=256.0 | 13.31 | 3.66 | **363.8%** | 6.68 | 1.84 | MIPPv2 (+263.8%) |
| K=512.0 | 13.30 | 3.56 | **373.1%** | 6.68 | 1.79 | MIPPv2 (+273.1%) |
| K=1024.0 | 13.36 | 3.58 | **373.3%** | 6.71 | 1.80 | MIPPv2 (+273.3%) |

### Platform: `m1_neon` (clang, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32.0 | 44.98 | 36.37 | **123.7%** | 14.84 | 12.00 | MIPPv2 (+23.7%) |
| K=64.0 | 45.09 | 41.57 | **108.5%** | 14.87 | 13.71 | MIPPv2 (+8.5%) |
| K=128.0 | 46.79 | 44.77 | **104.5%** | 15.44 | 14.77 | MIPPv2 (+4.5%) |
| K=256.0 | 47.63 | 46.56 | **102.3%** | 15.71 | 15.36 | MIPPv2 (+2.3%) |
| K=512.0 | 48.06 | 47.38 | **101.4%** | 15.85 | 15.63 | MIPPv2 (+1.4%) |
| K=1024.0 | 48.27 | 47.93 | **100.7%** | 15.93 | 15.81 | MIPPv2 (+0.7%) |

### Platform: `m1_neon` (gcc, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32.0 | 45.53 | 36.39 | **125.1%** | 15.02 | 12.01 | MIPPv2 (+25.1%) |
| K=64.0 | 46.97 | 41.58 | **113.0%** | 15.49 | 13.72 | MIPPv2 (+13.0%) |
| K=128.0 | 46.96 | 44.78 | **104.9%** | 15.49 | 14.77 | MIPPv2 (+4.9%) |
| K=256.0 | 47.72 | 46.56 | **102.5%** | 15.74 | 15.36 | MIPPv2 (+2.5%) |
| K=512.0 | 48.11 | 47.39 | **101.5%** | 15.87 | 15.63 | MIPPv2 (+1.5%) |
| K=1024.0 | 48.30 | 47.93 | **100.8%** | 15.93 | 15.81 | MIPPv2 (+0.8%) |

### Platform: `meteorlake_avx2` (clang, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32.0 | 68.47 | 65.13 | **105.1%** | 14.89 | 13.93 | MIPPv2 (+5.1%) |
| K=64.0 | 69.80 | 69.08 | **101.0%** | 15.18 | 14.84 | MIPPv2 (+1.0%) |
| K=128.0 | 70.16 | 70.70 | **99.2%** | 15.21 | 15.13 | BLIS (+0.8%) |
| K=256.0 | 71.39 | 71.79 | **99.4%** | 15.45 | 15.36 | BLIS (+0.6%) |
| K=512.0 | 71.16 | 70.84 | **100.4%** | 15.44 | 15.21 | Equivalent |
| K=1024.0 | 71.46 | 71.64 | **99.8%** | 15.53 | 15.35 | Equivalent |

### Platform: `meteorlake_avx2` (gcc, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32.0 | 62.17 | 65.36 | **95.1%** | 13.31 | 13.93 | BLIS (+4.9%) |
| K=64.0 | 64.60 | 69.09 | **93.5%** | 13.87 | 14.77 | BLIS (+6.5%) |
| K=128.0 | 67.33 | 70.78 | **95.1%** | 14.53 | 15.13 | BLIS (+4.9%) |
| K=256.0 | 69.60 | 72.12 | **96.5%** | 15.00 | 15.41 | BLIS (+3.5%) |
| K=512.0 | 60.87 | 70.48 | **86.4%** | 13.02 | 15.12 | BLIS (+13.6%) |
| K=1024.0 | 60.40 | 71.88 | **84.0%** | 12.91 | 15.37 | BLIS (+16.0%) |

### Platform: `rpi5_neon` (gcc, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32.0 | 17.57 | 12.67 | **138.7%** | 7.35 | 5.30 | MIPPv2 (+38.7%) |
| K=64.0 | 18.32 | 15.19 | **120.6%** | 7.66 | 6.35 | MIPPv2 (+20.6%) |
| K=128.0 | 18.72 | 16.92 | **110.6%** | 7.83 | 7.07 | MIPPv2 (+10.6%) |
| K=256.0 | 18.92 | 17.96 | **105.4%** | 7.91 | 7.51 | MIPPv2 (+5.4%) |
| K=512.0 | 18.83 | 18.52 | **101.7%** | 7.88 | 7.74 | MIPPv2 (+1.7%) |
| K=1024.0 | 18.27 | 18.78 | **97.3%** | 7.64 | 7.85 | BLIS (+2.7%) |

### Platform: `x100_rvv` (clang, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32.0 | 14.94 | 16.29 | **91.7%** | 6.24 | 6.80 | BLIS (+8.3%) |
| K=64.0 | 16.84 | 17.21 | **97.9%** | 7.03 | 7.18 | BLIS (+2.1%) |
| K=128.0 | 17.91 | 17.69 | **101.2%** | 7.48 | 7.38 | MIPPv2 (+1.2%) |
| K=256.0 | 18.49 | 17.90 | **103.3%** | 7.72 | 7.47 | MIPPv2 (+3.3%) |
| K=512.0 | 15.53 | 17.91 | **86.7%** | 6.48 | 7.48 | BLIS (+13.3%) |
| K=1024.0 | 15.62 | 18.01 | **86.7%** | 6.52 | 7.52 | BLIS (+13.3%) |

### Platform: `x100_rvv` (gcc, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32.0 | 14.03 | 16.34 | **85.9%** | 5.86 | 6.82 | BLIS (+14.1%) |
| K=64.0 | 15.27 | 17.22 | **88.6%** | 6.37 | 7.19 | BLIS (+11.4%) |
| K=128.0 | 15.89 | 17.69 | **89.8%** | 6.63 | 7.38 | BLIS (+10.2%) |
| K=256.0 | 16.22 | 17.91 | **90.6%** | 6.77 | 7.47 | BLIS (+9.4%) |
| K=512.0 | 13.70 | 17.95 | **76.3%** | 5.72 | 7.49 | BLIS (+23.7%) |
| K=1024.0 | 14.12 | 18.05 | **78.2%** | 5.90 | 7.54 | BLIS (+21.8%) |

### Platform: `x60_rvv` (clang, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32.0 | 6.44 | 9.00 | **71.5%** | 4.03 | 5.63 | BLIS (+28.5%) |
| K=64.0 | 7.06 | 9.83 | **71.9%** | 4.42 | 6.15 | BLIS (+28.1%) |
| K=128.0 | 7.43 | 10.28 | **72.2%** | 4.65 | 6.43 | BLIS (+27.8%) |
| K=256.0 | 6.19 | 6.92 | **89.5%** | 3.87 | 4.33 | BLIS (+10.5%) |
| K=512.0 | 6.46 | 7.15 | **90.4%** | 4.04 | 4.47 | BLIS (+9.6%) |
| K=1024.0 | 6.54 | 7.27 | **90.0%** | 4.10 | 4.55 | BLIS (+10.0%) |

### Platform: `x60_rvv` (gcc, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32.0 | 6.57 | 9.56 | **68.8%** | 4.12 | 5.98 | BLIS (+31.2%) |
| K=64.0 | 7.04 | 10.60 | **66.4%** | 4.41 | 6.63 | BLIS (+33.6%) |
| K=128.0 | 7.30 | 11.18 | **65.3%** | 4.57 | 7.00 | BLIS (+34.7%) |
| K=256.0 | 5.87 | 6.89 | **85.2%** | 3.67 | 4.31 | BLIS (+14.8%) |
| K=512.0 | 6.17 | 7.27 | **85.0%** | 3.86 | 4.55 | BLIS (+15.0%) |
| K=1024.0 | 6.29 | 7.40 | **85.0%** | 3.94 | 4.63 | BLIS (+15.0%) |

### Platform: `zen4_avx512` (clang, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32.0 | 83.66 | 77.06 | **108.6%** | 16.00 | 14.74 | MIPPv2 (+8.6%) |
| K=64.0 | 83.56 | 79.79 | **104.7%** | 15.99 | 15.34 | MIPPv2 (+4.7%) |
| K=128.0 | 83.43 | 81.31 | **102.6%** | 15.97 | 15.62 | MIPPv2 (+2.6%) |
| K=256.0 | 83.46 | 82.32 | **101.4%** | 15.98 | 15.83 | MIPPv2 (+1.4%) |
| K=512.0 | 83.39 | 82.85 | **100.7%** | 15.95 | 15.91 | MIPPv2 (+0.7%) |
| K=1024.0 | 83.38 | 83.02 | **100.4%** | 15.94 | 15.95 | Equivalent |

### Platform: `zen4_avx512` (gcc, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32.0 | 83.79 | 77.59 | **108.0%** | 15.94 | 14.80 | MIPPv2 (+8.0%) |
| K=64.0 | 83.77 | 79.93 | **104.8%** | 15.98 | 15.31 | MIPPv2 (+4.8%) |
| K=128.0 | 83.80 | 81.42 | **102.9%** | 15.97 | 15.62 | MIPPv2 (+2.9%) |
| K=256.0 | 83.77 | 82.41 | **101.6%** | 15.96 | 15.83 | MIPPv2 (+1.6%) |
| K=512.0 | 83.69 | 82.86 | **101.0%** | 15.95 | 15.91 | MIPPv2 (+1.0%) |
| K=1024.0 | 83.70 | 83.18 | **100.6%** | 15.95 | 15.94 | MIPPv2 (+0.6%) |

### Platform: `zen5_avx512` (clang, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32.0 | 79.81 | 77.44 | **103.1%** | 16.00 | 15.51 | MIPPv2 (+3.1%) |
| K=64.0 | 79.80 | 78.58 | **101.6%** | 16.00 | 15.75 | MIPPv2 (+1.6%) |
| K=128.0 | 79.78 | 79.19 | **100.8%** | 16.00 | 15.87 | MIPPv2 (+0.8%) |
| K=256.0 | 79.61 | 79.43 | **100.2%** | 16.00 | 15.93 | Equivalent |
| K=512.0 | 79.69 | 79.61 | **100.1%** | 16.00 | 15.96 | Equivalent |
| K=1024.0 | 79.74 | 79.73 | **100.0%** | 15.99 | 15.98 | Equivalent |

### Platform: `zen5_avx512` (gcc, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32.0 | 80.78 | 77.57 | **104.1%** | 15.99 | 15.51 | MIPPv2 (+4.1%) |
| K=64.0 | 80.18 | 78.62 | **102.0%** | 16.00 | 15.75 | MIPPv2 (+2.0%) |
| K=128.0 | 80.02 | 79.26 | **101.0%** | 15.99 | 15.87 | MIPPv2 (+1.0%) |
| K=256.0 | 79.80 | 79.47 | **100.4%** | 16.00 | 15.93 | Equivalent |
| K=512.0 | 79.87 | 79.65 | **100.3%** | 15.99 | 15.96 | Equivalent |
| K=1024.0 | 79.88 | 79.72 | **100.2%** | 15.99 | 15.98 | Equivalent |
