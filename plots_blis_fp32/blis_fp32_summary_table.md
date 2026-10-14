# GEMMBench — Comparative Analysis: MIPPv2 vs BLIS Reference Micro-Kernels (Single-Precision (FP32 / SGEMM))

Empirical comparison of sustained in-register compute density and throughput between MIPPv2 champions
and hand-tuned BLIS assembly micro-kernels across target microarchitectures in Single-Precision (FP32 / SGEMM).

## 1. Symmetric Microarchitectural Comparison (Direct Micro-Kernel Equivalents)

> [!NOTE]
> **Methodological Scope:**
> - **In-Register Scope:** This benchmark strictly measures in-register compute saturation ($M_R \times N_R \times K$) on hot L1 cache panels, isolating FMA pipeline throughput without end-to-end SGEMM cache blocking (GotoBLAS $J_C, P_C, I_C$), dynamic packing overheads, or fringe tiles.
> - **Architectural Symmetry:** Evaluates targets where BLIS provides a native, ISA-matched microkernel (AVX-512, AVX2, NEON, RVV 256b).
> - **Native Epilogue Layout Alignment:** Each microkernel is evaluated against its optimal hardware storage layout: MIPPv2 writes $C$ in row-major (`CRR`), while BLIS writes $C$ in native column-major (`CRC` on SKX, X60, and ARMv8A) to ensure both frameworks leverage contiguous vector stores and avoid scalar scatter/gather epilogue distortion.
> - **L1d/L2 Working Set Boundary:** While $K \in [32, 128]$ fits entirely within L1d cache, $K \ge 256$ spills the packed panel $B$ ($W_B = K \times N_R \times 4\text{ bytes}$) into L2 cache (e.g. 64–256 KiB on AVX-512 with $N_R=64$), testing sustained L2 streaming throughput rather than pure in-register FMA saturation.

| Microarchitecture | Compiler | Layout | MIPPv2 Champion | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **% of BLIS Peak** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Status |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `m1_neon` | `clang` | `CRR` | `x60_mr6_nr4_fmaddi_crr` | **96.56** | **95.52** | **101.1%** | 31.85 | 31.51 | 🟢 Parity (98–102%) |
| `m1_neon` | `gcc` | `CRR` | `x60_mr6_nr4_fmaddi_crr` | **96.61** | **95.53** | **101.1%** | 31.87 | 31.51 | 🟢 Parity (98–102%) |
| `meteorlake_avx2` | `clang` | `CRR` | `meteorlake_mr4_nr3_crr` | **145.61** | **148.30** | **98.2%** | 31.14 | 31.10 | 🟢 Parity (98–102%) |
| `meteorlake_avx2` | `gcc` | `CRR` | `meteorlake_mr4_nr3_crr` | **141.04** | **147.27** | **95.8%** | 30.15 | 30.83 | 🟢 Near-Parity (≥95%) |
| `rpi5_neon` | `clang` | `CRR` | `a76_mr6_nr4_fmaddi_crr` | **37.93** | **37.42** | **101.4%** | 15.86 | 15.65 | 🟢 Parity (98–102%) |
| `rpi5_neon` | `gcc` | `CRR` | `a76_mr6_nr4_fmaddi_crr` | **37.99** | **37.38** | **101.6%** | 15.89 | 15.63 | 🟢 Parity (98–102%) |
| `x100_rvv` | `clang` | `CRR` | `x100_register_blocked_apack4_crr` | **37.04** | **36.12*** | **102.5%** | 15.45 | 15.07 | 🔵 MIPPv2 (+2.5%) |
| `x100_rvv` | `gcc` | `CRR` | `x100_register_blocked_apack4_crr` | **32.41** | **36.16*** | **89.6%** | 13.54 | 15.09 | 🔴 Delta (<90%) |
| `x60_rvv` | `clang` | `CRR` | `a100_mr7_nr4_pipe_crr` | **14.85** | **21.95** | **67.7%** | 9.30 | 13.74 | 🔴 Delta (<90%) |
| `x60_rvv` | `gcc` | `CRR` | `a100_mr7_nr4_pipe_crr` | **13.13** | **21.67** | **60.6%** | 8.22 | 13.57 | 🔴 Delta (<90%) |
| `zen4_avx512` | `clang` | `CRR` | `firestorm_mr4_nr4_fmaddi_crr` | **170.48** | **169.75** | **100.4%** | 31.99 | 31.89 | 🟢 Parity (98–102%) |
| `zen4_avx512` | `gcc` | `CRR` | `firestorm_mr4_nr4_fmaddi_crr` | **171.09** | **169.69** | **100.8%** | 31.93 | 31.88 | 🟢 Parity (98–102%) |
| `zen5_avx512` | `clang` | `CRR` | `firestorm_mr4_nr4_fmaddi_crr` | **161.04** | **160.61** | **100.3%** | 31.99 | 31.96 | 🟢 Parity (98–102%) |
| `zen5_avx512` | `gcc` | `CRR` | `firestorm_mr4_nr4_fmaddi_crr` | **161.98** | **161.17** | **100.5%** | 31.99 | 31.96 | 🟢 Parity (98–102%) |

*\*Note: On SpacemiT X100 (RVV 256b, OOO dual-issue), BLIS runs the in-order X60 microkernel (`sgemm_x60_2vx14.c`).*

---

## 2. VLEN Agility & Cross-Vector Portability Study (SpacemiT A100 VLEN=1024b)

> [!IMPORTANT]
> **Asymmetry Disclosure & Portability Framing:**
> - Upstream BLIS provides no native 1024-bit vector microkernel; executing BLIS on A100 defaults to the 256-bit in-order X60 assembly kernel (`sgemm_x60_2vx14.c, 16×14`), utilizing only 25% of vector capacity per iteration.
> - MIPPv2's parameterized template vectorization automatically adapts to the 1024-bit VLEN across precision targets.
> - This comparison is presented not as a peer algorithmic match, but as an empirical demonstration of C++ template agility across variable vector lengths versus static assembly fragility.

| Microarchitecture | Compiler | MIPPv2 Champion | MIPPv2 Peak (GFLOP/s) | MIPPv2 (FLOP/cyc) | Hardware Peak (FLOP/cyc) | % Machine Peak | BLIS Fallback (X60 asm GFLOP/s) | Relative Speedup |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `a100_rvv` | `clang` | `a100_mr7_nr2_lmul2_pipe_crr` | **48.78** | 24.49 | 32.0 | **76.5%** | 13.46 (X60) | **3.63×** |
| `a100_rvv` | `gcc` | `a100_mr7_nr2_lmul2_pipe_crr` | **45.48** | 22.79 | 32.0 | **71.2%** | 13.49 (X60) | **3.37×** |

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
| K=32 | 43.02 | 11.77 | **365.4%** | 21.58 | 5.91 | MIPPv2 (+265.4%) |
| K=64 | 46.07 | 12.76 | **360.9%** | 23.11 | 6.40 | MIPPv2 (+260.9%) |
| K=128 | 47.72 | 13.20 | **361.5%** | 23.94 | 6.62 | MIPPv2 (+261.5%) |
| K=256 | 48.25 | 13.41 | **359.7%** | 24.20 | 6.73 | MIPPv2 (+259.7%) |
| K=512 | 48.72 | 13.45 | **362.3%** | 24.45 | 6.75 | MIPPv2 (+262.3%) |
| K=1024 | 48.77 | 13.45 | **362.7%** | 24.48 | 6.75 | MIPPv2 (+262.7%) |

### Platform: `a100_rvv` (gcc, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32 | 41.93 | 11.57 | **362.4%** | 21.02 | 5.80 | MIPPv2 (+262.4%) |
| K=64 | 44.01 | 12.81 | **343.7%** | 22.07 | 6.42 | MIPPv2 (+243.7%) |
| K=128 | 44.65 | 13.22 | **337.6%** | 22.38 | 6.63 | MIPPv2 (+237.6%) |
| K=256 | 45.11 | 13.43 | **335.9%** | 22.62 | 6.74 | MIPPv2 (+235.9%) |
| K=512 | 45.38 | 13.48 | **336.6%** | 22.73 | 6.77 | MIPPv2 (+236.6%) |
| K=1024 | 45.48 | 12.49 | **364.2%** | 22.79 | 6.28 | MIPPv2 (+264.2%) |

### Platform: `m1_neon` (clang, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32 | 89.97 | 66.47 | **135.3%** | 29.68 | 21.93 | MIPPv2 (+35.3%) |
| K=64 | 93.35 | 78.91 | **118.3%** | 30.80 | 26.03 | MIPPv2 (+18.3%) |
| K=128 | 93.58 | 87.03 | **107.5%** | 30.87 | 28.71 | MIPPv2 (+7.5%) |
| K=256 | 95.26 | 91.76 | **103.8%** | 31.42 | 30.27 | MIPPv2 (+3.8%) |
| K=512 | 96.12 | 94.06 | **102.2%** | 31.71 | 31.03 | MIPPv2 (+2.2%) |
| K=1024 | 96.55 | 95.50 | **101.1%** | 31.85 | 31.51 | MIPPv2 (+1.1%) |

### Platform: `m1_neon` (gcc, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32 | 91.07 | 66.31 | **137.3%** | 30.04 | 21.88 | MIPPv2 (+37.3%) |
| K=64 | 93.94 | 78.79 | **119.2%** | 30.99 | 25.99 | MIPPv2 (+19.2%) |
| K=128 | 93.95 | 86.95 | **108.0%** | 30.99 | 28.68 | MIPPv2 (+8.0%) |
| K=256 | 95.44 | 91.70 | **104.1%** | 31.49 | 30.25 | MIPPv2 (+4.1%) |
| K=512 | 96.21 | 94.08 | **102.3%** | 31.74 | 31.04 | MIPPv2 (+2.3%) |
| K=1024 | 96.60 | 95.51 | **101.1%** | 31.87 | 31.51 | MIPPv2 (+1.1%) |

### Platform: `meteorlake_avx2` (clang, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32 | 137.59 | 130.06 | **105.8%** | 29.52 | 27.56 | MIPPv2 (+5.8%) |
| K=64 | 141.83 | 138.05 | **102.7%** | 30.48 | 29.39 | MIPPv2 (+2.7%) |
| K=128 | 143.37 | 141.76 | **101.1%** | 30.65 | 30.10 | MIPPv2 (+1.1%) |
| K=256 | 144.79 | 144.31 | **100.3%** | 30.97 | 30.68 | Equivalent |
| K=512 | 143.15 | 145.82 | **98.2%** | 30.87 | 30.86 | BLIS (+1.8%) |
| K=1024 | 144.78 | 144.59 | **100.1%** | 31.09 | 30.69 | Equivalent |

### Platform: `meteorlake_avx2` (gcc, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32 | 125.19 | 128.88 | **97.1%** | 26.73 | 27.28 | BLIS (+2.9%) |
| K=64 | 131.56 | 135.05 | **97.4%** | 28.11 | 28.68 | BLIS (+2.6%) |
| K=128 | 136.31 | 140.56 | **97.0%** | 29.20 | 29.78 | BLIS (+3.0%) |
| K=256 | 139.84 | 143.14 | **97.7%** | 29.96 | 30.33 | BLIS (+2.3%) |
| K=512 | 121.74 | 146.06 | **83.4%** | 25.97 | 30.75 | BLIS (+16.6%) |
| K=1024 | 122.07 | 144.53 | **84.5%** | 26.05 | 30.71 | BLIS (+15.5%) |

### Platform: `rpi5_neon` (clang, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32 | 35.04 | 22.98 | **152.4%** | 14.65 | 9.61 | MIPPv2 (+52.4%) |
| K=64 | 36.52 | 28.78 | **126.9%** | 15.27 | 12.04 | MIPPv2 (+26.9%) |
| K=128 | 37.31 | 32.80 | **113.7%** | 15.60 | 13.72 | MIPPv2 (+13.7%) |
| K=256 | 37.78 | 35.34 | **106.9%** | 15.80 | 14.78 | MIPPv2 (+6.9%) |
| K=512 | 37.93 | 36.75 | **103.2%** | 15.86 | 15.37 | MIPPv2 (+3.2%) |
| K=1024 | 37.28 | 37.42 | **99.6%** | 15.59 | 15.65 | Equivalent |

### Platform: `rpi5_neon` (gcc, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32 | 35.13 | 22.88 | **153.5%** | 14.69 | 9.57 | MIPPv2 (+53.5%) |
| K=64 | 36.63 | 28.69 | **127.7%** | 15.32 | 12.00 | MIPPv2 (+27.7%) |
| K=128 | 37.42 | 32.74 | **114.3%** | 15.65 | 13.69 | MIPPv2 (+14.3%) |
| K=256 | 37.84 | 35.27 | **107.3%** | 15.82 | 14.75 | MIPPv2 (+7.3%) |
| K=512 | 37.99 | 36.71 | **103.5%** | 15.89 | 15.35 | MIPPv2 (+3.5%) |
| K=1024 | 36.90 | 37.38 | **98.7%** | 15.43 | 15.63 | BLIS (+1.3%) |

### Platform: `x100_rvv` (clang, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32 | 29.84 | 32.67 | **91.4%** | 12.45 | 13.63 | BLIS (+8.6%) |
| K=64 | 33.44 | 34.40 | **97.2%** | 13.95 | 14.35 | BLIS (+2.8%) |
| K=128 | 35.71 | 35.37 | **101.0%** | 14.90 | 14.76 | MIPPv2 (+1.0%) |
| K=256 | 37.02 | 35.77 | **103.5%** | 15.45 | 14.93 | MIPPv2 (+3.5%) |
| K=512 | 33.86 | 36.02 | **94.0%** | 14.13 | 15.04 | BLIS (+6.0%) |
| K=1024 | 31.46 | 36.11 | **87.1%** | 13.13 | 15.07 | BLIS (+12.9%) |

### Platform: `x100_rvv` (gcc, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32 | 28.20 | 32.73 | **86.1%** | 11.77 | 13.65 | BLIS (+13.9%) |
| K=64 | 30.38 | 34.42 | **88.3%** | 12.69 | 14.36 | BLIS (+11.7%) |
| K=128 | 31.71 | 35.41 | **89.6%** | 13.24 | 14.77 | BLIS (+10.4%) |
| K=256 | 32.41 | 35.85 | **90.4%** | 13.53 | 14.96 | BLIS (+9.6%) |
| K=512 | 29.03 | 36.05 | **80.5%** | 12.12 | 15.04 | BLIS (+19.5%) |
| K=1024 | 28.86 | 36.16 | **79.8%** | 12.04 | 15.09 | BLIS (+20.2%) |

### Platform: `x60_rvv` (clang, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32 | 12.88 | 18.30 | **70.4%** | 8.06 | 11.45 | BLIS (+29.6%) |
| K=64 | 14.12 | 20.23 | **69.8%** | 8.84 | 12.67 | BLIS (+30.2%) |
| K=128 | 14.85 | 21.36 | **69.5%** | 9.29 | 13.37 | BLIS (+30.5%) |
| K=256 | 11.93 | 21.94 | **54.4%** | 7.47 | 13.74 | BLIS (+45.6%) |
| K=512 | 13.42 | 17.14 | **78.3%** | 8.40 | 10.73 | BLIS (+21.7%) |
| K=1024 | 13.52 | 17.64 | **76.6%** | 8.46 | 11.04 | BLIS (+23.4%) |

### Platform: `x60_rvv` (gcc, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32 | 11.98 | 18.31 | **65.4%** | 7.50 | 11.46 | BLIS (+34.6%) |
| K=64 | 12.72 | 20.10 | **63.3%** | 7.96 | 12.58 | BLIS (+36.7%) |
| K=128 | 13.12 | 21.13 | **62.1%** | 8.22 | 13.23 | BLIS (+37.9%) |
| K=256 | 11.28 | 21.65 | **52.1%** | 7.06 | 13.55 | BLIS (+47.9%) |
| K=512 | 12.40 | 17.23 | **71.9%** | 7.76 | 10.79 | BLIS (+28.1%) |
| K=1024 | 12.57 | 17.66 | **71.2%** | 7.87 | 11.05 | BLIS (+28.8%) |

### Platform: `zen4_avx512` (clang, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32 | 170.38 | 155.45 | **109.6%** | 31.99 | 29.07 | MIPPv2 (+9.6%) |
| K=64 | 170.41 | 163.13 | **104.5%** | 31.99 | 30.63 | MIPPv2 (+4.5%) |
| K=128 | 169.78 | 166.63 | **101.9%** | 31.91 | 31.31 | MIPPv2 (+1.9%) |
| K=256 | 169.39 | 168.13 | **100.8%** | 31.88 | 31.63 | MIPPv2 (+0.8%) |
| K=512 | 169.35 | 169.30 | **100.0%** | 31.82 | 31.81 | Equivalent |
| K=1024 | 169.93 | 169.73 | **100.1%** | 31.93 | 31.89 | Equivalent |

### Platform: `zen4_avx512` (gcc, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32 | 170.74 | 154.89 | **110.2%** | 31.81 | 28.96 | MIPPv2 (+10.2%) |
| K=64 | 170.57 | 162.39 | **105.0%** | 31.90 | 30.50 | MIPPv2 (+5.0%) |
| K=128 | 169.49 | 165.95 | **102.1%** | 31.76 | 31.21 | MIPPv2 (+2.1%) |
| K=256 | 169.93 | 167.87 | **101.2%** | 31.90 | 31.60 | MIPPv2 (+1.2%) |
| K=512 | 169.85 | 168.68 | **100.7%** | 31.87 | 31.78 | MIPPv2 (+0.7%) |
| K=1024 | 170.13 | 169.29 | **100.5%** | 31.93 | 31.88 | Equivalent |

### Platform: `zen5_avx512` (clang, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32 | 161.00 | 155.89 | **103.3%** | 31.99 | 31.02 | MIPPv2 (+3.3%) |
| K=64 | 160.81 | 158.20 | **101.6%** | 31.99 | 31.50 | MIPPv2 (+1.6%) |
| K=128 | 160.74 | 159.42 | **100.8%** | 31.99 | 31.74 | MIPPv2 (+0.8%) |
| K=256 | 160.39 | 160.00 | **100.2%** | 31.99 | 31.87 | Equivalent |
| K=512 | 160.40 | 160.31 | **100.1%** | 31.99 | 31.93 | Equivalent |
| K=1024 | 160.49 | 160.47 | **100.0%** | 31.99 | 31.96 | Equivalent |

### Platform: `zen5_avx512` (gcc, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32 | 161.87 | 156.54 | **103.4%** | 31.99 | 31.02 | MIPPv2 (+3.4%) |
| K=64 | 161.72 | 158.82 | **101.8%** | 31.99 | 31.50 | MIPPv2 (+1.8%) |
| K=128 | 161.67 | 160.02 | **101.0%** | 31.99 | 31.74 | MIPPv2 (+1.0%) |
| K=256 | 161.13 | 160.60 | **100.3%** | 31.99 | 31.87 | Equivalent |
| K=512 | 161.17 | 161.00 | **100.1%** | 31.99 | 31.93 | Equivalent |
| K=1024 | 161.17 | 161.05 | **100.1%** | 31.99 | 31.96 | Equivalent |


---

## 5. Reference Assembly Kernels Provenance & Acknowledgements

The hand-tuned assembly microkernels used as baseline references in this benchmark originate from:

- **RISC-V Vector 1.0 (SpacemiT X60):** Authored and tuned by **Stepan Nassyr** via genetic loop exploration (`asmgen-gemm`), available in the [`stepannassyr/blis`](https://github.com/stepannassyr/blis/tree/x60/kernels) repository (`x60` branch, commit [`6ffc1d3`](https://github.com/stepannassyr/blis/commit/6ffc1d35fcafe74788b2aeaf00d5792d2618e0b9)).
- **x86-64 (AVX2, AVX-512) & AArch64 (NEON):** Sourced from the official upstream **FLAME BLIS** project ([`flame/blis`](https://github.com/flame/blis)), maintained by The University of Texas at Austin and contributors (commit [`a8037cf`](https://github.com/flame/blis/commit/a8037cfc4330030c726efaec1ffeb1f93d468aa3), based on BLIS 2.1).
