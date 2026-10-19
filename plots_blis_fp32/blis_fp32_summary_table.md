# GEMMBench — Comparative Analysis: MIPPv2 vs BLIS Reference Micro-Kernels (Single-Precision (FP32 / SGEMM))

Empirical comparison of sustained in-register compute density and throughput between MIPPv2 champions
and hand-tuned BLIS assembly micro-kernels across target microarchitectures in Single-Precision (FP32 / SGEMM).

## 1. Symmetric Microarchitectural Comparison (Direct Micro-Kernel Equivalents)

> [!NOTE]
> **Methodological Scope:**
> - **In-Register Scope:** This benchmark strictly measures in-register compute saturation ($M_R \times N_R \times K$) on hot L1 cache panels, isolating FMA pipeline throughput without end-to-end SGEMM cache blocking (GotoBLAS $J_C, P_C, I_C$), dynamic packing overheads, or fringe tiles.
> - **Architectural Symmetry:** Evaluates targets where BLIS provides a native, ISA-matched microkernel (AVX-512, AVX2, NEON, RVV 256b).
> - **Native Epilogue Layout Alignment:** Each microkernel is evaluated against its optimal hardware storage layout: MIPPv2 writes $C$ in row-major (`CRR`), while BLIS writes $C$ in native column-major (`CRC` on SKX, X60, and ARMv8A) to ensure both frameworks leverage contiguous vector stores and avoid scalar scatter/gather epilogue distortion.
> - **Small-$K$ Epilogue & Prologue Specialization ($K=32, 64$):** By default, MIPPv2 microkernels use compile-time `FastPath` specialization ($\alpha=1, \beta=0$) that emits direct vector stores without $\alpha$ scaling or $\beta$ branch checks, whereas BLIS assembly microkernels execute runtime $\alpha/\beta$ epilogue scaling and unconditional prologue $C$-tile prefetches (e.g., 24 `prfm pldl1keep` instructions in `armv8a`). At $K=32$, this fixed prologue/epilogue instruction overhead accounts for the 25–35% throughput gap on ARMv8-A and AVX-512 (use `--no-fastpath` to evaluate MIPPv2 with runtime $\alpha/\beta$ epilogue).
> - **L1d/L2 Working Set Boundary:** While $K \in [32, 128]$ fits entirely within L1d cache, $K \ge 256$ spills the packed panel $B$ ($W_B = K \times N_R \times 4\text{ bytes}$) into L2 cache (e.g. 64–256 KiB on AVX-512 with $N_R=64$), testing sustained L2 streaming throughput rather than pure in-register FMA saturation.

| Microarchitecture | Compiler | Layout | MIPPv2 Champion | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **% of BLIS Peak** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Status |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `m1_neon` | `clang` | `CRR` | `x60_mr6_nr4_fmaddi_crr` | **96.55** | **96.11** | **100.5%** | 31.85 | 31.71 | 🟢 Parity (98–102%) |
| `m1_neon` | `gcc` | `CRR` | `x60_mr6_nr4_fmaddi_crr` | **96.60** | **96.12** | **100.5%** | 31.87 | 31.71 | 🟢 Parity (98–102%) |
| `meteorlake_avx2` | `clang` | `CRR` | `meteorlake_mr4_nr3_crr` | **158.32** | **158.41** | **99.9%** | 31.14 | 31.15 | 🟢 Parity (98–102%) |
| `meteorlake_avx2` | `gcc` | `CRR` | `meteorlake_mr4_nr3_crr` | **153.76** | **157.40** | **97.7%** | 30.25 | 30.95 | 🟢 Near-Parity (≥95%) |
| `rpi5_neon` | `clang` | `CRR` | `a76_mr6_nr4_fmaddi_crr` | **37.93** | **37.71** | **100.6%** | 15.86 | 15.77 | 🟢 Parity (98–102%) |
| `rpi5_neon` | `gcc` | `CRR` | `a76_mr6_nr4_fmaddi_crr` | **37.99** | **37.71** | **100.7%** | 15.89 | 15.77 | 🟢 Parity (98–102%) |
| `x100_rvv` | `clang` | `CRR` | `x100_register_blocked_apack4_crr` | **37.02** | **36.11*** | **102.5%** | 15.45 | 15.07 | 🔵 MIPPv2 (+2.5%) |
| `x100_rvv` | `gcc` | `CRR` | `x100_register_blocked_apack4_crr` | **32.44** | **36.12*** | **89.8%** | 13.54 | 15.08 | 🔴 Delta (<90%) |
| `x60_rvv` | `clang` | `CRR` | `a100_mr7_nr4_pipe_crr` | **14.85** | **21.94** | **67.7%** | 9.29 | 13.74 | 🔴 Delta (<90%) |
| `x60_rvv` | `gcc` | `CRR` | `a100_mr7_nr4_pipe_crr` | **13.12** | **21.65** | **60.6%** | 8.22 | 13.55 | 🔴 Delta (<90%) |
| `zen4_avx512` | `clang` | `CRR` | `firestorm_mr4_nr4_fmaddi_crr` | **173.28** | **171.62** | **101.0%** | 31.99 | 31.89 | 🟢 Parity (98–102%) |
| `zen4_avx512` | `gcc` | `CRR` | `firestorm_mr4_nr4_fmaddi_crr` | **173.15** | **172.09** | **100.6%** | 31.93 | 31.88 | 🟢 Parity (98–102%) |
| `zen5_avx512` | `clang` | `CRR` | `firestorm_mr4_nr4_fmaddi_crr` | **163.12** | **162.87** | **100.2%** | 31.99 | 31.96 | 🟢 Parity (98–102%) |
| `zen5_avx512` | `gcc` | `CRR` | `firestorm_mr4_nr4_fmaddi_crr` | **163.84** | **162.91** | **100.6%** | 31.99 | 31.96 | 🟢 Parity (98–102%) |

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
| `a100_rvv` | `clang` | `a100_mr7_nr2_lmul2_pipe_crr` | **48.77** | 24.48 | 32.0 | **76.5%** | 13.45 (X60) | **3.63×** |
| `a100_rvv` | `gcc` | `a100_mr7_nr2_lmul2_pipe_crr` | **45.48** | 22.79 | 32.0 | **71.2%** | 13.48 (X60) | **3.37×** |

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
| K=32 | 89.97 | 77.68 | **115.8%** | 29.68 | 25.62 | MIPPv2 (+15.8%) |
| K=64 | 93.35 | 86.32 | **108.2%** | 30.80 | 28.48 | MIPPv2 (+8.2%) |
| K=128 | 93.58 | 91.32 | **102.5%** | 30.87 | 30.13 | MIPPv2 (+2.5%) |
| K=256 | 95.25 | 94.08 | **101.3%** | 31.42 | 31.03 | MIPPv2 (+1.3%) |
| K=512 | 96.12 | 95.27 | **100.9%** | 31.71 | 31.43 | MIPPv2 (+0.9%) |
| K=1024 | 96.55 | 96.11 | **100.5%** | 31.85 | 31.71 | Equivalent |

### Platform: `m1_neon` (gcc, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32 | 91.06 | 77.32 | **117.8%** | 30.04 | 25.51 | MIPPv2 (+17.8%) |
| K=64 | 93.94 | 86.04 | **109.2%** | 30.99 | 28.38 | MIPPv2 (+9.2%) |
| K=128 | 93.95 | 91.19 | **103.0%** | 30.99 | 30.08 | MIPPv2 (+3.0%) |
| K=256 | 95.44 | 94.00 | **101.5%** | 31.49 | 31.01 | MIPPv2 (+1.5%) |
| K=512 | 96.21 | 95.28 | **101.0%** | 31.74 | 31.43 | MIPPv2 (+1.0%) |
| K=1024 | 96.60 | 96.12 | **100.5%** | 31.87 | 31.71 | MIPPv2 (+0.5%) |

### Platform: `meteorlake_avx2` (clang, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32 | 151.45 | 141.54 | **107.0%** | 29.79 | 27.84 | MIPPv2 (+7.0%) |
| K=64 | 154.96 | 150.55 | **102.9%** | 30.47 | 29.60 | MIPPv2 (+2.9%) |
| K=128 | 156.46 | 154.32 | **101.4%** | 30.77 | 30.34 | MIPPv2 (+1.4%) |
| K=256 | 158.28 | 157.14 | **100.7%** | 31.12 | 30.90 | MIPPv2 (+0.7%) |
| K=512 | 156.96 | 158.41 | **99.1%** | 30.89 | 31.15 | BLIS (+0.9%) |
| K=1024 | 158.32 | 156.76 | **101.0%** | 31.14 | 30.83 | MIPPv2 (+1.0%) |

### Platform: `meteorlake_avx2` (gcc, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32 | 137.27 | 140.32 | **97.8%** | 26.99 | 27.59 | BLIS (+2.2%) |
| K=64 | 143.91 | 147.65 | **97.5%** | 28.30 | 29.03 | BLIS (+2.5%) |
| K=128 | 149.82 | 152.63 | **98.2%** | 29.47 | 30.01 | BLIS (+1.8%) |
| K=256 | 153.76 | 155.81 | **98.7%** | 30.25 | 30.63 | BLIS (+1.3%) |
| K=512 | 132.73 | 157.40 | **84.3%** | 26.13 | 30.95 | BLIS (+15.7%) |
| K=1024 | 134.66 | 157.21 | **85.7%** | 26.48 | 30.91 | BLIS (+14.3%) |

### Platform: `rpi5_neon` (clang, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32 | 35.06 | 27.41 | **127.9%** | 14.66 | 11.46 | MIPPv2 (+27.9%) |
| K=64 | 36.50 | 31.86 | **114.6%** | 15.26 | 13.32 | MIPPv2 (+14.6%) |
| K=128 | 37.31 | 34.74 | **107.4%** | 15.60 | 14.53 | MIPPv2 (+7.4%) |
| K=256 | 37.78 | 36.42 | **103.7%** | 15.80 | 15.23 | MIPPv2 (+3.7%) |
| K=512 | 37.93 | 37.31 | **101.6%** | 15.86 | 15.60 | MIPPv2 (+1.6%) |
| K=1024 | 36.99 | 37.71 | **98.1%** | 15.47 | 15.77 | BLIS (+1.9%) |

### Platform: `rpi5_neon` (gcc, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32 | 35.13 | 27.34 | **128.5%** | 14.69 | 11.43 | MIPPv2 (+28.5%) |
| K=64 | 36.63 | 31.86 | **114.9%** | 15.32 | 13.32 | MIPPv2 (+14.9%) |
| K=128 | 37.43 | 34.77 | **107.6%** | 15.65 | 14.54 | MIPPv2 (+7.6%) |
| K=256 | 37.84 | 36.42 | **103.9%** | 15.82 | 15.23 | MIPPv2 (+3.9%) |
| K=512 | 37.99 | 37.32 | **101.8%** | 15.89 | 15.61 | MIPPv2 (+1.8%) |
| K=1024 | 36.40 | 37.71 | **96.5%** | 15.23 | 15.77 | BLIS (+3.5%) |

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
| K=32 | 28.27 | 32.66 | **86.6%** | 11.80 | 13.63 | BLIS (+13.4%) |
| K=64 | 30.44 | 34.41 | **88.4%** | 12.70 | 14.36 | BLIS (+11.6%) |
| K=128 | 31.70 | 35.41 | **89.5%** | 13.23 | 14.78 | BLIS (+10.5%) |
| K=256 | 32.44 | 35.83 | **90.6%** | 13.54 | 14.95 | BLIS (+9.4%) |
| K=512 | 29.28 | 35.99 | **81.4%** | 12.23 | 15.03 | BLIS (+18.6%) |
| K=1024 | 28.81 | 36.12 | **79.7%** | 12.03 | 15.08 | BLIS (+20.3%) |

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
| K=32 | 173.28 | 157.58 | **110.0%** | 31.99 | 29.07 | MIPPv2 (+10.0%) |
| K=64 | 173.07 | 165.29 | **104.7%** | 31.99 | 30.62 | MIPPv2 (+4.7%) |
| K=128 | 170.49 | 168.48 | **101.2%** | 31.59 | 31.30 | MIPPv2 (+1.2%) |
| K=256 | 171.94 | 170.29 | **101.0%** | 31.88 | 31.63 | MIPPv2 (+1.0%) |
| K=512 | 172.03 | 171.33 | **100.4%** | 31.87 | 31.81 | Equivalent |
| K=1024 | 172.45 | 171.62 | **100.5%** | 31.95 | 31.89 | Equivalent |

### Platform: `zen4_avx512` (gcc, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32 | 172.96 | 157.07 | **110.1%** | 31.89 | 28.97 | MIPPv2 (+10.1%) |
| K=64 | 172.95 | 165.61 | **104.4%** | 31.89 | 30.54 | MIPPv2 (+4.4%) |
| K=128 | 172.21 | 168.71 | **102.1%** | 31.76 | 31.21 | MIPPv2 (+2.1%) |
| K=256 | 173.09 | 170.64 | **101.4%** | 31.92 | 31.61 | MIPPv2 (+1.4%) |
| K=512 | 172.97 | 171.54 | **100.8%** | 31.90 | 31.78 | MIPPv2 (+0.8%) |
| K=1024 | 173.15 | 172.09 | **100.6%** | 31.93 | 31.88 | MIPPv2 (+0.6%) |

### Platform: `zen5_avx512` (clang, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32 | 163.12 | 158.28 | **103.1%** | 31.99 | 31.02 | MIPPv2 (+3.1%) |
| K=64 | 162.84 | 160.46 | **101.5%** | 31.99 | 31.50 | MIPPv2 (+1.5%) |
| K=128 | 162.99 | 161.82 | **100.7%** | 31.99 | 31.74 | MIPPv2 (+0.7%) |
| K=256 | 162.58 | 162.47 | **100.1%** | 31.99 | 31.87 | Equivalent |
| K=512 | 162.72 | 162.85 | **99.9%** | 31.99 | 31.93 | Equivalent |
| K=1024 | 162.72 | 162.87 | **99.9%** | 31.99 | 31.96 | Equivalent |

### Platform: `zen5_avx512` (gcc, CRR)

| K Dimension | MIPPv2 (GFLOP/s) | BLIS (GFLOP/s) | **MIPPv2 / BLIS (%)** | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| K=32 | 163.83 | 158.50 | **103.4%** | 31.99 | 31.02 | MIPPv2 (+3.4%) |
| K=64 | 163.84 | 160.82 | **101.9%** | 31.99 | 31.50 | MIPPv2 (+1.9%) |
| K=128 | 163.73 | 161.98 | **101.1%** | 31.99 | 31.74 | MIPPv2 (+1.1%) |
| K=256 | 163.32 | 162.45 | **100.5%** | 31.99 | 31.87 | MIPPv2 (+0.5%) |
| K=512 | 163.31 | 162.75 | **100.3%** | 31.99 | 31.93 | Equivalent |
| K=1024 | 163.29 | 162.91 | **100.2%** | 31.99 | 31.96 | Equivalent |


---

## 5. Reference Assembly Kernels Provenance & Acknowledgements

The hand-tuned assembly microkernels used as baseline references in this benchmark originate from:

- **RISC-V Vector 1.0 (SpacemiT X60):** Authored and tuned by **Stepan Nassyr** via genetic loop exploration (`asmgen-gemm`), available in the [`stepannassyr/blis`](https://github.com/stepannassyr/blis/tree/x60/kernels) repository (`x60` branch, commit [`6ffc1d3`](https://github.com/stepannassyr/blis/commit/6ffc1d35fcafe74788b2aeaf00d5792d2618e0b9)).
- **x86-64 (AVX2, AVX-512) & AArch64 (NEON):** Sourced from the official upstream **FLAME BLIS** project ([`flame/blis`](https://github.com/flame/blis)), maintained by The University of Texas at Austin and contributors (commit [`a8037cf`](https://github.com/flame/blis/commit/a8037cfc4330030c726efaec1ffeb1f93d468aa3), based on BLIS 2.1).
