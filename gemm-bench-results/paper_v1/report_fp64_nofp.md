# GEMMBench — Comparative Analysis: MIPPv2 vs BLIS Reference Micro-Kernels (Double-Precision (FP64 / DGEMM))

**Ablation Variant:** `nofp` | **Analysis Window:** $K \in [128, 384]$ (inclusive)

Empirical comparison of sustained in-register compute density and throughput between MIPPv2 champions
and hand-tuned BLIS assembly micro-kernels across target microarchitectures in Double-Precision (FP64 / DGEMM).

## 1. Summary: Matched-$K$ Window Ratio ($K \in [128, 384]$, Variant: `nofp`)

> [!NOTE]
> **Statistical Methodology & Reporting Rules:**
> - **Pre-registered Champions:** MIPPv2 champions are locked prior to measurement in `uarch_config.json`; no post-hoc argmax across kernels.
> - **Matched-K Ratios:** Ratios are strictly evaluated at identical panel depth $K$ ($R_K = \text{FLOP/cyc}_{\text{MIPP}, K} / \text{FLOP/cyc}_{\text{BLIS}, K}$).
> - **Geometric Mean over Cache-Resident Window:** Summary ratio is $\text{geomean}(R_K)$ for $K \in [128, 384]$.
> - **Separate Compiler Rows:** GCC and Clang are reported on distinct rows. Post-hoc best-compiler envelope is provided in the Appendix.

| Microarchitecture | Compiler | Variant | Precision | MIPPv2 Champion | $N_K$ | **Ratio Geomean** | [Min, Max] | Parity Classification |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| `a100_rvv` | `clang` | `nofp` | `fp64` | `a100_mr7_nr2_lmul2_pipe_crr` | 3 | **349.8%** | [348.0%, 351.2%] | MIPPv2 Faster |
| `a100_rvv` | `gcc` | `nofp` | `fp64` | `a100_mr7_nr2_lmul2_pipe_crr` | 3 | **366.1%** | [362.7%, 368.7%] | MIPPv2 Faster |
| `m1_neon` | `clang` | `nofp` | `fp64` | `x60_mr6_nr4_fmaddi_crr` | 3 | **101.5%** | [100.8%, 102.4%] | Mixed Advantage |
| `m1_neon` | `gcc` | `nofp` | `fp64` | `x60_mr6_nr4_fmaddi_crr` | 3 | **101.6%** | [100.9%, 102.6%] | Mixed Advantage |
| `meteorlake_avx2` | `clang` | `nofp` | `fp64` | `meteorlake_mr4_nr3_crr` | 3 | **99.5%** | [98.0%, 100.5%] | Parity (±2% CI) |
| `meteorlake_avx2` | `gcc` | `nofp` | `fp64` | `meteorlake_mr4_nr3_crr` | 3 | **91.1%** | [81.0%, 96.7%] | BLIS Faster |
| `rpi5_neon` | `clang` | `nofp` | `fp64` | `a76_mr6_nr4_fmaddi_crr` | 3 | **101.5%** | [100.6%, 102.6%] | Mixed Advantage |
| `rpi5_neon` | `gcc` | `nofp` | `fp64` | `a76_mr6_nr4_fmaddi_crr` | 3 | **102.1%** | [101.0%, 103.5%] | Mixed Advantage |
| `x100_rvv` | `clang` | `nofp` | `fp64` | `x100_register_blocked_apack4_crr` | 3 | **101.5%** | [100.1%, 102.5%] | Mixed Advantage |
| `x100_rvv` | `gcc` | `nofp` | `fp64` | `x100_register_blocked_apack4_crr` | 3 | **88.8%** | [87.3%, 90.2%] | BLIS Faster |
| `x60_rvv` | `clang` | `nofp` | `fp64` | `a100_mr7_nr4_pipe_crr` | 3 | **84.5%** | [72.5%, 91.3%] | BLIS Faster |
| `x60_rvv` | `gcc` | `nofp` | `fp64` | `a100_mr7_nr4_pipe_crr` | 3 | **79.6%** | [71.1%, 85.6%] | BLIS Faster |
| `zen4_avx512` | `clang` | `nofp` | `fp64` | `firestorm_mr4_nr4_fmaddi_crr` | 3 | **100.7%** | [100.1%, 101.7%] | Parity (±2% CI) |
| `zen4_avx512` | `gcc` | `nofp` | `fp64` | `firestorm_mr4_nr4_fmaddi_crr` | 3 | **100.6%** | [100.2%, 101.1%] | Parity (±2% CI) |
| `zen5_avx512` | `clang` | `nofp` | `fp64` | `firestorm_mr4_nr4_fmaddi_crr` | 3 | **100.5%** | [100.3%, 100.8%] | Parity (±2% CI) |
| `zen5_avx512` | `gcc` | `nofp` | `fp64` | `firestorm_mr4_nr4_fmaddi_crr` | 3 | **100.5%** | [100.3%, 100.8%] | Parity (±2% CI) |

---

## 2. Visual Comparative Figures

- **Matched-K Ratio Comparison:** [nofp/plots_fp64/blis_vs_mippv2_ratio_comparison.svg](nofp/plots_fp64/blis_vs_mippv2_ratio_comparison.svg)
- **Scaling Grid Across $K \in [32, 1024]$:** [nofp/plots_fp64/blis_vs_mippv2_k_scaling_grid.svg](nofp/plots_fp64/blis_vs_mippv2_k_scaling_grid.svg)


---

## 3. Per-Size Matched-$K$ Scaling & Working-Set Analysis (Variant: `nofp`)

> [!TIP]
> **Cache Boundary Semantics:** Rows marked with `†` denote operating points where panel working set $K \times (M + N) \times \text{elem_bytes}$ exceeds L1d cache capacity, testing streaming memory throughput rather than pure in-register compute density.

### Platform: `a100_rvv` (clang, fp64, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 6.20 | 1.79 | **347.3%** | [346.2%, 347.4%] | 35.5 | 11.0 | MIPPv2 (+247.3%) |
| $K=128$ † | 6.33 | 1.82 | **348.0%** | [347.8%, 348.1%] | 71.0 | 22.0 | MIPPv2 (+248.0%) |
| $K=256$ † | 6.39 | 1.82 | **350.4%** | [348.9%, 354.1%] | 142.0 | 44.0 | MIPPv2 (+250.4%) |
| $K=384$ † | 6.41 | 1.83 | **351.2%** | [348.9%, 358.4%] | 213.0 | 66.0 | MIPPv2 (+251.2%) |

### Platform: `a100_rvv` (gcc, fp64, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 6.53 | 1.79 | **364.7%** | [358.4%, 366.4%] | 35.5 | 11.0 | MIPPv2 (+264.7%) |
| $K=128$ † | 6.61 | 1.82 | **362.7%** | [362.2%, 363.2%] | 71.0 | 22.0 | MIPPv2 (+262.7%) |
| $K=256$ † | 6.67 | 1.81 | **368.7%** | [364.7%, 376.1%] | 142.0 | 44.0 | MIPPv2 (+268.7%) |
| $K=384$ † | 6.69 | 1.82 | **366.8%** | [366.3%, 376.5%] | 213.0 | 66.0 | MIPPv2 (+266.8%) |

### Platform: `m1_neon` (clang, fp64, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 15.36 | 14.23 | **108.0%** | [108.0%, 108.0%] | 7.0 | 7.0 | MIPPv2 (+8.0%) |
| $K=128$ | 15.42 | 15.06 | **102.4%** | [102.4%, 102.4%] | 14.0 | 14.0 | MIPPv2 (+2.4%) |
| $K=256$ | 15.70 | 15.52 | **101.2%** | [101.2%, 101.2%] | 28.0 | 28.0 | Parity (±2%) |
| $K=384$ | 15.80 | 15.67 | **100.8%** | [100.8%, 100.8%] | 42.0 | 42.0 | Parity (±2%) |

### Platform: `m1_neon` (gcc, fp64, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 15.47 | 14.26 | **108.5%** | [108.5%, 108.5%] | 7.0 | 7.0 | MIPPv2 (+8.5%) |
| $K=128$ | 15.48 | 15.08 | **102.6%** | [102.6%, 102.6%] | 14.0 | 14.0 | MIPPv2 (+2.6%) |
| $K=256$ | 15.73 | 15.53 | **101.3%** | [101.3%, 101.3%] | 28.0 | 28.0 | Parity (±2%) |
| $K=384$ | 15.82 | 15.68 | **100.9%** | [100.9%, 100.9%] | 42.0 | 42.0 | Parity (±2%) |

### Platform: `meteorlake_avx2` (clang, fp64, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 15.20 | 14.93 | **101.8%** | [101.7%, 101.8%] | 8.0 | 7.0 | Parity (±2%) |
| $K=128$ | 15.33 | 15.25 | **100.5%** | [100.5%, 100.7%] | 16.0 | 14.0 | Parity (±2%) |
| $K=256$ | 15.53 | 15.52 | **100.1%** | [100.1%, 100.2%] | 32.0 | 28.0 | Parity (±2%) |
| $K=384$ | 15.34 | 15.65 | **98.0%** | [98.0%, 98.1%] | 48.0 | 42.0 | Parity (±2%) |

### Platform: `meteorlake_avx2` (gcc, fp64, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 14.09 | 14.86 | **94.8%** | [94.8%, 95.4%] | 8.0 | 7.0 | BLIS (+5.2%) |
| $K=128$ | 14.65 | 15.19 | **96.5%** | [96.4%, 96.5%] | 16.0 | 14.0 | BLIS (+3.5%) |
| $K=256$ | 15.00 | 15.52 | **96.7%** | [96.5%, 96.7%] | 32.0 | 28.0 | BLIS (+3.3%) |
| $K=384$ | 12.67 | 15.64 | **81.0%** | [81.0%, 81.4%] | 48.0 | 42.0 | BLIS (+19.0%) |

### Platform: `rpi5_neon` (clang, fp64, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 7.59 | 7.20 | **105.5%** | [105.2%, 105.5%] | 7.0 | 7.0 | MIPPv2 (+5.5%) |
| $K=128$ | 7.78 | 7.58 | **102.6%** | [102.6%, 102.7%] | 14.0 | 14.0 | MIPPv2 (+2.6%) |
| $K=256$ | 7.89 | 7.78 | **101.3%** | [101.3%, 101.4%] | 28.0 | 28.0 | Parity (±2%) |
| $K=384$ | 7.90 | 7.86 | **100.6%** | [100.6%, 100.6%] | 42.0 | 42.0 | Parity (±2%) |

### Platform: `rpi5_neon` (gcc, fp64, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 7.65 | 7.17 | **106.8%** | [106.5%, 106.9%] | 7.0 | 7.0 | MIPPv2 (+6.8%) |
| $K=128$ | 7.82 | 7.56 | **103.5%** | [103.3%, 103.5%] | 14.0 | 14.0 | MIPPv2 (+3.5%) |
| $K=256$ | 7.91 | 7.77 | **101.8%** | [101.7%, 102.0%] | 28.0 | 28.0 | Parity (±2%) |
| $K=384$ | 7.92 | 7.84 | **101.0%** | [101.0%, 101.1%] | 42.0 | 42.0 | Parity (±2%) |

### Platform: `x100_rvv` (clang, fp64, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 6.84 | 7.16 | **95.4%** | [95.4%, 96.2%] | 9.5 | 11.0 | BLIS (+4.6%) |
| $K=128$ | 7.38 | 7.37 | **100.1%** | [99.9%, 100.1%] | 19.0 | 22.0 | Parity (±2%) |
| $K=256$ | 7.65 | 7.47 | **102.5%** | [102.3%, 102.8%] | 38.0 | 44.0 | MIPPv2 (+2.5%) |
| $K=384$ † | 7.51 | 7.36 | **101.9%** | [101.3%, 103.8%] | 57.0 | 66.0 | Parity (±2%) |

### Platform: `x100_rvv` (gcc, fp64, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 6.22 | 7.16 | **87.0%** | [86.7%, 87.2%] | 9.5 | 11.0 | BLIS (+13.0%) |
| $K=128$ | 6.56 | 7.37 | **89.0%** | [89.0%, 89.1%] | 19.0 | 22.0 | BLIS (+11.0%) |
| $K=256$ | 6.74 | 7.47 | **90.2%** | [90.2%, 90.2%] | 38.0 | 44.0 | BLIS (+9.8%) |
| $K=384$ † | 6.48 | 7.42 | **87.3%** | [86.7%, 89.3%] | 57.0 | 66.0 | BLIS (+12.7%) |

### Platform: `x60_rvv` (clang, fp64, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 4.40 | 6.12 | **72.0%** | [72.0%, 72.1%] | 11.5 | 11.0 | BLIS (+28.0%) |
| $K=128$ | 4.64 | 6.40 | **72.5%** | [72.3%, 72.5%] | 23.0 | 22.0 | BLIS (+27.5%) |
| $K=256$ † | 3.85 | 4.22 | **91.1%** | [88.3%, 91.4%] | 46.0 | 44.0 | BLIS (+8.9%) |
| $K=384$ † | 3.99 | 4.37 | **91.3%** | [90.7%, 91.5%] | 69.0 | 66.0 | BLIS (+8.7%) |

### Platform: `x60_rvv` (gcc, fp64, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 4.39 | 6.09 | **72.0%** | [72.0%, 72.1%] | 11.5 | 11.0 | BLIS (+28.0%) |
| $K=128$ | 4.56 | 6.40 | **71.1%** | [71.1%, 71.2%] | 23.0 | 22.0 | BLIS (+28.9%) |
| $K=256$ † | 3.63 | 4.39 | **82.7%** | [82.0%, 84.0%] | 46.0 | 44.0 | BLIS (+17.3%) |
| $K=384$ † | 3.80 | 4.44 | **85.6%** | [85.1%, 86.2%] | 69.0 | 66.0 | BLIS (+14.4%) |

### Platform: `zen4_avx512` (clang, fp64, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 15.94 | 15.22 | **104.7%** | [104.7%, 104.7%] | 18.0 | 15.0 | MIPPv2 (+4.7%) |
| $K=128$ † | 15.86 | 15.60 | **101.7%** | [101.4%, 101.7%] | 36.0 | 30.0 | Parity (±2%) |
| $K=256$ † | 15.91 | 15.84 | **100.4%** | [100.4%, 100.6%] | 72.0 | 60.0 | Parity (±2%) |
| $K=384$ † | 15.91 | 15.89 | **100.1%** | [99.8%, 100.2%] | 108.0 | 90.0 | Parity (±2%) |

### Platform: `zen4_avx512` (gcc, fp64, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 15.99 | 15.26 | **104.8%** | [104.8%, 104.8%] | 18.0 | 15.0 | MIPPv2 (+4.8%) |
| $K=128$ † | 15.83 | 15.66 | **101.1%** | [101.0%, 101.5%] | 36.0 | 30.0 | Parity (±2%) |
| $K=256$ † | 15.91 | 15.84 | **100.4%** | [100.4%, 100.6%] | 72.0 | 60.0 | Parity (±2%) |
| $K=384$ † | 15.92 | 15.89 | **100.2%** | [100.2%, 100.2%] | 108.0 | 90.0 | Parity (±2%) |

### Platform: `zen5_avx512` (clang, fp64, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 15.99 | 15.73 | **101.7%** | [101.6%, 101.7%] | 18.0 | 15.0 | Parity (±2%) |
| $K=128$ | 15.99 | 15.86 | **100.8%** | [100.8%, 100.8%] | 36.0 | 30.0 | Parity (±2%) |
| $K=256$ † | 16.00 | 15.93 | **100.4%** | [100.4%, 100.4%] | 72.0 | 60.0 | Parity (±2%) |
| $K=384$ † | 16.00 | 15.95 | **100.3%** | [100.3%, 100.3%] | 108.0 | 90.0 | Parity (±2%) |

### Platform: `zen5_avx512` (gcc, fp64, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 16.00 | 15.73 | **101.7%** | [101.6%, 101.7%] | 18.0 | 15.0 | Parity (±2%) |
| $K=128$ | 16.00 | 15.86 | **100.8%** | [100.8%, 100.8%] | 36.0 | 30.0 | Parity (±2%) |
| $K=256$ † | 15.99 | 15.93 | **100.4%** | [100.4%, 100.4%] | 72.0 | 60.0 | Parity (±2%) |
| $K=384$ † | 16.00 | 15.95 | **100.3%** | [100.3%, 100.3%] | 108.0 | 90.0 | Parity (±2%) |


---

## Appendix: Best-Compiler Envelope (Post-Hoc)

> [!IMPORTANT]
> This appendix selects the post-hoc envelope across compilers (`max(gcc, clang)` at each measurement point). It is presented exclusively for secondary reference.

| Microarchitecture | Precision | Variant | Best Compiler by Ratio | Envelope Ratio Geomean | Envelope [Min, Max] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `a100_rvv` | `fp64` | `nofp` | `gcc` | **366.1%** | [362.7%, 368.7%] |
| `m1_neon` | `fp64` | `nofp` | `gcc` | **101.6%** | [100.9%, 102.6%] |
| `meteorlake_avx2` | `fp64` | `nofp` | `clang` | **99.5%** | [98.0%, 100.5%] |
| `rpi5_neon` | `fp64` | `nofp` | `gcc` | **102.1%** | [101.0%, 103.5%] |
| `x100_rvv` | `fp64` | `nofp` | `clang` | **101.5%** | [100.1%, 102.5%] |
| `x60_rvv` | `fp64` | `nofp` | `clang` | **84.5%** | [72.5%, 91.3%] |
| `zen4_avx512` | `fp64` | `nofp` | `clang` | **100.7%** | [100.1%, 101.7%] |
| `zen5_avx512` | `fp64` | `nofp` | `clang` | **100.5%** | [100.3%, 100.8%] |