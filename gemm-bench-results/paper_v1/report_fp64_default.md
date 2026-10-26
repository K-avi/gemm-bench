# GEMMBench — Comparative Analysis: MIPPv2 vs BLIS Reference Micro-Kernels (Double-Precision (FP64 / DGEMM))

**Ablation Variant:** `default` | **Analysis Window:** $K \in [128, 384]$ (inclusive)

Empirical comparison of sustained in-register compute density and throughput between MIPPv2 champions
and hand-tuned BLIS assembly micro-kernels across target microarchitectures in Double-Precision (FP64 / DGEMM).

## 1. Summary: Matched-$K$ Window Ratio ($K \in [128, 384]$, Variant: `default`)

> [!NOTE]
> **Statistical Methodology & Reporting Rules:**
> - **Pre-registered Champions:** MIPPv2 champions are locked prior to measurement in `uarch_config.json`; no post-hoc argmax across kernels.
> - **Matched-K Ratios:** Ratios are strictly evaluated at identical panel depth $K$ ($R_K = \text{FLOP/cyc}_{\text{MIPP}, K} / \text{FLOP/cyc}_{\text{BLIS}, K}$).
> - **Geometric Mean over Cache-Resident Window:** Summary ratio is $\text{geomean}(R_K)$ for $K \in [128, 384]$.
> - **Separate Compiler Rows:** GCC and Clang are reported on distinct rows. Post-hoc best-compiler envelope is provided in the Appendix.

| Microarchitecture | Compiler | Variant | Precision | MIPPv2 Champion | $N_K$ | **Ratio Geomean** | [Min, Max] | Parity Classification |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| `a100_rvv` | `clang` | `default` | `fp64` | `a100_mr7_nr2_lmul2_pipe_crr` | 3 | **352.6%** | [347.8%, 359.0%] | MIPPv2 Faster |
| `a100_rvv` | `gcc` | `default` | `fp64` | `a100_mr7_nr2_lmul2_pipe_crr` | 3 | **367.3%** | [362.7%, 374.3%] | MIPPv2 Faster |
| `m1_neon` | `clang` | `default` | `fp64` | `x60_mr6_nr4_fmaddi_crr` | 3 | **101.4%** | [100.8%, 102.3%] | Mixed Advantage |
| `m1_neon` | `gcc` | `default` | `fp64` | `x60_mr6_nr4_fmaddi_crr` | 3 | **101.7%** | [100.9%, 102.7%] | Mixed Advantage |
| `meteorlake_avx2` | `clang` | `default` | `fp64` | `meteorlake_mr4_nr3_crr` | 3 | **99.5%** | [97.9%, 100.6%] | Parity (±2% CI) |
| `meteorlake_avx2` | `gcc` | `default` | `fp64` | `meteorlake_mr4_nr3_crr` | 3 | **91.2%** | [81.1%, 96.7%] | BLIS Faster |
| `rpi5_neon` | `clang` | `default` | `fp64` | `a76_mr6_nr4_fmaddi_crr` | 3 | **101.7%** | [100.6%, 103.0%] | Mixed Advantage |
| `rpi5_neon` | `gcc` | `default` | `fp64` | `a76_mr6_nr4_fmaddi_crr` | 3 | **101.9%** | [100.8%, 103.3%] | Mixed Advantage |
| `x100_rvv` | `clang` | `default` | `fp64` | `x100_register_blocked_apack4_crr` | 3 | **103.0%** | [101.0%, 104.9%] | Mixed Advantage |
| `x100_rvv` | `gcc` | `default` | `fp64` | `x100_register_blocked_apack4_crr` | 3 | **90.5%** | [89.4%, 91.5%] | BLIS Faster |
| `x60_rvv` | `clang` | `default` | `fp64` | `a100_mr7_nr4_pipe_crr` | 3 | **83.7%** | [72.3%, 90.3%] | BLIS Faster |
| `x60_rvv` | `gcc` | `default` | `fp64` | `a100_mr7_nr4_pipe_crr` | 3 | **78.3%** | [68.3%, 84.4%] | BLIS Faster |
| `zen4_avx512` | `clang` | `default` | `fp64` | `firestorm_mr4_nr4_fmaddi_crr` | 3 | **100.8%** | [100.3%, 101.6%] | Parity (±2% CI) |
| `zen4_avx512` | `gcc` | `default` | `fp64` | `firestorm_mr4_nr4_fmaddi_crr` | 3 | **100.8%** | [100.2%, 101.6%] | Parity (±2% CI) |
| `zen5_avx512` | `clang` | `default` | `fp64` | `firestorm_mr4_nr4_fmaddi_crr` | 3 | **100.5%** | [100.3%, 100.8%] | Parity (±2% CI) |
| `zen5_avx512` | `gcc` | `default` | `fp64` | `firestorm_mr4_nr4_fmaddi_crr` | 3 | **100.5%** | [100.3%, 100.7%] | Parity (±2% CI) |

---

## 2. Visual Comparative Figures

- **Matched-K Ratio Comparison:** [default/plots_fp64/blis_vs_mippv2_ratio_comparison.svg](default/plots_fp64/blis_vs_mippv2_ratio_comparison.svg)
- **Scaling Grid Across $K \in [32, 1024]$:** [default/plots_fp64/blis_vs_mippv2_k_scaling_grid.svg](default/plots_fp64/blis_vs_mippv2_k_scaling_grid.svg)


---

## 3. Per-Size Matched-$K$ Scaling & Working-Set Analysis (Variant: `default`)

> [!TIP]
> **Cache Boundary Semantics:** Rows marked with `†` denote operating points where panel working set $K \times (M + N) \times \text{elem_bytes}$ exceeds L1d cache capacity, testing streaming memory throughput rather than pure in-register compute density.

### Platform: `a100_rvv` (clang, fp64, default)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=32$ | 5.99 | 1.72 | **348.7%** | [347.6%, 349.2%] | 17.8 | 5.5 | MIPPv2 (+248.7%) |
| $K=64$ | 6.19 | 1.79 | **346.2%** | [346.0%, 346.4%] | 35.5 | 11.0 | MIPPv2 (+246.2%) |
| $K=128$ † | 6.33 | 1.82 | **347.8%** | [347.5%, 348.1%] | 71.0 | 22.0 | MIPPv2 (+247.8%) |
| $K=256$ † | 6.40 | 1.82 | **351.0%** | [350.3%, 351.9%] | 142.0 | 44.0 | MIPPv2 (+251.0%) |
| $K=384$ † | 6.42 | 1.79 | **359.0%** | [354.0%, 360.6%] | 213.0 | 66.0 | MIPPv2 (+259.0%) |
| $K=512$ † | 6.40 | 1.80 | **356.1%** | [354.0%, 363.3%] | 284.0 | 88.0 | MIPPv2 (+256.1%) |
| $K=1024$ † | 6.40 | 1.80 | **355.9%** | [350.0%, 361.5%] | 568.0 | 176.0 | MIPPv2 (+255.9%) |

### Platform: `a100_rvv` (gcc, fp64, default)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=32$ | 6.35 | 1.72 | **368.2%** | [368.0%, 372.0%] | 17.8 | 5.5 | MIPPv2 (+268.2%) |
| $K=64$ | 6.42 | 1.79 | **358.8%** | [357.6%, 366.0%] | 35.5 | 11.0 | MIPPv2 (+258.8%) |
| $K=128$ † | 6.60 | 1.82 | **362.7%** | [362.5%, 363.2%] | 71.0 | 22.0 | MIPPv2 (+262.7%) |
| $K=256$ † | 6.68 | 1.83 | **365.0%** | [364.1%, 365.1%] | 142.0 | 44.0 | MIPPv2 (+265.0%) |
| $K=384$ † | 6.69 | 1.79 | **374.3%** | [371.5%, 375.1%] | 213.0 | 66.0 | MIPPv2 (+274.3%) |
| $K=512$ † | 6.67 | 1.76 | **379.1%** | [374.4%, 381.3%] | 284.0 | 88.0 | MIPPv2 (+279.1%) |
| $K=1024$ † | 6.69 | 1.80 | **371.9%** | [371.3%, 373.4%] | 568.0 | 176.0 | MIPPv2 (+271.9%) |

### Platform: `m1_neon` (clang, fp64, default)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=32$ | 14.80 | 12.85 | **115.1%** | [115.1%, 115.1%] | 3.5 | 3.5 | MIPPv2 (+15.1%) |
| $K=64$ | 15.37 | 14.25 | **107.9%** | [107.9%, 107.9%] | 7.0 | 7.0 | MIPPv2 (+7.9%) |
| $K=128$ | 15.42 | 15.07 | **102.3%** | [102.3%, 102.3%] | 14.0 | 14.0 | MIPPv2 (+2.3%) |
| $K=256$ | 15.70 | 15.52 | **101.2%** | [101.2%, 101.2%] | 28.0 | 28.0 | Parity (±2%) |
| $K=384$ | 15.80 | 15.68 | **100.8%** | [100.8%, 100.8%] | 42.0 | 42.0 | Parity (±2%) |
| $K=512$ | 15.85 | 15.71 | **100.9%** | [100.9%, 100.9%] | 56.0 | 56.0 | Parity (±2%) |
| $K=1024$ | 15.92 | 15.85 | **100.4%** | [100.3%, 100.6%] | 112.0 | 112.0 | Parity (±2%) |

### Platform: `m1_neon` (gcc, fp64, default)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=32$ | 15.02 | 12.87 | **116.7%** | [116.7%, 116.7%] | 3.5 | 3.5 | MIPPv2 (+16.7%) |
| $K=64$ | 15.49 | 14.26 | **108.6%** | [108.6%, 108.6%] | 7.0 | 7.0 | MIPPv2 (+8.6%) |
| $K=128$ | 15.49 | 15.08 | **102.7%** | [102.7%, 102.7%] | 14.0 | 14.0 | MIPPv2 (+2.7%) |
| $K=256$ | 15.74 | 15.53 | **101.4%** | [101.4%, 101.4%] | 28.0 | 28.0 | Parity (±2%) |
| $K=384$ | 15.83 | 15.68 | **100.9%** | [100.9%, 100.9%] | 42.0 | 42.0 | Parity (±2%) |
| $K=512$ | 15.87 | 15.71 | **101.0%** | [101.0%, 101.0%] | 56.0 | 56.0 | Parity (±2%) |
| $K=1024$ | 15.93 | 15.86 | **100.5%** | [100.5%, 100.5%] | 112.0 | 112.0 | Parity (±2%) |

### Platform: `meteorlake_avx2` (clang, fp64, default)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=32$ | 14.87 | 13.99 | **106.3%** | [105.7%, 106.3%] | 4.0 | 3.5 | MIPPv2 (+6.3%) |
| $K=64$ | 15.29 | 14.90 | **102.6%** | [102.5%, 102.6%] | 8.0 | 7.0 | MIPPv2 (+2.6%) |
| $K=128$ | 15.35 | 15.27 | **100.6%** | [100.5%, 100.7%] | 16.0 | 14.0 | Parity (±2%) |
| $K=256$ | 15.55 | 15.54 | **100.1%** | [100.0%, 100.1%] | 32.0 | 28.0 | Parity (±2%) |
| $K=384$ | 15.34 | 15.66 | **97.9%** | [97.9%, 98.1%] | 48.0 | 42.0 | BLIS (+2.1%) |
| $K=512$ † | 15.38 | 15.16 | **101.5%** | [101.3%, 101.8%] | 64.0 | 56.0 | Parity (±2%) |
| $K=1024$ † | 15.52 | 15.38 | **100.9%** | [100.8%, 101.0%] | 128.0 | 112.0 | Parity (±2%) |

### Platform: `meteorlake_avx2` (gcc, fp64, default)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=32$ | 13.53 | 13.96 | **96.9%** | [96.5%, 97.0%] | 4.0 | 3.5 | BLIS (+3.1%) |
| $K=64$ | 14.16 | 14.83 | **95.5%** | [95.4%, 96.1%] | 8.0 | 7.0 | BLIS (+4.5%) |
| $K=128$ | 14.70 | 15.22 | **96.5%** | [96.5%, 96.7%] | 16.0 | 14.0 | BLIS (+3.5%) |
| $K=256$ | 15.01 | 15.52 | **96.7%** | [96.7%, 96.9%] | 32.0 | 28.0 | BLIS (+3.3%) |
| $K=384$ | 12.69 | 15.65 | **81.1%** | [81.0%, 81.2%] | 48.0 | 42.0 | BLIS (+18.9%) |
| $K=512$ † | 12.79 | 15.02 | **85.1%** | [84.7%, 85.4%] | 64.0 | 56.0 | BLIS (+14.9%) |
| $K=1024$ † | 13.00 | 15.29 | **85.0%** | [85.1%, 85.4%] | 128.0 | 112.0 | BLIS (+15.0%) |

### Platform: `rpi5_neon` (clang, fp64, default)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=32$ | 7.29 | 6.55 | **111.3%** | [111.1%, 111.7%] | 3.5 | 3.5 | MIPPv2 (+11.3%) |
| $K=64$ | 7.61 | 7.19 | **105.9%** | [105.4%, 105.9%] | 7.0 | 7.0 | MIPPv2 (+5.9%) |
| $K=128$ | 7.79 | 7.56 | **103.0%** | [102.9%, 103.0%] | 14.0 | 14.0 | MIPPv2 (+3.0%) |
| $K=256$ | 7.89 | 7.78 | **101.5%** | [101.4%, 101.5%] | 28.0 | 28.0 | Parity (±2%) |
| $K=384$ | 7.90 | 7.85 | **100.6%** | [100.6%, 100.7%] | 42.0 | 42.0 | Parity (±2%) |
| $K=512$ | 7.86 | 7.88 | **99.8%** | [99.3%, 100.8%] | 56.0 | 56.0 | Parity (±2%) |
| $K=1024$ † | 7.71 | 7.91 | **97.5%** | [97.1%, 97.6%] | 112.0 | 112.0 | BLIS (+2.5%) |

### Platform: `rpi5_neon` (gcc, fp64, default)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=32$ | 7.33 | 6.55 | **111.9%** | [111.7%, 112.2%] | 3.5 | 3.5 | MIPPv2 (+11.9%) |
| $K=64$ | 7.66 | 7.20 | **106.3%** | [106.2%, 106.4%] | 7.0 | 7.0 | MIPPv2 (+6.3%) |
| $K=128$ | 7.82 | 7.57 | **103.3%** | [103.3%, 103.6%] | 14.0 | 14.0 | MIPPv2 (+3.3%) |
| $K=256$ | 7.91 | 7.78 | **101.6%** | [101.6%, 101.7%] | 28.0 | 28.0 | Parity (±2%) |
| $K=384$ | 7.92 | 7.86 | **100.8%** | [100.8%, 100.9%] | 42.0 | 42.0 | Parity (±2%) |
| $K=512$ | 7.92 | 7.86 | **100.7%** | [99.8%, 101.4%] | 56.0 | 56.0 | Parity (±2%) |
| $K=1024$ † | 7.55 | 7.90 | **95.5%** | [95.4%, 95.6%] | 112.0 | 112.0 | BLIS (+4.5%) |

### Platform: `x100_rvv` (clang, fp64, default)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=32$ | 6.18 | 6.80 | **90.9%** | [90.5%, 91.1%] | 4.8 | 5.5 | BLIS (+9.1%) |
| $K=64$ | 6.97 | 7.17 | **97.2%** | [97.1%, 97.3%] | 9.5 | 11.0 | BLIS (+2.8%) |
| $K=128$ | 7.45 | 7.38 | **101.0%** | [100.5%, 101.1%] | 19.0 | 22.0 | Parity (±2%) |
| $K=256$ | 7.71 | 7.47 | **103.3%** | [102.7%, 103.4%] | 38.0 | 44.0 | MIPPv2 (+3.3%) |
| $K=384$ † | 7.75 | 7.39 | **104.9%** | [104.1%, 105.7%] | 57.0 | 66.0 | MIPPv2 (+4.9%) |
| $K=512$ † | 6.42 | 7.45 | **86.2%** | [86.0%, 86.9%] | 76.0 | 88.0 | BLIS (+13.8%) |
| $K=1024$ † | 6.50 | 7.50 | **86.6%** | [86.3%, 86.9%] | 152.0 | 176.0 | BLIS (+13.4%) |

### Platform: `x100_rvv` (gcc, fp64, default)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=32$ | 5.81 | 6.79 | **85.5%** | [85.2%, 85.7%] | 4.8 | 5.5 | BLIS (+14.5%) |
| $K=64$ | 6.31 | 7.17 | **87.9%** | [87.9%, 88.0%] | 9.5 | 11.0 | BLIS (+12.1%) |
| $K=128$ | 6.60 | 7.38 | **89.4%** | [89.4%, 89.5%] | 19.0 | 22.0 | BLIS (+10.6%) |
| $K=256$ | 6.76 | 7.47 | **90.5%** | [90.5%, 90.5%] | 38.0 | 44.0 | BLIS (+9.5%) |
| $K=384$ † | 6.79 | 7.42 | **91.5%** | [88.9%, 91.6%] | 57.0 | 66.0 | BLIS (+8.5%) |
| $K=512$ † | 5.63 | 7.44 | **75.7%** | [75.3%, 76.8%] | 76.0 | 88.0 | BLIS (+24.3%) |
| $K=1024$ † | 5.81 | 7.52 | **77.3%** | [76.0%, 78.4%] | 152.0 | 176.0 | BLIS (+22.7%) |

### Platform: `x60_rvv` (clang, fp64, default)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=32$ | 4.04 | 5.62 | **71.8%** | [71.8%, 71.9%] | 5.8 | 5.5 | BLIS (+28.2%) |
| $K=64$ | 4.42 | 6.13 | **72.2%** | [72.1%, 72.2%] | 11.5 | 11.0 | BLIS (+27.8%) |
| $K=128$ | 4.65 | 6.43 | **72.3%** | [72.3%, 72.3%] | 23.0 | 22.0 | BLIS (+27.7%) |
| $K=256$ † | 3.80 | 4.24 | **89.7%** | [89.3%, 92.6%] | 46.0 | 44.0 | BLIS (+10.3%) |
| $K=384$ † | 3.99 | 4.41 | **90.3%** | [89.9%, 90.4%] | 69.0 | 66.0 | BLIS (+9.7%) |
| $K=512$ † | 4.03 | 4.46 | **90.4%** | [90.1%, 91.4%] | 92.0 | 88.0 | BLIS (+9.6%) |
| $K=1024$ † | 4.08 | 4.54 | **89.8%** | [89.6%, 90.2%] | 184.0 | 176.0 | BLIS (+10.2%) |

### Platform: `x60_rvv` (gcc, fp64, default)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=32$ | 4.12 | 5.76 | **71.5%** | [71.5%, 71.5%] | 5.8 | 5.5 | BLIS (+28.5%) |
| $K=64$ | 4.41 | 6.35 | **69.4%** | [69.4%, 69.4%] | 11.5 | 11.0 | BLIS (+30.6%) |
| $K=128$ | 4.57 | 6.69 | **68.3%** | [68.3%, 68.3%] | 23.0 | 22.0 | BLIS (+31.7%) |
| $K=256$ † | 3.68 | 4.41 | **83.4%** | [82.7%, 84.0%] | 46.0 | 44.0 | BLIS (+16.6%) |
| $K=384$ † | 3.80 | 4.50 | **84.4%** | [84.4%, 85.6%] | 69.0 | 66.0 | BLIS (+15.6%) |
| $K=512$ † | 3.85 | 4.58 | **84.2%** | [83.3%, 84.5%] | 92.0 | 88.0 | BLIS (+15.8%) |
| $K=1024$ † | 3.94 | 4.64 | **84.9%** | [84.5%, 85.0%] | 184.0 | 176.0 | BLIS (+15.1%) |

### Platform: `zen4_avx512` (clang, fp64, default)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=32$ | 15.98 | 14.75 | **108.4%** | [108.4%, 108.5%] | 9.0 | 7.5 | MIPPv2 (+8.4%) |
| $K=64$ | 15.99 | 15.27 | **104.7%** | [104.7%, 104.7%] | 18.0 | 15.0 | MIPPv2 (+4.7%) |
| $K=128$ † | 15.89 | 15.64 | **101.6%** | [101.0%, 101.6%] | 36.0 | 30.0 | Parity (±2%) |
| $K=256$ † | 15.92 | 15.83 | **100.6%** | [100.2%, 100.7%] | 72.0 | 60.0 | Parity (±2%) |
| $K=384$ † | 15.93 | 15.89 | **100.3%** | [100.2%, 100.3%] | 108.0 | 90.0 | Parity (±2%) |
| $K=512$ † | 15.92 | 15.92 | **100.0%** | [100.0%, 100.1%] | 144.0 | 120.0 | Parity (±2%) |
| $K=1024$ † | 15.94 | 15.95 | **99.9%** | [99.9%, 100.0%] | 288.0 | 240.0 | Parity (±2%) |

### Platform: `zen4_avx512` (gcc, fp64, default)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=32$ | 15.89 | 14.76 | **107.6%** | [107.6%, 108.0%] | 9.0 | 7.5 | MIPPv2 (+7.6%) |
| $K=64$ | 15.95 | 15.23 | **104.7%** | [104.7%, 104.7%] | 18.0 | 15.0 | MIPPv2 (+4.7%) |
| $K=128$ † | 15.87 | 15.62 | **101.6%** | [100.8%, 101.6%] | 36.0 | 30.0 | Parity (±2%) |
| $K=256$ † | 15.92 | 15.81 | **100.7%** | [100.7%, 100.8%] | 72.0 | 60.0 | Parity (±2%) |
| $K=384$ † | 15.91 | 15.87 | **100.2%** | [100.1%, 100.3%] | 108.0 | 90.0 | Parity (±2%) |
| $K=512$ † | 15.92 | 15.90 | **100.1%** | [99.3%, 100.2%] | 144.0 | 120.0 | Parity (±2%) |
| $K=1024$ † | 15.95 | 15.94 | **100.1%** | [99.9%, 100.1%] | 288.0 | 240.0 | Parity (±2%) |

### Platform: `zen5_avx512` (clang, fp64, default)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=32$ | 15.98 | 15.51 | **103.0%** | [103.0%, 103.1%] | 9.0 | 7.5 | MIPPv2 (+3.0%) |
| $K=64$ | 16.00 | 15.75 | **101.6%** | [101.5%, 101.6%] | 18.0 | 15.0 | Parity (±2%) |
| $K=128$ | 16.00 | 15.87 | **100.8%** | [100.8%, 100.8%] | 36.0 | 30.0 | Parity (±2%) |
| $K=256$ † | 16.00 | 15.93 | **100.4%** | [100.4%, 100.4%] | 72.0 | 60.0 | Parity (±2%) |
| $K=384$ † | 16.00 | 15.95 | **100.3%** | [100.3%, 100.3%] | 108.0 | 90.0 | Parity (±2%) |
| $K=512$ † | 16.00 | 15.96 | **100.2%** | [100.2%, 100.2%] | 144.0 | 120.0 | Parity (±2%) |
| $K=1024$ † | 16.00 | 15.98 | **100.1%** | [100.1%, 100.1%] | 288.0 | 240.0 | Parity (±2%) |

### Platform: `zen5_avx512` (gcc, fp64, default)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=32$ | 15.99 | 15.51 | **103.1%** | [102.5%, 103.1%] | 9.0 | 7.5 | MIPPv2 (+3.1%) |
| $K=64$ | 16.00 | 15.75 | **101.6%** | [101.5%, 101.6%] | 18.0 | 15.0 | Parity (±2%) |
| $K=128$ | 15.99 | 15.87 | **100.7%** | [100.7%, 100.8%] | 36.0 | 30.0 | Parity (±2%) |
| $K=256$ † | 16.00 | 15.93 | **100.4%** | [100.4%, 100.4%] | 72.0 | 60.0 | Parity (±2%) |
| $K=384$ † | 16.00 | 15.95 | **100.3%** | [100.3%, 100.3%] | 108.0 | 90.0 | Parity (±2%) |
| $K=512$ † | 16.00 | 15.96 | **100.2%** | [100.2%, 100.2%] | 144.0 | 120.0 | Parity (±2%) |
| $K=1024$ † | 16.00 | 15.98 | **100.1%** | [100.1%, 100.1%] | 288.0 | 240.0 | Parity (±2%) |


---

## Appendix: Best-Compiler Envelope (Post-Hoc)

> [!IMPORTANT]
> This appendix selects the post-hoc envelope across compilers (`max(gcc, clang)` at each measurement point). It is presented exclusively for secondary reference.

| Microarchitecture | Precision | Variant | Best Compiler by Ratio | Envelope Ratio Geomean | Envelope [Min, Max] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `a100_rvv` | `fp64` | `default` | `gcc` | **367.3%** | [362.7%, 374.3%] |
| `m1_neon` | `fp64` | `default` | `gcc` | **101.7%** | [100.9%, 102.7%] |
| `meteorlake_avx2` | `fp64` | `default` | `clang` | **99.5%** | [97.9%, 100.6%] |
| `rpi5_neon` | `fp64` | `default` | `gcc` | **101.9%** | [100.8%, 103.3%] |
| `x100_rvv` | `fp64` | `default` | `clang` | **103.0%** | [101.0%, 104.9%] |
| `x60_rvv` | `fp64` | `default` | `clang` | **83.7%** | [72.3%, 90.3%] |
| `zen4_avx512` | `fp64` | `default` | `gcc` | **100.8%** | [100.2%, 101.6%] |
| `zen5_avx512` | `fp64` | `default` | `clang` | **100.5%** | [100.3%, 100.8%] |