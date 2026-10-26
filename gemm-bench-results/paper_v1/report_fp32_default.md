# GEMMBench — Comparative Analysis: MIPPv2 vs BLIS Reference Micro-Kernels (Single-Precision (FP32 / SGEMM))

**Ablation Variant:** `default` | **Analysis Window:** $K \in [128, 384]$ (inclusive)

Empirical comparison of sustained in-register compute density and throughput between MIPPv2 champions
and hand-tuned BLIS assembly micro-kernels across target microarchitectures in Single-Precision (FP32 / SGEMM).

## 1. Summary: Matched-$K$ Window Ratio ($K \in [128, 384]$, Variant: `default`)

> [!NOTE]
> **Statistical Methodology & Reporting Rules:**
> - **Pre-registered Champions:** MIPPv2 champions are locked prior to measurement in `uarch_config.json`; no post-hoc argmax across kernels.
> - **Matched-K Ratios:** Ratios are strictly evaluated at identical panel depth $K$ ($R_K = \text{FLOP/cyc}_{\text{MIPP}, K} / \text{FLOP/cyc}_{\text{BLIS}, K}$).
> - **Geometric Mean over Cache-Resident Window:** Summary ratio is $\text{geomean}(R_K)$ for $K \in [128, 384]$.
> - **Separate Compiler Rows:** GCC and Clang are reported on distinct rows. Post-hoc best-compiler envelope is provided in the Appendix.

| Microarchitecture | Compiler | Variant | Precision | MIPPv2 Champion | $N_K$ | **Ratio Geomean** | [Min, Max] | Parity Classification |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| `a100_rvv` | `clang` | `default` | `fp32` | `a100_mr7_nr2_lmul2_pipe_crr` | 3 | **359.6%** | [358.8%, 360.3%] | MIPPv2 Faster |
| `a100_rvv` | `gcc` | `default` | `fp32` | `a100_mr7_nr2_lmul2_pipe_crr` | 3 | **333.0%** | [328.1%, 335.6%] | MIPPv2 Faster |
| `m1_neon` | `clang` | `default` | `fp32` | `x60_mr6_nr4_fmaddi_crr` | 3 | **101.5%** | [100.8%, 102.4%] | Mixed Advantage |
| `m1_neon` | `gcc` | `default` | `fp32` | `x60_mr6_nr4_fmaddi_crr` | 3 | **101.8%** | [101.0%, 103.0%] | Mixed Advantage |
| `meteorlake_avx2` | `clang` | `default` | `fp32` | `meteorlake_mr4_nr3_crr` | 3 | **100.8%** | [100.5%, 101.2%] | Parity (±2% CI) |
| `meteorlake_avx2` | `gcc` | `default` | `fp32` | `meteorlake_mr4_nr3_crr` | 3 | **98.6%** | [97.9%, 99.2%] | Parity (±2% CI) |
| `rpi5_neon` | `clang` | `default` | `fp32` | `a76_mr6_nr4_fmaddi_crr` | 3 | **101.6%** | [100.6%, 102.7%] | Mixed Advantage |
| `rpi5_neon` | `gcc` | `default` | `fp32` | `a76_mr6_nr4_fmaddi_crr` | 3 | **102.1%** | [101.0%, 103.5%] | Mixed Advantage |
| `x100_rvv` | `clang` | `default` | `fp32` | `x100_register_blocked_apack4_crr` | 3 | **102.5%** | [100.9%, 103.5%] | Mixed Advantage |
| `x100_rvv` | `gcc` | `default` | `fp32` | `x100_register_blocked_apack4_crr` | 3 | **90.3%** | [89.8%, 90.5%] | BLIS Faster |
| `x60_rvv` | `clang` | `default` | `fp32` | `x60_mr6_nr4_fmaddi_crr` | 3 | **74.0%** | [62.4%, 84.4%] | BLIS Faster |
| `x60_rvv` | `gcc` | `default` | `fp32` | `x60_mr6_nr4_fmaddi_crr` | 3 | **66.5%** | [52.0%, 80.1%] | BLIS Faster |
| `zen4_avx512` | `clang` | `default` | `fp32` | `firestorm_mr4_nr4_fmaddi_crr` | 3 | **100.6%** | [100.2%, 101.0%] | Parity (±2% CI) |
| `zen4_avx512` | `gcc` | `default` | `fp32` | `firestorm_mr4_nr4_fmaddi_crr` | 3 | **100.9%** | [100.4%, 101.5%] | Parity (±2% CI) |
| `zen5_avx512` | `clang` | `default` | `fp32` | `firestorm_mr4_nr4_fmaddi_crr` | 3 | **100.5%** | [100.3%, 100.8%] | Parity (±2% CI) |
| `zen5_avx512` | `gcc` | `default` | `fp32` | `firestorm_mr4_nr4_fmaddi_crr` | 3 | **100.5%** | [100.3%, 100.8%] | Parity (±2% CI) |

---

## 2. Visual Comparative Figures

- **Matched-K Ratio Comparison:** [default/plots_fp32/blis_vs_mippv2_ratio_comparison.svg](default/plots_fp32/blis_vs_mippv2_ratio_comparison.svg)
- **Scaling Grid Across $K \in [32, 1024]$:** [default/plots_fp32/blis_vs_mippv2_k_scaling_grid.svg](default/plots_fp32/blis_vs_mippv2_k_scaling_grid.svg)


---

## 3. Per-Size Matched-$K$ Scaling & Working-Set Analysis (Variant: `default`)

> [!TIP]
> **Cache Boundary Semantics:** Rows marked with `†` denote operating points where panel working set $K \times (M + N) \times \text{elem_bytes}$ exceeds L1d cache capacity, testing streaming memory throughput rather than pure in-register compute density.

### Platform: `a100_rvv` (clang, fp32, default)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=32$ | 21.56 | 5.88 | **367.0%** | [364.9%, 373.5%] | 16.9 | 3.8 | MIPPv2 (+267.0%) |
| $K=64$ | 22.98 | 6.40 | **359.3%** | [359.0%, 359.9%] | 33.8 | 7.5 | MIPPv2 (+259.3%) |
| $K=128$ † | 23.75 | 6.62 | **358.8%** | [358.5%, 359.1%] | 67.5 | 15.0 | MIPPv2 (+258.8%) |
| $K=256$ † | 24.20 | 6.73 | **359.7%** | [359.7%, 359.8%] | 135.0 | 30.0 | MIPPv2 (+259.7%) |
| $K=384$ † | 24.36 | 6.76 | **360.3%** | [360.2%, 360.6%] | 202.5 | 45.0 | MIPPv2 (+260.3%) |
| $K=512$ † | 24.45 | 6.77 | **361.4%** | [360.9%, 361.8%] | 270.0 | 60.0 | MIPPv2 (+261.4%) |
| $K=1024$ † | 24.45 | 6.22 | **393.0%** | [385.5%, 404.9%] | 540.0 | 120.0 | MIPPv2 (+293.0%) |

### Platform: `a100_rvv` (gcc, fp32, default)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=32$ | 20.90 | 5.43 | **384.9%** | [355.9%, 407.1%] | 16.9 | 3.8 | MIPPv2 (+284.9%) |
| $K=64$ | 21.98 | 6.37 | **345.3%** | [341.7%, 346.6%] | 33.8 | 7.5 | MIPPv2 (+245.3%) |
| $K=128$ † | 21.77 | 6.63 | **328.1%** | [327.7%, 337.2%] | 67.5 | 15.0 | MIPPv2 (+228.1%) |
| $K=256$ † | 22.62 | 6.74 | **335.6%** | [335.5%, 335.7%] | 135.0 | 30.0 | MIPPv2 (+235.6%) |
| $K=384$ † | 22.66 | 6.76 | **335.4%** | [335.3%, 335.8%] | 202.5 | 45.0 | MIPPv2 (+235.4%) |
| $K=512$ † | 22.72 | 6.77 | **335.9%** | [335.6%, 336.4%] | 270.0 | 60.0 | MIPPv2 (+235.9%) |
| $K=1024$ † | 22.75 | 6.27 | **363.0%** | [360.5%, 364.3%] | 540.0 | 120.0 | MIPPv2 (+263.0%) |

### Platform: `m1_neon` (clang, fp32, default)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=32$ | 29.59 | 25.56 | **115.7%** | [115.7%, 115.8%] | 2.8 | 2.5 | MIPPv2 (+15.7%) |
| $K=64$ | 30.75 | 28.43 | **108.2%** | [108.2%, 108.2%] | 5.5 | 5.0 | MIPPv2 (+8.2%) |
| $K=128$ | 30.83 | 30.11 | **102.4%** | [102.4%, 102.4%] | 11.0 | 10.0 | MIPPv2 (+2.4%) |
| $K=256$ | 31.40 | 31.03 | **101.2%** | [101.2%, 101.2%] | 22.0 | 20.0 | Parity (±2%) |
| $K=384$ | 31.60 | 31.34 | **100.8%** | [100.8%, 100.8%] | 33.0 | 30.0 | Parity (±2%) |
| $K=512$ | 31.70 | 31.41 | **100.9%** | [100.9%, 100.9%] | 44.0 | 40.0 | Parity (±2%) |
| $K=1024$ | 31.85 | 31.70 | **100.5%** | [100.5%, 100.5%] | 88.0 | 80.0 | Parity (±2%) |

### Platform: `m1_neon` (gcc, fp32, default)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=32$ | 30.04 | 25.52 | **117.7%** | [117.7%, 117.7%] | 2.8 | 2.5 | MIPPv2 (+17.7%) |
| $K=64$ | 30.99 | 28.39 | **109.1%** | [109.1%, 109.1%] | 5.5 | 5.0 | MIPPv2 (+9.1%) |
| $K=128$ | 30.99 | 30.09 | **103.0%** | [103.0%, 103.0%] | 11.0 | 10.0 | MIPPv2 (+3.0%) |
| $K=256$ | 31.49 | 31.01 | **101.5%** | [101.5%, 101.5%] | 22.0 | 20.0 | Parity (±2%) |
| $K=384$ | 31.65 | 31.34 | **101.0%** | [101.0%, 101.3%] | 33.0 | 30.0 | Parity (±2%) |
| $K=512$ | 31.74 | 31.42 | **101.0%** | [101.0%, 101.0%] | 44.0 | 40.0 | Parity (±2%) |
| $K=1024$ | 31.87 | 31.71 | **100.5%** | [100.5%, 100.5%] | 88.0 | 80.0 | Parity (±2%) |

### Platform: `meteorlake_avx2` (clang, fp32, default)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=32$ | 29.69 | 27.72 | **107.1%** | [107.0%, 107.1%] | 3.5 | 2.8 | MIPPv2 (+7.1%) |
| $K=64$ | 30.16 | 29.45 | **102.4%** | [102.4%, 102.7%] | 7.0 | 5.5 | MIPPv2 (+2.4%) |
| $K=128$ | 30.66 | 30.28 | **101.2%** | [100.8%, 101.2%] | 14.0 | 11.0 | Parity (±2%) |
| $K=256$ | 31.00 | 30.84 | **100.5%** | [100.5%, 100.7%] | 28.0 | 22.0 | Parity (±2%) |
| $K=384$ | 31.19 | 31.03 | **100.5%** | [100.4%, 100.5%] | 42.0 | 33.0 | Parity (±2%) |
| $K=512$ † | 30.84 | 31.12 | **99.1%** | [98.9%, 99.1%] | 56.0 | 44.0 | Parity (±2%) |
| $K=1024$ † | 31.04 | 30.75 | **100.9%** | [100.9%, 101.0%] | 112.0 | 88.0 | Parity (±2%) |

### Platform: `meteorlake_avx2` (gcc, fp32, default)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=32$ | 26.99 | 27.51 | **98.1%** | [98.1%, 98.2%] | 3.5 | 2.8 | Parity (±2%) |
| $K=64$ | 28.24 | 29.38 | **96.1%** | [96.1%, 96.4%] | 7.0 | 5.5 | BLIS (+3.9%) |
| $K=128$ | 29.53 | 30.16 | **97.9%** | [97.9%, 98.0%] | 14.0 | 11.0 | BLIS (+2.1%) |
| $K=256$ | 30.41 | 30.78 | **98.8%** | [98.7%, 98.9%] | 28.0 | 22.0 | Parity (±2%) |
| $K=384$ | 30.73 | 30.96 | **99.2%** | [98.9%, 99.3%] | 42.0 | 33.0 | Parity (±2%) |
| $K=512$ † | 26.28 | 30.73 | **85.5%** | [84.7%, 85.9%] | 56.0 | 44.0 | BLIS (+14.5%) |
| $K=1024$ † | 26.30 | 30.74 | **85.6%** | [85.3%, 85.7%] | 112.0 | 88.0 | BLIS (+14.4%) |

### Platform: `rpi5_neon` (clang, fp32, default)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=32$ | 14.58 | 13.03 | **111.9%** | [111.3%, 111.9%] | 2.8 | 2.5 | MIPPv2 (+11.9%) |
| $K=64$ | 15.22 | 14.42 | **105.5%** | [105.5%, 105.7%] | 5.5 | 5.0 | MIPPv2 (+5.5%) |
| $K=128$ | 15.58 | 15.16 | **102.7%** | [102.7%, 102.9%] | 11.0 | 10.0 | MIPPv2 (+2.7%) |
| $K=256$ | 15.79 | 15.57 | **101.4%** | [101.4%, 101.5%] | 22.0 | 20.0 | Parity (±2%) |
| $K=384$ | 15.81 | 15.71 | **100.6%** | [100.6%, 100.6%] | 33.0 | 30.0 | Parity (±2%) |
| $K=512$ | 15.85 | 15.78 | **100.4%** | [100.4%, 100.5%] | 44.0 | 40.0 | Parity (±2%) |
| $K=1024$ † | 15.36 | 15.82 | **97.1%** | [97.0%, 97.3%] | 88.0 | 80.0 | BLIS (+2.9%) |

### Platform: `rpi5_neon` (gcc, fp32, default)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=32$ | 14.66 | 13.05 | **112.4%** | [112.3%, 112.7%] | 2.8 | 2.5 | MIPPv2 (+12.4%) |
| $K=64$ | 15.32 | 14.34 | **106.8%** | [106.7%, 106.8%] | 5.5 | 5.0 | MIPPv2 (+6.8%) |
| $K=128$ | 15.65 | 15.12 | **103.5%** | [103.3%, 103.5%] | 11.0 | 10.0 | MIPPv2 (+3.5%) |
| $K=256$ | 15.82 | 15.55 | **101.8%** | [101.7%, 101.8%] | 22.0 | 20.0 | Parity (±2%) |
| $K=384$ | 15.84 | 15.70 | **101.0%** | [100.9%, 101.0%] | 33.0 | 30.0 | Parity (±2%) |
| $K=512$ | 15.88 | 15.78 | **100.7%** | [100.6%, 100.7%] | 44.0 | 40.0 | Parity (±2%) |
| $K=1024$ † | 15.15 | 15.81 | **95.9%** | [95.7%, 96.0%] | 88.0 | 80.0 | BLIS (+4.1%) |

### Platform: `x100_rvv` (clang, fp32, default)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=32$ | 12.35 | 13.61 | **90.7%** | [90.7%, 91.1%] | 4.4 | 3.8 | BLIS (+9.3%) |
| $K=64$ | 13.95 | 14.35 | **97.2%** | [97.2%, 97.4%] | 8.8 | 7.5 | BLIS (+2.8%) |
| $K=128$ | 14.88 | 14.75 | **100.9%** | [100.8%, 101.2%] | 17.5 | 15.0 | Parity (±2%) |
| $K=256$ | 15.45 | 14.93 | **103.5%** | [103.4%, 103.5%] | 35.0 | 30.0 | MIPPv2 (+3.5%) |
| $K=384$ | 15.52 | 15.02 | **103.3%** | [98.8%, 103.9%] | 52.5 | 45.0 | MIPPv2 (+3.3%) |
| $K=512$ † | 13.89 | 14.97 | **92.8%** | [92.3%, 93.0%] | 70.0 | 60.0 | BLIS (+7.2%) |
| $K=1024$ † | 13.09 | 15.07 | **86.9%** | [86.5%, 87.3%] | 140.0 | 120.0 | BLIS (+13.1%) |

### Platform: `x100_rvv` (gcc, fp32, default)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=32$ | 11.70 | 13.61 | **85.9%** | [85.8%, 86.5%] | 4.4 | 3.8 | BLIS (+14.1%) |
| $K=64$ | 12.69 | 14.32 | **88.6%** | [88.5%, 88.8%] | 8.8 | 7.5 | BLIS (+11.4%) |
| $K=128$ | 13.24 | 14.75 | **89.8%** | [89.8%, 90.0%] | 17.5 | 15.0 | BLIS (+10.2%) |
| $K=256$ | 13.53 | 14.94 | **90.5%** | [90.5%, 90.6%] | 35.0 | 30.0 | BLIS (+9.5%) |
| $K=384$ | 13.60 | 15.02 | **90.5%** | [87.5%, 90.7%] | 52.5 | 45.0 | BLIS (+9.5%) |
| $K=512$ † | 12.00 | 14.97 | **80.2%** | [79.1%, 82.1%] | 70.0 | 60.0 | BLIS (+19.8%) |
| $K=1024$ † | 11.96 | 15.05 | **79.5%** | [79.3%, 79.8%] | 140.0 | 120.0 | BLIS (+20.5%) |

### Platform: `x60_rvv` (clang, fp32, default)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=32$ | 9.34 | 11.42 | **81.8%** | [81.8%, 87.5%] | 4.8 | 3.8 | BLIS (+18.2%) |
| $K=64$ | 9.94 | 11.38 | **87.4%** | [86.9%, 87.4%] | 9.5 | 7.5 | BLIS (+12.6%) |
| $K=128$ | 10.26 | 13.35 | **76.9%** | [76.8%, 85.6%] | 19.0 | 15.0 | BLIS (+23.1%) |
| $K=256$ † | 7.57 | 12.13 | **62.4%** | [52.6%, 63.6%] | 38.0 | 30.0 | BLIS (+37.6%) |
| $K=384$ † | 8.12 | 9.63 | **84.4%** | [82.7%, 87.7%] | 57.0 | 45.0 | BLIS (+15.6%) |
| $K=512$ † | 8.27 | 9.89 | **83.6%** | [82.9%, 84.3%] | 76.0 | 60.0 | BLIS (+16.4%) |
| $K=1024$ † | 8.30 | 10.21 | **81.2%** | [79.2%, 82.3%] | 152.0 | 120.0 | BLIS (+18.8%) |

### Platform: `x60_rvv` (gcc, fp32, default)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=32$ | 8.65 | 11.40 | **75.9%** | [75.8%, 75.9%] | 4.8 | 3.8 | BLIS (+24.1%) |
| $K=64$ | 9.08 | 12.55 | **72.3%** | [72.3%, 72.3%] | 9.5 | 7.5 | BLIS (+27.7%) |
| $K=128$ | 9.31 | 13.20 | **70.5%** | [70.5%, 70.5%] | 19.0 | 15.0 | BLIS (+29.5%) |
| $K=256$ † | 6.99 | 13.43 | **52.0%** | [51.6%, 55.2%] | 38.0 | 30.0 | BLIS (+48.0%) |
| $K=384$ † | 7.90 | 9.87 | **80.1%** | [78.5%, 81.4%] | 57.0 | 45.0 | BLIS (+19.9%) |
| $K=512$ † | 7.96 | 10.02 | **79.5%** | [78.7%, 81.2%] | 76.0 | 60.0 | BLIS (+20.5%) |
| $K=1024$ † | 8.05 | 10.27 | **78.4%** | [77.5%, 78.7%] | 152.0 | 120.0 | BLIS (+21.6%) |

### Platform: `zen4_avx512` (clang, fp32, default)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=32$ | 31.99 | 29.23 | **109.5%** | [109.4%, 109.7%] | 8.5 | 5.5 | MIPPv2 (+9.5%) |
| $K=64$ | 31.99 | 30.65 | **104.4%** | [104.4%, 104.4%] | 17.0 | 11.0 | MIPPv2 (+4.4%) |
| $K=128$ † | 31.65 | 31.34 | **101.0%** | [101.0%, 101.1%] | 34.0 | 22.0 | Parity (±2%) |
| $K=256$ † | 31.86 | 31.66 | **100.6%** | [100.3%, 100.7%] | 68.0 | 44.0 | Parity (±2%) |
| $K=384$ † | 31.81 | 31.76 | **100.2%** | [100.1%, 100.7%] | 102.0 | 66.0 | Parity (±2%) |
| $K=512$ † | 31.84 | 31.82 | **100.1%** | [100.0%, 100.2%] | 136.0 | 88.0 | Parity (±2%) |
| $K=1024$ † | 31.91 | 31.89 | **100.1%** | [100.1%, 100.1%] | 272.0 | 176.0 | Parity (±2%) |

### Platform: `zen4_avx512` (gcc, fp32, default)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=32$ | 31.88 | 29.04 | **109.8%** | [109.5%, 109.8%] | 8.5 | 5.5 | MIPPv2 (+9.8%) |
| $K=64$ | 31.89 | 30.57 | **104.3%** | [104.3%, 104.4%] | 17.0 | 11.0 | MIPPv2 (+4.3%) |
| $K=128$ † | 31.71 | 31.24 | **101.5%** | [100.7%, 101.6%] | 34.0 | 22.0 | Parity (±2%) |
| $K=256$ † | 31.86 | 31.60 | **100.8%** | [100.8%, 100.8%] | 68.0 | 44.0 | Parity (±2%) |
| $K=384$ † | 31.86 | 31.73 | **100.4%** | [100.4%, 100.4%] | 102.0 | 66.0 | Parity (±2%) |
| $K=512$ † | 31.86 | 31.80 | **100.2%** | [100.1%, 100.2%] | 136.0 | 88.0 | Parity (±2%) |
| $K=1024$ † | 31.92 | 31.88 | **100.1%** | [100.1%, 100.1%] | 272.0 | 176.0 | Parity (±2%) |

### Platform: `zen5_avx512` (clang, fp32, default)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=32$ | 31.99 | 31.02 | **103.1%** | [103.1%, 103.1%] | 8.5 | 5.5 | MIPPv2 (+3.1%) |
| $K=64$ | 31.99 | 31.50 | **101.6%** | [101.6%, 101.6%] | 17.0 | 11.0 | Parity (±2%) |
| $K=128$ | 31.99 | 31.74 | **100.8%** | [100.8%, 100.8%] | 34.0 | 22.0 | Parity (±2%) |
| $K=256$ † | 31.99 | 31.87 | **100.4%** | [100.4%, 100.4%] | 68.0 | 44.0 | Parity (±2%) |
| $K=384$ † | 31.99 | 31.91 | **100.3%** | [100.3%, 100.3%] | 102.0 | 66.0 | Parity (±2%) |
| $K=512$ † | 31.99 | 31.93 | **100.2%** | [100.2%, 100.2%] | 136.0 | 88.0 | Parity (±2%) |
| $K=1024$ † | 31.99 | 31.96 | **100.1%** | [100.1%, 100.1%] | 272.0 | 176.0 | Parity (±2%) |

### Platform: `zen5_avx512` (gcc, fp32, default)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=32$ | 31.97 | 31.02 | **103.1%** | [103.1%, 103.1%] | 8.5 | 5.5 | MIPPv2 (+3.1%) |
| $K=64$ | 31.99 | 31.50 | **101.6%** | [101.2%, 101.6%] | 17.0 | 11.0 | Parity (±2%) |
| $K=128$ | 31.99 | 31.74 | **100.8%** | [100.8%, 100.8%] | 34.0 | 22.0 | Parity (±2%) |
| $K=256$ † | 31.99 | 31.87 | **100.4%** | [100.4%, 100.4%] | 68.0 | 44.0 | Parity (±2%) |
| $K=384$ † | 31.99 | 31.91 | **100.3%** | [100.3%, 100.3%] | 102.0 | 66.0 | Parity (±2%) |
| $K=512$ † | 31.99 | 31.93 | **100.2%** | [100.2%, 100.2%] | 136.0 | 88.0 | Parity (±2%) |
| $K=1024$ † | 31.99 | 31.96 | **100.1%** | [100.1%, 100.1%] | 272.0 | 176.0 | Parity (±2%) |


---

## Appendix: Best-Compiler Envelope (Post-Hoc)

> [!IMPORTANT]
> This appendix selects the post-hoc envelope across compilers (`max(gcc, clang)` at each measurement point). It is presented exclusively for secondary reference.

| Microarchitecture | Precision | Variant | Best Compiler by Ratio | Envelope Ratio Geomean | Envelope [Min, Max] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `a100_rvv` | `fp32` | `default` | `clang` | **359.6%** | [358.8%, 360.3%] |
| `m1_neon` | `fp32` | `default` | `gcc` | **101.8%** | [101.0%, 103.0%] |
| `meteorlake_avx2` | `fp32` | `default` | `clang` | **100.8%** | [100.5%, 101.2%] |
| `rpi5_neon` | `fp32` | `default` | `gcc` | **102.1%** | [101.0%, 103.5%] |
| `x100_rvv` | `fp32` | `default` | `clang` | **102.5%** | [100.9%, 103.5%] |
| `x60_rvv` | `fp32` | `default` | `clang` | **74.0%** | [62.4%, 84.4%] |
| `zen4_avx512` | `fp32` | `default` | `gcc` | **100.9%** | [100.4%, 101.5%] |
| `zen5_avx512` | `fp32` | `default` | `clang` | **100.5%** | [100.3%, 100.8%] |