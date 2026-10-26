# GEMMBench — Comparative Analysis: MIPPv2 vs BLIS Reference Micro-Kernels (Single-Precision (FP32 / SGEMM))

**Ablation Variant:** `coldc` | **Analysis Window:** $K \in [128, 384]$ (inclusive)

Empirical comparison of sustained in-register compute density and throughput between MIPPv2 champions
and hand-tuned BLIS assembly micro-kernels across target microarchitectures in Single-Precision (FP32 / SGEMM).

## 1. Summary: Matched-$K$ Window Ratio ($K \in [128, 384]$, Variant: `coldc`)

> [!NOTE]
> **Statistical Methodology & Reporting Rules:**
> - **Pre-registered Champions:** MIPPv2 champions are locked prior to measurement in `uarch_config.json`; no post-hoc argmax across kernels.
> - **Matched-K Ratios:** Ratios are strictly evaluated at identical panel depth $K$ ($R_K = \text{FLOP/cyc}_{\text{MIPP}, K} / \text{FLOP/cyc}_{\text{BLIS}, K}$).
> - **Geometric Mean over Cache-Resident Window:** Summary ratio is $\text{geomean}(R_K)$ for $K \in [128, 384]$.
> - **Separate Compiler Rows:** GCC and Clang are reported on distinct rows. Post-hoc best-compiler envelope is provided in the Appendix.

| Microarchitecture | Compiler | Variant | Precision | MIPPv2 Champion | $N_K$ | **Ratio Geomean** | [Min, Max] | Parity Classification |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| `a100_rvv` | `clang` | `coldc` | `fp32` | `a100_mr7_nr2_lmul2_pipe_crr` | 3 | **367.5%** | [367.3%, 367.7%] | MIPPv2 Faster |
| `a100_rvv` | `gcc` | `coldc` | `fp32` | `a100_mr7_nr2_lmul2_pipe_crr` | 3 | **336.1%** | [335.5%, 336.5%] | MIPPv2 Faster |
| `m1_neon` | `clang` | `coldc` | `fp32` | `x60_mr6_nr4_fmaddi_crr` | 3 | **101.7%** | [100.8%, 102.8%] | Mixed Advantage |
| `m1_neon` | `gcc` | `coldc` | `fp32` | `x60_mr6_nr4_fmaddi_crr` | 3 | **102.0%** | [101.0%, 103.3%] | Mixed Advantage |
| `meteorlake_avx2` | `clang` | `coldc` | `fp32` | `meteorlake_mr4_nr3_crr` | 3 | **100.8%** | [100.3%, 101.2%] | Parity (±2% CI) |
| `meteorlake_avx2` | `gcc` | `coldc` | `fp32` | `meteorlake_mr4_nr3_crr` | 3 | **96.1%** | [94.2%, 98.1%] | Mixed Advantage |
| `rpi5_neon` | `clang` | `coldc` | `fp32` | `a76_mr6_nr4_fmaddi_crr` | 3 | **99.8%** | [99.2%, 100.6%] | Parity (±2% CI) |
| `rpi5_neon` | `gcc` | `coldc` | `fp32` | `a76_mr6_nr4_fmaddi_crr` | 3 | **102.3%** | [101.1%, 103.8%] | Mixed Advantage |
| `x100_rvv` | `clang` | `coldc` | `fp32` | `x100_register_blocked_apack4_crr` | 3 | **106.4%** | [105.1%, 107.5%] | MIPPv2 Faster |
| `x100_rvv` | `gcc` | `coldc` | `fp32` | `x100_register_blocked_apack4_crr` | 3 | **90.5%** | [90.1%, 90.8%] | BLIS Faster |
| `x60_rvv` | `clang` | `coldc` | `fp32` | `x60_mr6_nr4_fmaddi_crr` | 3 | **74.9%** | [61.2%, 84.9%] | BLIS Faster |
| `x60_rvv` | `gcc` | `coldc` | `fp32` | `x60_mr6_nr4_fmaddi_crr` | 3 | **66.8%** | [52.1%, 80.7%] | BLIS Faster |
| `zen4_avx512` | `clang` | `coldc` | `fp32` | `firestorm_mr4_nr4_fmaddi_crr` | 3 | **99.3%** | [98.7%, 99.7%] | Parity (±2% CI) |
| `zen4_avx512` | `gcc` | `coldc` | `fp32` | `firestorm_mr4_nr4_fmaddi_crr` | 3 | **99.6%** | [99.1%, 100.4%] | Parity (±2% CI) |
| `zen5_avx512` | `clang` | `coldc` | `fp32` | `firestorm_mr4_nr4_fmaddi_crr` | 3 | **100.5%** | [100.3%, 100.9%] | Parity (±2% CI) |
| `zen5_avx512` | `gcc` | `coldc` | `fp32` | `firestorm_mr4_nr4_fmaddi_crr` | 3 | **100.6%** | [100.3%, 101.0%] | Parity (±2% CI) |

---

## 2. Visual Comparative Figures

- **Matched-K Ratio Comparison:** [coldc/plots_fp32/blis_vs_mippv2_ratio_comparison.svg](coldc/plots_fp32/blis_vs_mippv2_ratio_comparison.svg)
- **Scaling Grid Across $K \in [32, 1024]$:** [coldc/plots_fp32/blis_vs_mippv2_k_scaling_grid.svg](coldc/plots_fp32/blis_vs_mippv2_k_scaling_grid.svg)


---

## 3. Per-Size Matched-$K$ Scaling & Working-Set Analysis (Variant: `coldc`)

> [!TIP]
> **Cache Boundary Semantics:** Rows marked with `†` denote operating points where panel working set $K \times (M + N) \times \text{elem_bytes}$ exceeds L1d cache capacity, testing streaming memory throughput rather than pure in-register compute density.

### Platform: `a100_rvv` (clang, fp32, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ † | 23.39 | 6.37 | **367.3%** | [366.6%, 369.5%] | 67.5 | 15.0 | MIPPv2 (+267.3%) |
| $K=256$ † | 24.06 | 6.54 | **367.7%** | [366.5%, 368.7%] | 135.0 | 30.0 | MIPPv2 (+267.7%) |
| $K=384$ † | 24.26 | 6.60 | **367.4%** | [367.2%, 373.8%] | 202.5 | 45.0 | MIPPv2 (+267.4%) |

### Platform: `a100_rvv` (gcc, fp32, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ † | 21.94 | 6.52 | **336.3%** | [334.7%, 336.6%] | 67.5 | 15.0 | MIPPv2 (+236.3%) |
| $K=256$ † | 22.47 | 6.68 | **336.5%** | [336.2%, 336.9%] | 135.0 | 30.0 | MIPPv2 (+236.5%) |
| $K=384$ † | 22.57 | 6.73 | **335.5%** | [335.5%, 335.8%] | 202.5 | 45.0 | MIPPv2 (+235.5%) |

### Platform: `m1_neon` (clang, fp32, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 30.79 | 29.95 | **102.8%** | [102.8%, 102.8%] | 11.0 | 10.0 | MIPPv2 (+2.8%) |
| $K=256$ | 31.38 | 30.96 | **101.4%** | [101.4%, 101.4%] | 22.0 | 20.0 | Parity (±2%) |
| $K=384$ | 31.56 | 31.30 | **100.8%** | [100.8%, 100.9%] | 33.0 | 30.0 | Parity (±2%) |

### Platform: `m1_neon` (gcc, fp32, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 30.98 | 29.98 | **103.3%** | [103.3%, 103.4%] | 11.0 | 10.0 | MIPPv2 (+3.3%) |
| $K=256$ | 31.47 | 30.97 | **101.6%** | [101.6%, 101.6%] | 22.0 | 20.0 | Parity (±2%) |
| $K=384$ | 31.63 | 31.30 | **101.0%** | [101.0%, 101.1%] | 33.0 | 30.0 | Parity (±2%) |

### Platform: `meteorlake_avx2` (clang, fp32, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 30.42 | 30.07 | **101.2%** | [101.0%, 101.2%] | 14.0 | 11.0 | Parity (±2%) |
| $K=256$ | 30.97 | 30.69 | **100.9%** | [100.8%, 101.0%] | 28.0 | 22.0 | Parity (±2%) |
| $K=384$ | 31.09 | 31.00 | **100.3%** | [100.2%, 100.3%] | 42.0 | 33.0 | Parity (±2%) |

### Platform: `meteorlake_avx2` (gcc, fp32, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 28.74 | 29.91 | **96.1%** | [95.9%, 96.6%] | 14.0 | 11.0 | BLIS (+3.9%) |
| $K=256$ | 29.98 | 30.56 | **98.1%** | [98.0%, 98.2%] | 28.0 | 22.0 | Parity (±2%) |
| $K=384$ | 29.11 | 30.92 | **94.2%** | [94.0%, 96.4%] | 42.0 | 33.0 | BLIS (+5.8%) |

### Platform: `rpi5_neon` (clang, fp32, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 15.02 | 15.08 | **99.6%** | [99.3%, 100.2%] | 11.0 | 10.0 | Parity (±2%) |
| $K=256$ | 15.39 | 15.52 | **99.2%** | [98.1%, 99.6%] | 22.0 | 20.0 | Parity (±2%) |
| $K=384$ | 15.77 | 15.67 | **100.6%** | [100.5%, 100.6%] | 33.0 | 30.0 | Parity (±2%) |

### Platform: `rpi5_neon` (gcc, fp32, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 15.58 | 15.01 | **103.8%** | [103.7%, 103.8%] | 11.0 | 10.0 | MIPPv2 (+3.8%) |
| $K=256$ | 15.81 | 15.49 | **102.1%** | [101.7%, 102.1%] | 22.0 | 20.0 | MIPPv2 (+2.1%) |
| $K=384$ | 15.82 | 15.65 | **101.1%** | [101.0%, 101.2%] | 33.0 | 30.0 | Parity (±2%) |

### Platform: `x100_rvv` (clang, fp32, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 14.87 | 13.83 | **107.5%** | [107.2%, 107.6%] | 17.5 | 15.0 | MIPPv2 (+7.5%) |
| $K=256$ | 15.40 | 14.43 | **106.7%** | [106.6%, 106.9%] | 35.0 | 30.0 | MIPPv2 (+6.7%) |
| $K=384$ | 15.39 | 14.65 | **105.1%** | [104.1%, 106.1%] | 52.5 | 45.0 | MIPPv2 (+5.1%) |

### Platform: `x100_rvv` (gcc, fp32, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 13.20 | 14.65 | **90.1%** | [89.6%, 90.1%] | 17.5 | 15.0 | BLIS (+9.9%) |
| $K=256$ | 13.49 | 14.89 | **90.6%** | [90.6%, 90.7%] | 35.0 | 30.0 | BLIS (+9.4%) |
| $K=384$ | 13.59 | 14.97 | **90.8%** | [90.3%, 90.9%] | 52.5 | 45.0 | BLIS (+9.2%) |

### Platform: `x60_rvv` (clang, fp32, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 10.13 | 12.53 | **80.8%** | [80.7%, 89.3%] | 19.0 | 15.0 | BLIS (+19.2%) |
| $K=256$ † | 7.14 | 11.67 | **61.2%** | [55.6%, 63.6%] | 38.0 | 30.0 | BLIS (+38.8%) |
| $K=384$ † | 8.02 | 9.45 | **84.9%** | [84.0%, 87.6%] | 57.0 | 45.0 | BLIS (+15.1%) |

### Platform: `x60_rvv` (gcc, fp32, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 9.20 | 12.95 | **71.1%** | [71.0%, 71.1%] | 19.0 | 15.0 | BLIS (+28.9%) |
| $K=256$ † | 6.63 | 12.73 | **52.1%** | [52.1%, 52.2%] | 38.0 | 30.0 | BLIS (+47.9%) |
| $K=384$ † | 7.81 | 9.67 | **80.7%** | [80.4%, 81.3%] | 57.0 | 45.0 | BLIS (+19.3%) |

### Platform: `zen4_avx512` (clang, fp32, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ † | 30.39 | 30.80 | **98.7%** | [97.6%, 101.4%] | 34.0 | 22.0 | Parity (±2%) |
| $K=256$ † | 31.21 | 31.30 | **99.7%** | [99.0%, 100.7%] | 68.0 | 44.0 | Parity (±2%) |
| $K=384$ † | 31.31 | 31.47 | **99.5%** | [99.1%, 100.5%] | 102.0 | 66.0 | Parity (±2%) |

### Platform: `zen4_avx512` (gcc, fp32, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ † | 30.65 | 30.53 | **100.4%** | [99.8%, 102.4%] | 34.0 | 22.0 | Parity (±2%) |
| $K=256$ † | 31.13 | 31.37 | **99.2%** | [99.1%, 100.4%] | 68.0 | 44.0 | Parity (±2%) |
| $K=384$ † | 31.10 | 31.39 | **99.1%** | [98.2%, 99.9%] | 102.0 | 66.0 | Parity (±2%) |

### Platform: `zen5_avx512` (clang, fp32, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 31.96 | 31.66 | **100.9%** | [100.9%, 101.0%] | 34.0 | 22.0 | Parity (±2%) |
| $K=256$ † | 31.98 | 31.85 | **100.4%** | [100.4%, 100.5%] | 68.0 | 44.0 | Parity (±2%) |
| $K=384$ † | 31.98 | 31.89 | **100.3%** | [100.3%, 100.3%] | 102.0 | 66.0 | Parity (±2%) |

### Platform: `zen5_avx512` (gcc, fp32, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 31.96 | 31.65 | **101.0%** | [100.6%, 101.0%] | 34.0 | 22.0 | Parity (±2%) |
| $K=256$ † | 31.98 | 31.83 | **100.5%** | [100.4%, 100.6%] | 68.0 | 44.0 | Parity (±2%) |
| $K=384$ † | 31.99 | 31.90 | **100.3%** | [100.3%, 100.3%] | 102.0 | 66.0 | Parity (±2%) |


---

## Appendix: Best-Compiler Envelope (Post-Hoc)

> [!IMPORTANT]
> This appendix selects the post-hoc envelope across compilers (`max(gcc, clang)` at each measurement point). It is presented exclusively for secondary reference.

| Microarchitecture | Precision | Variant | Best Compiler by Ratio | Envelope Ratio Geomean | Envelope [Min, Max] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `a100_rvv` | `fp32` | `coldc` | `clang` | **367.5%** | [367.3%, 367.7%] |
| `m1_neon` | `fp32` | `coldc` | `gcc` | **102.0%** | [101.0%, 103.3%] |
| `meteorlake_avx2` | `fp32` | `coldc` | `clang` | **100.8%** | [100.3%, 101.2%] |
| `rpi5_neon` | `fp32` | `coldc` | `gcc` | **102.3%** | [101.1%, 103.8%] |
| `x100_rvv` | `fp32` | `coldc` | `clang` | **106.4%** | [105.1%, 107.5%] |
| `x60_rvv` | `fp32` | `coldc` | `clang` | **74.9%** | [61.2%, 84.9%] |
| `zen4_avx512` | `fp32` | `coldc` | `gcc` | **99.6%** | [99.1%, 100.4%] |
| `zen5_avx512` | `fp32` | `coldc` | `gcc` | **100.6%** | [100.3%, 101.0%] |