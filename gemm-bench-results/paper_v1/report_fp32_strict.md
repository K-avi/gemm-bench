# GEMMBench — Comparative Analysis: MIPPv2 vs BLIS Reference Micro-Kernels (Single-Precision (FP32 / SGEMM))

**Ablation Variant:** `strict` | **Analysis Window:** $K \in [128, 384]$ (inclusive)

Empirical comparison of sustained in-register compute density and throughput between MIPPv2 champions
and hand-tuned BLIS assembly micro-kernels across target microarchitectures in Single-Precision (FP32 / SGEMM).

## 1. Summary: Matched-$K$ Window Ratio ($K \in [128, 384]$, Variant: `strict`)

> [!NOTE]
> **Statistical Methodology & Reporting Rules:**
> - **Pre-registered Champions:** MIPPv2 champions are locked prior to measurement in `uarch_config.json`; no post-hoc argmax across kernels.
> - **Matched-K Ratios:** Ratios are strictly evaluated at identical panel depth $K$ ($R_K = \text{FLOP/cyc}_{\text{MIPP}, K} / \text{FLOP/cyc}_{\text{BLIS}, K}$).
> - **Geometric Mean over Cache-Resident Window:** Summary ratio is $\text{geomean}(R_K)$ for $K \in [128, 384]$.
> - **Separate Compiler Rows:** GCC and Clang are reported on distinct rows. Post-hoc best-compiler envelope is provided in the Appendix.

| Microarchitecture | Compiler | Variant | Precision | MIPPv2 Champion | $N_K$ | **Ratio Geomean** | [Min, Max] | Parity Classification |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| `a100_rvv` | `clang` | `strict` | `fp32` | `a100_mr7_nr2_lmul2_pipe_crr` | 3 | **360.0%** | [359.8%, 360.3%] | MIPPv2 Faster |
| `a100_rvv` | `gcc` | `strict` | `fp32` | `a100_mr7_nr2_lmul2_pipe_crr` | 3 | **336.0%** | [335.2%, 337.1%] | MIPPv2 Faster |
| `m1_neon` | `clang` | `strict` | `fp32` | `x60_mr6_nr4_fmaddi_crr` | 3 | **101.5%** | [100.8%, 102.4%] | Mixed Advantage |
| `m1_neon` | `gcc` | `strict` | `fp32` | `x60_mr6_nr4_fmaddi_crr` | 3 | **101.8%** | [101.0%, 103.0%] | Mixed Advantage |
| `meteorlake_avx2` | `clang` | `strict` | `fp32` | `meteorlake_mr4_nr3_crr` | 3 | **100.8%** | [100.4%, 101.4%] | Parity (±2% CI) |
| `meteorlake_avx2` | `gcc` | `strict` | `fp32` | `meteorlake_mr4_nr3_crr` | 3 | **97.8%** | [97.5%, 98.0%] | Mixed Advantage |
| `rpi5_neon` | `clang` | `strict` | `fp32` | `a76_mr6_nr4_fmaddi_crr` | 3 | **101.6%** | [100.6%, 102.8%] | Mixed Advantage |
| `rpi5_neon` | `gcc` | `strict` | `fp32` | `a76_mr6_nr4_fmaddi_crr` | 3 | **102.0%** | [100.9%, 103.5%] | Mixed Advantage |
| `x100_rvv` | `clang` | `strict` | `fp32` | `x100_register_blocked_apack4_crr` | 3 | **102.5%** | [100.9%, 103.3%] | Mixed Advantage |
| `x100_rvv` | `gcc` | `strict` | `fp32` | `x100_register_blocked_apack4_crr` | 3 | **90.4%** | [89.8%, 90.8%] | BLIS Faster |
| `x60_rvv` | `clang` | `strict` | `fp32` | `x60_mr6_nr4_fmaddi_crr` | 3 | **74.9%** | [64.2%, 85.1%] | BLIS Faster |
| `x60_rvv` | `gcc` | `strict` | `fp32` | `x60_mr6_nr4_fmaddi_crr` | 3 | **65.6%** | [51.1%, 79.6%] | BLIS Faster |
| `zen4_avx512` | `clang` | `strict` | `fp32` | `firestorm_mr4_nr4_fmaddi_crr` | 3 | **100.6%** | [100.3%, 101.0%] | Parity (±2% CI) |
| `zen4_avx512` | `gcc` | `strict` | `fp32` | `firestorm_mr4_nr4_fmaddi_crr` | 3 | **100.9%** | [100.4%, 101.5%] | Parity (±2% CI) |
| `zen5_avx512` | `clang` | `strict` | `fp32` | `firestorm_mr4_nr4_fmaddi_crr` | 3 | **100.5%** | [100.3%, 100.8%] | Parity (±2% CI) |
| `zen5_avx512` | `gcc` | `strict` | `fp32` | `firestorm_mr4_nr4_fmaddi_crr` | 3 | **100.5%** | [100.3%, 100.8%] | Parity (±2% CI) |

---

## 2. Visual Comparative Figures

- **Matched-K Ratio Comparison:** [strict/plots_fp32/blis_vs_mippv2_ratio_comparison.svg](strict/plots_fp32/blis_vs_mippv2_ratio_comparison.svg)
- **Scaling Grid Across $K \in [32, 1024]$:** [strict/plots_fp32/blis_vs_mippv2_k_scaling_grid.svg](strict/plots_fp32/blis_vs_mippv2_k_scaling_grid.svg)


---

## 3. Per-Size Matched-$K$ Scaling & Working-Set Analysis (Variant: `strict`)

> [!TIP]
> **Cache Boundary Semantics:** Rows marked with `†` denote operating points where panel working set $K \times (M + N) \times \text{elem_bytes}$ exceeds L1d cache capacity, testing streaming memory throughput rather than pure in-register compute density.

### Platform: `a100_rvv` (clang, fp32, strict)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ † | 23.84 | 6.62 | **360.1%** | [359.3%, 360.1%] | 67.5 | 15.0 | MIPPv2 (+260.1%) |
| $K=256$ † | 24.21 | 6.73 | **359.8%** | [359.7%, 359.9%] | 135.0 | 30.0 | MIPPv2 (+259.8%) |
| $K=384$ † | 24.36 | 6.76 | **360.3%** | [360.3%, 360.4%] | 202.5 | 45.0 | MIPPv2 (+260.3%) |

### Platform: `a100_rvv` (gcc, fp32, strict)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ † | 22.36 | 6.63 | **337.1%** | [328.2%, 337.5%] | 67.5 | 15.0 | MIPPv2 (+237.1%) |
| $K=256$ † | 22.62 | 6.74 | **335.6%** | [335.2%, 335.7%] | 135.0 | 30.0 | MIPPv2 (+235.6%) |
| $K=384$ † | 22.66 | 6.76 | **335.2%** | [335.1%, 335.2%] | 202.5 | 45.0 | MIPPv2 (+235.2%) |

### Platform: `m1_neon` (clang, fp32, strict)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 30.83 | 30.10 | **102.4%** | [102.4%, 102.4%] | 11.0 | 10.0 | MIPPv2 (+2.4%) |
| $K=256$ | 31.40 | 31.02 | **101.2%** | [101.2%, 101.2%] | 22.0 | 20.0 | Parity (±2%) |
| $K=384$ | 31.60 | 31.34 | **100.8%** | [100.8%, 100.8%] | 33.0 | 30.0 | Parity (±2%) |

### Platform: `m1_neon` (gcc, fp32, strict)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 30.99 | 30.09 | **103.0%** | [103.0%, 103.0%] | 11.0 | 10.0 | MIPPv2 (+3.0%) |
| $K=256$ | 31.48 | 31.01 | **101.5%** | [101.5%, 101.5%] | 22.0 | 20.0 | Parity (±2%) |
| $K=384$ | 31.65 | 31.33 | **101.0%** | [101.0%, 101.0%] | 33.0 | 30.0 | Parity (±2%) |

### Platform: `meteorlake_avx2` (clang, fp32, strict)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 30.73 | 30.29 | **101.4%** | [101.4%, 101.5%] | 14.0 | 11.0 | Parity (±2%) |
| $K=256$ | 31.09 | 30.90 | **100.6%** | [100.5%, 100.7%] | 28.0 | 22.0 | Parity (±2%) |
| $K=384$ | 31.22 | 31.09 | **100.4%** | [100.4%, 100.5%] | 42.0 | 33.0 | Parity (±2%) |

### Platform: `meteorlake_avx2` (gcc, fp32, strict)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 29.50 | 30.26 | **97.5%** | [97.4%, 97.6%] | 14.0 | 11.0 | BLIS (+2.5%) |
| $K=256$ | 30.23 | 30.85 | **98.0%** | [97.9%, 98.1%] | 28.0 | 22.0 | BLIS (+2.0%) |
| $K=384$ | 30.45 | 31.08 | **98.0%** | [98.0%, 98.2%] | 42.0 | 33.0 | BLIS (+2.0%) |

### Platform: `rpi5_neon` (clang, fp32, strict)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 15.58 | 15.16 | **102.8%** | [102.7%, 102.9%] | 11.0 | 10.0 | MIPPv2 (+2.8%) |
| $K=256$ | 15.79 | 15.57 | **101.4%** | [101.4%, 101.4%] | 22.0 | 20.0 | Parity (±2%) |
| $K=384$ | 15.81 | 15.71 | **100.6%** | [100.5%, 100.6%] | 33.0 | 30.0 | Parity (±2%) |

### Platform: `rpi5_neon` (gcc, fp32, strict)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 15.65 | 15.12 | **103.5%** | [103.4%, 103.6%] | 11.0 | 10.0 | MIPPv2 (+3.5%) |
| $K=256$ | 15.82 | 15.55 | **101.7%** | [101.7%, 101.8%] | 22.0 | 20.0 | Parity (±2%) |
| $K=384$ | 15.84 | 15.70 | **100.9%** | [100.9%, 100.9%] | 33.0 | 30.0 | Parity (±2%) |

### Platform: `x100_rvv` (clang, fp32, strict)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 14.88 | 14.75 | **100.9%** | [100.9%, 101.2%] | 17.5 | 15.0 | Parity (±2%) |
| $K=256$ | 15.43 | 14.93 | **103.3%** | [103.2%, 103.4%] | 35.0 | 30.0 | MIPPv2 (+3.3%) |
| $K=384$ | 15.50 | 15.01 | **103.2%** | [100.3%, 103.3%] | 52.5 | 45.0 | MIPPv2 (+3.2%) |

### Platform: `x100_rvv` (gcc, fp32, strict)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 13.24 | 14.75 | **89.8%** | [89.8%, 89.8%] | 17.5 | 15.0 | BLIS (+10.2%) |
| $K=256$ | 13.53 | 14.94 | **90.5%** | [90.5%, 90.6%] | 35.0 | 30.0 | BLIS (+9.5%) |
| $K=384$ | 13.63 | 15.02 | **90.8%** | [90.2%, 91.1%] | 52.5 | 45.0 | BLIS (+9.2%) |

### Platform: `x60_rvv` (clang, fp32, strict)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 10.26 | 13.35 | **76.9%** | [76.8%, 85.6%] | 19.0 | 15.0 | BLIS (+23.1%) |
| $K=256$ † | 7.44 | 11.59 | **64.2%** | [59.0%, 64.7%] | 38.0 | 30.0 | BLIS (+35.8%) |
| $K=384$ † | 8.17 | 9.60 | **85.1%** | [84.2%, 88.5%] | 57.0 | 45.0 | BLIS (+14.9%) |

### Platform: `x60_rvv` (gcc, fp32, strict)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 9.31 | 13.39 | **69.5%** | [69.5%, 70.5%] | 19.0 | 15.0 | BLIS (+30.5%) |
| $K=256$ † | 6.87 | 13.43 | **51.1%** | [49.5%, 54.6%] | 38.0 | 30.0 | BLIS (+48.9%) |
| $K=384$ † | 7.85 | 9.87 | **79.6%** | [79.4%, 80.3%] | 57.0 | 45.0 | BLIS (+20.4%) |

### Platform: `zen4_avx512` (clang, fp32, strict)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ † | 31.66 | 31.34 | **101.0%** | [100.9%, 101.1%] | 34.0 | 22.0 | Parity (±2%) |
| $K=256$ † | 31.84 | 31.66 | **100.6%** | [100.3%, 100.6%] | 68.0 | 44.0 | Parity (±2%) |
| $K=384$ † | 31.84 | 31.75 | **100.3%** | [100.1%, 100.4%] | 102.0 | 66.0 | Parity (±2%) |

### Platform: `zen4_avx512` (gcc, fp32, strict)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ † | 31.69 | 31.24 | **101.5%** | [99.8%, 101.5%] | 34.0 | 22.0 | Parity (±2%) |
| $K=256$ † | 31.86 | 31.60 | **100.8%** | [100.7%, 100.8%] | 68.0 | 44.0 | Parity (±2%) |
| $K=384$ † | 31.86 | 31.73 | **100.4%** | [100.4%, 100.4%] | 102.0 | 66.0 | Parity (±2%) |

### Platform: `zen5_avx512` (clang, fp32, strict)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 31.99 | 31.74 | **100.8%** | [100.7%, 100.8%] | 34.0 | 22.0 | Parity (±2%) |
| $K=256$ † | 31.99 | 31.87 | **100.4%** | [100.4%, 100.4%] | 68.0 | 44.0 | Parity (±2%) |
| $K=384$ † | 31.99 | 31.91 | **100.3%** | [100.3%, 100.3%] | 102.0 | 66.0 | Parity (±2%) |

### Platform: `zen5_avx512` (gcc, fp32, strict)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 31.99 | 31.74 | **100.8%** | [100.7%, 100.8%] | 34.0 | 22.0 | Parity (±2%) |
| $K=256$ † | 31.99 | 31.87 | **100.4%** | [100.4%, 100.4%] | 68.0 | 44.0 | Parity (±2%) |
| $K=384$ † | 31.99 | 31.91 | **100.3%** | [100.3%, 100.3%] | 102.0 | 66.0 | Parity (±2%) |


---

## Appendix: Best-Compiler Envelope (Post-Hoc)

> [!IMPORTANT]
> This appendix selects the post-hoc envelope across compilers (`max(gcc, clang)` at each measurement point). It is presented exclusively for secondary reference.

| Microarchitecture | Precision | Variant | Best Compiler by Ratio | Envelope Ratio Geomean | Envelope [Min, Max] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `a100_rvv` | `fp32` | `strict` | `clang` | **360.0%** | [359.8%, 360.3%] |
| `m1_neon` | `fp32` | `strict` | `gcc` | **101.8%** | [101.0%, 103.0%] |
| `meteorlake_avx2` | `fp32` | `strict` | `clang` | **100.8%** | [100.4%, 101.4%] |
| `rpi5_neon` | `fp32` | `strict` | `gcc` | **102.0%** | [100.9%, 103.5%] |
| `x100_rvv` | `fp32` | `strict` | `clang` | **102.5%** | [100.9%, 103.3%] |
| `x60_rvv` | `fp32` | `strict` | `clang` | **74.9%** | [64.2%, 85.1%] |
| `zen4_avx512` | `fp32` | `strict` | `gcc` | **100.9%** | [100.4%, 101.5%] |
| `zen5_avx512` | `fp32` | `strict` | `gcc` | **100.5%** | [100.3%, 100.8%] |