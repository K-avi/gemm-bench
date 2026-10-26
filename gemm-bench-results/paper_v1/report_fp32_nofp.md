# GEMMBench — Comparative Analysis: MIPPv2 vs BLIS Reference Micro-Kernels (Single-Precision (FP32 / SGEMM))

**Ablation Variant:** `nofp` | **Analysis Window:** $K \in [128, 384]$ (inclusive)

Empirical comparison of sustained in-register compute density and throughput between MIPPv2 champions
and hand-tuned BLIS assembly micro-kernels across target microarchitectures in Single-Precision (FP32 / SGEMM).

## 1. Summary: Matched-$K$ Window Ratio ($K \in [{k_min}, {k_max}]$)

> [!NOTE]
> **Statistical Methodology & Reporting Rules:**
> - **Pre-registered Champions:** MIPPv2 champions are locked prior to measurement in `uarch_config.json`; no post-hoc argmax across kernels.
> - **Matched-K Ratios:** Ratios are strictly evaluated at identical panel depth $K$ ($R_K = \text{FLOP/cyc}_{\text{MIPP}, K} / \text{FLOP/cyc}_{\text{BLIS}, K}$).
> - **Geometric Mean over Cache-Resident Window:** Summary ratio is $\text{geomean}(R_K)$ for $K \in [128, 384]$.
> - **Separate Compiler Rows:** GCC and Clang are reported on distinct rows. Post-hoc best-compiler envelope is provided in the Appendix.

| Microarchitecture | Compiler | Variant | Precision | MIPPv2 Champion | $N_K$ | **Ratio Geomean** | [Min, Max] | Parity Classification |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| `a100_rvv` | `clang` | `nofp` | `fp32` | `a100_mr7_nr2_lmul2_pipe_crr` | 3 | **359.8%** | [359.4%, 360.5%] | MIPPv2 Faster |
| `a100_rvv` | `clang` | `nofp` | `fp64` | `a100_mr7_nr2_lmul2_pipe_crr` | 3 | **349.8%** | [348.0%, 351.2%] | MIPPv2 Faster |
| `a100_rvv` | `gcc` | `nofp` | `fp32` | `a100_mr7_nr2_lmul2_pipe_crr` | 3 | **333.0%** | [328.3%, 335.5%] | MIPPv2 Faster |
| `a100_rvv` | `gcc` | `nofp` | `fp64` | `a100_mr7_nr2_lmul2_pipe_crr` | 3 | **366.1%** | [362.7%, 368.7%] | MIPPv2 Faster |
| `m1_neon` | `clang` | `nofp` | `fp32` | `x60_mr6_nr4_fmaddi_crr` | 3 | **101.5%** | [100.8%, 102.4%] | Mixed Advantage |
| `m1_neon` | `clang` | `nofp` | `fp64` | `x60_mr6_nr4_fmaddi_crr` | 3 | **101.5%** | [100.8%, 102.4%] | Mixed Advantage |
| `m1_neon` | `gcc` | `nofp` | `fp32` | `x60_mr6_nr4_fmaddi_crr` | 3 | **101.8%** | [101.0%, 102.9%] | Mixed Advantage |
| `m1_neon` | `gcc` | `nofp` | `fp64` | `x60_mr6_nr4_fmaddi_crr` | 3 | **101.6%** | [100.9%, 102.6%] | Mixed Advantage |
| `meteorlake_avx2` | `clang` | `nofp` | `fp32` | `meteorlake_mr4_nr3_crr` | 3 | **100.8%** | [100.5%, 101.3%] | Parity (±2% CI) |
| `meteorlake_avx2` | `clang` | `nofp` | `fp64` | `meteorlake_mr4_nr3_crr` | 3 | **99.5%** | [98.0%, 100.5%] | Parity (±2% CI) |
| `meteorlake_avx2` | `gcc` | `nofp` | `fp32` | `meteorlake_mr4_nr3_crr` | 3 | **98.5%** | [97.7%, 98.9%] | Mixed Advantage |
| `meteorlake_avx2` | `gcc` | `nofp` | `fp64` | `meteorlake_mr4_nr3_crr` | 3 | **91.1%** | [81.0%, 96.7%] | BLIS Faster |
| `rpi5_neon` | `clang` | `nofp` | `fp32` | `a76_mr6_nr4_fmaddi_crr` | 3 | **101.6%** | [100.7%, 102.6%] | Mixed Advantage |
| `rpi5_neon` | `clang` | `nofp` | `fp64` | `a76_mr6_nr4_fmaddi_crr` | 3 | **101.5%** | [100.6%, 102.6%] | Mixed Advantage |
| `rpi5_neon` | `gcc` | `nofp` | `fp32` | `a76_mr6_nr4_fmaddi_crr` | 3 | **102.0%** | [100.9%, 103.5%] | Mixed Advantage |
| `rpi5_neon` | `gcc` | `nofp` | `fp64` | `a76_mr6_nr4_fmaddi_crr` | 3 | **102.1%** | [101.0%, 103.5%] | Mixed Advantage |
| `x100_rvv` | `clang` | `nofp` | `fp32` | `x100_register_blocked_apack4_crr` | 3 | **102.6%** | [100.8%, 103.8%] | Mixed Advantage |
| `x100_rvv` | `clang` | `nofp` | `fp64` | `x100_register_blocked_apack4_crr` | 3 | **101.5%** | [100.1%, 102.5%] | Mixed Advantage |
| `x100_rvv` | `gcc` | `nofp` | `fp32` | `x100_register_blocked_apack4_crr` | 3 | **89.6%** | [89.1%, 90.2%] | BLIS Faster |
| `x100_rvv` | `gcc` | `nofp` | `fp64` | `x100_register_blocked_apack4_crr` | 3 | **88.8%** | [87.3%, 90.2%] | BLIS Faster |
| `x60_rvv` | `clang` | `nofp` | `fp32` | `x60_mr6_nr4_fmaddi_crr` | 3 | **72.8%** | [59.0%, 85.2%] | BLIS Faster |
| `x60_rvv` | `clang` | `nofp` | `fp64` | `a100_mr7_nr4_pipe_crr` | 3 | **84.5%** | [72.5%, 91.3%] | BLIS Faster |
| `x60_rvv` | `gcc` | `nofp` | `fp32` | `x60_mr6_nr4_fmaddi_crr` | 3 | **66.7%** | [52.6%, 80.4%] | BLIS Faster |
| `x60_rvv` | `gcc` | `nofp` | `fp64` | `a100_mr7_nr4_pipe_crr` | 3 | **79.6%** | [71.1%, 85.6%] | BLIS Faster |
| `zen4_avx512` | `clang` | `nofp` | `fp32` | `firestorm_mr4_nr4_fmaddi_crr` | 3 | **100.7%** | [100.4%, 100.9%] | Parity (±2% CI) |
| `zen4_avx512` | `clang` | `nofp` | `fp64` | `firestorm_mr4_nr4_fmaddi_crr` | 3 | **100.7%** | [100.1%, 101.7%] | Parity (±2% CI) |
| `zen4_avx512` | `gcc` | `nofp` | `fp32` | `firestorm_mr4_nr4_fmaddi_crr` | 3 | **100.7%** | [100.3%, 101.1%] | Parity (±2% CI) |
| `zen4_avx512` | `gcc` | `nofp` | `fp64` | `firestorm_mr4_nr4_fmaddi_crr` | 3 | **100.6%** | [100.2%, 101.1%] | Parity (±2% CI) |
| `zen5_avx512` | `clang` | `nofp` | `fp32` | `firestorm_mr4_nr4_fmaddi_crr` | 3 | **100.5%** | [100.3%, 100.8%] | Parity (±2% CI) |
| `zen5_avx512` | `clang` | `nofp` | `fp64` | `firestorm_mr4_nr4_fmaddi_crr` | 3 | **100.5%** | [100.3%, 100.8%] | Parity (±2% CI) |
| `zen5_avx512` | `gcc` | `nofp` | `fp32` | `firestorm_mr4_nr4_fmaddi_crr` | 3 | **100.5%** | [100.3%, 100.8%] | Parity (±2% CI) |
| `zen5_avx512` | `gcc` | `nofp` | `fp64` | `firestorm_mr4_nr4_fmaddi_crr` | 3 | **100.5%** | [100.3%, 100.8%] | Parity (±2% CI) |

---

## 2. Visual Comparative Figures

*Figures can be generated by supplying the `--plot` flag.*


---

## 3. Per-Size Matched-$K$ Scaling & Working-Set Analysis (Variant: `nofp`)

> [!TIP]
> **Cache Boundary Semantics:** Rows marked with `†` denote operating points where panel working set $K \times (M + N) \times \text{elem_bytes}$ exceeds L1d cache capacity, testing streaming memory throughput rather than pure in-register compute density.

### Platform: `a100_rvv` (clang, fp32, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 22.89 | 6.37 | **359.4%** | [358.9%, 360.1%] | 33.8 | 7.5 | MIPPv2 (+259.4%) |
| $K=128$ † | 23.78 | 6.62 | **359.4%** | [358.7%, 359.8%] | 67.5 | 15.0 | MIPPv2 (+259.4%) |
| $K=256$ † | 24.18 | 6.73 | **359.5%** | [359.3%, 359.6%] | 135.0 | 30.0 | MIPPv2 (+259.5%) |
| $K=384$ † | 24.34 | 6.75 | **360.5%** | [360.0%, 361.2%] | 202.5 | 45.0 | MIPPv2 (+260.5%) |

### Platform: `a100_rvv` (clang, fp64, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 6.20 | 1.79 | **347.3%** | [346.2%, 347.4%] | 35.5 | 11.0 | MIPPv2 (+247.3%) |
| $K=128$ † | 6.33 | 1.82 | **348.0%** | [347.8%, 348.1%] | 71.0 | 22.0 | MIPPv2 (+248.0%) |
| $K=256$ † | 6.39 | 1.82 | **350.4%** | [348.9%, 354.1%] | 142.0 | 44.0 | MIPPv2 (+250.4%) |
| $K=384$ † | 6.41 | 1.83 | **351.2%** | [348.9%, 358.4%] | 213.0 | 66.0 | MIPPv2 (+251.2%) |

### Platform: `a100_rvv` (gcc, fp32, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 21.81 | 6.41 | **340.4%** | [339.2%, 342.6%] | 33.8 | 7.5 | MIPPv2 (+240.4%) |
| $K=128$ † | 21.76 | 6.63 | **328.3%** | [326.2%, 337.6%] | 67.5 | 15.0 | MIPPv2 (+228.3%) |
| $K=256$ † | 22.59 | 6.73 | **335.5%** | [335.4%, 335.5%] | 135.0 | 30.0 | MIPPv2 (+235.5%) |
| $K=384$ † | 22.65 | 6.76 | **335.3%** | [335.1%, 335.3%] | 202.5 | 45.0 | MIPPv2 (+235.3%) |

### Platform: `a100_rvv` (gcc, fp64, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 6.53 | 1.79 | **364.7%** | [358.4%, 366.4%] | 35.5 | 11.0 | MIPPv2 (+264.7%) |
| $K=128$ † | 6.61 | 1.82 | **362.7%** | [362.2%, 363.2%] | 71.0 | 22.0 | MIPPv2 (+262.7%) |
| $K=256$ † | 6.67 | 1.81 | **368.7%** | [364.7%, 376.1%] | 142.0 | 44.0 | MIPPv2 (+268.7%) |
| $K=384$ † | 6.69 | 1.82 | **366.8%** | [366.3%, 376.5%] | 213.0 | 66.0 | MIPPv2 (+266.8%) |

### Platform: `m1_neon` (clang, fp32, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 30.72 | 28.42 | **108.1%** | [108.1%, 108.1%] | 5.5 | 5.0 | MIPPv2 (+8.1%) |
| $K=128$ | 30.83 | 30.10 | **102.4%** | [102.4%, 102.4%] | 11.0 | 10.0 | MIPPv2 (+2.4%) |
| $K=256$ | 31.41 | 31.02 | **101.2%** | [101.2%, 101.2%] | 22.0 | 20.0 | Parity (±2%) |
| $K=384$ | 31.60 | 31.35 | **100.8%** | [100.8%, 100.8%] | 33.0 | 30.0 | Parity (±2%) |

### Platform: `m1_neon` (clang, fp64, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 15.36 | 14.23 | **108.0%** | [108.0%, 108.0%] | 7.0 | 7.0 | MIPPv2 (+8.0%) |
| $K=128$ | 15.42 | 15.06 | **102.4%** | [102.4%, 102.4%] | 14.0 | 14.0 | MIPPv2 (+2.4%) |
| $K=256$ | 15.70 | 15.52 | **101.2%** | [101.2%, 101.2%] | 28.0 | 28.0 | Parity (±2%) |
| $K=384$ | 15.80 | 15.67 | **100.8%** | [100.8%, 100.8%] | 42.0 | 42.0 | Parity (±2%) |

### Platform: `m1_neon` (gcc, fp32, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 30.91 | 28.36 | **109.0%** | [109.0%, 109.1%] | 5.5 | 5.0 | MIPPv2 (+9.0%) |
| $K=128$ | 30.95 | 30.07 | **102.9%** | [102.9%, 102.9%] | 11.0 | 10.0 | MIPPv2 (+2.9%) |
| $K=256$ | 31.47 | 31.00 | **101.5%** | [101.5%, 101.5%] | 22.0 | 20.0 | Parity (±2%) |
| $K=384$ | 31.64 | 31.33 | **101.0%** | [101.0%, 101.0%] | 33.0 | 30.0 | Parity (±2%) |

### Platform: `m1_neon` (gcc, fp64, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 15.47 | 14.26 | **108.5%** | [108.5%, 108.5%] | 7.0 | 7.0 | MIPPv2 (+8.5%) |
| $K=128$ | 15.48 | 15.08 | **102.6%** | [102.6%, 102.6%] | 14.0 | 14.0 | MIPPv2 (+2.6%) |
| $K=256$ | 15.73 | 15.53 | **101.3%** | [101.3%, 101.3%] | 28.0 | 28.0 | Parity (±2%) |
| $K=384$ | 15.82 | 15.68 | **100.9%** | [100.9%, 100.9%] | 42.0 | 42.0 | Parity (±2%) |

### Platform: `meteorlake_avx2` (clang, fp32, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 29.88 | 29.44 | **101.5%** | [101.4%, 101.7%] | 7.0 | 5.5 | Parity (±2%) |
| $K=128$ | 30.58 | 30.20 | **101.3%** | [101.1%, 101.4%] | 14.0 | 11.0 | Parity (±2%) |
| $K=256$ | 31.02 | 30.83 | **100.6%** | [100.5%, 100.8%] | 28.0 | 22.0 | Parity (±2%) |
| $K=384$ | 31.17 | 31.00 | **100.5%** | [100.5%, 100.6%] | 42.0 | 33.0 | Parity (±2%) |

### Platform: `meteorlake_avx2` (clang, fp64, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 15.20 | 14.93 | **101.8%** | [101.7%, 101.8%] | 8.0 | 7.0 | Parity (±2%) |
| $K=128$ | 15.33 | 15.25 | **100.5%** | [100.5%, 100.7%] | 16.0 | 14.0 | Parity (±2%) |
| $K=256$ | 15.53 | 15.52 | **100.1%** | [100.1%, 100.2%] | 32.0 | 28.0 | Parity (±2%) |
| $K=384$ | 15.34 | 15.65 | **98.0%** | [98.0%, 98.1%] | 48.0 | 42.0 | Parity (±2%) |

### Platform: `meteorlake_avx2` (gcc, fp32, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 28.17 | 29.31 | **96.1%** | [96.0%, 96.4%] | 7.0 | 5.5 | BLIS (+3.9%) |
| $K=128$ | 29.45 | 30.13 | **97.7%** | [97.7%, 97.9%] | 14.0 | 11.0 | BLIS (+2.3%) |
| $K=256$ | 30.35 | 30.74 | **98.7%** | [98.6%, 98.9%] | 28.0 | 22.0 | Parity (±2%) |
| $K=384$ | 30.63 | 30.96 | **98.9%** | [98.7%, 99.1%] | 42.0 | 33.0 | Parity (±2%) |

### Platform: `meteorlake_avx2` (gcc, fp64, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 14.09 | 14.86 | **94.8%** | [94.8%, 95.4%] | 8.0 | 7.0 | BLIS (+5.2%) |
| $K=128$ | 14.65 | 15.19 | **96.5%** | [96.4%, 96.5%] | 16.0 | 14.0 | BLIS (+3.5%) |
| $K=256$ | 15.00 | 15.52 | **96.7%** | [96.5%, 96.7%] | 32.0 | 28.0 | BLIS (+3.3%) |
| $K=384$ | 12.67 | 15.64 | **81.0%** | [81.0%, 81.4%] | 48.0 | 42.0 | BLIS (+19.0%) |

### Platform: `rpi5_neon` (clang, fp32, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 15.19 | 14.34 | **105.9%** | [105.4%, 105.9%] | 5.5 | 5.0 | MIPPv2 (+5.9%) |
| $K=128$ | 15.56 | 15.16 | **102.6%** | [102.6%, 102.9%] | 11.0 | 10.0 | MIPPv2 (+2.6%) |
| $K=256$ | 15.78 | 15.56 | **101.4%** | [101.3%, 101.5%] | 22.0 | 20.0 | Parity (±2%) |
| $K=384$ | 15.81 | 15.70 | **100.7%** | [100.6%, 100.7%] | 33.0 | 30.0 | Parity (±2%) |

### Platform: `rpi5_neon` (clang, fp64, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 7.59 | 7.20 | **105.5%** | [105.2%, 105.5%] | 7.0 | 7.0 | MIPPv2 (+5.5%) |
| $K=128$ | 7.78 | 7.58 | **102.6%** | [102.6%, 102.7%] | 14.0 | 14.0 | MIPPv2 (+2.6%) |
| $K=256$ | 7.89 | 7.78 | **101.3%** | [101.3%, 101.4%] | 28.0 | 28.0 | Parity (±2%) |
| $K=384$ | 7.90 | 7.86 | **100.6%** | [100.6%, 100.6%] | 42.0 | 42.0 | Parity (±2%) |

### Platform: `rpi5_neon` (gcc, fp32, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 15.28 | 14.34 | **106.6%** | [106.5%, 106.7%] | 5.5 | 5.0 | MIPPv2 (+6.6%) |
| $K=128$ | 15.63 | 15.10 | **103.5%** | [103.4%, 103.5%] | 11.0 | 10.0 | MIPPv2 (+3.5%) |
| $K=256$ | 15.81 | 15.54 | **101.7%** | [101.7%, 101.8%] | 22.0 | 20.0 | Parity (±2%) |
| $K=384$ | 15.83 | 15.69 | **100.9%** | [100.9%, 100.9%] | 33.0 | 30.0 | Parity (±2%) |

### Platform: `rpi5_neon` (gcc, fp64, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 7.65 | 7.17 | **106.8%** | [106.5%, 106.9%] | 7.0 | 7.0 | MIPPv2 (+6.8%) |
| $K=128$ | 7.82 | 7.56 | **103.5%** | [103.3%, 103.5%] | 14.0 | 14.0 | MIPPv2 (+3.5%) |
| $K=256$ | 7.91 | 7.77 | **101.8%** | [101.7%, 102.0%] | 28.0 | 28.0 | Parity (±2%) |
| $K=384$ | 7.92 | 7.84 | **101.0%** | [101.0%, 101.1%] | 42.0 | 42.0 | Parity (±2%) |

### Platform: `x100_rvv` (clang, fp32, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 13.92 | 14.34 | **97.1%** | [96.8%, 97.3%] | 8.8 | 7.5 | BLIS (+2.9%) |
| $K=128$ | 14.86 | 14.74 | **100.8%** | [100.6%, 100.8%] | 17.5 | 15.0 | Parity (±2%) |
| $K=256$ | 15.43 | 14.93 | **103.4%** | [103.2%, 103.4%] | 35.0 | 30.0 | MIPPv2 (+3.4%) |
| $K=384$ | 15.58 | 15.02 | **103.8%** | [103.7%, 104.0%] | 52.5 | 45.0 | MIPPv2 (+3.8%) |

### Platform: `x100_rvv` (clang, fp64, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 6.84 | 7.16 | **95.4%** | [95.4%, 96.2%] | 9.5 | 11.0 | BLIS (+4.6%) |
| $K=128$ | 7.38 | 7.37 | **100.1%** | [99.9%, 100.1%] | 19.0 | 22.0 | Parity (±2%) |
| $K=256$ | 7.65 | 7.47 | **102.5%** | [102.3%, 102.8%] | 38.0 | 44.0 | MIPPv2 (+2.5%) |
| $K=384$ † | 7.51 | 7.36 | **101.9%** | [101.3%, 103.8%] | 57.0 | 66.0 | Parity (±2%) |

### Platform: `x100_rvv` (gcc, fp32, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 12.50 | 14.32 | **87.3%** | [87.0%, 87.5%] | 8.8 | 7.5 | BLIS (+12.7%) |
| $K=128$ | 13.13 | 14.74 | **89.1%** | [88.9%, 89.2%] | 17.5 | 15.0 | BLIS (+10.9%) |
| $K=256$ | 13.48 | 14.94 | **90.2%** | [90.2%, 90.2%] | 35.0 | 30.0 | BLIS (+9.8%) |
| $K=384$ | 13.45 | 15.02 | **89.6%** | [84.6%, 90.1%] | 52.5 | 45.0 | BLIS (+10.4%) |

### Platform: `x100_rvv` (gcc, fp64, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 6.22 | 7.16 | **87.0%** | [86.7%, 87.2%] | 9.5 | 11.0 | BLIS (+13.0%) |
| $K=128$ | 6.56 | 7.37 | **89.0%** | [89.0%, 89.1%] | 19.0 | 22.0 | BLIS (+11.0%) |
| $K=256$ | 6.74 | 7.47 | **90.2%** | [90.2%, 90.2%] | 38.0 | 44.0 | BLIS (+9.8%) |
| $K=384$ † | 6.48 | 7.42 | **87.3%** | [86.7%, 89.3%] | 57.0 | 66.0 | BLIS (+12.7%) |

### Platform: `x60_rvv` (clang, fp32, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 9.86 | 12.59 | **78.3%** | [78.3%, 87.4%] | 9.5 | 7.5 | BLIS (+21.7%) |
| $K=128$ | 10.22 | 13.33 | **76.7%** | [76.7%, 85.6%] | 19.0 | 15.0 | BLIS (+23.3%) |
| $K=256$ † | 7.53 | 12.75 | **59.0%** | [55.7%, 62.6%] | 38.0 | 30.0 | BLIS (+41.0%) |
| $K=384$ † | 8.13 | 9.55 | **85.2%** | [84.9%, 87.9%] | 57.0 | 45.0 | BLIS (+14.8%) |

### Platform: `x60_rvv` (clang, fp64, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 4.40 | 6.12 | **72.0%** | [72.0%, 72.1%] | 11.5 | 11.0 | BLIS (+28.0%) |
| $K=128$ | 4.64 | 6.40 | **72.5%** | [72.3%, 72.5%] | 23.0 | 22.0 | BLIS (+27.5%) |
| $K=256$ † | 3.85 | 4.22 | **91.1%** | [88.3%, 91.4%] | 46.0 | 44.0 | BLIS (+8.9%) |
| $K=384$ † | 3.99 | 4.37 | **91.3%** | [90.7%, 91.5%] | 69.0 | 66.0 | BLIS (+8.7%) |

### Platform: `x60_rvv` (gcc, fp32, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 9.03 | 12.52 | **72.1%** | [72.1%, 72.1%] | 9.5 | 7.5 | BLIS (+27.9%) |
| $K=128$ | 9.28 | 13.18 | **70.4%** | [70.4%, 70.4%] | 19.0 | 15.0 | BLIS (+29.6%) |
| $K=256$ † | 6.68 | 12.71 | **52.6%** | [50.1%, 52.6%] | 38.0 | 30.0 | BLIS (+47.4%) |
| $K=384$ † | 7.93 | 9.86 | **80.4%** | [79.8%, 80.8%] | 57.0 | 45.0 | BLIS (+19.6%) |

### Platform: `x60_rvv` (gcc, fp64, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 4.39 | 6.09 | **72.0%** | [72.0%, 72.1%] | 11.5 | 11.0 | BLIS (+28.0%) |
| $K=128$ | 4.56 | 6.40 | **71.1%** | [71.1%, 71.2%] | 23.0 | 22.0 | BLIS (+28.9%) |
| $K=256$ † | 3.63 | 4.39 | **82.7%** | [82.0%, 84.0%] | 46.0 | 44.0 | BLIS (+17.3%) |
| $K=384$ † | 3.80 | 4.44 | **85.6%** | [85.1%, 86.2%] | 69.0 | 66.0 | BLIS (+14.4%) |

### Platform: `zen4_avx512` (clang, fp32, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 31.75 | 30.68 | **103.5%** | [103.4%, 104.0%] | 17.0 | 11.0 | MIPPv2 (+3.5%) |
| $K=128$ † | 31.63 | 31.34 | **100.9%** | [100.9%, 101.6%] | 34.0 | 22.0 | Parity (±2%) |
| $K=256$ † | 31.84 | 31.62 | **100.7%** | [100.6%, 100.8%] | 68.0 | 44.0 | Parity (±2%) |
| $K=384$ † | 31.86 | 31.74 | **100.4%** | [100.2%, 100.4%] | 102.0 | 66.0 | Parity (±2%) |

### Platform: `zen4_avx512` (clang, fp64, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 15.94 | 15.22 | **104.7%** | [104.7%, 104.7%] | 18.0 | 15.0 | MIPPv2 (+4.7%) |
| $K=128$ † | 15.86 | 15.60 | **101.7%** | [101.4%, 101.7%] | 36.0 | 30.0 | Parity (±2%) |
| $K=256$ † | 15.91 | 15.84 | **100.4%** | [100.4%, 100.6%] | 72.0 | 60.0 | Parity (±2%) |
| $K=384$ † | 15.91 | 15.89 | **100.1%** | [99.8%, 100.2%] | 108.0 | 90.0 | Parity (±2%) |

### Platform: `zen4_avx512` (gcc, fp32, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 31.70 | 30.44 | **104.1%** | [103.9%, 104.4%] | 17.0 | 11.0 | MIPPv2 (+4.1%) |
| $K=128$ † | 31.60 | 31.24 | **101.1%** | [100.9%, 101.5%] | 34.0 | 22.0 | Parity (±2%) |
| $K=256$ † | 31.82 | 31.62 | **100.6%** | [100.6%, 100.6%] | 68.0 | 44.0 | Parity (±2%) |
| $K=384$ † | 31.83 | 31.74 | **100.3%** | [100.2%, 100.4%] | 102.0 | 66.0 | Parity (±2%) |

### Platform: `zen4_avx512` (gcc, fp64, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 15.99 | 15.26 | **104.8%** | [104.8%, 104.8%] | 18.0 | 15.0 | MIPPv2 (+4.8%) |
| $K=128$ † | 15.83 | 15.66 | **101.1%** | [101.0%, 101.5%] | 36.0 | 30.0 | Parity (±2%) |
| $K=256$ † | 15.91 | 15.84 | **100.4%** | [100.4%, 100.6%] | 72.0 | 60.0 | Parity (±2%) |
| $K=384$ † | 15.92 | 15.89 | **100.2%** | [100.2%, 100.2%] | 108.0 | 90.0 | Parity (±2%) |

### Platform: `zen5_avx512` (clang, fp32, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 31.99 | 31.50 | **101.6%** | [101.6%, 101.6%] | 17.0 | 11.0 | Parity (±2%) |
| $K=128$ | 31.99 | 31.74 | **100.8%** | [100.8%, 100.8%] | 34.0 | 22.0 | Parity (±2%) |
| $K=256$ † | 31.99 | 31.87 | **100.4%** | [100.4%, 100.4%] | 68.0 | 44.0 | Parity (±2%) |
| $K=384$ † | 31.99 | 31.91 | **100.3%** | [100.3%, 100.3%] | 102.0 | 66.0 | Parity (±2%) |

### Platform: `zen5_avx512` (clang, fp64, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 15.99 | 15.73 | **101.7%** | [101.6%, 101.7%] | 18.0 | 15.0 | Parity (±2%) |
| $K=128$ | 15.99 | 15.86 | **100.8%** | [100.8%, 100.8%] | 36.0 | 30.0 | Parity (±2%) |
| $K=256$ † | 16.00 | 15.93 | **100.4%** | [100.4%, 100.4%] | 72.0 | 60.0 | Parity (±2%) |
| $K=384$ † | 16.00 | 15.95 | **100.3%** | [100.3%, 100.3%] | 108.0 | 90.0 | Parity (±2%) |

### Platform: `zen5_avx512` (gcc, fp32, nofp)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=64$ | 31.99 | 31.50 | **101.6%** | [101.4%, 101.6%] | 17.0 | 11.0 | Parity (±2%) |
| $K=128$ | 31.99 | 31.74 | **100.8%** | [100.8%, 100.9%] | 34.0 | 22.0 | Parity (±2%) |
| $K=256$ † | 31.99 | 31.87 | **100.4%** | [100.4%, 100.4%] | 68.0 | 44.0 | Parity (±2%) |
| $K=384$ † | 31.99 | 31.91 | **100.3%** | [100.3%, 100.3%] | 102.0 | 66.0 | Parity (±2%) |

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
| `a100_rvv` | `fp32` | `nofp` | `clang` | **359.8%** | [359.4%, 360.5%] |
| `a100_rvv` | `fp64` | `nofp` | `gcc` | **366.1%** | [362.7%, 368.7%] |
| `m1_neon` | `fp32` | `nofp` | `gcc` | **101.8%** | [101.0%, 102.9%] |
| `m1_neon` | `fp64` | `nofp` | `gcc` | **101.6%** | [100.9%, 102.6%] |
| `meteorlake_avx2` | `fp32` | `nofp` | `clang` | **100.8%** | [100.5%, 101.3%] |
| `meteorlake_avx2` | `fp64` | `nofp` | `clang` | **99.5%** | [98.0%, 100.5%] |
| `rpi5_neon` | `fp32` | `nofp` | `gcc` | **102.0%** | [100.9%, 103.5%] |
| `rpi5_neon` | `fp64` | `nofp` | `gcc` | **102.1%** | [101.0%, 103.5%] |
| `x100_rvv` | `fp32` | `nofp` | `clang` | **102.6%** | [100.8%, 103.8%] |
| `x100_rvv` | `fp64` | `nofp` | `clang` | **101.5%** | [100.1%, 102.5%] |
| `x60_rvv` | `fp32` | `nofp` | `clang` | **72.8%** | [59.0%, 85.2%] |
| `x60_rvv` | `fp64` | `nofp` | `clang` | **84.5%** | [72.5%, 91.3%] |
| `zen4_avx512` | `fp32` | `nofp` | `gcc` | **100.7%** | [100.3%, 101.1%] |
| `zen4_avx512` | `fp64` | `nofp` | `clang` | **100.7%** | [100.1%, 101.7%] |
| `zen5_avx512` | `fp32` | `nofp` | `gcc` | **100.5%** | [100.3%, 100.8%] |
| `zen5_avx512` | `fp64` | `nofp` | `clang` | **100.5%** | [100.3%, 100.8%] |