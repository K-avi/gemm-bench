# GEMMBench — Comparative Analysis: MIPPv2 vs BLIS Reference Micro-Kernels (Double-Precision (FP64 / DGEMM))

**Ablation Variant:** `coldc` | **Analysis Window:** $K \in [128, 384]$ (inclusive)

Empirical comparison of sustained in-register compute density and throughput between MIPPv2 champions
and hand-tuned BLIS assembly micro-kernels across target microarchitectures in Double-Precision (FP64 / DGEMM).

## 1. Summary: Matched-$K$ Window Ratio ($K \in [{k_min}, {k_max}]$)

> [!NOTE]
> **Statistical Methodology & Reporting Rules:**
> - **Pre-registered Champions:** MIPPv2 champions are locked prior to measurement in `uarch_config.json`; no post-hoc argmax across kernels.
> - **Matched-K Ratios:** Ratios are strictly evaluated at identical panel depth $K$ ($R_K = \text{FLOP/cyc}_{\text{MIPP}, K} / \text{FLOP/cyc}_{\text{BLIS}, K}$).
> - **Geometric Mean over Cache-Resident Window:** Summary ratio is $\text{geomean}(R_K)$ for $K \in [128, 384]$.
> - **Separate Compiler Rows:** GCC and Clang are reported on distinct rows. Post-hoc best-compiler envelope is provided in the Appendix.

| Microarchitecture | Compiler | Variant | Precision | MIPPv2 Champion | $N_K$ | **Ratio Geomean** | [Min, Max] | Parity Classification |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| `a100_rvv` | `clang` | `coldc` | `fp32` | `a100_mr7_nr2_lmul2_pipe_crr` | 3 | **367.5%** | [367.3%, 367.7%] | MIPPv2 Faster |
| `a100_rvv` | `clang` | `coldc` | `fp64` | `a100_mr7_nr2_lmul2_pipe_crr` | 3 | **355.8%** | [354.6%, 356.9%] | MIPPv2 Faster |
| `a100_rvv` | `gcc` | `coldc` | `fp32` | `a100_mr7_nr2_lmul2_pipe_crr` | 3 | **336.1%** | [335.5%, 336.5%] | MIPPv2 Faster |
| `a100_rvv` | `gcc` | `coldc` | `fp64` | `a100_mr7_nr2_lmul2_pipe_crr` | 3 | **365.9%** | [364.3%, 367.3%] | MIPPv2 Faster |
| `m1_neon` | `clang` | `coldc` | `fp32` | `x60_mr6_nr4_fmaddi_crr` | 3 | **101.7%** | [100.8%, 102.8%] | Mixed Advantage |
| `m1_neon` | `clang` | `coldc` | `fp64` | `x60_mr6_nr4_fmaddi_crr` | 3 | **101.5%** | [100.8%, 102.6%] | Mixed Advantage |
| `m1_neon` | `gcc` | `coldc` | `fp32` | `x60_mr6_nr4_fmaddi_crr` | 3 | **102.0%** | [101.0%, 103.3%] | Mixed Advantage |
| `m1_neon` | `gcc` | `coldc` | `fp64` | `x60_mr6_nr4_fmaddi_crr` | 3 | **101.8%** | [101.0%, 103.1%] | Mixed Advantage |
| `meteorlake_avx2` | `clang` | `coldc` | `fp32` | `meteorlake_mr4_nr3_crr` | 3 | **100.8%** | [100.3%, 101.2%] | Parity (±2% CI) |
| `meteorlake_avx2` | `clang` | `coldc` | `fp64` | `meteorlake_mr4_nr3_crr` | 3 | **100.4%** | [99.2%, 101.0%] | Parity (±2% CI) |
| `meteorlake_avx2` | `gcc` | `coldc` | `fp32` | `meteorlake_mr4_nr3_crr` | 3 | **96.1%** | [94.2%, 98.1%] | Mixed Advantage |
| `meteorlake_avx2` | `gcc` | `coldc` | `fp64` | `meteorlake_mr4_nr3_crr` | 3 | **91.2%** | [82.6%, 96.4%] | BLIS Faster |
| `rpi5_neon` | `clang` | `coldc` | `fp32` | `a76_mr6_nr4_fmaddi_crr` | 3 | **99.8%** | [99.2%, 100.6%] | Parity (±2% CI) |
| `rpi5_neon` | `clang` | `coldc` | `fp64` | `a76_mr6_nr4_fmaddi_crr` | 3 | **100.5%** | [99.6%, 101.1%] | Parity (±2% CI) |
| `rpi5_neon` | `gcc` | `coldc` | `fp32` | `a76_mr6_nr4_fmaddi_crr` | 3 | **102.3%** | [101.1%, 103.8%] | Mixed Advantage |
| `rpi5_neon` | `gcc` | `coldc` | `fp64` | `a76_mr6_nr4_fmaddi_crr` | 3 | **102.1%** | [101.0%, 103.6%] | Mixed Advantage |
| `x100_rvv` | `clang` | `coldc` | `fp32` | `x100_register_blocked_apack4_crr` | 3 | **106.4%** | [105.1%, 107.5%] | MIPPv2 Faster |
| `x100_rvv` | `clang` | `coldc` | `fp64` | `x100_register_blocked_apack4_crr` | 3 | **105.9%** | [104.6%, 107.1%] | MIPPv2 Faster |
| `x100_rvv` | `gcc` | `coldc` | `fp32` | `x100_register_blocked_apack4_crr` | 3 | **90.5%** | [90.1%, 90.8%] | BLIS Faster |
| `x100_rvv` | `gcc` | `coldc` | `fp64` | `x100_register_blocked_apack4_crr` | 3 | **90.1%** | [89.3%, 90.8%] | BLIS Faster |
| `x60_rvv` | `clang` | `coldc` | `fp32` | `x60_mr6_nr4_fmaddi_crr` | 3 | **74.9%** | [61.2%, 84.9%] | BLIS Faster |
| `x60_rvv` | `clang` | `coldc` | `fp64` | `a100_mr7_nr4_pipe_crr` | 3 | **87.7%** | [76.8%, 93.9%] | BLIS Faster |
| `x60_rvv` | `gcc` | `coldc` | `fp32` | `x60_mr6_nr4_fmaddi_crr` | 3 | **66.8%** | [52.1%, 80.7%] | BLIS Faster |
| `x60_rvv` | `gcc` | `coldc` | `fp64` | `a100_mr7_nr4_pipe_crr` | 3 | **80.4%** | [71.6%, 85.2%] | BLIS Faster |
| `zen4_avx512` | `clang` | `coldc` | `fp32` | `firestorm_mr4_nr4_fmaddi_crr` | 3 | **99.3%** | [98.7%, 99.7%] | Parity (±2% CI) |
| `zen4_avx512` | `clang` | `coldc` | `fp64` | `firestorm_mr4_nr4_fmaddi_crr` | 3 | **98.0%** | [97.2%, 98.5%] | Mixed Advantage |
| `zen4_avx512` | `gcc` | `coldc` | `fp32` | `firestorm_mr4_nr4_fmaddi_crr` | 3 | **99.6%** | [99.1%, 100.4%] | Parity (±2% CI) |
| `zen4_avx512` | `gcc` | `coldc` | `fp64` | `firestorm_mr4_nr4_fmaddi_crr` | 3 | **97.3%** | [94.2%, 99.0%] | Mixed Advantage |
| `zen5_avx512` | `clang` | `coldc` | `fp32` | `firestorm_mr4_nr4_fmaddi_crr` | 3 | **100.5%** | [100.3%, 100.9%] | Parity (±2% CI) |
| `zen5_avx512` | `clang` | `coldc` | `fp64` | `firestorm_mr4_nr4_fmaddi_crr` | 3 | **100.5%** | [100.3%, 100.7%] | Parity (±2% CI) |
| `zen5_avx512` | `gcc` | `coldc` | `fp32` | `firestorm_mr4_nr4_fmaddi_crr` | 3 | **100.6%** | [100.3%, 101.0%] | Parity (±2% CI) |
| `zen5_avx512` | `gcc` | `coldc` | `fp64` | `firestorm_mr4_nr4_fmaddi_crr` | 3 | **100.5%** | [100.2%, 100.7%] | Parity (±2% CI) |

---

## 2. Visual Comparative Figures

*Figures can be generated by supplying the `--plot` flag.*


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

### Platform: `a100_rvv` (clang, fp64, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ † | 6.29 | 1.77 | **354.6%** | [352.5%, 355.5%] | 71.0 | 22.0 | MIPPv2 (+254.6%) |
| $K=256$ † | 6.37 | 1.79 | **356.9%** | [351.8%, 358.8%] | 142.0 | 44.0 | MIPPv2 (+256.9%) |
| $K=384$ † | 6.40 | 1.80 | **355.9%** | [354.7%, 359.3%] | 213.0 | 66.0 | MIPPv2 (+255.9%) |

### Platform: `a100_rvv` (gcc, fp32, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ † | 21.94 | 6.52 | **336.3%** | [334.7%, 336.6%] | 67.5 | 15.0 | MIPPv2 (+236.3%) |
| $K=256$ † | 22.47 | 6.68 | **336.5%** | [336.2%, 336.9%] | 135.0 | 30.0 | MIPPv2 (+236.5%) |
| $K=384$ † | 22.57 | 6.73 | **335.5%** | [335.5%, 335.8%] | 202.5 | 45.0 | MIPPv2 (+235.5%) |

### Platform: `a100_rvv` (gcc, fp64, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ † | 6.60 | 1.81 | **364.3%** | [363.6%, 364.7%] | 71.0 | 22.0 | MIPPv2 (+264.3%) |
| $K=256$ † | 6.65 | 1.82 | **366.0%** | [364.8%, 366.2%] | 142.0 | 44.0 | MIPPv2 (+266.0%) |
| $K=384$ † | 6.67 | 1.82 | **367.3%** | [367.1%, 367.8%] | 213.0 | 66.0 | MIPPv2 (+267.3%) |

### Platform: `m1_neon` (clang, fp32, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 30.79 | 29.95 | **102.8%** | [102.8%, 102.8%] | 11.0 | 10.0 | MIPPv2 (+2.8%) |
| $K=256$ | 31.38 | 30.96 | **101.4%** | [101.4%, 101.4%] | 22.0 | 20.0 | Parity (±2%) |
| $K=384$ | 31.56 | 31.30 | **100.8%** | [100.8%, 100.9%] | 33.0 | 30.0 | Parity (±2%) |

### Platform: `m1_neon` (clang, fp64, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 15.40 | 15.01 | **102.6%** | [102.6%, 102.6%] | 14.0 | 14.0 | MIPPv2 (+2.6%) |
| $K=256$ | 15.69 | 15.50 | **101.2%** | [101.2%, 101.2%] | 28.0 | 28.0 | Parity (±2%) |
| $K=384$ | 15.79 | 15.66 | **100.8%** | [100.8%, 100.8%] | 42.0 | 42.0 | Parity (±2%) |

### Platform: `m1_neon` (gcc, fp32, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 30.98 | 29.98 | **103.3%** | [103.3%, 103.4%] | 11.0 | 10.0 | MIPPv2 (+3.3%) |
| $K=256$ | 31.47 | 30.97 | **101.6%** | [101.6%, 101.6%] | 22.0 | 20.0 | Parity (±2%) |
| $K=384$ | 31.63 | 31.30 | **101.0%** | [101.0%, 101.1%] | 33.0 | 30.0 | Parity (±2%) |

### Platform: `m1_neon` (gcc, fp64, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 15.49 | 15.03 | **103.1%** | [103.1%, 103.1%] | 14.0 | 14.0 | MIPPv2 (+3.1%) |
| $K=256$ | 15.74 | 15.50 | **101.5%** | [101.5%, 101.5%] | 28.0 | 28.0 | Parity (±2%) |
| $K=384$ | 15.81 | 15.66 | **101.0%** | [101.0%, 101.0%] | 42.0 | 42.0 | Parity (±2%) |

### Platform: `meteorlake_avx2` (clang, fp32, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 30.42 | 30.07 | **101.2%** | [101.0%, 101.2%] | 14.0 | 11.0 | Parity (±2%) |
| $K=256$ | 30.97 | 30.69 | **100.9%** | [100.8%, 101.0%] | 28.0 | 22.0 | Parity (±2%) |
| $K=384$ | 31.09 | 31.00 | **100.3%** | [100.2%, 100.3%] | 42.0 | 33.0 | Parity (±2%) |

### Platform: `meteorlake_avx2` (clang, fp64, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 15.28 | 15.12 | **101.0%** | [100.8%, 101.1%] | 16.0 | 14.0 | Parity (±2%) |
| $K=256$ | 15.51 | 15.38 | **100.8%** | [100.8%, 100.9%] | 32.0 | 28.0 | Parity (±2%) |
| $K=384$ | 15.31 | 15.44 | **99.2%** | [98.7%, 99.1%] | 48.0 | 42.0 | Parity (±2%) |

### Platform: `meteorlake_avx2` (gcc, fp32, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 28.74 | 29.91 | **96.1%** | [95.9%, 96.6%] | 14.0 | 11.0 | BLIS (+3.9%) |
| $K=256$ | 29.98 | 30.56 | **98.1%** | [98.0%, 98.2%] | 28.0 | 22.0 | Parity (±2%) |
| $K=384$ | 29.11 | 30.92 | **94.2%** | [94.0%, 96.4%] | 42.0 | 33.0 | BLIS (+5.8%) |

### Platform: `meteorlake_avx2` (gcc, fp64, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 14.31 | 15.03 | **95.2%** | [95.2%, 95.8%] | 16.0 | 14.0 | BLIS (+4.8%) |
| $K=256$ | 14.81 | 15.36 | **96.4%** | [96.4%, 96.6%] | 32.0 | 28.0 | BLIS (+3.6%) |
| $K=384$ | 12.75 | 15.43 | **82.6%** | [82.2%, 82.6%] | 48.0 | 42.0 | BLIS (+17.4%) |

### Platform: `rpi5_neon` (clang, fp32, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 15.02 | 15.08 | **99.6%** | [99.3%, 100.2%] | 11.0 | 10.0 | Parity (±2%) |
| $K=256$ | 15.39 | 15.52 | **99.2%** | [98.1%, 99.6%] | 22.0 | 20.0 | Parity (±2%) |
| $K=384$ | 15.77 | 15.67 | **100.6%** | [100.5%, 100.6%] | 33.0 | 30.0 | Parity (±2%) |

### Platform: `rpi5_neon` (clang, fp64, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 7.56 | 7.48 | **101.1%** | [100.7%, 101.1%] | 14.0 | 14.0 | Parity (±2%) |
| $K=256$ | 7.71 | 7.74 | **99.6%** | [99.6%, 99.7%] | 28.0 | 28.0 | Parity (±2%) |
| $K=384$ | 7.88 | 7.81 | **100.9%** | [100.8%, 101.0%] | 42.0 | 42.0 | Parity (±2%) |

### Platform: `rpi5_neon` (gcc, fp32, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 15.58 | 15.01 | **103.8%** | [103.7%, 103.8%] | 11.0 | 10.0 | MIPPv2 (+3.8%) |
| $K=256$ | 15.81 | 15.49 | **102.1%** | [101.7%, 102.1%] | 22.0 | 20.0 | MIPPv2 (+2.1%) |
| $K=384$ | 15.82 | 15.65 | **101.1%** | [101.0%, 101.2%] | 33.0 | 30.0 | Parity (±2%) |

### Platform: `rpi5_neon` (gcc, fp64, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 7.79 | 7.52 | **103.6%** | [103.3%, 103.8%] | 14.0 | 14.0 | MIPPv2 (+3.6%) |
| $K=256$ | 7.88 | 7.74 | **101.7%** | [101.6%, 102.0%] | 28.0 | 28.0 | Parity (±2%) |
| $K=384$ | 7.91 | 7.83 | **101.0%** | [100.9%, 101.2%] | 42.0 | 42.0 | Parity (±2%) |

### Platform: `x100_rvv` (clang, fp32, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 14.87 | 13.83 | **107.5%** | [107.2%, 107.6%] | 17.5 | 15.0 | MIPPv2 (+7.5%) |
| $K=256$ | 15.40 | 14.43 | **106.7%** | [106.6%, 106.9%] | 35.0 | 30.0 | MIPPv2 (+6.7%) |
| $K=384$ | 15.39 | 14.65 | **105.1%** | [104.1%, 106.1%] | 52.5 | 45.0 | MIPPv2 (+5.1%) |

### Platform: `x100_rvv` (clang, fp64, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 7.41 | 6.92 | **107.1%** | [106.4%, 107.6%] | 19.0 | 22.0 | MIPPv2 (+7.1%) |
| $K=256$ | 7.63 | 7.20 | **106.0%** | [105.8%, 106.4%] | 38.0 | 44.0 | MIPPv2 (+6.0%) |
| $K=384$ † | 7.52 | 7.19 | **104.6%** | [102.9%, 106.6%] | 57.0 | 66.0 | MIPPv2 (+4.6%) |

### Platform: `x100_rvv` (gcc, fp32, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 13.20 | 14.65 | **90.1%** | [89.6%, 90.1%] | 17.5 | 15.0 | BLIS (+9.9%) |
| $K=256$ | 13.49 | 14.89 | **90.6%** | [90.6%, 90.7%] | 35.0 | 30.0 | BLIS (+9.4%) |
| $K=384$ | 13.59 | 14.97 | **90.8%** | [90.3%, 90.9%] | 52.5 | 45.0 | BLIS (+9.2%) |

### Platform: `x100_rvv` (gcc, fp64, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 6.61 | 7.32 | **90.2%** | [90.1%, 90.2%] | 19.0 | 22.0 | BLIS (+9.8%) |
| $K=256$ | 6.76 | 7.44 | **90.8%** | [90.7%, 90.9%] | 38.0 | 44.0 | BLIS (+9.2%) |
| $K=384$ † | 6.59 | 7.37 | **89.3%** | [88.9%, 91.9%] | 57.0 | 66.0 | BLIS (+10.7%) |

### Platform: `x60_rvv` (clang, fp32, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 10.13 | 12.53 | **80.8%** | [80.7%, 89.3%] | 19.0 | 15.0 | BLIS (+19.2%) |
| $K=256$ † | 7.14 | 11.67 | **61.2%** | [55.6%, 63.6%] | 38.0 | 30.0 | BLIS (+38.8%) |
| $K=384$ † | 8.02 | 9.45 | **84.9%** | [84.0%, 87.6%] | 57.0 | 45.0 | BLIS (+15.1%) |

### Platform: `x60_rvv` (clang, fp64, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 4.58 | 5.96 | **76.8%** | [76.2%, 77.0%] | 23.0 | 22.0 | BLIS (+23.2%) |
| $K=256$ † | 3.83 | 4.08 | **93.9%** | [93.3%, 94.4%] | 46.0 | 44.0 | BLIS (+6.1%) |
| $K=384$ † | 3.92 | 4.18 | **93.7%** | [93.3%, 94.3%] | 69.0 | 66.0 | BLIS (+6.3%) |

### Platform: `x60_rvv` (gcc, fp32, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 9.20 | 12.95 | **71.1%** | [71.0%, 71.1%] | 19.0 | 15.0 | BLIS (+28.9%) |
| $K=256$ † | 6.63 | 12.73 | **52.1%** | [52.1%, 52.2%] | 38.0 | 30.0 | BLIS (+47.9%) |
| $K=384$ † | 7.81 | 9.67 | **80.7%** | [80.4%, 81.3%] | 57.0 | 45.0 | BLIS (+19.3%) |

### Platform: `x60_rvv` (gcc, fp64, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 4.51 | 6.29 | **71.6%** | [71.3%, 71.6%] | 23.0 | 22.0 | BLIS (+28.4%) |
| $K=256$ † | 3.66 | 4.30 | **85.2%** | [84.9%, 86.3%] | 46.0 | 44.0 | BLIS (+14.8%) |
| $K=384$ † | 3.77 | 4.42 | **85.2%** | [84.8%, 85.5%] | 69.0 | 66.0 | BLIS (+14.8%) |

### Platform: `zen4_avx512` (clang, fp32, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ † | 30.39 | 30.80 | **98.7%** | [97.6%, 101.4%] | 34.0 | 22.0 | Parity (±2%) |
| $K=256$ † | 31.21 | 31.30 | **99.7%** | [99.0%, 100.7%] | 68.0 | 44.0 | Parity (±2%) |
| $K=384$ † | 31.31 | 31.47 | **99.5%** | [99.1%, 100.5%] | 102.0 | 66.0 | Parity (±2%) |

### Platform: `zen4_avx512` (clang, fp64, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ † | 15.16 | 15.60 | **97.2%** | [97.2%, 97.6%] | 36.0 | 30.0 | BLIS (+2.8%) |
| $K=256$ † | 15.56 | 15.79 | **98.5%** | [98.5%, 98.6%] | 72.0 | 60.0 | Parity (±2%) |
| $K=384$ † | 15.60 | 15.86 | **98.3%** | [95.7%, 98.6%] | 108.0 | 90.0 | Parity (±2%) |

### Platform: `zen4_avx512` (gcc, fp32, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ † | 30.65 | 30.53 | **100.4%** | [99.8%, 102.4%] | 34.0 | 22.0 | Parity (±2%) |
| $K=256$ † | 31.13 | 31.37 | **99.2%** | [99.1%, 100.4%] | 68.0 | 44.0 | Parity (±2%) |
| $K=384$ † | 31.10 | 31.39 | **99.1%** | [98.2%, 99.9%] | 102.0 | 66.0 | Parity (±2%) |

### Platform: `zen4_avx512` (gcc, fp64, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ † | 14.68 | 15.58 | **94.2%** | [93.1%, 96.7%] | 36.0 | 30.0 | BLIS (+5.8%) |
| $K=256$ † | 15.62 | 15.79 | **99.0%** | [93.6%, 99.0%] | 72.0 | 60.0 | Parity (±2%) |
| $K=384$ † | 15.64 | 15.85 | **98.7%** | [98.5%, 98.9%] | 108.0 | 90.0 | Parity (±2%) |

### Platform: `zen5_avx512` (clang, fp32, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 31.96 | 31.66 | **100.9%** | [100.9%, 101.0%] | 34.0 | 22.0 | Parity (±2%) |
| $K=256$ † | 31.98 | 31.85 | **100.4%** | [100.4%, 100.5%] | 68.0 | 44.0 | Parity (±2%) |
| $K=384$ † | 31.98 | 31.89 | **100.3%** | [100.3%, 100.3%] | 102.0 | 66.0 | Parity (±2%) |

### Platform: `zen5_avx512` (clang, fp64, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 15.98 | 15.87 | **100.7%** | [100.6%, 100.8%] | 36.0 | 30.0 | Parity (±2%) |
| $K=256$ † | 15.99 | 15.93 | **100.4%** | [100.4%, 100.4%] | 72.0 | 60.0 | Parity (±2%) |
| $K=384$ † | 15.99 | 15.95 | **100.3%** | [100.3%, 100.3%] | 108.0 | 90.0 | Parity (±2%) |

### Platform: `zen5_avx512` (gcc, fp32, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 31.96 | 31.65 | **101.0%** | [100.6%, 101.0%] | 34.0 | 22.0 | Parity (±2%) |
| $K=256$ † | 31.98 | 31.83 | **100.5%** | [100.4%, 100.6%] | 68.0 | 44.0 | Parity (±2%) |
| $K=384$ † | 31.99 | 31.90 | **100.3%** | [100.3%, 100.3%] | 102.0 | 66.0 | Parity (±2%) |

### Platform: `zen5_avx512` (gcc, fp64, coldc)

| Panel Depth $K$ | MIPPv2 (FLOP/cyc) | BLIS (FLOP/cyc) | **Ratio MIPP / BLIS** | 95% Bootstrap CI | WS MIPP (KiB) | WS BLIS (KiB) | Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=128$ | 15.98 | 15.87 | **100.7%** | [100.7%, 100.7%] | 36.0 | 30.0 | Parity (±2%) |
| $K=256$ † | 15.99 | 15.93 | **100.4%** | [100.4%, 100.4%] | 72.0 | 60.0 | Parity (±2%) |
| $K=384$ † | 15.99 | 15.95 | **100.2%** | [100.2%, 100.3%] | 108.0 | 90.0 | Parity (±2%) |


---

## Appendix: Best-Compiler Envelope (Post-Hoc)

> [!IMPORTANT]
> This appendix selects the post-hoc envelope across compilers (`max(gcc, clang)` at each measurement point). It is presented exclusively for secondary reference.

| Microarchitecture | Precision | Variant | Best Compiler by Ratio | Envelope Ratio Geomean | Envelope [Min, Max] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `a100_rvv` | `fp32` | `coldc` | `clang` | **367.5%** | [367.3%, 367.7%] |
| `a100_rvv` | `fp64` | `coldc` | `gcc` | **365.9%** | [364.3%, 367.3%] |
| `m1_neon` | `fp32` | `coldc` | `gcc` | **102.0%** | [101.0%, 103.3%] |
| `m1_neon` | `fp64` | `coldc` | `gcc` | **101.8%** | [101.0%, 103.1%] |
| `meteorlake_avx2` | `fp32` | `coldc` | `clang` | **100.8%** | [100.3%, 101.2%] |
| `meteorlake_avx2` | `fp64` | `coldc` | `clang` | **100.4%** | [99.2%, 101.0%] |
| `rpi5_neon` | `fp32` | `coldc` | `gcc` | **102.3%** | [101.1%, 103.8%] |
| `rpi5_neon` | `fp64` | `coldc` | `gcc` | **102.1%** | [101.0%, 103.6%] |
| `x100_rvv` | `fp32` | `coldc` | `clang` | **106.4%** | [105.1%, 107.5%] |
| `x100_rvv` | `fp64` | `coldc` | `clang` | **105.9%** | [104.6%, 107.1%] |
| `x60_rvv` | `fp32` | `coldc` | `clang` | **74.9%** | [61.2%, 84.9%] |
| `x60_rvv` | `fp64` | `coldc` | `clang` | **87.7%** | [76.8%, 93.9%] |
| `zen4_avx512` | `fp32` | `coldc` | `gcc` | **99.6%** | [99.1%, 100.4%] |
| `zen4_avx512` | `fp64` | `coldc` | `clang` | **98.0%** | [97.2%, 98.5%] |
| `zen5_avx512` | `fp32` | `coldc` | `gcc` | **100.6%** | [100.3%, 101.0%] |
| `zen5_avx512` | `fp64` | `coldc` | `clang` | **100.5%** | [100.3%, 100.7%] |