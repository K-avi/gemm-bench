# GEMMBench — Unified Paper Summary (v1.0)

**Empirical Analysis Window:** $K \in [128, 384]$ | **Protocol:** 3 Independent Process Runs $\times$ 15 Repetitions

This document provides the canonical synthesis of MIPPv2 vs BLIS reference micro-kernel performance across all 8 target platforms,
both major toolchains (GCC and Clang), two IEEE floating-point precisions (FP64 and FP32), and three methodological ablation variants.

## 1. Primary Synthesis Table: MIPPv2 / BLIS Throughput Ratio

> [!NOTE]
> **Table Legend & Methodology Rules:**
> - **Analysis Window:** Summary ratios represent $\text{geomean}(R_K)$ across panel depths $K \in [128, 384]$.
> - **Ratio Definition:** $R_K = \frac{\text{FLOP/cyc}_{\text{MIPPv2}, K}}{\text{FLOP/cyc}_{\text{BLIS}, K}}$. Values $>100\%$ indicate MIPPv2 advantage.
> - **Pre-registered Champions:** MIPPv2 champions are locked prior to measurement in `uarch_config.json` (no post-hoc argmax).
> - **Parity Notation:** `≈` indicates parity (95% CI overlaps $[0.98, 1.02]$ at all $K$ in window); `+` indicates strict MIPPv2 advantage ($\text{CI}_{\text{low}} > 1.02$ at all $K$); `-` indicates strict BLIS advantage ($\text{CI}_{\text{high}} < 0.98$ at all $K$); `±` indicates a mixed regime across $K$ in the window (e.g. transition between $>2\%$ advantage and parity).

| Platform | Compiler | Prec. | MIPPv2 Champion | Default | No-FastPath | Cold-C | Strict IEEE (FP32) |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| `zen4_avx512` | `gcc` | FP64 | `mippv2_firestorm_mr4_nr4_fmaddi_crr` | **100.8%** ≈ | **100.6%** ≈ | **97.3%** ± | — |
| `zen4_avx512` | `clang` | FP64 | `mippv2_firestorm_mr4_nr4_fmaddi_crr` | **100.8%** ≈ | **100.7%** ≈ | **98.0%** ± | — |
| `zen4_avx512` | `gcc` | FP32 | `mippv2_firestorm_mr4_nr4_fmaddi_crr` | **100.9%** ≈ | **100.7%** ≈ | **99.5%** ≈ | **100.9%** ≈ |
| `zen4_avx512` | `clang` | FP32 | `mippv2_firestorm_mr4_nr4_fmaddi_crr` | **100.6%** ≈ | **100.6%** ≈ | **99.3%** ≈ | **100.6%** ≈ |
| `zen5_avx512` | `gcc` | FP64 | `mippv2_firestorm_mr4_nr4_fmaddi_crr` | **100.5%** ≈ | **100.5%** ≈ | **100.5%** ≈ | — |
| `zen5_avx512` | `clang` | FP64 | `mippv2_firestorm_mr4_nr4_fmaddi_crr` | **100.5%** ≈ | **100.5%** ≈ | **100.5%** ≈ | — |
| `zen5_avx512` | `gcc` | FP32 | `mippv2_firestorm_mr4_nr4_fmaddi_crr` | **100.5%** ≈ | **100.5%** ≈ | **100.6%** ≈ | **100.5%** ≈ |
| `zen5_avx512` | `clang` | FP32 | `mippv2_firestorm_mr4_nr4_fmaddi_crr` | **100.5%** ≈ | **100.5%** ≈ | **100.5%** ≈ | **100.5%** ≈ |
| `meteorlake_avx2` | `gcc` | FP64 | `mippv2_meteorlake_mr4_nr3_crr` | **91.2%** - | **91.1%** - | **91.2%** - | — |
| `meteorlake_avx2` | `clang` | FP64 | `mippv2_meteorlake_mr4_nr3_crr` | **99.5%** ≈ | **99.6%** ≈ | **100.2%** ≈ | — |
| `meteorlake_avx2` | `gcc` | FP32 | `mippv2_meteorlake_mr4_nr3_crr` | **98.7%** ≈ | **98.5%** ± | **96.1%** ± | **97.9%** ± |
| `meteorlake_avx2` | `clang` | FP32 | `mippv2_meteorlake_mr4_nr3_crr` | **100.7%** ≈ | **100.8%** ≈ | **100.8%** ≈ | **100.8%** ≈ |
| `m1_neon` | `gcc` | FP64 | `mippv2_x60_mr6_nr4_fmaddi_crr` | **101.7%** ± | **101.6%** ± | **101.8%** ± | — |
| `m1_neon` | `clang` | FP64 | `mippv2_x60_mr6_nr4_fmaddi_crr` | **101.4%** ± | **101.5%** ± | **101.5%** ± | — |
| `m1_neon` | `gcc` | FP32 | `mippv2_x60_mr6_nr4_fmaddi_crr` | **101.8%** ± | **101.8%** ± | **102.0%** ± | **101.8%** ± |
| `m1_neon` | `clang` | FP32 | `mippv2_x60_mr6_nr4_fmaddi_crr` | **101.5%** ± | **101.5%** ± | **101.7%** ± | **101.5%** ± |
| `rpi5_neon` | `gcc` | FP64 | `mippv2_a76_mr6_nr4_fmaddi_crr` | **101.9%** ± | **102.1%** ± | **102.1%** ± | — |
| `rpi5_neon` | `clang` | FP64 | `mippv2_a76_mr6_nr4_fmaddi_crr` | **101.7%** ± | **101.5%** ± | **100.5%** ≈ | — |
| `rpi5_neon` | `gcc` | FP32 | `mippv2_a76_mr6_nr4_fmaddi_crr` | **102.1%** ± | **102.0%** ± | **102.3%** ± | **102.0%** ± |
| `rpi5_neon` | `clang` | FP32 | `mippv2_a76_mr6_nr4_fmaddi_crr` | **101.6%** ± | **101.6%** ± | **99.8%** ≈ | **101.6%** ± |
| `x100_rvv` | `gcc` | FP64 | `mippv2_x100_register_blocked_apack4_crr` | **90.5%** - | **88.8%** - | **90.1%** - | — |
| `x100_rvv` | `clang` | FP64 | `mippv2_x100_register_blocked_apack4_crr` | **103.0%** ± | **101.5%** ± | **105.9%** + | — |
| `x100_rvv` | `gcc` | FP32 | `mippv2_x100_register_blocked_apack4_crr` | **90.3%** - | **89.6%** - | **90.5%** - | **90.4%** - |
| `x100_rvv` | `clang` | FP32 | `mippv2_x100_register_blocked_apack4_crr` | **102.5%** ± | **102.6%** ± | **106.4%** + | **102.5%** ± |
| `a100_rvv` | `gcc` | FP64 | `mippv2_a100_mr7_nr2_lmul2_pipe_crr` | **367.2%** + | **364.9%** + | **365.9%** + | — |
| `a100_rvv` | `clang` | FP64 | `mippv2_a100_mr7_nr2_lmul2_pipe_crr` | **352.5%** + | **349.4%** + | **355.4%** + | — |
| `a100_rvv` | `gcc` | FP32 | `mippv2_a100_mr7_nr2_lmul2_pipe_crr` | **333.2%** + | **333.2%** + | **336.1%** + | **335.9%** + |
| `a100_rvv` | `clang` | FP32 | `mippv2_a100_mr7_nr2_lmul2_pipe_crr` | **359.6%** + | **359.8%** + | **367.5%** + | **360.0%** + |
| `x60_rvv` | `gcc` | FP64 | `mippv2_a100_mr7_nr4_pipe_crr` | **78.3%** - | **79.6%** - | **80.4%** - | — |
| `x60_rvv` | `clang` | FP64 | `mippv2_a100_mr7_nr4_pipe_crr` | **83.7%** - | **84.5%** - | **87.7%** - | — |
| `x60_rvv` | `gcc` | FP32 | `mippv2_x60_mr6_nr4_fmaddi_crr` | **66.5%** - | **66.7%** - | **66.9%** - | **65.6%** - |
| `x60_rvv` | `clang` | FP32 | `mippv2_x60_mr6_nr4_fmaddi_crr` | **74.0%** - | **72.8%** - | **74.9%** - | **74.9%** - |

---

## 2. Key Findings & Microarchitectural Analysis

### 2.1 Cross-ISA Abstraction Parity (x86-64 & ARM64)
- **AMD Zen 4 & Zen 5 (AVX-512):** MIPPv2 achieves strictly **100.5% – 100.9%** parity with BLIS hand-written assembly micro-kernels across both GCC and Clang, in both FP64 and FP32. The MIPPv2 abstraction adds **zero measurable overhead** over expert native assembly.
- **Intel Meteor Lake (AVX2):** Clang achieves full parity (**100.8%** in FP32, **99.5%** in FP64). Under GCC, MIPPv2 reaches **98.6%** parity across $K \in [128, 384]$ in FP32 and **96.5% – 96.7%** up to $K=256$ in FP64, before experiencing a sharp drop at $K=384$ in FP64 (**81.1%**, pulling the window geomean to **91.2%**) where the MIPPv2 panel working set ($48.0\text{ KiB}$) reaches Redwood Cove's $48\text{ KiB}$ L1d capacity ceiling ($42.0\text{ KiB}$ for BLIS; the same L1d boundary drop occurs for GCC FP32 at $K=512$, $56.0\text{ KiB}$).
- **Apple M1 & Raspberry Pi 5 (ARM NEON):** MIPPv2 matches or slightly exceeds BLIS (**101.4% – 102.1%**; $+2.4\%$ to $+3.5\%$ at $K=128$ and strict parity at $K \in \{256, 384\}$), confirming that modern out-of-order execution cores comfortably absorb register-blocked load/FMA sequences generated via high-level C++ vector intrinsics.

### 2.2 RISC-V Vector Extensions (RVV 1.0)
- **SpacemiT X100 (256-bit VLEN, out-of-order):** Clang produces an effective kernel achieving **102.5% – 103.0%** of BLIS throughput in both precisions. GCC generates less optimal RVV instruction scheduling on X100, reaching **90.3% – 90.5%** of BLIS.
- **SpacemiT A100 (1024-bit VLEN):** MIPPv2 achieves **333.0% – 367.3%** of BLIS reference throughput. Because upstream BLIS does not provide a native 1024-bit micro-kernel, BLIS falls back to an unscaled 256-bit kernel, whereas MIPPv2 cleanly scales across vector lengths via its parameterized VLEN compile-time abstraction.
- **SpacemiT X60 (256-bit VLEN, in-order):** BLIS includes an artisanal assembly kernel handcrafted specifically for the dual-issue in-order X60 core. MIPPv2 reaches **74.0% – 83.7%** on Clang and **66.5% – 78.3%** on GCC, highlighting the sensitivity of in-order RISC-V pipelines without register renaming. In FP32, a pronounced dip occurs at $K=256$ (**52.0% – 62.4%**) before rebounding at $K=384$ (**80.1% – 84.4%**): at $K=256$, MIPPv2's $6 \times 32$ panel ($38.0\text{ KiB}$) exceeds the $32\text{ KiB}$ 4-way L1d cache while BLIS's $16 \times 14$ panel ($30.0\text{ KiB}$) still fits in L1d before spilling at $K=384$ ($45.0\text{ KiB}$).

---

## 3. Methodological Ablation Insights

### 3.1 FastPath Specialization (`default` vs `nofp`)
The compile-time FastPath optimization specializes the matrix epilogue for $\alpha=1.0, \beta=0.0$, eliminating runtime branch checks and register re-accumulation. In the cache-resident window ($K \in [128, 384]$), the throughput delta is negligible ($<0.3\%$ on x86 and ARM), demonstrating that the inner computational loop completely dominates execution time while the epilogue cost is amortized across $K$.

### 3.2 Working Set Memory Footprint (`default` vs `coldc`)
Rotating the output accumulator tile $C$ through a large out-of-cache memory pool (`--c-cold`) leaves relative throughput virtually unchanged on Zen 5, M1, RPi5, and X60 ($\Delta < 1\%$), while exposing two tile-geometry effects at $K=128$:
1. **Wide Output Tile RFO Latency on Zen 4 (FP64):** On `zen4_avx512` FP64, the ratio shifts from **100.8%** (`default`) to **97.3% – 98.0%** (`coldc`), driven by $K=128$ (**94.2%** on GCC) where DRAM Read-For-Ownership (RFO) write-allocate latency on MIPPv2's wide $4 \times 32$ output tile is not yet fully amortized (recovering to **98.5% – 99.0%** at $K \in \{256, 384\}$).
2. **Contiguous vs. Column-Strided Epilogue Advantage on X100:** On `x100_rvv` Clang, MIPPv2's advantage **increases** under `--c-cold`, rising from **103.0%** to **105.9%** in FP64 and from **102.5%** to **106.4%** in FP32. MIPPv2's row-major $3 \times 16$ / $3 \times 32$ tile writes to only 3 contiguous cache lines, whereas BLIS's column-strided $8 \times 14$ / $16 \times 14$ tile scatters writes across 14 distinct cache lines, incurring higher DRAM miss penalties when $C$ is cold.

### 3.3 Strict IEEE-754 Math (`default` vs `strict`)
Enforcing strict non-associative IEEE-754 semantics (`-fno-fast-math -ffp-contract=fast`) preserves parity across x86 and ARM platforms. Because MIPPv2 structures its accumulation tree explicitly in vector registers, the compiler does not rely on aggressive fast-math algebraic transformations to emit optimal FMA chains.

---

## 4. Threats to Validity

1. **Isolated Micro-Kernel Scope:** Measurements focus strictly on the computational GEMM micro-kernel ($C_{MR \times NR} += A_{MR \times K} \cdot B_{K \times NR}$), operating on pre-packed panel buffers in L1/L2 cache. This isolates vector compute density from macro-kernel cache partitioning (GotoBLAS/BLIS 5-loop framework), as established by Goto & van de Geijn (2008) and Van Zee & van de Geijn (2015).
2. **Asymmetric Tile Geometries:** BLIS reference kernels employ specific microarchitecture-tuned register blockings ($16 \times 14$ DP / $32 \times 12$ SP on AVX-512 `skx`, $6 \times 8$ DP / $6 \times 16$ SP on AVX2 `haswell`, $6 \times 8$ DP / $8 \times 12$ SP on NEON `armv8a`, and $8 \times 14$ DP / $16 \times 14$ SP on RVV `x60`), whereas MIPPv2 champions use platform-parameterized register tilings ($4 \times 4$ vectors on AVX-512, $4 \times 3$ on AVX2, $6 \times 4$ on NEON, $3 \times 2$ `apack4` on X100, $7 \times 2$ `lmul2` on A100). Working-set footprints at each $K$ and L1d cache spill boundaries (`†`) are tracked in the per-variant analytical reports.
3. **Pre-Registered Champion Selection:** To eliminate post-hoc selection bias across exploratory kernel variants, a single MIPPv2 champion per target and precision is locked in `uarch_config.json` prior to the benchmark campaign. Post-hoc best-compiler envelopes (`max(gcc, clang)`) are isolated in report appendices.
4. **RISC-V Baselines & In-Order Core Sensitivity:** Upstream BLIS lacks a native 1024-bit RVV micro-kernel for SpacemiT A100 and an out-of-order tuned micro-kernel for SpacemiT X100; both evaluate Stepan Nassyr's 256-bit X60 assembly micro-kernel. On the dual-issue in-order X60 core, static assembly scheduling exploits pipeline latencies inaccessible to general-purpose C++ compilers. On SpacemiT K3 (`a100`), threads are migrated to the AI core cluster (`cpu8..15`) via `/proc/set_ai_thread`; primary `FLOP/cycle` ratios are derived directly from per-thread hardware cycle counters (`PERF_COUNT_HW_CPU_CYCLES`).
