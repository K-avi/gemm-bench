# Investigation: Upstream BLIS Zen 4 AVX-512 Microkernel Baseline

**Date:** 2026-10-22  
**Target Architecture:** AMD Zen 4 (`znver4`, Family 19h Model 61h/70h, e.g., Ryzen 9 7900X / 7945HX)  
**Reference Commit:** `flame/blis@a8037cfc4330030c726efaec1ffeb1f93d468aa3`

---

## 1. Executive Summary

Upstream `flame/blis` at commit `a8037cfc4330030c726efaec1ffeb1f93d468aa3` (and mainline BLIS) **does not contain** a dedicated `kernels/zen4/` or `config/zen4/` directory.

- AMD Zen support in upstream `flame/blis` is limited to `config/zen`, `config/zen2`, and `config/zen3`, all of which target **256-bit AVX2/FMA3** microkernels ($MR \times NR = 6 \times 8$ for DGEMM, $6 \times 16$ for SGEMM).
- AVX-512 level-3 microkernels in upstream `flame/blis` reside exclusively under `kernels/skx/3/` (Intel Skylake-X).
- Dedicated Zen 4 AVX-512 microkernels exist primarily in AMD's downstream vendor fork **`amd/blis`** ([AOCL-BLIS](https://github.com/amd/blis)).

---

## 2. Upstream BLIS Repository Structure Analysis

### 2.1 Absence of `kernels/zen4/` in Upstream
Inspection of the `flame/blis` repository tree reveals:
- `config/`: Contains `zen`, `zen2`, `zen3`, `haswell`, `skx`, `knl`, `armsve`, `generic`, etc. There is no `config/zen4`.
- `kernels/`: Contains `zen/` (aliasing AVX2 Haswell kernels), `haswell/` (256-bit FMA3 asm), and `skx/` (512-bit AVX-512 asm).
- Under upstream BLIS dynamic dispatch, a Zen 4 processor executes the **Zen 3** configuration (`zen3`), relying on 256-bit AVX2 instructions (`vfmadd231pd` with 256-bit `%ymm` registers), thus capping theoretical FP64 peak at 16 FLOP/cycle instead of exploiting the 512-bit ISA capability.

### 2.2 Upstream AVX-512 Path: `kernels/skx/3/`
The only AVX-512 Level-3 GEMM microkernels present in upstream `flame/blis` are in `kernels/skx/3/`:
- **Files:** `bli_gemm_skx_asm_d24x8.c` / `bli_dgemm_skx_asm_16x12_l2.c`, `bli_sgemm_skx_asm_32x12.c` / `bli_sgemm_skx_asm_16x16.c`.
- **Target Microarchitecture:** Intel Skylake-X / Ice Lake / Sapphire Rapids.
- **Microarchitectural characteristics:**
  - Optimized for Intel's native 512-bit execution units on Port 0 and Port 5.
  - Geometry: $MR \times NR = 24 \times 8$ or $16 \times 12$ for DGEMM (32 `%zmm` registers used: 24 or 12 accumulator registers, remaining for $A$ broadcast and $B$ vector streaming).
  - Designed around Intel's thermal frequency throttling curves for heavy 512-bit instructions.

---

## 3. Downstream AMD Fork: AOCL-BLIS (`amd/blis`)

AMD maintains its own optimized distribution of BLIS ([github.com/amd/blis](https://github.com/amd/blis)):
- Introduces explicit `zen4` (`znver4`) configuration.
- Exploits Zen 4's microarchitecture: Zen 4 implements AVX-512 via **dual 256-bit execution units** (double-pumped 512-bit FMA across two cycles). Zen 4 does not suffer from Intel-style heavy thermal frequency throttling when executing AVX-512 code.
- Features tailored register allocations for DGEMM/SGEMM as well as low-precision extensions (BF16, Int8 VNNI).

---

## 4. Architectural Comparison: MIPPv2 vs BLIS Baselines on Zen 4

| Feature | MIPPv2 Champion (`zen4_mr4_nr4_fmaddi_crr`) | Upstream BLIS Default (`zen3`) | Upstream BLIS `skx` (`kernels/skx/3/`) | AOCL-BLIS (`amd/blis`) |
|---|---|---|---|---|
| **ISA** | AVX-512F (512-bit `%zmm`) | AVX2 / FMA3 (256-bit `%ymm`) | AVX-512F (512-bit `%zmm`) | AVX-512F (512-bit `%zmm`) |
| **DGEMM Tile ($MR \times NR$)** | $4 \times 32$ ($4 \times 4\cdot\text{VL}$) | $6 \times 8$ ($6 \times 2\cdot\text{VL}$) | $24 \times 8$ or $16 \times 12$ | $16 \times 8$ / tuned |
| **SGEMM Tile ($MR \times NR$)** | $4 \times 64$ ($4 \times 4\cdot\text{VL}$) | $6 \times 16$ ($6 \times 2\cdot\text{VL}$) | $32 \times 12$ or $16 \times 16$ | Tuned znver4 |
| **Theoretical Peak** | 16 DP / 32 SP FLOP/cycle | 16 DP / 32 SP FLOP/cycle | 16 DP / 32 SP FLOP/cycle | 16 DP / 32 SP FLOP/cycle |
| **Implementation** | C++ MIPPv2 SIMD abstraction | Inline Assembly (`.s` / GNU Asm) | Inline Assembly (`.s` / GNU Asm) | Inline Assembly / JIT |
| **Availability in `gemm_bench`** | Implemented & benchmarked | Implemented (`haswell_d6x8.c`) | Not integrated | Downstream external fork |

---

## 5. Integration Assessment into `libs/blis_reference/`

If a dedicated AVX-512 BLIS baseline is desired for Zen 4 / Zen 5 benchmarking:

### Option A: Integrate Upstream `kernels/skx/3/` (`skx` AVX-512 Assembly)
- **Source:** `flame/blis/kernels/skx/3/` at commit `a8037cfc4330030c726efaec1ffeb1f93d468aa3`.
- **Feasibility:** Moderate. Can be extracted into a standalone file `libs/blis_reference/src/skx_d24x8.c` similar to `haswell_d6x8.c`.
- **Pros:** Completely upstream standard BLIS code; direct microkernel comparison against native AVX-512 assembly.
- **Cons:** Tiling ($24 \times 8$ or $16 \times 12$) was tuned for Intel Skylake-X cache and execution ports, not Zen 4's dual 256-bit double-pumped pipeline.

### Option B: Integrate AOCL-BLIS `znver4` Microkernel
- **Source:** `amd/blis` downstream repository.
- **Feasibility:** High effort. AOCL microkernels frequently depend on AMD-specific header frameworks, runtime dispatch layers, or custom packing layouts.
- **Cons:** Deviates from the official reference BLIS repository (`flame/blis`), introducing vendor-fork specificity into the paper's baseline.

### Option C: Retain Current Baseline (Status Quo)
- Keep `blis_native` referencing the standard upstream BLIS Haswell/Zen3 AVX2 kernel (`haswell_d6x8.c`).
- In the publication, explicitly clarify that on Zen 4 / Zen 5, upstream mainline BLIS dispatches to its AVX2 `zen3` kernel, whereas MIPPv2 compiles directly to AVX-512F.

---

## 6. Recommendation

**Status:** Strictly documentary. **No code integrated into `libs/blis_reference/`.**  
Awaiting explicit user instruction before any integration attempt.
