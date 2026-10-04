# GEMM Benchmark suite across modern microarchitectures using MIPPv2

> [!WARNING]
> **Work in progress (WIP)**: This repository is under active development. Microkernels, benchmark drivers, and analysis tools are actively worked on. Anything might change 
> at anytime without notice.

A high-performance General Matrix Multiply (DGEMM) microbenchmark and exploration suite built on top of [MIPPv2](https://github.com/aff3ct/MIPP/tree/develop) (MyIntrinsics++ v2).

This suite evaluates DGEMM microkernels written in C++ with MIPPv2 across **9 distinct CPU microarchitectures** :  **x86_64 AVX2 & AVX-512**, **AArch64 NEON**, and **RISC-V Vector** (RVV).

> [!NOTE]
> **Origins & context**: This benchmark was originally written as part of a research internship at **LIP6** (Sorbonne Université), focusing on the design, implementation, and benchmarking of **RISC-V Vector 1.0 (RVV)** support for **MIPPv2** on the **SpacemiT X100** architecture and **AVX2** on the **Intel Skylake** architecture (because it's the one in my laptop).

---

## Microarchitectural Performance & Peak Efficiency

The benchmark evaluates single-thread, compute-bound double-precision GEMM throughput against the hardware theoretical peak ($\text{FLOP/cycle} \times \text{Frequency}$).

Across all target ISA families (x86_64 AVX2 & AVX-512, ARM NEON, RISC-V Vector), some microkernels achieve **>90% of theoretical peak performance** (reaching up to **>99%** on some uarchs):

### CRR Layout (Col-Major A, Row-Major B, C) — GotoBLAS / BLIS packing

CRR microkernels use a column-major packed $A$ panel (leading dimension $= M_R$), matching the access pattern of the GotoBLAS / BLIS macro-kernel inner loop. The benchmark evaluates a single register tile ($M_R \times N_R$) across panel depth $K \in \{32, 64, 128, 256, 512, 1024\}$.

| SIMD | Microarchitecture | CPU / SoC Model | Freq (GHz) | Peak FLOP/cyc | Comp | Achieved GFLOP/s | DGEMM FLOP/cyc | % of Peak |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **NEON** | **Apple M1 Firestorm** | M1 Ultra (P-core) | 3.036 | 16.0 | GCC | 48.3 | 15.90 | **99.4 %** |
| **NEON** | **Cortex-A76** | Raspberry Pi 5 | 2.400 | 8.0 | Clang | 18.9 | 7.88 | **98.5 %** |
| **AVX-512** | **AMD Zen 4** | Ryzen 9 7945HX | 5.463 | 16.0 | GCC | 85.3 | 15.62 | **97.6 %** |
| **AVX2** | **Intel Meteor Lake** | Redwood Cove P-Core | 4.780** | 16.0 | Clang | 74.5 | 15.58 | **97.4 %** |
| **AVX-512** | **AMD Zen 5 Strix Point** | Ryzen AI 9 HX 370 | 5.158 | 16.0 | GCC | 80.1 | 15.54 | **97.1 %** |
| **RVV 1.0** | **SpacemiT X100** | K3 SoC (VLEN=256) | 2.400 | 8.0 | Clang | 18.5 | 7.71 | **96.4 %** |
| **AVX2** | **Intel Skylake** | Core i5-6200U | 2.300 | 16.0 | Clang | 35.2 | 15.32 | **95.7 %** |
| **RVV 1.0** | **SpacemiT A100** | K3 SoC (VLEN=1024) | 2.000 | 8.0 | GCC | 13.2 | 6.60 | **82.5 %** |
| **RVV 1.0** | **SpacemiT X60** | BPI-F3 SoC (VLEN=256)| 1.600 | 8.0 | Clang* | 7.7 | 4.79 | **59.9 %** |

> [!IMPORTANT]
> **Intel Meteor Lake frequency calibration (sustained AVX2 operating frequency)**:
> - **5.10 GHz** is the single-core burst ceiling of the Redwood Cove P-core under light load.
> - Under sustained dense 256-bit AVX2 FMA execution, power and thermal management stabilize operating frequency at **4.78 GHz**, as calibrated empirically via `cpufp` (76.27 GFLOP/s empirical ceiling).
> - Evaluating against 4.78 GHz yields 15.58 FLOP/cycle (97.4% theoretical efficiency, 97.6% of empirical `cpufp` peak). 

> [!NOTE]
> **Zen 5 Strix Point Datapath Note**:
> Server/desktop Zen 5 incorporates dual native 512-bit FMA units ($32\text{ DP FLOP/cycle}$), AMD's mobile Strix Point SoC implements dual 256-bit physical datapaths (*double-pumped* 512-bit FMA, identical to Zen 4), yielding a physical ceiling of **$16\text{ DP FLOP/cycle}$**.
> If you own a Zen5 with 512-bit native FMA units and don't know what to do with it, I'd be very happy if you provided benchmarks results on the existing microkernels :o .

### Methodological Scope & Current Limitations

To interpret the reported throughput figures accurately, several methodological and architectural boundaries of the current benchmark suite must be highlighted:

1. **Isolated Microkernel Scope (In-Cache / Single-Panel)**:
   - Benchmarks evaluate the innermost $K$-loop on pre-packed micropanels $A$ ($M_R \times K$, column-major) and $B$ ($K \times N_R$, row-major) accumulating into an in-register tile ($M_R \times N_R$) and writing to $C$.
   - Overheads of the 5-loop cache-blocking hierarchy (GotoBLAS / BLIS: $J_C, P_C, I_C, J_R, I_R$) and dynamic matrix repacking into memory buffers ($M_C \times K_C$, $K_C \times N_C$) are **not** included. End-to-end DGEMM performance within BLIS is part of ongoing integration work.
2. **Canonical GEMM Regime & Throughput Formulation**:
   - Reported peak throughputs strictly evaluate the compute-bound canonical form $\alpha = 1.0, \beta = 0.0$ ($C \leftarrow AB$), where throughput is computed as $\text{GFLOP/s} = \frac{2 \times M \times N \times K}{\text{Time (seconds)} \times 10^9}$. Non-canonical operations ($\beta \neq 0.0$) introduce load-scale-accumulate overheads on $C$ that are supported functionally but not evaluated for peak throughput.
3. **Tile Geometry & Edge Handling (Tails / Fringes)**:
   - Peak pipeline saturation requires dimensions $M, N$ to be exact multiples of the register tile $M_R, N_R$ and depth $K$ to align with loop unrolling (typically 4). While all microkernels include functional scalar/vector remainder handling for unit-test validation, fringe handling in production BLAS is typically offloaded to dedicated edge kernels or zero-padded buffers.
4. **Single-Thread & Precision Scope**:
   - All measurements are strictly single-thread IEEE-754 double precision (`double` / FP64). Multi-threaded scaling and memory bus contention are not covered in this phase.
5. **Clock Frequency Baseline & Empirical FPU Ceilings (`cpufp`)**:
   - Efficiencies are reported against both the architectural theoretical ceiling ($\text{FLOP/cycle} \times \text{Frequency}$) and empirical double-precision peak throughput measured via [`cpufp`](https://github.com/pigirons/cpufp).
   - Rather than relying on transient single-core boost clocks that throttle under dense vector FMA workloads, sustained operating frequencies and empirical FP64 ceilings are benchmarked directly and recorded in [`uarch_config.json`](uarch_config.json).

### Robust Measurement Protocol & Statistical Rigor

To prevent common benchmarking pitfalls in microkernel evaluation (frequency throttling, compiler optimization artifacts, timing jitter, and cold cache effects), the suite implements a multi-layer measurement protocol:

1. **Multi-Repetition Sampling & Median Filtering**:
   - Each measurement point collects $R = 15$ timed samples (configurable via `-r / --repetitions`).
   - Each sample measures steady-state execution over hundreds of thousands to millions of kernel invocations.
   - Timings report the **median** duration (less sensitive to occasional background OS scheduling interrupts than the mean) alongside min, max, stddev, and intra-run coefficient of variation ($\text{CV} = \frac{\sigma}{\mu} \times 100\,\%$, reported as `GFLOPS_cv_pct`).
2. **Multi-Run Replication & Inter-Run CV Monitoring**:
   - Benchmarks execute $N_{\text{runs}} = 3$ independent processes per configuration (`--runs 3`).
   - The driver selects the best median execution time across passes and computes the inter-run variation (`InterRun_CV_pct`, sample coefficient of variation across run medians). An automatic warning (`⚠ inter-run CV > 5%`) is raised if runs exhibit significant variance.
3. **Compiler Optimization & Monotonic Timing Barriers**:
   - High-resolution monotonic timestamps (`std::chrono::steady_clock::now()`) are strictly isolated by `asm volatile("" ::: "memory")` compiler barriers to prevent reordering across timing boundaries.
   - Dead-code elimination (DCE) and loop-invariant hoisting are prevented via a clobbered read barrier on the destination matrix pointer (`asm volatile("" :: "r"(C.data) : "memory")`), forcing the compiler to re-evaluate the kernel on every iteration.
4. **Hardware Cycle Counting & In-Process FLOP/Cycle**:
   - On Linux systems, the harness uses unprivileged hardware performance counters (`perf_event_open` with `PERF_COUNT_HW_CPU_CYCLES`, accessible when `perf_event_paranoid <= 2`).
   - This records the exact CPU cycles elapsed during the timed loop, computing true $\text{FLOP/cycle} = \frac{2 \times M \times N \times K}{\text{Cycles}}$ and measured effective operating frequency ($\text{Eff\_GHz} = \frac{\text{Cycles}}{\text{Time} \times 10^9}$) directly, removing reliance on nominal boost clock assumptions.
5. **Passive CPU Frequency Monitoring & Throttle Detection**:
   - Does not require `sudo` / root privileges, enabling reliable execution on shared HPC clusters (e.g. the Dalek cluster) where governors cannot be forced to `performance`.
   - The driver logs governor state and reads `/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq`, warning if core frequency drops by $> 5\%$ during benchmarks.
6. **Thermal Cooldown Pauses**:
   - A configurable pause (default: $2.0\,\text{s}$ via `--cooldown 2.0`) is inserted between matrix sizes for $K \ge 256$, preventing heat buildup and thermal frequency degradation during intensive sweeps.
7. **Hardware Core Pinning**:
   - Benchmark processes are pinned to a dedicated physical core via `taskset` (configurable via `-c / --core`, with per-platform defaults such as Core 1 on x86, Core 2 on RPi5, Core 3 on M1, and Core 8 on A100) to prevent OS thread migration and cache-thrashing penalties.
8. **Numerical Accuracy Verification (`--validate`)**:
   - Includes on-the-fly verification against a pure scalar IEEE-754 FP64 reference computation before benchmarking to ensure mathematical correctness within machine precision tolerance bounds.

---

## Hardware Platforms & Compiler Toolchains

To ensure empirical reproducibility, benchmarks are compiled with modern toolchains across all evaluated platforms. The table below lists the exact compiler versions and ISA configurations on each test machine:

| Microarchitecture | Platform Tag | SoC / Model | ISA / Vector Unit | `g++` Version | `clang++` Version |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SpacemiT X100** | `x100` | SpacemiT K3 | RVV 1.0 (256-bit, VLEN=256) | 15.2.0 (Bianbu 15.2.0-16ubuntu1bb3) | 21.1.8 (Ubuntu 21.1.8-6ubuntu1) |
| **SpacemiT A100** | `a100` | SpacemiT K3 | RVV 1.0 (1024-bit via `ai`) | 15.2.0 (Bianbu 15.2.0-16ubuntu1bb3) | 21.1.8 (Ubuntu 21.1.8-6ubuntu1) |
| **SpacemiT X60** | `x60` | Banana Pi BPI-F3 | RVV 1.0 (256-bit, VLEN=256) | 15.2.0 (RISCstar 15.2-r1) | 21.1.8 (cross-built on X100)* |
| **Raspberry Pi 5** | `rpi5` | Broadcom BCM2712 (Cortex-A76) | ARMv8-A NEON (128-bit) | 15.2.0 (Ubuntu 15.2.0-16ubuntu1) | 21.1.8 (Ubuntu 21.1.8-6ubuntu1) |
| **Apple M1** | `m1` | Apple M1 Ultra (Firestorm) | ARMv8-A NEON (128-bit) | 16.1.1 (Red Hat 16.1.1-1) | 22.1.5 (Fedora 22.1.5-1.fc44) |
| **AMD Zen 4** | `zen4` | Ryzen 9 7945HX | x86_64 AVX-512 | 13.3.0 (Ubuntu 13.3.0-6ubuntu2) | 18.1.3 (Ubuntu 18.1.3-1ubuntu1) |
| **AMD Zen 5** | `zen5` | Ryzen AI 9 HX 370 | x86_64 AVX-512 (256-bit DP) | 13.3.0 (Ubuntu 13.3.0-6ubuntu2) | 18.1.3 (Ubuntu 18.1.3-1ubuntu1) |
| **Intel Meteor Lake** | `meteorlake` | Core Ultra (Redwood Cove P-core) | x86_64 AVX2 / FMA | 13.3.0 (Ubuntu 13.3.0-6ubuntu2) | 18.1.3 (Ubuntu 18.1.3-1ubuntu1) |
| **Intel Skylake** | `skylake` | Core i5-6200U (local laptop) | x86_64 AVX2 / FMA | 14.2.1 (GCC 14.2.1 20250405) | 21.1.8 (Clang 21.1.8) |

> [!NOTE]
> \* **SpacemiT X60 Clang Support**: The native distribution on the BPI-F3 board includes Bianbu Clang 18.1.8, which lacks functional RVV 1.0 intrinsic support. Because the X60 and X100 share the exact same vector ISA (`-march=rv64gcv_zvl256b -mrvv-vector-bits=zvl`) and access the shared NFS filesystem on the Dalek cluster, X60 Clang binaries are compiled on the X100 node with Ubuntu Clang 21.1.8 and then executed natively on the X60 hardware.

---

## Performance Visualizations

The executive comparison below summarizes peak double-precision efficiency achieved across all 9 target CPU microarchitectures under the **CRR layout** (GotoBLAS / BLIS column-major packed $A$ panel), grouped by SIMD family:

<p align="center">
  <img src="plots_crr/cross_uarch_crr_efficiency_comparison.svg" alt="GEMMBench Cross-Microarchitecture CRR Efficiency Comparison" width="100%">
</p>

Detailed individual microarchitecture breakdown plots and RRR comparisons can be generated locally via the plotting suite (see [Plotting & Performance Analysis](#plotting--performance-analysis) below).

---

## Prerequisites & Building

### 1. Requirements
- **C++20 Compiler**: `g++` (>= 12.0) or `clang++` (>= 15.0).
- **CMake**: version 3.20 or newer.
- **Python**: 3.9+ with `matplotlib`, `pandas`, `numpy`.
- **MIPPv2 Headers**: Obtain the headers from the [MIPP develop branch](https://github.com/aff3ct/MIPP/tree/develop) or download from [MIPP Releases](https://github.com/aff3ct/MIPP/releases).

### 2. Standalone Build
```bash
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DMIPPV2_INCLUDE_DIR=/path/to/mipp/include

cmake --build build -j$(nproc)
```

### 3. Build and Run Unit Tests
Unit tests use Catch2 v3 (automatically fetched if not present on system):
```bash
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DMIPPV2_INCLUDE_DIR=/path/to/mipp/include \
  -DGEMMBENCH_BUILD_TESTS=ON

cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

---

## Running Benchmarks

Benchmarks are driven by `./run_benchmarks.sh <platform> [options] [compilers...]`.

Always specify `-o <uarch>_<simd>` to ensure non-colliding, standardized result files:

```bash
# Benchmark Intel Skylake (AVX2) with GCC and Clang
./run_benchmarks.sh avx2 -o skylake_avx2 gcc clang

# Benchmark Intel Meteor Lake (AVX2) on specific matrix sizes
./run_benchmarks.sh avx2 -o meteorlake_avx2 -s 32 -s 64 -s 96 -s 128 gcc clang

# Benchmark AMD Zen 4 (AVX-512) pinned to core 0 with 15 repetitions and 3 runs
./run_benchmarks.sh avx512 -o zen4_avx512 -c 0 -r 15 --runs 3 gcc clang

# Benchmark SpacemiT X100 with clean rebuild
./run_benchmarks.sh rvv_x100 -o x100_rvv --rebuild clang

# Benchmark CRR kernels only (BLIS-like K sweep on native MR×NR tile)
./run_benchmarks.sh avx512 --crr -o zen4_avx512 gcc clang
```

### Benchmark Driver CLI Options

| Flag | Argument | Description | Default |
| :--- | :--- | :--- | :--- |
| `-o, --output, --prefix` | `<prefix>` | Suffix tag for generated CSV files (`results/gemm_results_<prefix>_...`) | `""` |
| `--crr, --crr-only` | — | Benchmark only CRR microkernels (GotoBLAS / BLIS register tile layout) | `false` |
| `--rrr, --rrr-only` | — | Benchmark only legacy RRR baseline microkernels | `false` |
| `-r, --repetitions` | `<N>` | Number of timed repetitions per size/kernel sample | `15` |
| `--runs, --repeats` | `<N>` | Number of full benchmark passes to compute inter-run CV | `3` |
| `--cooldown` | `<sec>` | Cooldown sleep between sizes for $K \ge 256$ (thermal throttle mitigation) | `2.0` |
| `--validate` | — | Validate numerical correctness against scalar reference before benchmarking | `false` |
| `-c, --core` | `<id>` | Pin benchmark execution to a specific CPU core via `taskset` | Config default (e.g. core 1) |
| `--no-pin` | — | Disable core affinity pinning | `false` |
| `-s, --size` | `<K>` | Evaluate specific panel depth $K$ (can be repeated) | Config defaults |
| `-k, --kernel` | `<name>` | Benchmark only a specific microkernel | All in config |
| `--explo` | — | Enable exploratory kernels (`-DGEMMBENCH_ENABLE_EXPLO=ON`) | `false` |
| `--run-all` | — | Run all versions (including scalar baseline) and exploratory kernels | `false` |
| `--rebuild` | — | Force clean rebuild of target executables before benchmarking | `false` |
| `--tests` | — | Build and run Catch2 unit tests before benchmarking | `false` |
| `--tests-only` | — | Build and run Catch2 unit tests only, then exit | `false` |

---

## Plotting & Performance Analysis

All figures are generated as clean, publication-ready vector **SVG** graphics with automatically trimmed bounding boxes (`bbox_inches="tight"`).

### RRR Plots

The analytical suite `plot_results.py` reads the hierarchical hardware specification in `uarch_config.json`, scans the benchmark datasets, identifies the champion kernel for each microarchitecture, and outputs comparison plots:

```bash
# Generate all RRR plots from reference datasets
python3 plot_results.py --input-dir gemm-bench-results --output plots

# Filter specific architectures or SIMD extensions
python3 plot_results.py --input-dir gemm-bench-results --uarch zen4 zen5 m1
python3 plot_results.py --input-dir gemm-bench-results --simd avx512 neon

# Override clock frequencies (in GHz) for dynamic boost clock analysis
python3 plot_results.py --input-dir gemm-bench-results --freq zen4=5.4,m1=3.0
```

### CRR Plots

`plot_results_crr.py` analyzes CRR micro-kernel benchmark results with automatic champion selection and FLOP/cycle efficiency. It can optionally generate side-by-side comparisons against RRR baselines:

```bash
# Generate all CRR plots (auto-detects gemm-bench-results/crr/)
python3 plot_results_crr.py --output plots_crr

# With side-by-side CRR vs RRR comparison
python3 plot_results_crr.py --output plots_crr --compare-rrr gemm-bench-results

# Filter specific architectures
python3 plot_results_crr.py --uarch zen4 m1 x100 --output plots_crr
```

### Output Figures
- **`plots/`** (RRR):
  - `cross_uarch_efficiency_comparison.svg`: Multi-architecture comparison in **% of theoretical peak**.
  - `cross_uarch_flop_cycle_comparison.svg`: Multi-architecture comparison in **DP FLOP/cycle**.
  - `<simd>_<uarch>_overview.svg`: Per-architecture detailed breakdown.
- **`plots_crr/`** (CRR):
  - `cross_uarch_crr_efficiency_comparison.svg`: CRR efficiency across architectures.
  - `cross_uarch_crr_flop_cycle_comparison.svg`: CRR FLOP/cycle comparison.
  - `cross_uarch_crr_vs_rrr_comparison.svg`: CRR vs RRR speedup (with `--compare-rrr`).
  - `<simd>_<uarch>_crr_overview.svg`: Per-architecture CRR breakdown.

---

## Developer Tutorial: Adding a New Microkernel

This guide walks through implementing, registering, testing, benchmarking, and analyzing a new GEMM microkernel across the entire pipeline. The repository supports two principal layouts:
- **CRR (Recommended / State-of-the-Art)**: Column-major packed $A$ ($M_R \times K$, leading dimension $M_R$), Row-major $B$ ($K \times N_R$), Row-major $C$ ($M_R \times N_R$). This matches the GotoBLAS / BLIS register-tiled microkernel convention.
- **RRR (Legacy)**: Row-major packed $A, B, C$.

---

### Step 1: Implement the Microkernel in C++

CRR microkernels are implemented in [`include/microkernels/GemmUKernel_opti_crr.hpp`](include/microkernels/GemmUKernel_opti_crr.hpp). RRR microkernels reside in [`include/microkernels/GemmUKernel_opti_rrr.hpp`](include/microkernels/GemmUKernel_opti_rrr.hpp). Exploratory kernels go in [`include/microkernels/GemmUKernel_explo.hpp`](include/microkernels/GemmUKernel_explo.hpp).

#### 1. Choose Register Tile Geometry ($M_R \times N_R$)
- $M_R$ rows $\times$ $N_R$ columns, with $N_R = N_V \times VL$ ($N_V$ vectors wide, $VL = \text{vlen}$).
- **Accumulator vector registers**: $M_R \times N_V = M_R \times \lceil N_R / VL \rceil$.
  - Example: a $4 \times 3VL$ tile on AVX2 ($M_R=4, N_V=3$, 4 doubles/vector): $4 \times 3 = 12$ accumulator registers, leaving 4 registers for $B$ broadcast operands and memory addresses within the 16 architectural registers of x86-64.
  - On 32-register architectures (AVX-512, AArch64 NEON, RVV), tiles like $4 \times 4VL$ (16 accumulators) or $6 \times 4VL$ (24 accumulators) provide optimal pipeline depth.
- Ensure the accumulator budget plus operand registers fits within architectural limits with zero stack spilling.

#### 2. CRR Implementation Pattern
A production CRR microkernel features:
1. **Single-Tile Fast Path**: Dedicated compute loop for $M = M_R$ and $N = N_R$, unrolled along $K$ (typically 4-way with `#pragma GCC unroll 4`) with a scalar cleanup loop for remainder $K \pmod 4$.
2. **Multi-Tile Fallback & Fringes**: Outer $i, j$ loops with edge handling for complete matrix operations.
3. **Epilogue Vector Stores**: `Epilogue::store<T, lmul, FastPath>()` handling both the canonical fast path ($\alpha = 1, \beta = 0$) and full $(\alpha, \beta)$ scaling/accumulation.

```cpp
// In include/microkernels/GemmUKernel_opti_crr.hpp (inside template <typename T> class GemmUKernel):
template <int lmul = 1, bool FastPath = false>
static inline void
gemm_mippv2_mykernel_crr_core(const PackedColMajor<T> &__restrict A,
                              const PackedRowMajor<T> &__restrict B,
                              PackedRowMajor<T> &__restrict C,
                              T alpha = T{1}, T beta = T{0}) {
  using namespace mipp;

  constexpr size_t VL = N<T, lmul>();
  constexpr size_t MR = 4;
  constexpr size_t NR = 3 * VL;

  const size_t a_ld = A.ld;
  const size_t b_ld = B.ld;
  const size_t c_ld = C.ld;
  const size_t K = A.cols;

  // 1. Fast Path: Single micro-tile (in-cache microkernel benchmark mode)
  if (__builtin_expect(A.rows == MR && B.cols == NR, 1)) {
    const T *__restrict a_k = A.data;
    const T *__restrict b_k = B.data;

    auto c00 = set0<T, lmul>(); auto c01 = set0<T, lmul>(); auto c02 = set0<T, lmul>();
    auto c10 = set0<T, lmul>(); auto c11 = set0<T, lmul>(); auto c12 = set0<T, lmul>();
    auto c20 = set0<T, lmul>(); auto c21 = set0<T, lmul>(); auto c22 = set0<T, lmul>();
    auto c30 = set0<T, lmul>(); auto c31 = set0<T, lmul>(); auto c32 = set0<T, lmul>();

    const size_t K4 = (K / 4) * 4;
    for (size_t k = 0; k < K4; k += 4) {
      #pragma GCC unroll 4
      for (size_t s = 0; s < 4; ++s) {
        const auto b0 = load<T, lmul>(b_k + s * b_ld);
        const auto b1 = load<T, lmul>(b_k + s * b_ld + VL);
        const auto b2 = load<T, lmul>(b_k + s * b_ld + 2 * VL);

        const T *a_s = a_k + s * a_ld;
        const auto a0 = set1<T, lmul>(a_s[0]);
        c00 = fmadd(a0, b0, c00); c01 = fmadd(a0, b1, c01); c02 = fmadd(a0, b2, c02);

        const auto a1 = set1<T, lmul>(a_s[1]);
        c10 = fmadd(a1, b0, c10); c11 = fmadd(a1, b1, c11); c12 = fmadd(a1, b2, c12);

        const auto a2 = set1<T, lmul>(a_s[2]);
        c20 = fmadd(a2, b0, c20); c21 = fmadd(a2, b1, c21); c22 = fmadd(a2, b2, c22);

        const auto a3 = set1<T, lmul>(a_s[3]);
        c30 = fmadd(a3, b0, c30); c31 = fmadd(a3, b1, c31); c32 = fmadd(a3, b2, c32);
      }
      b_k += 4 * b_ld;
      a_k += 4 * a_ld;
    }

    // Remainder loop along K
    for (size_t k = K4; k < K; ++k) {
      const auto b0 = load<T, lmul>(b_k);
      const auto b1 = load<T, lmul>(b_k + VL);
      const auto b2 = load<T, lmul>(b_k + 2 * VL);
      b_k += b_ld;

      const auto a0 = set1<T, lmul>(a_k[0]);
      c00 = fmadd(a0, b0, c00); c01 = fmadd(a0, b1, c01); c02 = fmadd(a0, b2, c02);
      // ... repeat for a1, a2, a3 ...
      a_k += a_ld;
    }

    // Write back via epilogue
    Epilogue::store<T, lmul, FastPath>(C[0] + 0 * VL, c00, alpha, beta);
    Epilogue::store<T, lmul, FastPath>(C[0] + 1 * VL, c01, alpha, beta);
    Epilogue::store<T, lmul, FastPath>(C[0] + 2 * VL, c02, alpha, beta);
    // ... repeat stores for rows C[1], C[2], C[3] ...
    return;
  }

  // 2. Multi-tile loops and tail cleanup (for full matrix support) ...
}

// Dispatch wrappers:
template <int lmul = 1>
static inline void
gemm_mippv2_mykernel_crr(const PackedColMajor<T> &__restrict A,
                         const PackedRowMajor<T> &__restrict B,
                         PackedRowMajor<T> &__restrict C) {
  gemm_mippv2_mykernel_crr_core<lmul, true>(A, B, C);
}

template <int lmul = 1>
static inline void
gemm_mippv2_mykernel_crr(const PackedColMajor<T> &__restrict A,
                         const PackedRowMajor<T> &__restrict B,
                         PackedRowMajor<T> &__restrict C,
                         T alpha, T beta) {
  if (__builtin_expect(alpha == T{1} && beta == T{0}, 1)) {
    gemm_mippv2_mykernel_crr_core<lmul, true>(A, B, C);
  } else {
    gemm_mippv2_mykernel_crr_core<lmul, false>(A, B, C, alpha, beta);
  }
}
```

---

### Step 2: Register in `include/microkernels/GemmUKernel.h`

1. **Add to `UKernelType`**:
   ```cpp
   enum class UKernelType {
     // ...
     mippv2_mykernel_crr,
   };
   ```

2. **Add Descriptor to `kernelTable`**:
   ```cpp
   {UKernelType::mippv2_mykernel_crr, "mippv2_mykernel_crr", CRR, 64, 4, 3},
   ```
   - **Packing constant**: `CRR` (`{PLayout::Col, PLayout::Row, PLayout::Row}`) for Col-Row-Row, or `RRR` for Row-Row-Row.
   - **Tile parameters**: `MR` (e.g. `4`) and `NR_vec` (e.g. `3` for $3 \times VL$). These values are used by `src/main.cpp` and `run_benchmarks.sh` to configure auto-sizing in CRR mode ($M = M_R, N = N_R \times VL$).

---

### Step 3: Wire Dispatch in `src/main.cpp`

[`src/main.cpp`](src/main.cpp) defines **three distinct overloads** of `run()` based on matrix layout combinations:
1. `run(cfg, gemm, PackedRowMajor<T> A, PackedRowMajor<T> B, PackedRowMajor<T> C)` $\rightarrow$ **RRR** kernels.
2. `run(cfg, gemm, PackedRowMajor<T> A, PackedColMajor<T> B, PackedRowMajor<T> C)` $\rightarrow$ **RCR** kernels.
3. `run(cfg, gemm, PackedColMajor<T> A, PackedRowMajor<T> B, PackedRowMajor<T> C)` $\rightarrow$ **CRR** kernels.

> [!CAUTION]
> You **must** add your CRR kernel dispatch branch inside the **3rd overload** (`PackedColMajor<T> A`):
> ```cpp
> case UKernelType::mippv2_mykernel_crr:
>   BENCH_KERNEL_FASTPATH(
>       gemm.gemm_mippv2_mykernel_crr(A, B, C),
>       gemm.gemm_mippv2_mykernel_crr(A, B, C, cfg.alpha, cfg.beta));
> ```
> Adding a CRR kernel to the RRR overload will cause a runtime exception (`"Kernel incompatible with RRR packing"`).

---

### Step 4: Add Unit Tests in `tests/GemmUKernelTest.cpp`

Unit tests use Catch2 v3. For a CRR kernel, register test cases in the CRR sections of [`tests/GemmUKernelTest.cpp`](tests/GemmUKernelTest.cpp):

1. **Nominal & Edge Dimension Tests**:
   ```cpp
   // Inside testGemmSuite<T, PackedColMajor<T>, PackedRowMajor<T>, PackedRowMajor<T>>()
   TEST_KERNEL("MIPPv2 MyKernel CRR",
               gemm.gemm_mippv2_mykernel_crr(A, B, C_test));
   ```
2. **$\alpha, \beta$ Scaling & Accumulation Tests**:
   ```cpp
   // Inside testGemmSuiteAlphaBeta() for CRR
   SECTION("mippv2_mykernel_crr") {
     resetMatrix(C_test, C_init);
     gemm.gemm_mippv2_mykernel_crr(A, B, C_test, p.alpha, p.beta);
     checkMatrixEqual(C_ref, C_test);
   }
   ```
3. **Execute the Test Suite**:
   ```bash
   cmake -B build -DGEMMBENCH_BUILD_TESTS=ON && cmake --build build
   ctest --test-dir build --output-on-failure
   ```

---

### Step 5: Benchmark Driver Integration

1. In [`run_benchmarks.sh`](run_benchmarks.sh), add the kernel identifier to `UNIVERSAL_KERNELS`:
   ```bash
   mippv2_mykernel_crr
   ```
2. If the kernel is designated as the champion for an architecture, update `TARGET_CHAMPIONS` in [`tools/run_cluster_benchmarks.sh`](tools/run_cluster_benchmarks.sh):
   ```bash
   [myarch]="mippv2_mykernel_rrr mippv2_mykernel_crr"
   ```
3. Run the hardened benchmark driver locally or on cluster nodes:
   ```bash
   # Benchmark CRR microkernel only (K sweep on native register tile)
   ./run_benchmarks.sh avx2 -o myarch_avx2 --crr-only -k mippv2_mykernel_crr gcc clang
   ```

---

### Step 6: Configure in `uarch_config.json` & Plot

In [`uarch_config.json`](uarch_config.json), configure the expected champion kernels and CSV patterns:
```json
"myarch": {
  "name": "My Architecture",
  "cpu_model": "Core Model",
  "frequency_ghz": 3.5,
  "peak_flop_per_cycle": 16.0,
  "expected_best_kernel": "mippv2_mykernel",
  "expected_best_kernel_crr": "mippv2_mykernel_crr",
  "csv_pattern": "*gemm_results_*myarch*_{compiler}.csv",
  "crr_csv_pattern": "*gemm_results_*myarch*crr*_{compiler}.csv"
}
```

Generate updated publication figures:
```bash
# Generate CRR overview, cross-uarch comparison, and RRR speedup SVG plots:
python3 plot_results_crr.py --output plots_crr --compare-rrr gemm-bench-results/rrr
```

---

**Current Focus & Roadmap (Prioritized TODO)**:
1. **Direct Microkernel Comparison Against BLIS (`bli_dgemm_ukernel_*`)**: Integrate native BLIS microkernels directly into the `GemmBench` in-cache harness (under iso-packing Col-Row-Row layouts) across supported microarchitectures (x86_64, AArch64) to quantify the exact performance delta between MIPPv2 C++ microkernels and hand-tuned assembly microkernels.
2. **Cluster-Wide Benchmark Refresh (Dalek)**: Re-run the full multi-architecture sweep under the hardened protocol ($R = 15$, $N_{\text{runs}} = 3$, passive throttling checks) to update all reference datasets.
3. **Single-Precision Support (SGEMM / FP32)**: Extend the benchmark suite and microkernel generators to evaluate FP32 as well.
4. **Comparison with Hand-Written Native Intrinsics**: Implement intrinsics baselines (native RVV `riscv_vector.h`, AVX-512 `immintrin.h`, NEON `arm_neon.h`) to measure the exact abstraction overhead introduced by MIPPv2.
5. **Cross-SIMD Abstraction Comparisons**: Benchmark against alternative SIMD abstraction frameworks, like **Google Highway**, **EVE**, and **`std::simd`**.
6. **Non-Canonical GEMM Evaluation**: Benchmark throughput and epilogue overhead for non-canonical forms ($\beta \neq 0.0$, $\alpha \neq 1.0$) to quantify the cost of $C$ tile reloads and scaling.

> [!NOTE]
> **Completed Milestones**:
> - Empirical hardware FPU ceiling and sustained clock calibration via `cpufp` integrated into `uarch_config.json`.
> - Statistical robustness framework (median filtering across 15 repetitions, 3-run replication with inter-run CV tracking, compiler memory barriers, passive CPU throttling monitoring, thermal cooldown pauses).

> [!NOTE]
> BLIS/OpenBLAS for RVV: Ongoing vendor and community work on RVV-optimized BLAS kernels reportedly outperforms current upstream public releases of BLIS and OpenBLAS, but these patches are not yet fully merged/streamlined in mainline distributions. Benchmarking methodology should evaluate both upstream releases and patched branches.

---

## Acknowledgments

- **Adrien Cassagne** ([@kouchy](https://github.com/kouchy/)) for access to the single-board computers (SBCs) and nodes on the [Dalek Cluster](https://dalek.proj.lip6.fr/) (*"Dalek: An Unconventional and Energy-Aware Heterogeneous Cluster"*, [arXiv:2508.10481](https://arxiv.org/abs/2508.10481)).
- **SpacemiT**, for the [SpacemiT K3 Technical Whitepaper](https://forum.spacemit.com/uploads/short-url/60aJ8cYNmrFWqHn4ddwwSzMLjlY.pdf).
- **camel-cdr**, for the [RVV Benchmark Results](https://camel-cdr.github.io/rvv-bench-results/index.html) project.
- The authors of *["Great Expectations: Benchmarking the Real-World Performance of RVV 1.0 in HPC"](https://arxiv.org/abs/2608.28097)* ([arXiv:2608.28097](https://arxiv.org/abs/2608.28097)).

---

## Generative AI Disclosure

Generative AI assistants (**Google Gemini** and **OpenAI ChatGPT**) were utilized during the development of this repository to accelerate boilerplate tasks and infrastructure setup, specifically:
- Mechanically generating C++ microkernel implementations across varying register tile geometries ($M_R \times N_R$) and accumulator layouts.
- Automation scripts (shell runners and Python data parsing / plotting pipelines).
- Benchmark driver CLI scaffolding, argument handling, and boilerplate wiring in C++.
- Build system configuration and test suite plumbing.

### Engineering Methodology & Kernel Design

The foundational DGEMM microkernel designs tuned for **Intel Skylake** (AVX2 + FMA) and the **SpacemiT K3 X100** core are the result of multiple weeks of work and reflection on the underlying microarchitectures and platforms. And the best **X100** microkernels are, I think, some of the best results of my internship.

For subsequent target microarchitectures, the methodology was empirical, the exploration workflow relied on pre-established invariants:
- **Layout Exploration**: Both the $RRR$ format (Row-major $A, B, C$) and the $CRR$ format (Column-major $A$, Row-major $B, C$) are implemented. CRR uses the GotoBLAS / BLIS packing convention where $A$ is stored contiguously along $M_R$ with leading dimension $= M_R$, matching the inner kernel's access pattern. CRR microkernels achieve comparable or higher peak efficiency than RRR on most microarchitectures (up to **99.4%** on M1).
- **Inner-Loop Access Patterns**: Preserving either the scalar broadcast pattern (`mipp::set1` + `mipp::fmadd`) or the indexed FMA pattern (`mipp::fmaddi`) on matrix $A$.

Under these, the exploration focused on sweeping register block dimensions ($M_R \times N_R$) that made sense given the uarch characteristics. Register file size, number of FMA pipelines/execution units and FMA latency being the three things that answer the questions :
- How many accumulators can I have at most? 
- How many accumulators do I want? 
- How many accumulators do I need at least?

In practice, it often boiled down to telling Gemini to implement a bunch of tile sizes I thought might yield good results. Then I ran the benchmarks on the target hardware and iterated over the ones that performed well "Gemini please make `gemm_mippv2_firestorm_mr8_nr2_fmaddi` using `gemm_mippv2_firestorm_mr6_nr4_fmaddi` as reference" and so on. I feel like the most inefficient part of the automation was sometimes the agent sitting between the keyboard and the chair. Oh well.
