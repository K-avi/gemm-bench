# GEMM Benchmark Suite across Modern Microarchitectures (MIPPv2)

> [!WARNING]
> **Work in Progress (WIP)**: This repository is under active development. Microkernels, benchmark drivers, and analysis tools are continuously refined.

A high-performance General Matrix Multiply (DGEMM) microbenchmark and exploration suite built on top of [MIPPv2](https://github.com/aff3ct/MIPP/tree/develop) (MyIntrinsics++ v2).

This suite systematically evaluates microarchitectural efficiency, vector register tiling strategies, and compiler code generation across **9 distinct CPU microarchitectures** spanning **x86_64** (AVX2, AVX-512), **AArch64** (NEON), and **RISC-V** (RVV 1.0).

---

## Microarchitectural Performance & Peak Efficiency

The benchmark evaluates single-thread, compute-bound double-precision GEMM throughput against the hardware theoretical peak ($\text{FLOP/cycle} \times \text{Frequency}$). Across modern out-of-order execution engines, the tuned microkernels achieve **90% to 99% of theoretical hardware saturation**:

| SIMD | Microarchitecture | CPU / SoC Model | Freq (GHz) | Peak FLOP/cyc | Champion Kernel | Compiler | Peak GFLOP/s | FLOP/cyc | % of Peak |
| :--- | :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **AVX-512** | **AMD Zen 4** | Ryzen 9 7900X | 5.4 | 16.0 | `mippv2_firestorm_mr4_nr4_fmaddi` | GCC | 85.4 | 15.82 | **98.9 %** |
| **AVX-512** | **AMD Zen 5 Strix Point** | Ryzen AI 9 HX 370 | 5.1 | 16.0 | `mippv2_firestorm_mr4_nr4_fmaddi` | GCC | 80.4 | 15.77 | **98.6 %** |
| **NEON** | **Apple M1 Firestorm** | M1 Ultra (P-core) | 3.0 | 16.0 | `mippv2_x60_mr6_nr4_fmaddi` | GCC | 47.5 | 15.83 | **98.9 %** |
| **NEON** | **Raspberry Pi 5** | Cortex-A76 (BCM2712) | 2.4 | 8.0 | `mippv2_a76_mr6_nr3` | GCC | 18.0 | 7.49 | **93.6 %** |
| **RVV 1.0** | **SpacemiT X100** | K3 SoC (VLEN=256) | 2.2 | 8.0 | `mippv2_x100_register_blocked_apack4` | Clang | 16.3 | 7.42 | **92.7 %** |
| **AVX2** | **Intel Meteor Lake** | Redwood Cove P-Core | 5.1 | 16.0 | `mippv2_meteorlake_mr4_nr3` | GCC | 74.8 | 14.66 | **91.6 %** |
| **AVX2** | **Intel Skylake** | Core i5-6200U | 2.3 | 16.0 | `mippv2_meteorlake_mr4_nr3` | Clang | 33.4 | 14.53 | **90.8 %** |
| **RVV 1.0** | **SpacemiT A100** | K3 SoC (VLEN=1024) | 1.8 | 8.0 | `mippv2_a100_mr7_nr2_lmul2_pipe` | GCC | 11.7 | 6.48 | **81.0 %** |
| **RVV 1.0** | **SpacemiT X60** | BPI-F3 SoC (VLEN=256)| 1.6 | 8.0 | `mippv2_a100_mr7_nr4_pipe` | GCC | 6.2 | 3.89 | **48.7 %** |

> [!NOTE]
> **Zen 5 Strix Point Datapath Note**:
> While server/desktop Zen 5 (Turin / Granite Ridge) incorporates dual native 512-bit FMA units ($32\text{ DP FLOP/cycle}$), AMD's mobile Strix Point SoC implements dual 256-bit physical datapaths (*double-pumped* 512-bit FMA, identical to Zen 4), yielding a physical ceiling of **$16\text{ DP FLOP/cycle}$**. At $15.77\text{ FLOP/cycle}$, the execution pipes are fully saturated.

---

## Repository Structure

```text
gemm_bench/
├── bench_configs/              # Per-platform compilation flags and hardware settings
│   ├── avx2.sh                 # x86_64 AVX2 / FMA configuration
│   ├── avx512.sh               # x86_64 AVX-512 configuration
│   ├── neon.sh                 # ARMv8-A NEON configuration (M1, Cortex-A76)
│   ├── rvv_x100.sh             # SpacemiT X100 (256-bit RVV)
│   ├── rvv_x60.sh              # SpacemiT X60 (256-bit RVV)
│   └── rvv_a100.sh             # SpacemiT A100 (1024-bit RVV)
├── gemm-bench-results/         # Reference benchmark CSV datasets (<uarch>_<simd>_<compiler>.csv)
├── include/
│   ├── Alloc.h                 # Cache-aligned SIMD memory allocators (std::aligned_alloc)
│   ├── BenchConfig.h           # Benchmark CLI configuration parser
│   ├── Epilogue.hpp            # In-place C accumulation epilogue (alpha, beta, C update)
│   ├── GemmUKernel.h           # Microkernel declarations, descriptors, and dispatch tables
│   ├── GemmUKernel_opti.hpp    # Optimized architecture-specific microkernels (AVX, NEON, RVV)
│   ├── GemmUKernel_explo.hpp   # Experimental and exploratory microkernels
│   ├── StorageType.h           # Matrix storage wrappers (RowMajor, ColMajor, Packed)
│   └── mipp_custom.h           # Custom vector-scalar FMA extensions (vfmacc.vf)
├── plots/                      # Generated vector graphics (SVG)
├── src/
│   └── main.cpp                # Benchmark runner driver
├── tests/                      # Catch2 unit tests (accuracy, non-square, alignment)
├── tools/
│   └── run_cluster_benchmarks.sh # Slurm multi-node cluster runner (Dalek LIP6)
├── plot_results.py             # Analytical plotting suite (FLOP/cyc, % of peak, champion detection)
├── run_benchmarks.sh           # Unified benchmark driver
├── uarch_config.json           # Microarchitectural hardware specs & expected champions
└── CMakeLists.txt              # Build configuration
```

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

### Local Execution (`run_benchmarks.sh`)
Benchmarks are driven by `./run_benchmarks.sh <platform> [options] [compilers...]`.

Always specify `-o <uarch>_<simd>` to ensure non-colliding, standardized result files:

```bash
# Benchmark Intel Skylake (AVX2) with GCC and Clang
./run_benchmarks.sh avx2 -o skylake_avx2 gcc clang

# Benchmark Intel Meteor Lake (AVX2) on specific matrix sizes
./run_benchmarks.sh avx2 -o meteorlake_avx2 -s 32 -s 64 -s 96 -s 128 gcc clang

# Benchmark AMD Zen 4 (AVX-512) pinned to core 0
./run_benchmarks.sh avx512 -o zen4_avx512 -c 0 gcc clang

# Benchmark SpacemiT X100 with clean rebuild
./run_benchmarks.sh rvv_x100 -o x100_rvv --rebuild clang
```

Result files are saved in `results/gemm_results_<prefix>_<compiler>.csv` (e.g. `results/gemm_results_skylake_avx2_gcc.csv`).

### Remote Slurm Cluster Execution (`tools/run_cluster_benchmarks.sh`)
When benchmarking across heterogeneous cluster nodes (such as the LIP6 Dalek cluster):
```bash
# Run benchmarks on Raspberry Pi 5 and Zen 5 in parallel
./tools/run_cluster_benchmarks.sh rpi5 zen5

# Run only Catch2 unit tests on all available cluster nodes
./tools/run_cluster_benchmarks.sh all --tests-only

# Run custom matrix sizes sequentially
./tools/run_cluster_benchmarks.sh --sequential -s "32 64 96 128" meteorlake zen4
```

---

## Plotting & Performance Analysis

The analytical suite [`plot_results.py`](file:///home/ivan/Files/projets/gemm_bench/plot_results.py) reads the hierarchical hardware specification in [`uarch_config.json`](file:///home/ivan/Files/projets/gemm_bench/uarch_config.json), scans the benchmark datasets, identifies the champion kernel for each microarchitecture, and outputs comparison plots:

```bash
# Generate all plots from reference datasets
python3 plot_results.py --input-dir gemm-bench-results --output plots

# Filter specific architectures or SIMD extensions
python3 plot_results.py --input-dir gemm-bench-results --uarch zen4 zen5 m1
python3 plot_results.py --input-dir gemm-bench-results --simd avx512 neon

# Override clock frequencies (in GHz) for dynamic boost clock analysis
python3 plot_results.py --input-dir gemm-bench-results --freq zen4=5.4,m1=3.0
```

### Output Figures (`plots/`)
- `cross_uarch_efficiency_comparison.svg`: Multi-architecture comparison in **% of theoretical peak** ($\log_2$ scale).
- `cross_uarch_flop_cycle_comparison.svg`: Multi-architecture comparison in **DP FLOP/cycle**.
- `<simd>_<uarch>_overview.svg`: Per-architecture detailed breakdown showing top kernels, scalar baseline, and dual-axis throughput (GFLOP/s and FLOP/cycle).

---

## Developer Tutorial: Adding a New Microkernel

This end-to-end tutorial demonstrates how to implement, register, test, benchmark, and analyze a new GEMM microkernel across the entire pipeline.

### Step 1: Implement the Kernel in C++
Microkernels are implemented in [`include/GemmUKernel_opti.hpp`](file:///home/ivan/Files/projets/gemm_bench/include/GemmUKernel_opti.hpp) (or [`include/GemmUKernel_explo.hpp`](file:///home/ivan/Files/projets/gemm_bench/include/GemmUKernel_explo.hpp) for exploratory variants).

1. **Choose Register Tile Geometry ($M_R \times N_R$)**:
   - Accumulator registers required = $M_R \times \lceil N_R / \text{vlen} \rceil$.
   - For example, a $4 \times 3$ tile (4 rows, 3 vectors wide): 12 accumulator registers.
   - Ensure the total register budget ($M_R \times N_R/\text{vlen} + \text{operands}$) fits within the architectural register file (16 registers on AVX2, 32 on AVX-512 / NEON / RVV) with zero spilling.

2. **Inner Loop Pattern**:
   ```cpp
   template <typename T>
   static inline void gemm_myarch_mr4_nr3(
       const PackedRowMajor<T> &__restrict A,
       const PackedRowMajor<T> &__restrict B,
       PackedRowMajor<T> &__restrict C,
       const T alpha = 1.0,
       const T beta = 0.0)
   {
       constexpr int MR = 4;
       constexpr int NR_VEC = 3; // 3 SIMD vectors wide
       constexpr int VEC_SIZE = mipp::N<T>();

       for (int i = 0; i < A.rows; i += MR) {
           for (int j = 0; j < B.cols; j += NR_VEC * VEC_SIZE) {
               // 1. Initialize accumulators
               mipp::reg c[MR][NR_VEC];
               for (int r = 0; r < MR; ++r)
                   for (int v = 0; v < NR_VEC; ++v)
                       c[r][v] = mipp::setzero<T>();

               // 2. Compute-bound inner accumulation loop over K
               for (int k = 0; k < A.cols; ++k) {
                   // Broadcast A elements
                   mipp::reg a0 = mipp::set1<T>(A(i + 0, k));
                   mipp::reg a1 = mipp::set1<T>(A(i + 1, k));
                   mipp::reg a2 = mipp::set1<T>(A(i + 2, k));
                   mipp::reg a3 = mipp::set1<T>(A(i + 3, k));

                   // Load B vectors and issue FMAs
                   for (int v = 0; v < NR_VEC; ++v) {
                       mipp::reg b_vec = mipp::load<T>(&B(k, j + v * VEC_SIZE));
                       c[0][v] = mipp::fmadd(a0, b_vec, c[0][v]);
                       c[1][v] = mipp::fmadd(a1, b_vec, c[1][v]);
                       c[2][v] = mipp::fmadd(a2, b_vec, c[2][v]);
                       c[3][v] = mipp::fmadd(a3, b_vec, c[3][v]);
                   }
               }

               // 3. Epilogue: write back with alpha and beta scaling
               // (see include/Epilogue.hpp for standard accumulation epilogue)
           }
       }
   }
   ```

### Step 2: Register in `include/GemmUKernel.h`
1. Add the enum identifier in `UKernelType`:
   ```cpp
   enum class UKernelType {
       // ...
       mippv2_myarch_mr4_nr3,
   };
   ```

2. Add a `KernelDescriptor` to `allDescriptors`:
   ```cpp
   {UKernelType::mippv2_myarch_mr4_nr3, "mippv2_myarch_mr4_nr3", RRR, 64},
   ```
   - Specify layout: `RRR` (Row-major $A, B, C$) or `CRR` (Col-major $A$, Row-major $B, C$).
   - Specify default tile size: typically `64` (or `66` for SpacemiT X100).

3. Declare the dispatch wrapper in the `Gemm` struct:
   ```cpp
   void gemm_mippv2_myarch_mr4_nr3(const PackedRowMajor<T> &A,
                                   const PackedRowMajor<T> &B,
                                   PackedRowMajor<T> &C,
                                   const T alpha = 1.0,
                                   const T beta = 0.0) {
       gemm_myarch_mr4_nr3(A, B, C, alpha, beta);
   }
   ```

### Step 3: Wire in `src/main.cpp`
Add the kernel dispatch branch in [`src/main.cpp`](file:///home/ivan/Files/projets/gemm_bench/src/main.cpp#L125) using `BENCH_KERNEL_FASTPATH`:
```cpp
BENCH_KERNEL_FASTPATH(UKernelType::mippv2_myarch_mr4_nr3, gemm_mippv2_myarch_mr4_nr3);
```

### Step 4: Add Unit Tests in `tests/GemmUKernelTest.cpp`
Add validation test cases for $C = A \times B$ and $C = \alpha (A \times B) + \beta C$:
```cpp
TEST_KERNEL("MIPPv2 MyArch MR4 NR3",
            gemm.gemm_mippv2_myarch_mr4_nr3(A, B, C_test));
```
Run `ctest --test-dir build --output-on-failure` to verify correctness and numerical precision.

### Step 5: Shell Driver Integration (`run_benchmarks.sh`)
In [`run_benchmarks.sh`](file:///home/ivan/Files/projets/gemm_bench/run_benchmarks.sh), add the kernel name to the corresponding platform's `KERNELS` list:
```bash
    mippv2_myarch_mr4_nr3
```

### Step 6: Benchmark Execution
Execute the benchmark specifying the output prefix:
```bash
./run_benchmarks.sh avx2 -o myarch_avx2 -k mippv2_myarch_mr4_nr3 gcc clang
```

### Step 7: Configure in `uarch_config.json` & Plot
In [`uarch_config.json`](file:///home/ivan/Files/projets/gemm_bench/uarch_config.json), add or update the target entry:
```json
"myarch": {
  "name": "My Architecture",
  "cpu_model": "Custom Core",
  "frequency_ghz": 3.5,
  "peak_flop_per_cycle": 16.0,
  "expected_best_kernel": "mippv2_myarch_mr4_nr3",
  "csv_pattern": "*gemm_results_*myarch*_{compiler}.csv"
}
```
Run `python3 plot_results.py --input-dir results` to generate updated cross-architecture graphs.

---

## Acknowledgments

- **Adrien Cassagne** ([@kouchy](https://github.com/kouchy/)) for access to the single-board computers (SBCs) and nodes on the [Dalek Cluster](https://dalek.proj.lip6.fr/) (*"Dalek: An Unconventional and Energy-Aware Heterogeneous Cluster"*, [arXiv:2508.10481](https://arxiv.org/abs/2508.10481)).
- **SpacemiT**, for the [SpacemiT K3 Technical Whitepaper](https://forum.spacemit.com/uploads/short-url/60aJ8cYNmrFWqHn4ddwwSzMLjlY.pdf).
- **camel-cdr**, for the [RVV Benchmark Results](https://camel-cdr.github.io/rvv-bench-results/index.html) project.
- The authors of *["Great Expectations: Benchmarking the Real-World Performance of RVV 1.0 in HPC"](https://arxiv.org/abs/2608.28097)* ([arXiv:2608.28097](https://arxiv.org/abs/2608.28097)).
