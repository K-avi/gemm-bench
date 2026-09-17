# GEMM Benchmark Suite for SpacemiT RISC-V & MIPPv2

> [!WARNING]
> **Work in Progress (WIP)**: This repository is under development. Microkernels, benchmark (and anything else to be honest) might change any time.

This repostitory contains a high-performance General Matrix Multiply (DGEMM) microbenchmark suite, built on top of [MIPPv2](https://github.com/aff3ct/MIPP/tree/develop) (MyIntrinsics++ v2).
It currently contains microbenchmarks for AMD Zen 4 (AVX-512), Intel Meteor Lake (Redwood Cove P-Core, AVX2 + FMA), Intel Skylake (AVX2 + FMA), and SpacemiT X100 (RVV 1.0).

> [!NOTE]
> **Origins & Context**: Developed during a research internship at **LIP6** (Sorbonne Université), this benchmark evaluates MIPPv2 as a zero-overhead abstraction layer for compute-bound microkernels (DGEMM). Key objectives included assessing cross-platform efficiency across different microarchitectures (Intel Skylake vs. SpacemiT X100) and determining whether MIPPv2 features (such as emulated LMUL on fixed-width SIMD) simplify microkernel design space exploration.
>
> **Repurposing & Roadmap**: This microkernel benchmark is being repurposed and actively expanded into a standalone **GEMM benchmark** and optimized kernel suite across multiple architectures:
> - **AMD Zen 4 (Ryzen 9 7945HX)**: AVX-512 microkernel ($M_R=4, N_R=32$) reaching **85.06 GFLOP/s (97% of theoretical peak)**.
> - **Intel Meteor Lake (Redwood Cove P-Core)**: AVX2 microkernel ($M_R=4, N_R=12$) reaching **75.96 GFLOP/s (93% of theoretical peak)**.
> - **SpacemiT K3 X100**: RVV 1.0 microkernel with vector-scalar FMA reaching **16.43 GFLOP/s (93% of theoretical peak)**.
> - **Intel Skylake**: AVX2 microkernel ($M_R=2, N_R=8$) reaching **33.84 GFLOP/s (92% of theoretical peak)**.
> - Next microkernel targets: **AMD Zen 5** (AVX-512) and **ARM Cortex-A76 / Apple Silicon M1 Firestorm** (ARM NEON).


---

## Performance Overview

- **AMD Zen 4 (Ryzen 9 7945HX, 5.46 GHz max boost)**: Reaches **85.06 GFLOP/s** (**97.3% of theoretical peak**, 87.40 GFLOP/s) on a single core using the `mippv2_zen4_mr4_nr4_fmaddi` microkernel.
- **Intel Meteor Lake (Redwood Cove P-Core, 5.10 GHz)**: Reaches **75.96 GFLOP/s** (**93.1% of theoretical peak**, 81.60 GFLOP/s) on a single core using the `mippv2_meteorlake_mr4_nr3` microkernel.
- **SpacemiT K3 X100 (RVV 1.0, 1.60 GHz)**: Reaches **16.43 GFLOP/s** (**93.3% of theoretical peak**, 17.6 GFLOP/s) on a single core using the `mippv2_x100_register_blocked_apack4` microkernel.

Achieving >93% hardware utilization on both modern x86 and emerging RISC-V architectures demonstrates that pure C++ using a well-designed SIMD abstraction library (MIPPv2) can match hand-crafted assembly while retaining portability.

> [!NOTE]
> **Scope & Experimental Caveats**:
> These benchmarks currently evaluate **isolated microkernels under idealized conditions**:
> - **Synthetic constraints:** Matrices are strictly square, power-of-two dimensions ($32 \le M,N,K \le 512$) with 64-byte SIMD/cache-line aligned allocations.
> - **No fringe/tail handling:** Dimensions are exact multiples of the vector register tiles ($M_R, N_R$), avoiding peeling or masking overheads.
> - **Absence of multi-level cache hierarchy:** Without full BLIS/GotoBLAS-style cache tiling ($M_C, K_C, N_C$ packing across L1/L2/LLC), throughput naturally drops as matrix dimensions exceed local cache capacity.
> Bridging this gap—transitioning from isolated microarchitectural microkernels to an arbitrary-dimension, cache-blocked GEMM engine—is the primary motivation for this standalone repository.

### Architecture Comparison: Skylake vs. SpacemiT X100

| Skylake (AVX2 / FMA) | SpacemiT K3 X100 (RVV 1.0, VLEN=256) |
| :---: | :---: |
| **Sustained Throughput (GFLOP/s)** | **Sustained Throughput (GFLOP/s)** |
| ![Skylake Main GFLOPS](assets/main_gflops_skylake.svg) | ![X100 Main GFLOPS](assets/main_gflops_x100.svg) |
| **Speedup vs. Scalar Baseline (`ijk`)** | **Speedup vs. Scalar Baseline (`ijk`)** |
| ![Skylake Speedup vs Baseline](assets/speedup_vs_scalar_skylake.svg) | ![X100 Speedup vs Baseline](assets/speedup_vs_scalar_x100.svg) |

---

## Repository Structure

```text
gemm_bench/
├── assets/                  # Performance summary figures (SVG)
├── docs/                    # In-depth microarchitectural analysis documents (WIP, not pushed yet!)
├── include/
│   ├── Alloc.h              # Cache/SIMD-aligned memory allocators
│   ├── BenchConfig.h        # Benchmark configuration types
│   ├── GemmUKernel.h        # GEMM microkernel implementations
│   ├── StorageType.h        # Row-major, column-major, and packed matrix wrappers
│   └── mipp_custom.h        # Vector-scalar FMA (vfmacc.vf) extension header
├── results/
│   └── reference/           # Canonical benchmark datasets (Skylake & X100)
├── src/
│   └── main.cpp             # Benchmark driver CLI
├── tests/                   # Catch2 unit and validation tests
├── x100_uarch_exp/
│   └── broadcast_bench.cpp  # FMA & vector-broadcast microarchitectural probe
├── plot_results.py          # Fancy plotting script
├── run_benchmarks.sh        # Automated build & benchmark runner for x86
└── run_benchmarks_rvv.sh    # Automated build & benchmark runner for RVV
```

---

## Prerequisites & Dependencies

1. **C++20 Compiler**:
   - `clang++` (>= 15.0) or `g++` (>= 12.0) with RVV 1.0 support (`-march=rv64gcv`).
2. **Build System**:
   - `CMake` (>= 3.20).
3. **MIPPv2 Headers**:
   MIPPv2 headers are not vendored into this repository. You can obtain them from the official [MIPP repository on GitHub (`develop` branch)](https://github.com/aff3ct/MIPP/tree/develop) using either method:

   - **Method 1: Standalone Release Archive (Recommended)**
     Download the pre-generated headers tarball directly from the [MIPP Releases page](https://github.com/aff3ct/MIPP/releases) (e.g., v2.0.1 or v2.0.0):
     ```bash
     wget https://github.com/aff3ct/MIPP/releases/download/v2.0.1/mipp-v2.0.1-headers.tar.gz
     tar -xzf mipp-v2.0.1-headers.tar.gz
     # The extracted directory contains include/mipp.hpp, include/mipp.h, etc.
     ```

   - **Method 2: Clone from `develop` Branch & Generate**
     Clone the upstream `develop` branch and run the generator:
     ```bash
     git clone -b develop https://github.com/aff3ct/MIPP.git
     cd MIPP
     pip install -r generator/requirements.txt
     cd generator
     ./gen_mipp_headers.py --simd-ext sse avx avx512 rvv scalar
     # Generated headers will be located in MIPP/include/
     ```

4. **Python Plotting Dependencies**:
   ```bash
   pip install matplotlib pandas numpy
   ```

---

## Building

### Basic Build
Point CMake to your MIPPv2 `include` directory:

```bash
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DMIPPV2_INCLUDE_DIR=/path/to/mipp/include

cmake --build build -j$(nproc)
```

### Build with Tests
Unit tests use Catch2 v3 (automatically fetched via CMake if not installed on your system):

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

### Automated x86 Execution
The `run_benchmarks.sh` script compiles multiple target configurations (scalar, SSE4.2, AVX2, native) and sweeps matrix dimensions $32 \times 32$ to $512 \times 512$:

```bash
./run_benchmarks.sh
```
Results will be exported to `results/gemm_results.csv`.

### Automated RISC-V Vector (SpacemiT X100 / X60) Execution
This script functions the same way as `run_benchmarks.sh` except it builds for RVV1.0 with fixed size VLEN=256. It also runs every ukernel for matrices ranging from 32x32 to 512x512 to map performance.

```bash
./run_benchmarks_rvv.sh
```

---

## Reproducing Figures

To regenerate the SVG figures using the included reference datasets:

```bash
python3 plot_results.py \
  --skylake-gcc results/reference/skylake_gcc.csv \
  --skylake-clang results/reference/skylake_clang.csv \
  --x100-gcc results/reference/x100_gcc.csv \
  --x100-clang results/reference/x100_clang.csv \
  --output plots
```

---

## Kernel Taxonomy & Microarchitecture Notes

> [!NOTE]
> **Kernel Pruning & Design Space Exploration**:
> This repository contains a broad variety of microkernels implemented during microkernel design space exploration (evaluating unroll factors, register tile geometries, packing schemes, and memory access orders).
> - Suboptimal or redundant kernels will eventually be pruned to maintain a focused collection of the highest-performing microkernels for each target architecture.
> - Scalar kernels (such as `ijk`, `ikj`, and naive blocked variants) are retained strictly as reference baselines for speedup calculations and vectorization efficiency analysis.

- **Scalar Baselines**: Standard triply-nested loops (`ijk`, `ikj`), basic loop blocking, and register-blocked scalar implementations.
- **Generic MIPPv2**: Direct vectorized loops parameterized across vector lengths and unrolling factors (`mippv2_lmul1/2/4/8`, `mippv2_blocked`).
- **Register Blocked**: Microkernels with accumulators kept in vector register files to maximize arithmetic intensity.
- **Panel / Packed**: GotoBLAS/BLIS-style contiguous packing for continuous streaming into vector execution units.
- **Intel Meteor Lake P-Core Microkernel (`mippv2_meteorlake_mr4_nr3`)**: Optimal $4 \times 3$ register tile ($M_R=4, N_R=12$ doubles = 48 FP64 elements) designed for the Redwood Cove core. Balances FMA throughput on Ports 0/1 against the Port 5 vector-broadcast bottleneck (3:1 ratio), uses C++ lexical block scoping for zero register spills within the 16-register AVX2 budget, and features decoupled $4 \times 2$ cleanup tiles reaching **75.96 GFLOP/s (93.1% of theoretical peak)**.
- **AMD Zen 4 Microkernels (`mippv2_zen4_mr4_nr4` & `mippv2_zen4_mr4_nr4_fmaddi`)**: $4 \times 4$ register tile ($M_R=4, N_R=32$ doubles = 128 FP64 elements) utilizing 16 ZMM accumulators on AVX-512. Designed for Zen 4's dual 256-bit FPU pipelines (double-pumped 512-bit FMA). Reaches **85.06 GFLOP/s (97.3% of theoretical peak)**.
- **SpacemiT-Tuned Microkernels**: Microkernels exploring register pressure and dynamic vector-length configurations for SpacemiT cores ($VLEN=256$):
  - **Vector-Scalar FMA**: Uses `include/mipp_custom.h` to emit `vfmacc.vf` instructions, eliminating explicit vector broadcast (`vfmv.v.f`) overhead inside the inner accumulation loop.
  - **Register File Budgeting**: RVV provides 32 architectural vector registers. Custom tiles allocate up to 16–24 registers for accumulators while keeping A-matrix scalars and B-matrix vectors resident.

---

## Acknowledgments

- **Adrien Cassagne** ([@kouchy](https://github.com/kouchy/)) for providing access to single-board computers (SBCs) equipped with SpacemiT K1 and K3 SoCs on the [Dalek Cluster](https://dalek.proj.lip6.fr/) (*"Dalek: An Unconventional and Energy-Aware Heterogeneous Cluster"*, [arXiv:2508.10481](https://arxiv.org/abs/2508.10481)).
- **SpacemiT**, for publishing the [SpacemiT K3 Technical Whitepaper](https://forum.spacemit.com/uploads/short-url/60aJ8cYNmrFWqHn4ddwwSzMLjlY.pdf), which provided the key microarchitectural insights into the X100 core RVV implementation.
- **camel-cdr**, for creating and maintaining the [RVV Benchmark Results](https://camel-cdr.github.io/rvv-bench-results/index.html) project, an invaluable empirical reference for RISC-V Vector performance evaluation.
- The authors of *["Great Expectations: Benchmarking the Real-World Performance of RVV 1.0 in HPC"](https://arxiv.org/abs/2608.28097)* ([arXiv:2608.28097](https://arxiv.org/abs/2608.28097)), whose recent empirical findings and analysis served as one of the motivation for open-sourcing this benchmark suite :-)

---

## Generative AI Disclosure

Generative AI assistants (**Google Gemini** and **OpenAI ChatGPT**) were utilized during the development of this repository to accelerate boilerplate tasks and infrastructure setup, specifically:
- Automation scripts (shell runners and Python data parsing / plotting pipelines).
- Benchmark driver CLI scaffolding, argument handling, and boilerplate wiring in C++.
- Build system configuration and test suite plumbing.

However, the actual GEMM microkernel tuned for Skylake (AVX2 + FMA) and especially the SpacemiT K3 X100 core are the result of multiple weeks of dedicated work, hardware-level profiling, and reflection on the underlying microarchitectures and platforms.
