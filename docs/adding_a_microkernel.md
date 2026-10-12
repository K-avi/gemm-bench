
## Developer Tutorial: Adding a New Microkernel

This guide walks through implementing, registering, testing, benchmarking, and analyzing a new GEMM microkernel across the entire pipeline. The repository supports two principal layouts:
- **CRR (GotoBLAS / BLIS packing)**: Column-major packed $A$ ($M_R \times K$, leading dimension $M_R$), Row-major $B$ ($K \times N_R$), Row-major $C$ ($M_R \times N_R$). This matches the GotoBLAS / BLIS register-tiled microkernel convention.
- **RRR (Legacy)**: Row-major packed $A, B, C$.

---

### Step 1: Implement the Microkernel in C++

CRR microkernels are implemented in [`include/microkernels/GemmUKernel_opti_crr.hpp`](../include/microkernels/GemmUKernel_opti_crr.hpp). RRR microkernels reside in [`include/microkernels/GemmUKernel_opti_rrr.hpp`](../include/microkernels/GemmUKernel_opti_rrr.hpp). Exploratory kernels go in [`include/microkernels/GemmUKernel_explo.hpp`](../include/microkernels/GemmUKernel_explo.hpp).

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

[`src/main.cpp`](../src/main.cpp) defines **three distinct overloads** of `run()` based on matrix layout combinations:
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

Unit tests use Catch2 v3. For a CRR kernel, register test cases in the CRR sections of [`tests/GemmUKernelTest.cpp`](../tests/GemmUKernelTest.cpp):

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

1. In [`run_benchmarks.sh`](../run_benchmarks.sh), add the kernel identifier to `UNIVERSAL_KERNELS`:
   ```bash
   mippv2_mykernel_crr
   ```
2. If the kernel is designated as the champion for an architecture, update `TARGET_CHAMPIONS` in [`tools/run_cluster_benchmarks.sh`](../tools/run_cluster_benchmarks.sh):
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

In [`uarch_config.json`](../uarch_config.json), configure the expected champion kernels and CSV patterns:
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

