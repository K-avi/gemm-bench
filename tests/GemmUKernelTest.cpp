#include <catch2/catch_test_macros.hpp>

#include "Alloc.h"
#include "BenchConfig.h"
#include "GemmUKernel.h"

#include <cmath>

using namespace GEMMBench;

template <typename T>
constexpr T CANARY_VALUE() {
  return static_cast<T>(-1337.125);
}

template <typename M>
static void fillPaddingWithCanary(M &mat) {
  using T = typename M::value_type;
  const T canary = CANARY_VALUE<T>();
  for (size_t i = 0; i < mat.padded_rows; ++i) {
    for (size_t j = 0; j < mat.padded_cols; ++j) {
      if (i >= mat.rows || j >= mat.cols) {
        mat[i][j] = canary;
      }
    }
  }
}

template <typename M>
static void checkPaddingIntact(const M &mat) {
  using T = typename M::value_type;
  const T canary = CANARY_VALUE<T>();
  for (size_t i = 0; i < mat.padded_rows; ++i) {
    for (size_t j = 0; j < mat.padded_cols; ++j) {
      if (i >= mat.rows || j >= mat.cols) {
        INFO("Padding clobbered at (" << i << ", " << j << ") got=" << mat[i][j]
                                     << " expected canary=" << canary);
        REQUIRE(mat[i][j] == canary);
      }
    }
  }
}

template <typename T>
constexpr T defaultEps(size_t K) {
  if constexpr (std::is_same_v<T, float>) {
    return static_cast<T>(1e-4 + K * 1e-6);
  } else {
    return static_cast<T>(1e-10 + K * 1e-13);
  }
}

template <typename MA, typename MB>
static void
checkMatrixEqual(const MA &A, const MB &B, size_t K = 32) {
  using T = typename MA::value_type;

  static_assert(std::is_same_v<T, typename MB::value_type>);
  const T eps = defaultEps<T>(K);

  REQUIRE(A.rows == B.rows);
  REQUIRE(A.cols == B.cols);

  for (size_t i = 0; i < A.rows; ++i) {
    for (size_t j = 0; j < A.cols; ++j) {
      const T a = A(i, j);
      const T b = B(i, j);

      const T diff = std::abs(a - b);

      const bool close = diff <= eps || diff <= eps * std::abs(b);

      INFO("Mismatch at (" << i << ", " << j << ") "
                           << "expected=" << b << " got=" << a
                           << " diff=" << diff);

      REQUIRE(close);
    }
  }
}

template <typename MRef, typename MTest>
static void
checkMatrixAndPadding(const MRef &ref, const MTest &test, size_t K = 32) {
  checkMatrixEqual(test, ref, K);
  checkPaddingIntact(test);
}

template <typename MDest, typename MSrc>
static void resetMatrix(MDest &dest, const MSrc &src) {
  REQUIRE(dest.rows == src.rows);
  REQUIRE(dest.cols == src.cols);
  for (size_t i = 0; i < dest.rows; ++i) {
    for (size_t j = 0; j < dest.cols; ++j) {
      dest(i, j) = src(i, j);
    }
  }
  fillPaddingWithCanary(dest);
}

#define TEST_KERNEL(NAME, CALL)                                                \
  SECTION(NAME) {                                                              \
    std::fill_n(C_test.data, C_test.padded_rows * C_test.padded_cols, T{});    \
    fillPaddingWithCanary(C_test);                                             \
                                                                               \
    CALL;                                                                      \
                                                                               \
    checkMatrixAndPadding(C_ref, C_test, K);                                   \
  }

template <typename T, class AMatrix, class BMatrix, class CMatrix>
static void testGemmSuite() {

  constexpr size_t M = 32;
  constexpr size_t N = 32;
  constexpr size_t K = 32;

  auto &cfg = GEMMBench::config();

  SimdAlloc<T> alloc(cfg.packet_size, cfg.lmul, cfg.alignment, cfg.seed);

  auto A = alloc.template allocatePacked<AMatrix>(M, K, InitMode::Random);

  auto B = alloc.template allocatePacked<BMatrix>(K, N, InitMode::Random);

  auto C_ref = alloc.template allocatePacked<CMatrix>(M, N, InitMode::Zero);

  auto C_test = alloc.template allocatePacked<CMatrix>(M, N, InitMode::Zero);

  GemmUKernel<T> gemm;

  /*
   * Reference implementation.
   *
   * The naive ijk kernel is intentionally used
   * as the correctness oracle.
   */
  gemm.gemm_ijk(A, B, C_ref);


#ifdef GEMMBENCH_ENABLE_EXPLO
  TEST_KERNEL("IKJ scalar GEMM", gemm.gemm_ikj(A, B, C_test));

  TEST_KERNEL("Blocked scalar GEMM", gemm.gemm_blocked(A, B, C_test));

  TEST_KERNEL("Blocked scalar GEMM unroll2",
              gemm.gemm_blocked_unroll2(A, B, C_test));

  TEST_KERNEL("Blocked scalar GEMM unroll4",
              gemm.gemm_blocked_unroll4(A, B, C_test));
#endif

  // Champion kernels
  TEST_KERNEL("MIPPv2 Meteorlake MR4 NR3",
              gemm.gemm_mippv2_meteorlake_mr4_nr3(A, B, C_test));

  TEST_KERNEL("MIPPv2 Cortex-A76 MR6 NR3",
              gemm.gemm_mippv2_a76_mr6_nr3(A, B, C_test));

  TEST_KERNEL("MIPPv2 Cortex-A76 MR6 NR4 fmaddi",
              gemm.gemm_mippv2_a76_mr6_nr4_fmaddi(A, B, C_test));

  TEST_KERNEL("MIPPv2 Zen 4 MR4 NR4 fmaddi",
              gemm.gemm_mippv2_zen4_mr4_nr4_fmaddi(A, B, C_test));

  TEST_KERNEL("MIPPv2 Firestorm MR4 NR4 fmaddi",
              gemm.gemm_mippv2_firestorm_mr4_nr4_fmaddi(A, B, C_test));

  TEST_KERNEL("MIPPv2 Firestorm MR6 NR4 fmaddi",
              gemm.gemm_mippv2_firestorm_mr6_nr4_fmaddi(A, B, C_test));

  TEST_KERNEL("MIPPv2 x60 MR6 NR4 fmaddi",
              gemm.gemm_mippv2_x60_mr6_nr4_fmaddi(A, B, C_test));

  TEST_KERNEL("MIPPv2 x100 register blocked apack4",
              gemm.gemm_mippv2_x100_register_blocked_apack4(A, B, C_test));

  TEST_KERNEL("MIPPv2 A100 MR7 NR4 pipe",
              gemm.gemm_mippv2_a100_mr7_nr4_pipe(A, B, C_test));

  TEST_KERNEL("MIPPv2 A100 MR7 NR2 lmul2 pipe",
              gemm.gemm_mippv2_a100_mr7_nr2_lmul2_pipe(A, B, C_test));

#ifdef GEMMBENCH_ENABLE_EXPLO
  TEST_KERNEL("Blocked register blocked scalar GEMM",
              gemm.gemm_blocked_register_blocked(A, B, C_test));

  TEST_KERNEL("MIPPv2 GEMM", gemm.gemm_mippv2(A, B, C_test));

  TEST_KERNEL("MIPPv2 GEMM lmul=2", gemm.template gemm_mippv2<2>(A, B, C_test));

  TEST_KERNEL("MIPPv2 GEMM lmul=4", gemm.template gemm_mippv2<4>(A, B, C_test));

  if constexpr (mipp::N<T, 8>() <= 64) {
    TEST_KERNEL("MIPPv2 GEMM lmul=8", gemm.template gemm_mippv2<8>(A, B, C_test));
  }

  TEST_KERNEL("Blocked MIPPv2 GEMM", gemm.gemm_mippv2_blocked(A, B, C_test));

  TEST_KERNEL("Blocked MIPPv2 GEMM lmul=2",
              gemm.template gemm_mippv2_blocked<2>(A, B, C_test));

  TEST_KERNEL("Blocked MIPPv2 GEMM lmul=4",
              gemm.template gemm_mippv2_blocked<4>(A, B, C_test));

  if constexpr (mipp::N<T, 8>() <= 64) {
    TEST_KERNEL("Blocked MIPPv2 GEMM lmul=8",
                gemm.template gemm_mippv2_blocked<8>(A, B, C_test));
  }

  TEST_KERNEL("Blocked MIPPv2 GEMM unroll2",
              gemm.gemm_mippv2_blocked_unroll2(A, B, C_test));

  TEST_KERNEL("Blocked MIPPv2 GEMM unroll4",
              gemm.gemm_mippv2_blocked_unroll4(A, B, C_test));

  TEST_KERNEL("Blocked MIPPv2 GEMM unroll jam4",
              gemm.gemm_mippv2_blocked_unroll_jam4(A, B, C_test));

  TEST_KERNEL("MIPPv2 Skylake register blocked",
              gemm.gemm_mippv2_skylake_register_blocked(A, B, C_test));

  TEST_KERNEL("MIPPv2 blocked register blocked",
              gemm.gemm_mippv2_blocked_register_blocked(A, B, C_test));

  TEST_KERNEL(
      "MIPPv2 Skylake LMUL register blocked",
      gemm.template gemm_mippv2_skylake_lmul_register_blocked<1>(A, B, C_test));

  TEST_KERNEL(
      "MIPPv2 Skylake LMUL2 register blocked",
      gemm.template gemm_mippv2_skylake_lmul_register_blocked<2>(A, B, C_test));

  TEST_KERNEL(
      "MIPPv2 Skylake LMUL4 register blocked",
      gemm.template gemm_mippv2_skylake_lmul_register_blocked<4>(A, B, C_test));

  TEST_KERNEL("MIPPv2 Firestorm MR4 NR4",
              gemm.gemm_mippv2_firestorm_mr4_nr4(A, B, C_test));

  TEST_KERNEL("MIPPv2 Firestorm MR6 NR4",
              gemm.gemm_mippv2_firestorm_mr6_nr4(A, B, C_test));

  TEST_KERNEL("MIPPv2 Zen 4 MR4 NR4",
              gemm.gemm_mippv2_zen4_mr4_nr4(A, B, C_test));

  TEST_KERNEL("MIPPv2 x100 register blocked LMUL1",
              gemm.template gemm_mippv2_x100_register_blocked<1>(A, B, C_test));
  TEST_KERNEL("MIPPv2 x100 register blocked LMUL2",
              gemm.template gemm_mippv2_x100_register_blocked<2>(A, B, C_test));
  if constexpr (2 * mipp::N<T, 4>() <= 64) {
    TEST_KERNEL("MIPPv2 x100 register blocked LMUL4",
                gemm.template gemm_mippv2_x100_register_blocked<4>(A, B, C_test));
  }
  if constexpr (2 * mipp::N<T, 8>() <= 64) {
    TEST_KERNEL("MIPPv2 x100 register blocked LMUL8",
                gemm.template gemm_mippv2_x100_register_blocked<8>(A, B, C_test));
  }
#endif
#undef TEST_KERNEL

  alloc.freePacked(A);
  alloc.freePacked(B);
  alloc.freePacked(C_ref);
  alloc.freePacked(C_test);
}

// ---------------------------------------------------------------------------
// testGemmSuiteCRR — parameterized version
//
// Exercises correctness of CRR champion kernels for arbitrary (M, N, K).
// This is the primary vehicle for covering cleanup paths:
//   - tail rows  : M % MR != 0  (e.g. M = 33, 35, 37)
//   - tail cols  : N % (MR*VL) != 0  (e.g. N = 17, 25)
//   - K remainder: K % 4 != 0  (e.g. K = 35)
//   - FP accumulation at large K (e.g. K = 512)
// ---------------------------------------------------------------------------
template <typename T>
static void testGemmSuiteCRR(size_t M = 32, size_t N = 32, size_t K = 32,
                              uint64_t seed = 42) {

  auto &cfg = GEMMBench::config();

  SimdAlloc<T> alloc(cfg.packet_size, cfg.lmul, cfg.alignment, seed);

  //
  // CRR matrices
  //
  auto A =
      alloc.template allocatePacked<PackedColMajor<T>>(M, K, InitMode::Random);

  auto B =
      alloc.template allocatePacked<PackedRowMajor<T>>(K, N, InitMode::Random);

  auto C_test =
      alloc.template allocatePacked<PackedRowMajor<T>>(M, N, InitMode::Zero);

  auto C_ref =
      alloc.template allocatePacked<PackedRowMajor<T>>(M, N, InitMode::Zero);

  //
  // Reference matrices (RRR)
  //
  auto A_ref =
      alloc.template allocatePacked<PackedRowMajor<T>>(M, K, InitMode::Zero);

  // Copy A (col-major) -> A_ref (row-major) via layout-agnostic operator()
  for (size_t i = 0; i < M; ++i)
    for (size_t k = 0; k < K; ++k)
      A_ref(i, k) = A(i, k);

  GemmUKernel<T> gemm;

  //
  // Scalar reference: naive ijk on the row-major copy of A.
  //
  gemm.gemm_ijk(A_ref, B, C_ref);

#define TEST_KERNEL(NAME, CALL)                                                \
  SECTION(NAME) {                                                              \
    std::fill_n(C_test.data, C_test.padded_rows * C_test.padded_cols, T{});    \
    fillPaddingWithCanary(C_test);                                             \
                                                                               \
    CALL;                                                                      \
                                                                               \
    checkMatrixAndPadding(C_ref, C_test, K);                                   \
  }

  // CRR Champion kernels
  TEST_KERNEL("MIPPv2 Meteorlake MR4 NR3 CRR",
              gemm.gemm_mippv2_meteorlake_mr4_nr3_crr(A, B, C_test));
  TEST_KERNEL("MIPPv2 Cortex-A76 MR6 NR4 FMADDI CRR",
              gemm.gemm_mippv2_a76_mr6_nr4_fmaddi_crr(A, B, C_test));
  TEST_KERNEL("MIPPv2 Zen 4 MR4 NR4 FMADDI CRR",
              gemm.gemm_mippv2_zen4_mr4_nr4_fmaddi_crr(A, B, C_test));
  TEST_KERNEL("MIPPv2 Firestorm MR4 NR4 FMADDI CRR",
              gemm.gemm_mippv2_firestorm_mr4_nr4_fmaddi_crr(A, B, C_test));
  TEST_KERNEL("MIPPv2 Firestorm MR6 NR4 FMADDI CRR",
              gemm.gemm_mippv2_firestorm_mr6_nr4_fmaddi_crr(A, B, C_test));
  TEST_KERNEL("MIPPv2 X60 MR6 NR4 FMADDI CRR",
              gemm.gemm_mippv2_x60_mr6_nr4_fmaddi_crr(A, B, C_test));
  TEST_KERNEL("MIPPv2 X100 Register Blocked Apack4 CRR",
              gemm.gemm_mippv2_x100_register_blocked_apack4_crr(A, B, C_test));
  TEST_KERNEL("MIPPv2 A100 MR7 NR4 Pipe CRR",
              gemm.gemm_mippv2_a100_mr7_nr4_pipe_crr(A, B, C_test));
  TEST_KERNEL("MIPPv2 A100 MR7 NR2 LMUL2 Pipe CRR",
              gemm.gemm_mippv2_a100_mr7_nr2_lmul2_pipe_crr(A, B, C_test));

#ifdef GEMMBENCH_ENABLE_EXPLO
  TEST_KERNEL("MIPPv2 Skylake panel",
              gemm.gemm_mippv2_skylake_panel(A, B, C_test));

  TEST_KERNEL("MIPPv2 Skylake panel LMUL1",
              gemm.template gemm_mippv2_skylake_panel_lmul<1>(A, B, C_test));

  TEST_KERNEL("MIPPv2 Skylake panel LMUL2",
              gemm.template gemm_mippv2_skylake_panel_lmul<2>(A, B, C_test));

  TEST_KERNEL("MIPPv2 Skylake panel LMUL4",
              gemm.template gemm_mippv2_skylake_panel_lmul<4>(A, B, C_test));

  TEST_KERNEL("MIPPv2 panel x100 LMUL1",
              gemm.template gemm_mippv2_panel_x100<1>(A, B, C_test));

  TEST_KERNEL("MIPPv2 panel x100 LMUL2",
              gemm.template gemm_mippv2_panel_x100<2>(A, B, C_test));

  TEST_KERNEL("MIPPv2 panel x100 LMUL4",
              gemm.template gemm_mippv2_panel_x100<4>(A, B, C_test));

  if constexpr (mipp::N<T, 8>() <= 64) {
    TEST_KERNEL("MIPPv2 Skylake panel LMUL8",
                gemm.template gemm_mippv2_skylake_panel_lmul<8>(A, B, C_test));

    TEST_KERNEL("MIPPv2 panel x100 LMUL8",
                gemm.template gemm_mippv2_panel_x100<8>(A, B, C_test));
  }
#else
  (void)C_test;
#endif

#undef TEST_KERNEL

  alloc.freePacked(A);
  alloc.freePacked(B);
  alloc.freePacked(C_test);
  alloc.freePacked(C_ref);
  alloc.freePacked(A_ref);
}

// ---------------------------------------------------------------------------
// testGemmNonSquareCRR — mirrors testGemmNonSquare for CRR kernels
//
// Specifically designed to exercise:
//   - tail rows (M not a multiple of MR=4,6,7)
//   - tail cols (N not a multiple of NR*VL)
//   - non-square K
// ---------------------------------------------------------------------------
template <typename T>
static void testGemmNonSquareCRR(size_t M, size_t N, size_t K,
                                  uint64_t seed = 42) {

  auto &cfg = GEMMBench::config();
  SimdAlloc<T> alloc(cfg.packet_size, cfg.lmul, cfg.alignment, seed);

  auto A =
      alloc.template allocatePacked<PackedColMajor<T>>(M, K, InitMode::Random);
  auto B =
      alloc.template allocatePacked<PackedRowMajor<T>>(K, N, InitMode::Random);
  auto C_ref =
      alloc.template allocatePacked<PackedRowMajor<T>>(M, N, InitMode::Zero);
  auto C_test =
      alloc.template allocatePacked<PackedRowMajor<T>>(M, N, InitMode::Zero);
  auto A_ref =
      alloc.template allocatePacked<PackedRowMajor<T>>(M, K, InitMode::Zero);

  for (size_t i = 0; i < M; ++i)
    for (size_t k = 0; k < K; ++k)
      A_ref(i, k) = A(i, k);

  GemmUKernel<T> gemm;
  gemm.gemm_ijk(A_ref, B, C_ref);

#define TEST_KERNEL(NAME, CALL)                                                \
  SECTION(NAME) {                                                              \
    std::fill_n(C_test.data, C_test.padded_rows * C_test.padded_cols, T{});    \
    fillPaddingWithCanary(C_test);                                             \
    CALL;                                                                      \
    checkMatrixAndPadding(C_ref, C_test, K);                                   \
  }

  // CRR Champion kernels — same set as testGemmSuiteCRR
  TEST_KERNEL("MIPPv2 Meteorlake MR4 NR3 CRR",
              gemm.gemm_mippv2_meteorlake_mr4_nr3_crr(A, B, C_test));
  TEST_KERNEL("MIPPv2 Cortex-A76 MR6 NR4 FMADDI CRR",
              gemm.gemm_mippv2_a76_mr6_nr4_fmaddi_crr(A, B, C_test));
  TEST_KERNEL("MIPPv2 Zen 4 MR4 NR4 FMADDI CRR",
              gemm.gemm_mippv2_zen4_mr4_nr4_fmaddi_crr(A, B, C_test));
  TEST_KERNEL("MIPPv2 Firestorm MR4 NR4 FMADDI CRR",
              gemm.gemm_mippv2_firestorm_mr4_nr4_fmaddi_crr(A, B, C_test));
  TEST_KERNEL("MIPPv2 Firestorm MR6 NR4 FMADDI CRR",
              gemm.gemm_mippv2_firestorm_mr6_nr4_fmaddi_crr(A, B, C_test));
  TEST_KERNEL("MIPPv2 X60 MR6 NR4 FMADDI CRR",
              gemm.gemm_mippv2_x60_mr6_nr4_fmaddi_crr(A, B, C_test));
  TEST_KERNEL("MIPPv2 X100 Register Blocked Apack4 CRR",
              gemm.gemm_mippv2_x100_register_blocked_apack4_crr(A, B, C_test));
  TEST_KERNEL("MIPPv2 A100 MR7 NR4 Pipe CRR",
              gemm.gemm_mippv2_a100_mr7_nr4_pipe_crr(A, B, C_test));
  TEST_KERNEL("MIPPv2 A100 MR7 NR2 LMUL2 Pipe CRR",
              gemm.gemm_mippv2_a100_mr7_nr2_lmul2_pipe_crr(A, B, C_test));

#undef TEST_KERNEL

  alloc.freePacked(A);
  alloc.freePacked(B);
  alloc.freePacked(C_ref);
  alloc.freePacked(C_test);
  alloc.freePacked(A_ref);
}

template <typename T> static void testAlphaBetaSuite() {
  constexpr size_t M = 32;
  constexpr size_t N = 32;
  constexpr size_t K = 32;

  auto &cfg = GEMMBench::config();
  SimdAlloc<T> alloc(cfg.packet_size, cfg.lmul, cfg.alignment, cfg.seed);

  auto A =
      alloc.template allocatePacked<PackedRowMajor<T>>(M, K, InitMode::Random);
  auto B =
      alloc.template allocatePacked<PackedRowMajor<T>>(K, N, InitMode::Random);
  auto C_ref =
      alloc.template allocatePacked<PackedRowMajor<T>>(M, N, InitMode::Zero);
  auto C_test =
      alloc.template allocatePacked<PackedRowMajor<T>>(M, N, InitMode::Zero);
  auto C_init =
      alloc.template allocatePacked<PackedRowMajor<T>>(M, N, InitMode::Random);

  GemmUKernel<T> gemm;

  struct Param {
    T alpha;
    T beta;
    const char *name;
  };

  Param params[] = {
      {T{1.0}, T{0.0}, "alpha=1.0, beta=0.0"},
      {T{1.0}, T{1.0}, "alpha=1.0, beta=1.0"},
      {T{2.5}, T{0.5}, "alpha=2.5, beta=0.5"},
      {T{0.0}, T{1.0}, "alpha=0.0, beta=1.0"},
      {T{0.0}, T{0.0}, "alpha=0.0, beta=0.0 (BLAS zeroing)"},
      {T{-1.0}, T{1.0}, "alpha=-1.0, beta=1.0 (subtraction)"},
      {T{-1.0}, T{-1.0}, "alpha=-1.0, beta=-1.0 (neg accumulation)"},
  };

  for (const auto &p : params) {
    DYNAMIC_SECTION("AlphaBeta RRR - " << p.name) {
      resetMatrix(C_ref, C_init);
      resetMatrix(C_test, C_init);

      gemm.gemm_ijk(A, B, C_ref, p.alpha, p.beta);

      // Champions
      SECTION("mippv2_meteorlake_mr4_nr3") {
        resetMatrix(C_test, C_init);
        gemm.gemm_mippv2_meteorlake_mr4_nr3(A, B, C_test, p.alpha, p.beta);
        checkMatrixEqual(C_ref, C_test);
      }

      SECTION("mippv2_a76_mr6_nr3") {
        resetMatrix(C_test, C_init);
        gemm.gemm_mippv2_a76_mr6_nr3(A, B, C_test, p.alpha, p.beta);
        checkMatrixEqual(C_ref, C_test);
      }

      SECTION("mippv2_a76_mr6_nr4_fmaddi") {
        resetMatrix(C_test, C_init);
        gemm.gemm_mippv2_a76_mr6_nr4_fmaddi(A, B, C_test, p.alpha, p.beta);
        checkMatrixEqual(C_ref, C_test);
      }

      SECTION("mippv2_firestorm_mr4_nr4_fmaddi") {
        resetMatrix(C_test, C_init);
        gemm.gemm_mippv2_firestorm_mr4_nr4_fmaddi(A, B, C_test, p.alpha,
                                                  p.beta);
        checkMatrixEqual(C_ref, C_test);
      }

      SECTION("mippv2_firestorm_mr6_nr4_fmaddi") {
        resetMatrix(C_test, C_init);
        gemm.gemm_mippv2_firestorm_mr6_nr4_fmaddi(A, B, C_test, p.alpha,
                                                  p.beta);
        checkMatrixEqual(C_ref, C_test);
      }

      SECTION("mippv2_x60_mr6_nr4_fmaddi") {
        resetMatrix(C_test, C_init);
        gemm.gemm_mippv2_x60_mr6_nr4_fmaddi(A, B, C_test, p.alpha, p.beta);
        checkMatrixEqual(C_ref, C_test);
      }

      SECTION("mippv2_zen4_mr4_nr4_fmaddi") {
        resetMatrix(C_test, C_init);
        gemm.gemm_mippv2_zen4_mr4_nr4_fmaddi(A, B, C_test, p.alpha, p.beta);
        checkMatrixEqual(C_ref, C_test);
      }

      SECTION("mippv2_x100_register_blocked_apack4") {
        resetMatrix(C_test, C_init);
        gemm.gemm_mippv2_x100_register_blocked_apack4(A, B, C_test, p.alpha,
                                                      p.beta);
        checkMatrixEqual(C_ref, C_test);
      }

      SECTION("mippv2_a100_mr7_nr4_pipe") {
        resetMatrix(C_test, C_init);
        gemm.gemm_mippv2_a100_mr7_nr4_pipe(A, B, C_test, p.alpha, p.beta);
        checkMatrixEqual(C_ref, C_test);
      }

      SECTION("mippv2_a100_mr7_nr2_lmul2_pipe") {
        resetMatrix(C_test, C_init);
        gemm.gemm_mippv2_a100_mr7_nr2_lmul2_pipe(A, B, C_test, p.alpha, p.beta);
        checkMatrixEqual(C_ref, C_test);
      }

#ifdef GEMMBENCH_ENABLE_EXPLO
      SECTION("blocked_register_blocked") {
        gemm.gemm_blocked_register_blocked(A, B, C_test, p.alpha, p.beta);
        checkMatrixEqual(C_ref, C_test);
      }

      SECTION("mippv2_skylake_register_blocked") {
        resetMatrix(C_test, C_init);
        gemm.gemm_mippv2_skylake_register_blocked(A, B, C_test, p.alpha,
                                                  p.beta);
        checkMatrixEqual(C_ref, C_test);
      }

      SECTION("mippv2_skylake_lmul1") {
        resetMatrix(C_test, C_init);
        gemm.template gemm_mippv2_skylake_lmul_register_blocked<1>(
            A, B, C_test, p.alpha, p.beta);
        checkMatrixEqual(C_ref, C_test);
      }

      SECTION("mippv2_skylake_lmul2") {
        resetMatrix(C_test, C_init);
        gemm.template gemm_mippv2_skylake_lmul_register_blocked<2>(
            A, B, C_test, p.alpha, p.beta);
        checkMatrixEqual(C_ref, C_test);
      }

      SECTION("mippv2_skylake_lmul4") {
        resetMatrix(C_test, C_init);
        gemm.template gemm_mippv2_skylake_lmul_register_blocked<4>(
            A, B, C_test, p.alpha, p.beta);
        checkMatrixEqual(C_ref, C_test);
      }

      SECTION("mippv2_firestorm_mr4_nr4") {
        resetMatrix(C_test, C_init);
        gemm.gemm_mippv2_firestorm_mr4_nr4(A, B, C_test, p.alpha, p.beta);
        checkMatrixEqual(C_ref, C_test);
      }

      SECTION("mippv2_firestorm_mr6_nr4") {
        resetMatrix(C_test, C_init);
        gemm.gemm_mippv2_firestorm_mr6_nr4(A, B, C_test, p.alpha, p.beta);
        checkMatrixEqual(C_ref, C_test);
      }

      SECTION("mippv2_zen4_mr4_nr4") {
        resetMatrix(C_test, C_init);
        gemm.gemm_mippv2_zen4_mr4_nr4(A, B, C_test, p.alpha, p.beta);
        checkMatrixEqual(C_ref, C_test);
      }

      SECTION("mippv2_x100_register_blocked_lmul1") {
        resetMatrix(C_test, C_init);
        gemm.template gemm_mippv2_x100_register_blocked<1>(A, B, C_test,
                                                           p.alpha, p.beta);
        checkMatrixEqual(C_ref, C_test);
      }

      SECTION("mippv2_x100_register_blocked_lmul2") {
        resetMatrix(C_test, C_init);
        gemm.template gemm_mippv2_x100_register_blocked<2>(A, B, C_test,
                                                           p.alpha, p.beta);
        checkMatrixEqual(C_ref, C_test);
      }

      SECTION("mippv2_x100_register_blocked_lmul4") {
        resetMatrix(C_test, C_init);
        gemm.template gemm_mippv2_x100_register_blocked<4>(A, B, C_test,
                                                           p.alpha, p.beta);
        checkMatrixEqual(C_ref, C_test);
      }
#endif
    }
  }

  alloc.freePacked(A);
  alloc.freePacked(B);
  alloc.freePacked(C_ref);
  alloc.freePacked(C_test);
  alloc.freePacked(C_init);
}

template <typename T>
static void testAlphaBetaSuiteCRR(size_t M = 32, size_t N = 32, size_t K = 32,
                                   uint64_t seed = 42) {

  auto &cfg = GEMMBench::config();
  SimdAlloc<T> alloc(cfg.packet_size, cfg.lmul, cfg.alignment, seed);

  auto A =
      alloc.template allocatePacked<PackedColMajor<T>>(M, K, InitMode::Random);
  auto A_ref =
      alloc.template allocatePacked<PackedRowMajor<T>>(M, K, InitMode::Zero);
  auto B =
      alloc.template allocatePacked<PackedRowMajor<T>>(K, N, InitMode::Random);
  auto C_ref =
      alloc.template allocatePacked<PackedRowMajor<T>>(M, N, InitMode::Zero);
  auto C_test =
      alloc.template allocatePacked<PackedRowMajor<T>>(M, N, InitMode::Zero);
  auto C_init =
      alloc.template allocatePacked<PackedRowMajor<T>>(M, N, InitMode::Random);

  for (size_t i = 0; i < M; ++i)
    for (size_t k = 0; k < K; ++k)
      A_ref(i, k) = A(i, k);

  GemmUKernel<T> gemm;

  struct Param {
    T alpha;
    T beta;
    const char *name;
  };

  Param params[] = {
      {T{1.0}, T{0.0}, "alpha=1.0, beta=0.0"},
      {T{1.0}, T{1.0}, "alpha=1.0, beta=1.0"},
      {T{2.5}, T{0.5}, "alpha=2.5, beta=0.5"},
      {T{0.0}, T{1.0}, "alpha=0.0, beta=1.0"},
      {T{0.0}, T{0.0}, "alpha=0.0, beta=0.0 (BLAS zeroing)"},
      {T{-1.0}, T{1.0}, "alpha=-1.0, beta=1.0 (subtraction)"},
      {T{-1.0}, T{-1.0}, "alpha=-1.0, beta=-1.0 (neg accumulation)"},
  };

  for (const auto &p : params) {
    DYNAMIC_SECTION("AlphaBeta CRR - " << p.name) {
      resetMatrix(C_ref, C_init);
      resetMatrix(C_test, C_init);

      gemm.gemm_ijk(A_ref, B, C_ref, p.alpha, p.beta);

      // CRR Champion kernels
      SECTION("mippv2_meteorlake_mr4_nr3_crr") {
        resetMatrix(C_test, C_init);
        gemm.gemm_mippv2_meteorlake_mr4_nr3_crr(A, B, C_test, p.alpha, p.beta);
        checkMatrixEqual(C_ref, C_test);
      }
      SECTION("mippv2_a76_mr6_nr4_fmaddi_crr") {
        resetMatrix(C_test, C_init);
        gemm.gemm_mippv2_a76_mr6_nr4_fmaddi_crr(A, B, C_test, p.alpha, p.beta);
        checkMatrixEqual(C_ref, C_test);
      }
      SECTION("mippv2_zen4_mr4_nr4_fmaddi_crr") {
        resetMatrix(C_test, C_init);
        gemm.gemm_mippv2_zen4_mr4_nr4_fmaddi_crr(A, B, C_test, p.alpha, p.beta);
        checkMatrixEqual(C_ref, C_test);
      }
      SECTION("mippv2_firestorm_mr4_nr4_fmaddi_crr") {
        resetMatrix(C_test, C_init);
        gemm.gemm_mippv2_firestorm_mr4_nr4_fmaddi_crr(A, B, C_test, p.alpha, p.beta);
        checkMatrixEqual(C_ref, C_test);
      }
      SECTION("mippv2_firestorm_mr6_nr4_fmaddi_crr") {
        resetMatrix(C_test, C_init);
        gemm.gemm_mippv2_firestorm_mr6_nr4_fmaddi_crr(A, B, C_test, p.alpha, p.beta);
        checkMatrixEqual(C_ref, C_test);
      }
      SECTION("mippv2_x60_mr6_nr4_fmaddi_crr") {
        resetMatrix(C_test, C_init);
        gemm.gemm_mippv2_x60_mr6_nr4_fmaddi_crr(A, B, C_test, p.alpha, p.beta);
        checkMatrixEqual(C_ref, C_test);
      }
      SECTION("mippv2_x100_register_blocked_apack4_crr") {
        resetMatrix(C_test, C_init);
        gemm.gemm_mippv2_x100_register_blocked_apack4_crr(A, B, C_test, p.alpha, p.beta);
        checkMatrixEqual(C_ref, C_test);
      }
      SECTION("mippv2_a100_mr7_nr4_pipe_crr") {
        resetMatrix(C_test, C_init);
        gemm.gemm_mippv2_a100_mr7_nr4_pipe_crr(A, B, C_test, p.alpha, p.beta);
        checkMatrixEqual(C_ref, C_test);
      }
      SECTION("mippv2_a100_mr7_nr2_lmul2_pipe_crr") {
        resetMatrix(C_test, C_init);
        gemm.gemm_mippv2_a100_mr7_nr2_lmul2_pipe_crr(A, B, C_test, p.alpha, p.beta);
        checkMatrixEqual(C_ref, C_test);
      }

#ifdef GEMMBENCH_ENABLE_EXPLO
      SECTION("mippv2_skylake_panel") {
        resetMatrix(C_test, C_init);
        gemm.gemm_mippv2_skylake_panel(A, B, C_test, p.alpha, p.beta);
        checkMatrixEqual(C_ref, C_test);
      }

      SECTION("mippv2_skylake_panel_lmul1") {
        resetMatrix(C_test, C_init);
        gemm.template gemm_mippv2_skylake_panel_lmul<1>(A, B, C_test, p.alpha,
                                                        p.beta);
        checkMatrixEqual(C_ref, C_test);
      }

      SECTION("mippv2_skylake_panel_lmul2") {
        resetMatrix(C_test, C_init);
        gemm.template gemm_mippv2_skylake_panel_lmul<2>(A, B, C_test, p.alpha,
                                                        p.beta);
        checkMatrixEqual(C_ref, C_test);
      }

      SECTION("mippv2_skylake_panel_lmul4") {
        resetMatrix(C_test, C_init);
        gemm.template gemm_mippv2_skylake_panel_lmul<4>(A, B, C_test, p.alpha,
                                                        p.beta);
        checkMatrixEqual(C_ref, C_test);
      }

      SECTION("mippv2_panel_x100_lmul1") {
        resetMatrix(C_test, C_init);
        gemm.template gemm_mippv2_panel_x100<1>(A, B, C_test, p.alpha, p.beta);
        checkMatrixEqual(C_ref, C_test);
      }

      SECTION("mippv2_panel_x100_lmul2") {
        resetMatrix(C_test, C_init);
        gemm.template gemm_mippv2_panel_x100<2>(A, B, C_test, p.alpha, p.beta);
        checkMatrixEqual(C_ref, C_test);
      }

      SECTION("mippv2_panel_x100_lmul4") {
        resetMatrix(C_test, C_init);
        gemm.template gemm_mippv2_panel_x100<4>(A, B, C_test, p.alpha, p.beta);
        checkMatrixEqual(C_ref, C_test);
      }
#else
      (void)C_test;
#endif
    }
  }

  alloc.freePacked(A);
  alloc.freePacked(A_ref);
  alloc.freePacked(B);
  alloc.freePacked(C_ref);
  alloc.freePacked(C_test);
  alloc.freePacked(C_init);
}

TEST_CASE("GEMM RRR kernels", "[gemm]") {
  testGemmSuite<double, PackedRowMajor<double>, PackedRowMajor<double>,
                PackedRowMajor<double>>();
}

// Each dimension gets its own TEST_CASE to avoid Catch2 section ID collision.
// (Catch2 identifies sections by name + source line; calling the same function
// multiple times in one TEST_CASE creates duplicate IDs.)

TEST_CASE("GEMM CRR kernels 32x32x32 (nominal)", "[gemm][crr]") {
  testGemmSuiteCRR<double>(32, 32, 32);
}

// Tail rows: M % MR != 0
// MR=4 -> tail=1, MR=6 -> tail=1, MR=7 -> tail=2
TEST_CASE("GEMM CRR kernels 33x32x32 (tail rows)", "[gemm][crr]") {
  testGemmSuiteCRR<double>(33, 32, 32);
}
// MR=4 -> tail=3, MR=6 -> tail=5, MR=7 -> tail=2
TEST_CASE("GEMM CRR kernels 35x32x32 (tail rows all)", "[gemm][crr]") {
  testGemmSuiteCRR<double>(35, 32, 32);
}

// Tail cols: N % NR != 0, N must be VL-aligned.
// Kernels handle N down to VL granularity (BLIS handles sub-VL via edge kernel).
// Dynamic VL alignment ensures portability across NEON (VL=2), AVX2/RVV256 (VL=4),
// AVX-512 (VL=8), and RVV1024 (VL=16).
TEST_CASE("GEMM CRR kernels tail cols (1*VL)", "[gemm][crr]") {
  constexpr size_t VL = mipp::N<double>();
  testGemmSuiteCRR<double>(32, 1 * VL, 32);
}
TEST_CASE("GEMM CRR kernels tail cols (3*VL)", "[gemm][crr]") {
  constexpr size_t VL = mipp::N<double>();
  testGemmSuiteCRR<double>(32, 3 * VL, 32);
}

// K non-multiple of 4 (meteorlake K4 peel loop)
TEST_CASE("GEMM CRR kernels 32x32x35 (K%4 != 0)", "[gemm][crr]") {
  testGemmSuiteCRR<double>(32, 32, 35);
}

// Large K: validate FP accumulation
TEST_CASE("GEMM CRR kernels 32x32x512 (large K)", "[gemm][crr]") {
  testGemmSuiteCRR<double>(32, 32, 512);
}

// Combined: all tails simultaneously (N VL-aligned)
TEST_CASE("GEMM CRR kernels combined tails (37 x 3*VL x 35)", "[gemm][crr]") {
  constexpr size_t VL = mipp::N<double>();
  testGemmSuiteCRR<double>(37, 3 * VL, 35);
}

// Different seed: stochastic coverage
TEST_CASE("GEMM CRR kernels 32x32x32 seed=0xDEAD", "[gemm][crr]") {
  testGemmSuiteCRR<double>(32, 32, 32, 0xDEADBEEFULL);
}

// Non-square CRR: exercises tail rows + tail cols simultaneously
// N is VL-aligned (kernels don't handle sub-VL boundaries)
TEST_CASE("GEMM CRR Non-square 48x64x32", "[gemm][crr]") {
  testGemmNonSquareCRR<double>(48, 64, 32);
}
TEST_CASE("GEMM CRR Non-square (33 x 5*VL x 35) (tail rows + K%4)", "[gemm][crr]") {
  constexpr size_t VL = mipp::N<double>();
  testGemmNonSquareCRR<double>(33, 5 * VL, 35);
}
TEST_CASE("GEMM CRR Non-square 49x64x32 (tail row only)", "[gemm][crr]") {
  testGemmNonSquareCRR<double>(49, 64, 32);
}

// Isolated nominal single-tile unit tests matching benchmark fast-path without tails
TEST_CASE("GEMM CRR nominal single tile MR6 (6 x 4*VL x 128)", "[gemm][crr]") {
  constexpr size_t VL = mipp::N<double>();
  testGemmSuiteCRR<double>(6, 4 * VL, 128);
}
TEST_CASE("GEMM CRR nominal single tile MR7 (7 x 4*VL x 128)", "[gemm][crr]") {
  constexpr size_t VL = mipp::N<double>();
  testGemmSuiteCRR<double>(7, 4 * VL, 128);
}

TEST_CASE("GEMM Alpha Beta scaling and accumulation", "[gemm]") {
  testAlphaBetaSuite<double>();
  testAlphaBetaSuiteCRR<double>(32, 32, 32, 42);
}

// Separate TEST_CASE for tail-row alpha/beta to avoid Catch2 section collision
TEST_CASE("GEMM Alpha Beta CRR tail rows", "[gemm][crr]") {
  testAlphaBetaSuiteCRR<double>(33, 32, 32, 0xDEADBEEFULL);
}

template <typename T> static void testGemmNonSquare() {
  constexpr size_t M = 48;
  constexpr size_t N = 64;
  constexpr size_t K = 32;

  auto &cfg = GEMMBench::config();
  SimdAlloc<T> alloc(cfg.packet_size, cfg.lmul, cfg.alignment, cfg.seed);

  auto A = alloc.template allocatePacked<PackedRowMajor<T>>(M, K, InitMode::Random);
  auto B = alloc.template allocatePacked<PackedRowMajor<T>>(K, N, InitMode::Random);
  auto C_ref = alloc.template allocatePacked<PackedRowMajor<T>>(M, N, InitMode::Zero);
  auto C_test = alloc.template allocatePacked<PackedRowMajor<T>>(M, N, InitMode::Zero);

  GemmUKernel<T> gemm;
  gemm.gemm_ijk(A, B, C_ref);

#define TEST_KERNEL(NAME, CALL)                                                \
  SECTION(NAME) {                                                              \
    std::fill_n(C_test.data, C_test.padded_rows * C_test.padded_cols, T{});    \
    fillPaddingWithCanary(C_test);                                             \
    CALL;                                                                      \
    checkMatrixAndPadding(C_ref, C_test, K);                                   \
  }

  // Champions
  TEST_KERNEL("MIPPv2 Meteorlake MR4 NR3",
              gemm.gemm_mippv2_meteorlake_mr4_nr3(A, B, C_test));
  TEST_KERNEL("MIPPv2 Cortex-A76 MR6 NR3",
              gemm.gemm_mippv2_a76_mr6_nr3(A, B, C_test));
  TEST_KERNEL("MIPPv2 x60 MR6 NR4 fmaddi",
              gemm.gemm_mippv2_x60_mr6_nr4_fmaddi(A, B, C_test));
  TEST_KERNEL("MIPPv2 A100 MR7 NR4 pipe",
              gemm.gemm_mippv2_a100_mr7_nr4_pipe(A, B, C_test));

#ifdef GEMMBENCH_ENABLE_EXPLO
  TEST_KERNEL("Blocked register blocked",
              gemm.gemm_blocked_register_blocked(A, B, C_test));
  TEST_KERNEL("MIPPv2 Skylake register blocked",
              gemm.gemm_mippv2_skylake_register_blocked(A, B, C_test));
  TEST_KERNEL("MIPPv2 Firestorm MR4 NR4",
              gemm.gemm_mippv2_firestorm_mr4_nr4(A, B, C_test));
  TEST_KERNEL("MIPPv2 Zen 4 MR4 NR4",
              gemm.gemm_mippv2_zen4_mr4_nr4(A, B, C_test));
  TEST_KERNEL("MIPPv2 x100 register blocked LMUL1",
              gemm.template gemm_mippv2_x100_register_blocked<1>(A, B, C_test));
#endif
#undef TEST_KERNEL

  alloc.freePacked(A);
  alloc.freePacked(B);
  alloc.freePacked(C_ref);
  alloc.freePacked(C_test);
}

TEST_CASE("GEMM Non-square dimensions (M!=N!=K)", "[gemm]") {
  testGemmNonSquare<double>();
}

// ===========================================================================
// FP32 (Single Precision float) Tests
// ===========================================================================
TEST_CASE("GEMM RRR kernels float", "[gemm][float]") {
  testGemmSuite<float, PackedRowMajor<float>, PackedRowMajor<float>,
                PackedRowMajor<float>>();
}

TEST_CASE("GEMM CRR kernels float (nominal)", "[gemm][crr][float]") {
  constexpr size_t VL = mipp::N<float>();
  testGemmSuiteCRR<float>(32, 4 * VL, 32);
}

TEST_CASE("GEMM CRR kernels float (tail rows)", "[gemm][crr][float]") {
  constexpr size_t VL = mipp::N<float>();
  testGemmSuiteCRR<float>(33, 4 * VL, 32);
}

TEST_CASE("GEMM Alpha Beta scaling float", "[gemm][float]") {
  constexpr size_t VL = mipp::N<float>();
  testAlphaBetaSuite<float>();
  testAlphaBetaSuiteCRR<float>(32, 4 * VL, 32, 42);
}