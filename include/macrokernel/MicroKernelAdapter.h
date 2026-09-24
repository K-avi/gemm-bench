#pragma once

#include "Alloc.h"
#include "StorageType.h"
#include "macrokernel/UArchMacroTraits.h"
#include "microkernels/GemmUKernel.h"
#include <concepts>
#include <cstddef>

namespace GEMMBench {

/**
 * @brief C++20 Concept defining a MicroKernel policy for MacroKernel
 * integration.
 *
 * A MicroKernelPolicy specifies:
 * - MR: Number of rows in the register tile.
 * - NR: Number of columns in the register tile.
 * - compute(): In-place multiplication of packed A (MR x K) and packed B (K x
 * NR) into C (MR x NR with stride ldc).
 */
template <typename Policy, typename T>
concept MicroKernelPolicy = requires {
  { Policy::MR } -> std::convertible_to<size_t>;
  { Policy::NR } -> std::convertible_to<size_t>;
  requires requires(const T *a_pack, const T *b_pack, T *c, size_t k,
                    size_t ldc) {
    Policy::compute_overwrite(a_pack, b_pack, c, k, ldc);
    Policy::compute_accumulate(a_pack, b_pack, c, k, ldc);
  };
};

// =============================================================================
// Helper: Zero-Cost POD View Creator
// =============================================================================

namespace gemm_bench_detail {

template <typename T>
inline PackedRowMajor<T> make_row_major_view(const T *data, size_t rows,
                                             size_t cols, size_t ld) {
  PackedRowMajor<T> m;
  m.data = const_cast<T *>(data);
  m.rows = rows;
  m.cols = cols;
  m.padded_rows = rows;
  m.padded_cols = cols;
  m.ld = ld;
  return m;
}

template <typename T>
inline PackedRowMajor<T> make_row_major_view(T *data, size_t rows, size_t cols,
                                             size_t ld) {
  PackedRowMajor<T> m;
  m.data = data;
  m.rows = rows;
  m.cols = cols;
  m.padded_rows = rows;
  m.padded_cols = cols;
  m.ld = ld;
  return m;
}

} // namespace gemm_bench_detail

// =============================================================================
// Reference / Baseline Policies
// =============================================================================

/**
 * @brief Scalar IJK MicroKernel Policy (useful as verification baseline).
 */
template <typename T, size_t InMR = 4, size_t InNR = 4>
struct Scalar_IJK_Policy {
  static constexpr size_t MR = InMR;
  static constexpr size_t NR = InNR;

  static inline void compute_overwrite(const T *__restrict a_pack,
                                       const T *__restrict b_pack,
                                       T *__restrict c, size_t k, size_t ldc) {
    auto A_view = gemm_bench_detail::make_row_major_view(a_pack, MR, k, k);
    auto B_view = gemm_bench_detail::make_row_major_view(b_pack, k, NR, NR);
    auto C_view = gemm_bench_detail::make_row_major_view(c, MR, NR, ldc);
    GemmUKernel<T>::gemm_ijk(A_view, B_view, C_view, T{1}, T{0});
  }

  static inline void compute_accumulate(const T *__restrict a_pack,
                                        const T *__restrict b_pack,
                                        T *__restrict c, size_t k, size_t ldc) {
    auto A_view = gemm_bench_detail::make_row_major_view(a_pack, MR, k, k);
    auto B_view = gemm_bench_detail::make_row_major_view(b_pack, k, NR, NR);
    auto C_view = gemm_bench_detail::make_row_major_view(c, MR, NR, ldc);
    GemmUKernel<T>::gemm_ijk(A_view, B_view, C_view, T{1}, T{1});
  }
};

// =============================================================================
// Macro: Policy Generator
// =============================================================================

#define GEMMBENCH_DECLARE_UKERNEL_POLICY(PolicyName, InMR, InNR, KernelCall)   \
  template <typename T> struct PolicyName {                                    \
    static constexpr size_t MR = (InMR);                                       \
    static constexpr size_t NR = (InNR);                                       \
                                                                               \
    static inline void compute_overwrite(const T *__restrict a_pack,           \
                                         const T *__restrict b_pack,           \
                                         T *__restrict c, size_t k,            \
                                         size_t ldc) {                         \
      auto A = gemm_bench_detail::make_row_major_view(a_pack, MR, k, k);       \
      auto B = gemm_bench_detail::make_row_major_view(b_pack, k, NR, NR);      \
      auto C = gemm_bench_detail::make_row_major_view(c, MR, NR, ldc);         \
      GemmUKernel<T>::KernelCall(A, B, C);                                     \
    }                                                                          \
                                                                               \
    static inline void compute_accumulate(const T *__restrict a_pack,          \
                                          const T *__restrict b_pack,          \
                                          T *__restrict c, size_t k,           \
                                          size_t ldc) {                        \
      auto A = gemm_bench_detail::make_row_major_view(a_pack, MR, k, k);       \
      auto B = gemm_bench_detail::make_row_major_view(b_pack, k, NR, NR);      \
      auto C = gemm_bench_detail::make_row_major_view(c, MR, NR, ldc);         \
      GemmUKernel<T>::KernelCall(A, B, C, T{1}, T{1});                         \
    }                                                                          \
  };

// =============================================================================
// Architecture Champion Policies (Zero-Cost Adapters for the 9 Target UArchs)
// =============================================================================

/**
 * @brief Intel Meteor Lake & Skylake Champion Policy (MR=4, NR=12 with VL=4
 * FP64) Champion kernel: mippv2_meteorlake_mr4_nr3
 */
GEMMBENCH_DECLARE_UKERNEL_POLICY(MeteorLake_MR4_NR3_Policy, 4, 12,
                                 gemm_mippv2_meteorlake_mr4_nr3)

/**
 * @brief AMD Zen 4 & Zen 5 Champion Policy (MR=4, NR=32 with AVX-512)
 * Champion kernel: mippv2_firestorm_mr4_nr4_fmaddi
 */
GEMMBENCH_DECLARE_UKERNEL_POLICY(Zen4_Firestorm_MR4_NR4_FMADDI_Policy, 4, 32,
                                 gemm_mippv2_firestorm_mr4_nr4_fmaddi)

/**
 * @brief Apple M1 Firestorm P-core Champion Policy (MR=6, NR=8 with NEON)
 * Champion kernel: mippv2_x60_mr6_nr4_fmaddi
 */
GEMMBENCH_DECLARE_UKERNEL_POLICY(AppleM1_Champion_Policy, 6, 8,
                                 gemm_mippv2_x60_mr6_nr4_fmaddi)

/**
 * @brief ARM Cortex-A76 Champion Policy (MR=6, NR=6 with NEON)
 * Champion kernel: mippv2_a76_mr6_nr3
 */
GEMMBENCH_DECLARE_UKERNEL_POLICY(CortexA76_MR6_NR3_Policy, 6, 6,
                                 gemm_mippv2_a76_mr6_nr3)

/**
 * @brief SpacemiT X100 Champion Policy (MR=3, NR=8 with RVV 256b VL=4)
 * Champion kernel: mippv2_x100_register_blocked_apack4
 */
GEMMBENCH_DECLARE_UKERNEL_POLICY(
    X100_Champion_Policy, 3, 8,
    template gemm_mippv2_x100_register_blocked_apack4<1>)

/**
 * @brief SpacemiT A100 Champion Policy (MR=7, NR=64 with RVV 1024b lmul=2,
 * VL=32) Champion kernel: mippv2_a100_mr7_nr2_lmul2_pipe
 */
GEMMBENCH_DECLARE_UKERNEL_POLICY(
    A100_Champion_Policy, 7, 64,
    template gemm_mippv2_a100_mr7_nr2_lmul2_pipe<2>)

/**
 * @brief SpacemiT X60 Champion Policy (MR=7, NR=16 with RVV 256b VL=4)
 * Champion kernel: mippv2_a100_mr7_nr4_pipe
 */
GEMMBENCH_DECLARE_UKERNEL_POLICY(X60_Champion_Policy, 7, 16,
                                 template gemm_mippv2_a100_mr7_nr4_pipe<1>)

#undef GEMMBENCH_DECLARE_UKERNEL_POLICY

// =============================================================================
// Type-level Champion Policy Mapping for TargetUArch
// =============================================================================

template <TargetUArch Arch, typename T> struct UArchChampionPolicy;

template <typename T> struct UArchChampionPolicy<TargetUArch::Skylake, T> {
  using type = MeteorLake_MR4_NR3_Policy<T>;
};

template <typename T> struct UArchChampionPolicy<TargetUArch::MeteorLake, T> {
  using type = MeteorLake_MR4_NR3_Policy<T>;
};

template <typename T> struct UArchChampionPolicy<TargetUArch::Zen4, T> {
  using type = Zen4_Firestorm_MR4_NR4_FMADDI_Policy<T>;
};

template <typename T> struct UArchChampionPolicy<TargetUArch::Zen5, T> {
  using type = Zen4_Firestorm_MR4_NR4_FMADDI_Policy<T>;
};

template <typename T> struct UArchChampionPolicy<TargetUArch::AppleM1, T> {
  using type = AppleM1_Champion_Policy<T>;
};

template <typename T> struct UArchChampionPolicy<TargetUArch::CortexA76, T> {
  using type = CortexA76_MR6_NR3_Policy<T>;
};

template <typename T>
struct UArchChampionPolicy<TargetUArch::SpacemiT_X100, T> {
  using type = X100_Champion_Policy<T>;
};

template <typename T>
struct UArchChampionPolicy<TargetUArch::SpacemiT_A100, T> {
  using type = A100_Champion_Policy<T>;
};

template <typename T> struct UArchChampionPolicy<TargetUArch::SpacemiT_X60, T> {
  using type = X60_Champion_Policy<T>;
};

template <TargetUArch Arch, typename T>
using UArchChampionPolicy_t = typename UArchChampionPolicy<Arch, T>::type;

} // namespace GEMMBench
