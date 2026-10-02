#include "Alloc.h"
#include "GemmUKernel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "BenchConfig.h"

using namespace GEMMBench;

struct BenchmarkStats {
  double time_median_s = 0.0;
  double time_min_s    = 0.0;
  double time_max_s    = 0.0;
  double gflops_median = 0.0;
  double gflops_peak   = 0.0;
  double gflops_stddev = 0.0;
  double cv_percent    = 0.0;
  size_t repetitions   = 0;
};

inline BenchmarkStats compute_stats(std::vector<double> &samples, double ops) {
  const size_t R = samples.size();
  std::sort(samples.begin(), samples.end());

  const double median_time = samples[R / 2];
  const double min_time    = samples.front();
  const double max_time    = samples.back();

  double sum = 0.0;
  for (double t : samples) sum += t;
  const double mean_time = sum / R;

  double var_sum = 0.0;
  for (double t : samples) var_sum += (t - mean_time) * (t - mean_time);
  const double stddev_time = (R > 1) ? std::sqrt(var_sum / (R - 1)) : 0.0;

  BenchmarkStats s;
  s.time_median_s = median_time;
  s.time_min_s    = min_time;
  s.time_max_s    = max_time;
  s.gflops_median = (median_time > 0.0) ? ((ops / median_time) / 1e9) : 0.0;
  s.gflops_peak   = (min_time > 0.0) ? ((ops / min_time) / 1e9) : 0.0;
  s.gflops_stddev = (mean_time > 0.0) ? ((ops / mean_time / 1e9) * (stddev_time / mean_time)) : 0.0;
  s.cv_percent    = (mean_time > 0.0) ? ((stddev_time / mean_time) * 100.0) : 0.0;
  s.repetitions   = R;
  return s;
}

#define BENCH_KERNEL(CALL)                                                     \
  do {                                                                         \
    for (size_t w = 0; w < cfg.warmup; ++w) {                                  \
      CALL;                                                                    \
    }                                                                          \
                                                                               \
    const size_t reps = (cfg.repetitions > 0) ? cfg.repetitions : 1;           \
    std::vector<double> samples_s(reps);                                       \
    for (size_t r = 0; r < reps; ++r) {                                        \
      const auto start = std::chrono::high_resolution_clock::now();            \
      for (size_t i = 0; i < cfg.iterations; ++i) {                            \
        CALL;                                                                  \
      }                                                                        \
      const auto end = std::chrono::high_resolution_clock::now();              \
      asm volatile("" ::: "memory");                                           \
      const auto duration =                                                    \
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);   \
      samples_s[r] = (duration.count() * 1e-9) / cfg.iterations;               \
    }                                                                          \
    const double ops = 2.0 * cfg.M * cfg.N * cfg.K;                           \
    return compute_stats(samples_s, ops);                                      \
  } while (false)

#define BENCH_KERNEL_FASTPATH(CALL_FAST, CALL_FULL)                            \
  do {                                                                         \
    if (cfg.alpha == 1.0 && cfg.beta == 0.0) {                                 \
      BENCH_KERNEL(CALL_FAST);                                                 \
    } else {                                                                   \
      BENCH_KERNEL(CALL_FULL);                                                 \
    }                                                                          \
  } while (false)

template <typename T>
inline BenchmarkStats run(const BenchConfig &cfg, const GemmUKernel<T> &gemm,
                          const PackedRowMajor<T> &A, const PackedRowMajor<T> &B,
                          PackedRowMajor<T> &C) {
  switch (cfg.kernel) {
  // Production / Champion kernels
  case UKernelType::IJK:
    BENCH_KERNEL(gemm.gemm_ijk(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_meteorlake_mr4_nr3:
    BENCH_KERNEL_FASTPATH(
        gemm.gemm_mippv2_meteorlake_mr4_nr3(A, B, C),
        gemm.gemm_mippv2_meteorlake_mr4_nr3(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_a76_mr6_nr3:
    BENCH_KERNEL_FASTPATH(
        gemm.gemm_mippv2_a76_mr6_nr3(A, B, C),
        gemm.gemm_mippv2_a76_mr6_nr3(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_a76_mr6_nr4_fmaddi:
    BENCH_KERNEL_FASTPATH(
        gemm.gemm_mippv2_a76_mr6_nr4_fmaddi(A, B, C),
        gemm.gemm_mippv2_a76_mr6_nr4_fmaddi(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_zen4_mr4_nr4_fmaddi:
    BENCH_KERNEL_FASTPATH(
        gemm.gemm_mippv2_zen4_mr4_nr4_fmaddi(A, B, C),
        gemm.gemm_mippv2_zen4_mr4_nr4_fmaddi(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_firestorm_mr4_nr4_fmaddi:
    BENCH_KERNEL_FASTPATH(
        gemm.gemm_mippv2_firestorm_mr4_nr4_fmaddi(A, B, C),
        gemm.gemm_mippv2_firestorm_mr4_nr4_fmaddi(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_firestorm_mr6_nr4_fmaddi:
    BENCH_KERNEL_FASTPATH(
        gemm.gemm_mippv2_firestorm_mr6_nr4_fmaddi(A, B, C),
        gemm.gemm_mippv2_firestorm_mr6_nr4_fmaddi(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_x60_mr6_nr4_fmaddi:
    BENCH_KERNEL_FASTPATH(
        gemm.gemm_mippv2_x60_mr6_nr4_fmaddi(A, B, C),
        gemm.gemm_mippv2_x60_mr6_nr4_fmaddi(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_x100_register_blocked_apack4:
    BENCH_KERNEL_FASTPATH(
        gemm.gemm_mippv2_x100_register_blocked_apack4(A, B, C),
        gemm.gemm_mippv2_x100_register_blocked_apack4(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_a100_mr7_nr4_pipe:
    BENCH_KERNEL_FASTPATH(
        gemm.gemm_mippv2_a100_mr7_nr4_pipe(A, B, C),
        gemm.gemm_mippv2_a100_mr7_nr4_pipe(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_a100_mr7_nr2_lmul2_pipe:
    BENCH_KERNEL_FASTPATH(
        gemm.gemm_mippv2_a100_mr7_nr2_lmul2_pipe(A, B, C),
        gemm.gemm_mippv2_a100_mr7_nr2_lmul2_pipe(A, B, C, cfg.alpha, cfg.beta));

#ifdef GEMMBENCH_ENABLE_EXPLO
  case UKernelType::blocked_register_blocked:
    BENCH_KERNEL(
        gemm.gemm_blocked_register_blocked(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_skylake_register_blocked:
    BENCH_KERNEL_FASTPATH(
        gemm.gemm_mippv2_skylake_register_blocked(A, B, C),
        gemm.gemm_mippv2_skylake_register_blocked(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_skylake_lmul_register_blocked:
    BENCH_KERNEL(gemm.template gemm_mippv2_skylake_lmul_register_blocked<1>(
        A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_skylake_lmul2_register_blocked:
    BENCH_KERNEL(gemm.template gemm_mippv2_skylake_lmul_register_blocked<2>(
        A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_skylake_lmul4_register_blocked:
    BENCH_KERNEL(gemm.template gemm_mippv2_skylake_lmul_register_blocked<4>(
        A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_firestorm_mr4_nr4:
    BENCH_KERNEL_FASTPATH(
        gemm.gemm_mippv2_firestorm_mr4_nr4(A, B, C),
        gemm.gemm_mippv2_firestorm_mr4_nr4(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_firestorm_mr6_nr4:
    BENCH_KERNEL_FASTPATH(
        gemm.gemm_mippv2_firestorm_mr6_nr4(A, B, C),
        gemm.gemm_mippv2_firestorm_mr6_nr4(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_zen4_mr4_nr4:
    BENCH_KERNEL_FASTPATH(
        gemm.gemm_mippv2_zen4_mr4_nr4(A, B, C),
        gemm.gemm_mippv2_zen4_mr4_nr4(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_x100_register_blocked_lmul1:
    BENCH_KERNEL_FASTPATH(
        gemm.template gemm_mippv2_x100_register_blocked<1>(A, B, C),
        gemm.template gemm_mippv2_x100_register_blocked<1>(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_x100_register_blocked_lmul2:
    BENCH_KERNEL_FASTPATH(
        gemm.template gemm_mippv2_x100_register_blocked<2>(A, B, C),
        gemm.template gemm_mippv2_x100_register_blocked<2>(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_x100_register_blocked_lmul4:
    BENCH_KERNEL_FASTPATH(
        gemm.template gemm_mippv2_x100_register_blocked<4>(A, B, C),
        gemm.template gemm_mippv2_x100_register_blocked<4>(A, B, C, cfg.alpha, cfg.beta));
  case UKernelType::IKJ:
    BENCH_KERNEL(gemm.gemm_ikj(A, B, C));

  case UKernelType::blocked:
    BENCH_KERNEL(gemm.gemm_blocked(A, B, C));

  case UKernelType::blocked_unroll2:
    BENCH_KERNEL(gemm.gemm_blocked_unroll2(A, B, C));

  case UKernelType::blocked_unroll4:
    BENCH_KERNEL(gemm.gemm_blocked_unroll4(A, B, C));

  case UKernelType::mippv2:
    BENCH_KERNEL(gemm.template gemm_mippv2<1>(A, B, C));

  case UKernelType::mippv2_lmul2:
    BENCH_KERNEL(gemm.template gemm_mippv2<2>(A, B, C));

  case UKernelType::mippv2_lmul4:
    BENCH_KERNEL(gemm.template gemm_mippv2<4>(A, B, C));

  case UKernelType::mippv2_lmul8:
    BENCH_KERNEL(gemm.template gemm_mippv2<8>(A, B, C));

  case UKernelType::mippv2_blocked:
    BENCH_KERNEL(gemm.template gemm_mippv2_blocked<1>(A, B, C));

  case UKernelType::mippv2_blocked_lmul2:
    BENCH_KERNEL(gemm.template gemm_mippv2_blocked<2>(A, B, C));

  case UKernelType::mippv2_blocked_lmul4:
    BENCH_KERNEL(gemm.template gemm_mippv2_blocked<4>(A, B, C));

  case UKernelType::mippv2_blocked_lmul8:
    BENCH_KERNEL(gemm.template gemm_mippv2_blocked<8>(A, B, C));

  case UKernelType::mippv2_blocked_unroll2:
    BENCH_KERNEL(gemm.gemm_mippv2_blocked_unroll2(A, B, C));

  case UKernelType::mippv2_blocked_unroll4:
    BENCH_KERNEL(gemm.gemm_mippv2_blocked_unroll4(A, B, C));

  case UKernelType::mippv2_blocked_unroll_jam4:
    BENCH_KERNEL(gemm.gemm_mippv2_blocked_unroll_jam4(A, B, C));

  case UKernelType::mippv2_blocked_register_blocked:
    BENCH_KERNEL(gemm.gemm_mippv2_blocked_register_blocked(A, B, C));

  case UKernelType::mippv2_skylake_lmul8_register_blocked:
    BENCH_KERNEL(
        gemm.template gemm_mippv2_skylake_lmul_register_blocked<8>(A, B, C));

  case UKernelType::mippv2_x100_lmul1:
    BENCH_KERNEL(gemm.template gemm_mippv2_x100<1>(A, B, C));

  case UKernelType::mippv2_x100_lmul2:
    BENCH_KERNEL(gemm.template gemm_mippv2_x100<2>(A, B, C));
  case UKernelType::mippv2_x100_lmul4:
    BENCH_KERNEL(gemm.template gemm_mippv2_x100<4>(A, B, C));

  case UKernelType::mippv2_x100_lmul8:
    BENCH_KERNEL(gemm.template gemm_mippv2_x100<8>(A, B, C));

  case UKernelType::mippv2_x100_register_blocked_lmul8:
    BENCH_KERNEL(gemm.template gemm_mippv2_x100_register_blocked<8>(A, B, C));
#endif

  default:
    throw std::runtime_error("Kernel incompatible with RRR packing");
  }
}

template <typename T>
inline BenchmarkStats run(const BenchConfig &cfg, const GemmUKernel<T> &gemm,
                          const PackedRowMajor<T> &A, const PackedColMajor<T> &B,
                          PackedRowMajor<T> &C) {
  switch (cfg.kernel) {
#ifdef GEMMBENCH_ENABLE_EXPLO
  case UKernelType::IJK_RC:
    BENCH_KERNEL(gemm.gemm_ijk_rc(A, B, C));

  case UKernelType::IKJ_RC:
    BENCH_KERNEL(gemm.gemm_ikj_rc(A, B, C));

  case UKernelType::mippv2_dot:
    BENCH_KERNEL(gemm.gemm_mippv2_dot(A, B, C));
#endif

  default:
    throw std::runtime_error("Kernel incompatible with RCR packing");
  }
}

template <typename T>
inline BenchmarkStats run(const BenchConfig &cfg, const GemmUKernel<T> &gemm,
                          const PackedColMajor<T> &A, const PackedRowMajor<T> &B,
                          PackedRowMajor<T> &C) {
  switch (cfg.kernel) {
  case UKernelType::mippv2_meteorlake_mr4_nr3_crr:
    BENCH_KERNEL_FASTPATH(
        gemm.gemm_mippv2_meteorlake_mr4_nr3_crr(A, B, C),
        gemm.gemm_mippv2_meteorlake_mr4_nr3_crr(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_a76_mr6_nr4_fmaddi_crr:
    BENCH_KERNEL_FASTPATH(
        gemm.gemm_mippv2_a76_mr6_nr4_fmaddi_crr(A, B, C),
        gemm.gemm_mippv2_a76_mr6_nr4_fmaddi_crr(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_zen4_mr4_nr4_fmaddi_crr:
    BENCH_KERNEL_FASTPATH(
        gemm.gemm_mippv2_zen4_mr4_nr4_fmaddi_crr(A, B, C),
        gemm.gemm_mippv2_zen4_mr4_nr4_fmaddi_crr(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_firestorm_mr4_nr4_fmaddi_crr:
    BENCH_KERNEL_FASTPATH(
        gemm.gemm_mippv2_firestorm_mr4_nr4_fmaddi_crr(A, B, C),
        gemm.gemm_mippv2_firestorm_mr4_nr4_fmaddi_crr(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_firestorm_mr6_nr4_fmaddi_crr:
    BENCH_KERNEL_FASTPATH(
        gemm.gemm_mippv2_firestorm_mr6_nr4_fmaddi_crr(A, B, C),
        gemm.gemm_mippv2_firestorm_mr6_nr4_fmaddi_crr(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_x60_mr6_nr4_fmaddi_crr:
    BENCH_KERNEL_FASTPATH(
        gemm.gemm_mippv2_x60_mr6_nr4_fmaddi_crr(A, B, C),
        gemm.gemm_mippv2_x60_mr6_nr4_fmaddi_crr(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_x100_register_blocked_apack4_crr:
    BENCH_KERNEL_FASTPATH(
        gemm.gemm_mippv2_x100_register_blocked_apack4_crr(A, B, C),
        gemm.gemm_mippv2_x100_register_blocked_apack4_crr(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_a100_mr7_nr4_pipe_crr:
    BENCH_KERNEL_FASTPATH(
        gemm.gemm_mippv2_a100_mr7_nr4_pipe_crr(A, B, C),
        gemm.gemm_mippv2_a100_mr7_nr4_pipe_crr(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_a100_mr7_nr2_lmul2_pipe_crr:
    BENCH_KERNEL_FASTPATH(
        gemm.gemm_mippv2_a100_mr7_nr2_lmul2_pipe_crr(A, B, C),
        gemm.gemm_mippv2_a100_mr7_nr2_lmul2_pipe_crr(A, B, C, cfg.alpha, cfg.beta));

#ifdef GEMMBENCH_ENABLE_EXPLO
  case UKernelType::mippv2_skylake_panel:
    BENCH_KERNEL(gemm.gemm_mippv2_skylake_panel(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_skylake_panel_lmul:
    BENCH_KERNEL(gemm.template gemm_mippv2_skylake_panel_lmul<1>(
        A, B, C, cfg.alpha, cfg.beta));
  case UKernelType::mippv2_skylake_panel_lmul2:
    BENCH_KERNEL(gemm.template gemm_mippv2_skylake_panel_lmul<2>(
        A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_skylake_panel_lmul4:
    BENCH_KERNEL(gemm.template gemm_mippv2_skylake_panel_lmul<4>(
        A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_panel_x100_lmul1:
    BENCH_KERNEL_FASTPATH(
        gemm.template gemm_mippv2_panel_x100<1>(A, B, C),
        gemm.template gemm_mippv2_panel_x100<1>(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_panel_x100_lmul2:
    BENCH_KERNEL_FASTPATH(
        gemm.template gemm_mippv2_panel_x100<2>(A, B, C),
        gemm.template gemm_mippv2_panel_x100<2>(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_panel_x100_lmul4:
    BENCH_KERNEL_FASTPATH(
        gemm.template gemm_mippv2_panel_x100<4>(A, B, C),
        gemm.template gemm_mippv2_panel_x100<4>(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_skylake_panel_lmul8:
    BENCH_KERNEL(gemm.template gemm_mippv2_skylake_panel_lmul<8>(A, B, C));

  case UKernelType::mippv2_panel_x100_lmul8:
    BENCH_KERNEL(gemm.template gemm_mippv2_panel_x100<8>(A, B, C));
#endif

  default:
    throw std::runtime_error("Kernel incompatible with CRR packing");
  }
}

template <typename T>
void printResults(const BenchConfig &cfg,
                  const BenchmarkStats &stats) {

  if (!cfg.csv_mode) {
    std::cout << "M: " << cfg.M << "\n"
              << "N: " << cfg.N << "\n"
              << "K: " << cfg.K << "\n"
              << "Alpha: " << cfg.alpha << "\n"
              << "Beta: " << cfg.beta << "\n"
              << "Time(s) [median]: " << std::setprecision(12) << stats.time_median_s << "\n"
              << "Time(s) [min]:    " << std::setprecision(12) << stats.time_min_s << "\n"
              << "GFLOP/s [median]: " << std::fixed << std::setprecision(4) << stats.gflops_median << "\n"
              << "GFLOP/s [peak]:   " << std::fixed << std::setprecision(4) << stats.gflops_peak << "\n"
              << "GFLOP/s [stddev]: " << std::fixed << std::setprecision(4) << stats.gflops_stddev << "\n"
              << "CV (%):           " << std::fixed << std::setprecision(2) << stats.cv_percent << " %\n"
              << "Repetitions:      " << stats.repetitions << "\n";
  } else {
    std::cout << cfg.kernel_name << "," << cfg.M << "," << cfg.N << "," << cfg.K
              << "," << cfg.alpha << "," << cfg.beta << ","
              << std::defaultfloat << std::setprecision(12) << stats.time_median_s << ","
              << std::fixed << std::setprecision(4) << stats.gflops_median << ","
              << std::defaultfloat << std::setprecision(12) << stats.time_min_s << ","
              << std::fixed << std::setprecision(4) << stats.gflops_peak << ","
              << std::fixed << std::setprecision(4) << stats.gflops_stddev << ","
              << std::fixed << std::setprecision(2) << stats.cv_percent << ","
              << stats.repetitions << "\n";
  }
}

template <typename T, typename APacked, typename BPacked, typename CPacked>
void runBenchmark(BenchConfig &cfg) {
  GemmUKernel<T> gemm;

  const auto &desc = getKernelDescriptor(cfg.kernel);
  const size_t simd_width = cfg.packet_size * cfg.lmul;

  if (!cfg.legacy_mode) {
    if (cfg.M == 0) cfg.M = desc.MR;
    if (cfg.N == 0) cfg.N = desc.NR_vec * simd_width;
    if (cfg.K == 0) cfg.K = 128;
  } else {
    if (cfg.M == 0) cfg.M = 32;
    if (cfg.N == 0) cfg.N = 32;
    if (cfg.K == 0) cfg.K = 32;
  }

  SimdAlloc<T> alloc(cfg.packet_size, cfg.lmul, cfg.alignment, cfg.seed, desc.defaultTileSize);

  auto A = cfg.legacy_mode
               ? alloc.template allocatePacked<APacked>(cfg.M, cfg.K, InitMode::Random)
               : alloc.template allocatePackedTile<APacked>(cfg.M, cfg.K, InitMode::Random);

  auto B = cfg.legacy_mode
               ? alloc.template allocatePacked<BPacked>(cfg.K, cfg.N, InitMode::Random)
               : alloc.template allocatePackedTile<BPacked>(cfg.K, cfg.N, InitMode::Random);

  auto C = cfg.legacy_mode
               ? ((cfg.beta != 0.0)
                      ? alloc.template allocatePacked<CPacked>(cfg.M, cfg.N, InitMode::Random)
                      : alloc.template allocatePacked<CPacked>(cfg.M, cfg.N, InitMode::Zero))
               : ((cfg.beta != 0.0)
                      ? alloc.template allocatePackedTile<CPacked>(cfg.M, cfg.N, InitMode::Random)
                      : alloc.template allocatePackedTile<CPacked>(cfg.M, cfg.N, InitMode::Zero));

  if (!cfg.csv_mode) {
    std::cout << "Mode: " << (cfg.legacy_mode ? "Legacy (strided/padded)" : "BLIS Tile (compacted)") << "\n";
    std::cout << "A.ld = " << A.ld << "\n";
    std::cout << "B.ld = " << B.ld << "\n";
    std::cout << "C.ld = " << C.ld << "\n";
  }

  const auto stats = run<T>(cfg, gemm, A, B, C);

  printResults<T>(cfg, stats);

  alloc.freePacked(A);
  alloc.freePacked(B);
  alloc.freePacked(C);
}

void dispatchBenchmark(BenchConfig &cfg) {
  const auto &desc = getKernelDescriptor(cfg.kernel);

  if (desc.defaultPacking.A == PLayout::Row &&
      desc.defaultPacking.B == PLayout::Row) {
    runBenchmark<double, PackedRowMajor<double>, PackedRowMajor<double>,
                 PackedRowMajor<double>>(cfg);
  }

  else if (desc.defaultPacking.A == PLayout::Row &&
           desc.defaultPacking.B == PLayout::Col) {
    runBenchmark<double, PackedRowMajor<double>, PackedColMajor<double>,
                 PackedRowMajor<double>>(cfg);
  } else if (desc.defaultPacking.A == PLayout::Col &&
             desc.defaultPacking.B == PLayout::Row) {
    runBenchmark<double, PackedColMajor<double>, PackedRowMajor<double>,
                 PackedRowMajor<double>>(cfg);
  }

  else {
    throw std::runtime_error("Unsupported packing");
  }
}
int main(int argc, char **argv) {
  try {
    auto cfg = parseArgs(argc, argv);

    dispatchBenchmark(cfg);
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << '\n';
    return EXIT_FAILURE;
  }

  return 0;
}