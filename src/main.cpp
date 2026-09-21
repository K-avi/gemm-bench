#include "Alloc.h"
#include "GemmUKernel.h"

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

#include "BenchConfig.h"

using namespace GEMMBench;

#define BENCH_KERNEL(CALL)                                                     \
  do {                                                                         \
    for (size_t i = 0; i < cfg.warmup; ++i)                                    \
      CALL;                                                                    \
                                                                               \
    const auto start = std::chrono::high_resolution_clock::now();              \
                                                                               \
    for (size_t i = 0; i < cfg.iterations; ++i)                                \
      CALL;                                                                    \
                                                                               \
    const auto end = std::chrono::high_resolution_clock::now();                \
                                                                               \
    const auto duration =                                                      \
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);     \
                                                                               \
    return duration.count();                                                   \
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
inline long long run(const BenchConfig &cfg, const GemmUKernel<T> &gemm,
                     const PackedRowMajor<T> &A, const PackedRowMajor<T> &B,
                     PackedRowMajor<T> &C) {
  switch (cfg.kernel) {
  // Production / Optimized kernels
  case UKernelType::IJK:
    BENCH_KERNEL(gemm.gemm_ijk(A, B, C, cfg.alpha, cfg.beta));

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

  case UKernelType::mippv2_meteorlake_mr4_nr3:
    BENCH_KERNEL_FASTPATH(
        gemm.gemm_mippv2_meteorlake_mr4_nr3(A, B, C),
        gemm.gemm_mippv2_meteorlake_mr4_nr3(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_a76_mr6_nr3:
    BENCH_KERNEL_FASTPATH(
        gemm.gemm_mippv2_a76_mr6_nr3(A, B, C),
        gemm.gemm_mippv2_a76_mr6_nr3(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_firestorm_mr4_nr4:
    BENCH_KERNEL_FASTPATH(
        gemm.gemm_mippv2_firestorm_mr4_nr4(A, B, C),
        gemm.gemm_mippv2_firestorm_mr4_nr4(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_firestorm_mr4_nr4_fmaddi:
    BENCH_KERNEL_FASTPATH(
        gemm.gemm_mippv2_firestorm_mr4_nr4_fmaddi(A, B, C),
        gemm.gemm_mippv2_firestorm_mr4_nr4_fmaddi(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_firestorm_mr6_nr4:
    BENCH_KERNEL_FASTPATH(
        gemm.gemm_mippv2_firestorm_mr6_nr4(A, B, C),
        gemm.gemm_mippv2_firestorm_mr6_nr4(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_firestorm_mr6_nr4_fmaddi:
    BENCH_KERNEL_FASTPATH(
        gemm.gemm_mippv2_firestorm_mr6_nr4_fmaddi(A, B, C),
        gemm.gemm_mippv2_firestorm_mr6_nr4_fmaddi(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_zen4_mr4_nr4:
    BENCH_KERNEL_FASTPATH(
        gemm.gemm_mippv2_zen4_mr4_nr4(A, B, C),
        gemm.gemm_mippv2_zen4_mr4_nr4(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_zen4_mr4_nr4_fmaddi:
    BENCH_KERNEL_FASTPATH(
        gemm.gemm_mippv2_zen4_mr4_nr4_fmaddi(A, B, C),
        gemm.gemm_mippv2_zen4_mr4_nr4_fmaddi(A, B, C, cfg.alpha, cfg.beta));

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

  case UKernelType::mippv2_x100_register_blocked_apack4:
    BENCH_KERNEL_FASTPATH(
        gemm.gemm_mippv2_x100_register_blocked_apack4(A, B, C),
        gemm.gemm_mippv2_x100_register_blocked_apack4(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_x60_mr6_nr4_fmaddi:
    BENCH_KERNEL_FASTPATH(
        gemm.gemm_mippv2_x60_mr6_nr4_fmaddi(A, B, C),
        gemm.gemm_mippv2_x60_mr6_nr4_fmaddi(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_a100_mr7_nr4_pipe:
    BENCH_KERNEL_FASTPATH(
        gemm.gemm_mippv2_a100_mr7_nr4_pipe(A, B, C),
        gemm.gemm_mippv2_a100_mr7_nr4_pipe(A, B, C, cfg.alpha, cfg.beta));

  case UKernelType::mippv2_a100_mr7_nr2_lmul2_pipe:
    BENCH_KERNEL_FASTPATH(
        gemm.gemm_mippv2_a100_mr7_nr2_lmul2_pipe(A, B, C),
        gemm.gemm_mippv2_a100_mr7_nr2_lmul2_pipe(A, B, C, cfg.alpha, cfg.beta));

#ifdef GEMMBENCH_ENABLE_EXPLO
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
inline long long run(const BenchConfig &cfg, const GemmUKernel<T> &gemm,
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
inline long long run(const BenchConfig &cfg, const GemmUKernel<T> &gemm,
                     const PackedColMajor<T> &A, const PackedRowMajor<T> &B,
                     PackedRowMajor<T> &C) {
  switch (cfg.kernel) {
  // Production / Optimized kernels
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

#ifdef GEMMBENCH_ENABLE_EXPLO
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
                  long long time) { // time is in nanoseconds

  const double ops = 2.0 * cfg.M * cfg.N * cfg.K;

  const double time_s = (time * 1e-9) / cfg.iterations;

  const double gflops = ops / time_s / 1e9;

  if (!cfg.csv_mode) {
    std::cout << "M: " << cfg.M << "\n"
              << "N: " << cfg.N << "\n"
              << "K: " << cfg.K << "\n"
              << "Alpha: " << cfg.alpha << "\n"
              << "Beta: " << cfg.beta << "\n"
              << "Time(s): " << std::setprecision(12) << time_s << "\n"
              << "GFLOP/s: " << gflops << "\n";
  } else {
    std::cout << cfg.kernel_name << "," << cfg.M << "," << cfg.N << "," << cfg.K
              << "," << cfg.alpha << "," << cfg.beta << "," << time_s << ","
              << gflops << "\n";
  }
}

template <typename T, typename APacked, typename BPacked, typename CPacked>
void runBenchmark(const BenchConfig &cfg) {
  GemmUKernel<T> gemm;

  const auto &desc = getKernelDescriptor(cfg.kernel);
  SimdAlloc<T> alloc(cfg.packet_size, cfg.lmul, cfg.alignment, cfg.seed, desc.defaultTileSize);

  auto A =
      alloc.template allocatePacked<APacked>(cfg.M, cfg.K, InitMode::Random);

  auto B =
      alloc.template allocatePacked<BPacked>(cfg.K, cfg.N, InitMode::Random);

  auto C = (cfg.beta != 0.0)
               ? alloc.template allocatePacked<CPacked>(cfg.M, cfg.N,
                                                        InitMode::Random)
               : alloc.template allocatePacked<CPacked>(cfg.M, cfg.N,
                                                        InitMode::Zero);

  if (!cfg.csv_mode) {
    std::cout << "A.ld = " << A.ld << "\n";
    std::cout << "B.ld = " << B.ld << "\n";
    std::cout << "C.ld = " << C.ld << "\n";
  }

  const auto time = run<T>(cfg, gemm, A, B, C);

  printResults<T>(cfg, time);

  alloc.freePacked(A);
  alloc.freePacked(B);
  alloc.freePacked(C);
}

void dispatchBenchmark(const BenchConfig &cfg) {
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