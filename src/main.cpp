#include "Alloc.h"
#include "GemmUKernel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__)
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include "BenchConfig.h"

using namespace GEMMBench;

// User-mode CPU cycle counter (Linux perf_event_open, no root needed when
// perf_event_paranoid <= 2). If unavailable (non-Linux, VM without PMU,
// restricted kernel), ok() is false and all cycle-derived metrics are 0.
class CycleCounter {
public:
  CycleCounter() {
#if defined(__linux__)
    perf_event_attr pe;
    std::memset(&pe, 0, sizeof(pe));
    pe.type = PERF_TYPE_HARDWARE;
    pe.size = sizeof(pe);
    pe.config = PERF_COUNT_HW_CPU_CYCLES;
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    fd_ = static_cast<int>(syscall(SYS_perf_event_open, &pe, 0, -1, -1, 0));
    if (fd_ >= 0) {
      ioctl(fd_, PERF_EVENT_IOC_RESET, 0);
      ioctl(fd_, PERF_EVENT_IOC_ENABLE, 0);
    } else {
      static bool warned = false;
      if (!warned) {
        warned = true;
        std::cerr << "Notice: perf_event_open CPU cycles counter unavailable (errno: "
                  << errno << " - " << std::strerror(errno)
                  << "). Cycle-derived metrics (Eff_GHz, FLOP_per_cycle) will be 0.\n";
      }
    }
#endif
  }
  ~CycleCounter() {
#if defined(__linux__)
    if (fd_ >= 0) close(fd_);
#endif
  }
  CycleCounter(const CycleCounter &) = delete;
  CycleCounter &operator=(const CycleCounter &) = delete;

  bool ok() const { return fd_ >= 0; }

  uint64_t read_cycles() const {
#if defined(__linux__)
    uint64_t v = 0;
    if (fd_ >= 0 && ::read(fd_, &v, sizeof(v)) == static_cast<ssize_t>(sizeof(v)))
      return v;
#endif
    return 0;
  }

private:
  int fd_ = -1;
};

struct BenchmarkStats {
  double time_median_s = 0.0;
  double time_min_s    = 0.0;
  double time_max_s    = 0.0;
  double gflops_median = 0.0;
  double gflops_peak   = 0.0;
  double gflops_stddev = 0.0;
  double cv_percent    = 0.0;
  size_t repetitions   = 0;
  // Measured with the user-mode cycle counter, taken from the same sample
  // as the median time. 0 when the counter is unavailable.
  double cycles_per_call = 0.0;
  double eff_ghz         = 0.0;
  double flop_per_cycle  = 0.0;
};

inline BenchmarkStats compute_stats(const std::vector<double> &times,
                                    const std::vector<double> &cycles,
                                    double ops) {
  const size_t R = times.size();
  std::vector<size_t> order(R);
  std::iota(order.begin(), order.end(), size_t{0});
  std::sort(order.begin(), order.end(),
            [&](size_t a, size_t b) { return times[a] < times[b]; });
  std::vector<double> samples(R);
  for (size_t i = 0; i < R; ++i) samples[i] = times[order[i]];

  const size_t median_idx  = order[R / 2];
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
  const double cyc = cycles[median_idx];
  if (cyc > 0.0 && median_time > 0.0) {
    s.cycles_per_call = cyc;
    s.eff_ghz         = cyc / median_time / 1e9;
    s.flop_per_cycle  = ops / cyc;
  }
  return s;
}

// Timing note: the "memory" clobber forces the compiler to assume A/B/C may
// have changed/been read, so the kernel call cannot be hoisted or elided.
// steady_clock is monotonic (high_resolution_clock may alias system_clock).
#define BENCH_KERNEL(CALL)                                                     \
  do {                                                                         \
    for (size_t w = 0; w < cfg.warmup; ++w) {                                  \
      CALL;                                                                    \
    }                                                                          \
                                                                               \
    CycleCounter cyc_counter;                                                  \
    const size_t reps = (cfg.repetitions > 0) ? cfg.repetitions : 1;           \
    std::vector<double> samples_s(reps);                                       \
    std::vector<double> samples_cyc(reps, 0.0);                                \
    for (size_t r = 0; r < reps; ++r) {                                        \
      asm volatile("" ::: "memory");                                           \
      const uint64_t cyc0 = cyc_counter.read_cycles();                         \
      const auto start = std::chrono::steady_clock::now();                     \
      asm volatile("" ::: "memory");                                           \
      for (size_t i = 0; i < cfg.iterations; ++i) {                            \
        CALL;                                                                  \
        asm volatile("" :: "r"(C.data) : "memory");                            \
      }                                                                        \
      asm volatile("" ::: "memory");                                           \
      const auto end = std::chrono::steady_clock::now();                       \
      const uint64_t cyc1 = cyc_counter.read_cycles();                         \
      asm volatile("" ::: "memory");                                           \
      const auto duration =                                                    \
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);   \
      samples_s[r] = (duration.count() * 1e-9) / cfg.iterations;               \
      if (cyc_counter.ok() && cyc1 > cyc0)                                     \
        samples_cyc[r] = static_cast<double>(cyc1 - cyc0) / cfg.iterations;    \
    }                                                                          \
    const double ops = 2.0 * cfg.M * cfg.N * cfg.K;                           \
    return compute_stats(samples_s, samples_cyc, ops);                         \
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
              << "Time(s) [max]:    " << std::setprecision(12) << stats.time_max_s << "\n"
              << "GFLOP/s [median]: " << std::fixed << std::setprecision(4) << stats.gflops_median << "\n"
              << "GFLOP/s [peak]:   " << std::fixed << std::setprecision(4) << stats.gflops_peak << "\n"
              << "GFLOP/s [stddev]: " << std::fixed << std::setprecision(4) << stats.gflops_stddev << "\n"
              << "CV (%):           " << std::fixed << std::setprecision(2) << stats.cv_percent << " %\n"
              << "Repetitions:      " << stats.repetitions << "\n";
    if (stats.cycles_per_call > 0.0) {
      std::cout << "Cycles / call:    " << std::fixed << std::setprecision(1) << stats.cycles_per_call << "\n"
                << "Eff Freq (GHz):   " << std::fixed << std::setprecision(3) << stats.eff_ghz << "\n"
                << "FLOP / cycle:     " << std::fixed << std::setprecision(4) << stats.flop_per_cycle << "\n";
    }
  } else {
    std::cout << cfg.kernel_name << "," << cfg.M << "," << cfg.N << "," << cfg.K
              << "," << cfg.alpha << "," << cfg.beta << ","
              << std::defaultfloat << std::setprecision(12) << stats.time_median_s << ","
              << std::fixed << std::setprecision(4) << stats.gflops_median << ","
              << std::defaultfloat << std::setprecision(12) << stats.time_min_s << ","
              << std::fixed << std::setprecision(4) << stats.gflops_peak << ","
              << std::fixed << std::setprecision(4) << stats.gflops_stddev << ","
              << std::fixed << std::setprecision(2) << stats.cv_percent << ","
              << stats.repetitions << ","
              << std::defaultfloat << std::setprecision(12) << stats.time_max_s << ","
              << std::fixed << std::setprecision(1) << stats.cycles_per_call << ","
              << std::fixed << std::setprecision(4) << stats.eff_ghz << ","
              << std::fixed << std::setprecision(4) << stats.flop_per_cycle << "\n";
  }
}

template <typename T, typename APacked, typename BPacked, typename CPacked>
void validateKernel(const BenchConfig &cfg, const GemmUKernel<T> &gemm,
                    const APacked &A, const BPacked &B, CPacked &C,
                    SimdAlloc<T> &alloc) {
  auto C_val = cfg.legacy_mode
                   ? alloc.template allocatePacked<CPacked>(cfg.M, cfg.N, InitMode::Zero)
                   : alloc.template allocatePackedTile<CPacked>(cfg.M, cfg.N, InitMode::Zero);
  auto C_ref = cfg.legacy_mode
                   ? alloc.template allocatePacked<CPacked>(cfg.M, cfg.N, InitMode::Zero)
                   : alloc.template allocatePackedTile<CPacked>(cfg.M, cfg.N, InitMode::Zero);

  for (size_t i = 0; i < cfg.M; ++i) {
    for (size_t j = 0; j < cfg.N; ++j) {
      C_val(i, j) = C(i, j);
      T acc = T{0};
      for (size_t k = 0; k < cfg.K; ++k) {
        acc += A(i, k) * B(k, j);
      }
      C_ref(i, j) = (cfg.beta == T{0}) ? (cfg.alpha * acc)
                                       : (cfg.alpha * acc + cfg.beta * C(i, j));
    }
  }

  BenchConfig val_cfg = cfg;
  val_cfg.iterations = 1;
  val_cfg.repetitions = 1;
  val_cfg.warmup = 0;
  run<T>(val_cfg, gemm, A, B, C_val);

  double max_diff = 0.0;
  for (size_t i = 0; i < cfg.M; ++i) {
    for (size_t j = 0; j < cfg.N; ++j) {
      double diff = std::abs(static_cast<double>(C_val(i, j) - C_ref(i, j)));
      if (diff > max_diff) max_diff = diff;
    }
  }
  alloc.freePacked(C_val);
  alloc.freePacked(C_ref);

  const double tol = std::numeric_limits<T>::epsilon() * static_cast<double>(cfg.K) * 100.0;
  if (max_diff > tol) {
    std::cerr << "VALIDATION FAILED for " << cfg.kernel_name
              << ": max diff = " << max_diff << " (tolerance = " << tol << ")\n";
    throw std::runtime_error("Numerical validation failed");
  } else if (!cfg.csv_mode) {
    std::cout << "Validation: OK (max diff = " << max_diff << ")\n";
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

  if (cfg.validate) {
    validateKernel<T, APacked, BPacked, CPacked>(cfg, gemm, A, B, C, alloc);
  }

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