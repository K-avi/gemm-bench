#include "Alloc.h"
#include "GemmUKernel.h"

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

#include "BenchConfig.h"

using namespace GEMMBench;

static constexpr UKernelType kernel_tilesize66[] = {
    UKernelType::mippv2_x100_register_blocked_lmul1,
    UKernelType::mippv2_x100_register_blocked_lmul2,
    UKernelType::mippv2_x100_register_blocked_lmul4,
    UKernelType::mippv2_x100_register_blocked_lmul8,
    UKernelType::mippv2_x100_register_blocked_apack4,
    UKernelType::mippv2_panel_x100_lmul1,
    UKernelType::mippv2_panel_x100_lmul2,
    UKernelType::mippv2_panel_x100_lmul4,
    UKernelType::mippv2_panel_x100_lmul8};

static inline bool isKernelTileSize66(UKernelType kernel) {
  for (const auto &k : kernel_tilesize66) {
    if (kernel == k)
      return true;
  }
  return false;
}

bool csv_mode = false;
char name_buff[256] = {0};

static UKernelType parseKernel(const std::string &name) {
  if (name == "ijk")
    return UKernelType::IJK;

  if (name == "ikj")
    return UKernelType::IKJ;

  if (name == "blocked")
    return UKernelType::blocked;

  if (name == "blocked_unroll2")
    return UKernelType::blocked_unroll2;

  if (name == "blocked_unroll4")
    return UKernelType::blocked_unroll4;

  if (name == "mippv2")
    return UKernelType::mippv2;

  if (name == "mippv2_lmul2")
    return UKernelType::mippv2_lmul2;

  if (name == "mippv2_lmul4")
    return UKernelType::mippv2_lmul4;

  if (name == "mippv2_lmul8")
    return UKernelType::mippv2_lmul8;

  if (name == "mippv2_blocked")
    return UKernelType::mippv2_blocked;

  if (name == "mippv2_blocked_lmul2")
    return UKernelType::mippv2_blocked_lmul2;

  if (name == "mippv2_blocked_lmul4")
    return UKernelType::mippv2_blocked_lmul4;

  if (name == "mippv2_blocked_lmul8")
    return UKernelType::mippv2_blocked_lmul8;

  if (name == "mippv2_blocked_unroll2")
    return UKernelType::mippv2_blocked_unroll2;

  if (name == "mippv2_blocked_unroll4")
    return UKernelType::mippv2_blocked_unroll4;

  if (name == "mippv2_blocked_unroll_jam4")
    return UKernelType::mippv2_blocked_unroll_jam4;

  if (name == "mippv2_skylake_register_blocked")
    return UKernelType::mippv2_skylake_register_blocked;

  if (name == "blocked_register_blocked")
    return UKernelType::blocked_register_blocked;

  if (name == "mippv2_blocked_register_blocked")
    return UKernelType::mippv2_blocked_register_blocked;

  if (name == "mippv2_skylake_lmul_register_blocked")
    return UKernelType::mippv2_skylake_lmul_register_blocked;

  if (name == "mippv2_skylake_lmul2_register_blocked")
    return UKernelType::mippv2_skylake_lmul2_register_blocked;

  if (name == "mippv2_skylake_lmul4_register_blocked")
    return UKernelType::mippv2_skylake_lmul4_register_blocked;
  if (name == "mippv2_skylake_lmul8_register_blocked")
    return UKernelType::mippv2_skylake_lmul8_register_blocked;

  // if(name == "mippv2_register_blocked_hoh")
  //     return UKernelType::mippv2_register_blocked_hoh;

  if (name == "mippv2_skylake_panel")
    return UKernelType::mippv2_skylake_panel;

  if (name == "mippv2_skylake_panel_lmul")
    return UKernelType::mippv2_skylake_panel_lmul;

  if (name == "mippv2_skylake_panel_lmul2")
    return UKernelType::mippv2_skylake_panel_lmul2;

  if (name == "mippv2_skylake_panel_lmul4")
    return UKernelType::mippv2_skylake_panel_lmul4;

  if (name == "mippv2_skylake_panel_lmul8")
    return UKernelType::mippv2_skylake_panel_lmul8;

  if (name == "mippv2_dot")
    return UKernelType::mippv2_dot;
  if (name == "mippv2_x100_lmul1")
    return UKernelType::mippv2_x100_lmul1;
  if (name == "mippv2_x100_lmul2")
    return UKernelType::mippv2_x100_lmul2;
  if (name == "mippv2_x100_lmul4")
    return UKernelType::mippv2_x100_lmul4;
  if (name == "mippv2_x100_lmul8")
    return UKernelType::mippv2_x100_lmul8;

  if (name == "mippv2_x100_register_blocked_lmul1")
    return UKernelType::mippv2_x100_register_blocked_lmul1;
  if (name == "mippv2_x100_register_blocked_lmul2")
    return UKernelType::mippv2_x100_register_blocked_lmul2;
  if (name == "mippv2_x100_register_blocked_lmul4")
    return UKernelType::mippv2_x100_register_blocked_lmul4;
  if (name == "mippv2_x100_register_blocked_lmul8")
    return UKernelType::mippv2_x100_register_blocked_lmul8;

  if (name == "mippv2_x100_register_blocked_apack4")
    return UKernelType::mippv2_x100_register_blocked_apack4;

  if (name == "mippv2_panel_x100_lmul1")
    return UKernelType::mippv2_panel_x100_lmul1;
  if (name == "mippv2_panel_x100_lmul2")
    return UKernelType::mippv2_panel_x100_lmul2;
  if (name == "mippv2_panel_x100_lmul4")
    return UKernelType::mippv2_panel_x100_lmul4;
  if (name == "mippv2_panel_x100_lmul8")
    return UKernelType::mippv2_panel_x100_lmul8;

  throw std::runtime_error("Unknown GEMM kernel: " + name);
}

static BenchConfig parseArgs(int argc, char **argv) {
  BenchConfig cfg;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    auto next = [&]() { return std::stoul(argv[++i]); };

    if (arg == "--m")
      cfg.M = next();

    else if (arg == "--n")
      cfg.N = next();

    else if (arg == "--k")
      cfg.K = next();

    else if (arg == "--iterations")
      cfg.iterations = next();

    else if (arg == "--warmup")
      cfg.warmup = next();

    else if (arg == "--packet-size")
      cfg.packet_size = next();

    else if (arg == "--lmul")
      cfg.lmul = next();

    else if (arg == "--alignment")
      cfg.alignment = next();

    else if (arg == "--kernel") {
      cfg.kernel = parseKernel(argv[++i]);
      std::strncpy(name_buff, argv[i], sizeof(name_buff) - 1);
    } else if (arg == "--csv")
      csv_mode = true;

    else {
      std::cerr << "Unknown argument: " << arg << '\n';
      std::exit(EXIT_FAILURE);
    }
  }

  return cfg;
}

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

template <typename T>
inline long long run(const BenchConfig &cfg, const GemmUKernel<T> &gemm,
                     const PackedRowMajor<T> &A, const PackedRowMajor<T> &B,
                     PackedRowMajor<T> &C) {
  switch (cfg.kernel) {
  case UKernelType::IJK:
    BENCH_KERNEL(gemm.gemm_ijk(A, B, C));

  case UKernelType::IKJ:
    BENCH_KERNEL(gemm.gemm_ikj(A, B, C));

  case UKernelType::blocked:
    BENCH_KERNEL(gemm.gemm_blocked(A, B, C));

  case UKernelType::blocked_register_blocked:
    BENCH_KERNEL(gemm.gemm_blocked_register_blocked(A, B, C));

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

  case UKernelType::mippv2_skylake_register_blocked:
    BENCH_KERNEL(gemm.gemm_mippv2_skylake_register_blocked(A, B, C));

  case UKernelType::mippv2_blocked_register_blocked:
    BENCH_KERNEL(gemm.gemm_mippv2_blocked_register_blocked(A, B, C));

  case UKernelType::mippv2_skylake_lmul_register_blocked:
    BENCH_KERNEL(gemm.template gemm_mippv2_skylake_lmul_register_blocked<1>(A, B, C));

  case UKernelType::mippv2_skylake_lmul2_register_blocked:
    BENCH_KERNEL(gemm.template gemm_mippv2_skylake_lmul2_register_blocked<2>(A, B, C));

  case UKernelType::mippv2_skylake_lmul4_register_blocked:
    BENCH_KERNEL(gemm.template gemm_mippv2_skylake_lmul_register_blocked<4>(A, B, C));

  case UKernelType::mippv2_x100_lmul1:
    BENCH_KERNEL(gemm.template gemm_mippv2_x100<1>(A, B, C));

  case UKernelType::mippv2_x100_lmul2:
    BENCH_KERNEL(gemm.template gemm_mippv2_x100<2>(A, B, C));
  case UKernelType::mippv2_x100_lmul4:
    BENCH_KERNEL(gemm.template gemm_mippv2_x100<4>(A, B, C));

  case UKernelType::mippv2_x100_lmul8:
    BENCH_KERNEL(gemm.template gemm_mippv2_x100<8>(A, B, C));

  case UKernelType::mippv2_x100_register_blocked_lmul1:
    BENCH_KERNEL(gemm.template gemm_mippv2_x100_register_blocked<1>(A, B, C));

  case UKernelType::mippv2_x100_register_blocked_lmul2:
    BENCH_KERNEL(gemm.template gemm_mippv2_x100_register_blocked<2>(A, B, C));
  case UKernelType::mippv2_x100_register_blocked_lmul4:
    BENCH_KERNEL(gemm.template gemm_mippv2_x100_register_blocked<4>(A, B, C));

  case UKernelType::mippv2_x100_register_blocked_lmul8:
    BENCH_KERNEL(gemm.template gemm_mippv2_x100_register_blocked<8>(A, B, C));

  case UKernelType::mippv2_x100_register_blocked_apack4:
    BENCH_KERNEL(gemm.gemm_mippv2_x100_register_blocked_apack4(A, B, C));

  default:
    throw std::runtime_error("Kernel incompatible with RRR packing");
  }
}

template <typename T>
inline long long run(const BenchConfig &cfg, const GemmUKernel<T> &gemm,
                     const PackedRowMajor<T> &A, const PackedColMajor<T> &B,
                     PackedRowMajor<T> &C) {
  switch (cfg.kernel) {
  case UKernelType::IJK_RC:
    BENCH_KERNEL(gemm.gemm_ijk_rc(A, B, C));

  case UKernelType::IKJ_RC:
    BENCH_KERNEL(gemm.gemm_ikj_rc(A, B, C));

  case UKernelType::mippv2_dot:
    BENCH_KERNEL(gemm.gemm_mippv2_dot(A, B, C));

  default:
    throw std::runtime_error("Kernel incompatible with RCR packing");
  }
}

template <typename T>
inline long long run(const BenchConfig &cfg, const GemmUKernel<T> &gemm,
                     const PackedColMajor<T> &A, const PackedRowMajor<T> &B,
                     PackedRowMajor<T> &C) {
  switch (cfg.kernel) {
  case UKernelType::mippv2_skylake_panel:
    BENCH_KERNEL(gemm.gemm_mippv2_skylake_panel(A, B, C));

  case UKernelType::mippv2_skylake_panel_lmul:
    BENCH_KERNEL(gemm.template gemm_mippv2_skylake_panel_lmul<1>(A, B, C));
  case UKernelType::mippv2_skylake_panel_lmul2:
    BENCH_KERNEL(gemm.template gemm_mippv2_skylake_panel_lmul<2>(A, B, C));

  case UKernelType::mippv2_skylake_panel_lmul4:
    BENCH_KERNEL(gemm.template gemm_mippv2_skylake_panel_lmul<4>(A, B, C));

  case UKernelType::mippv2_skylake_panel_lmul8:
    BENCH_KERNEL(gemm.template gemm_mippv2_skylake_panel_lmul<8>(A, B, C));

  case UKernelType::mippv2_panel_x100_lmul1:
    BENCH_KERNEL(gemm.template gemm_mippv2_panel_x100<1>(A, B, C));
  case UKernelType::mippv2_panel_x100_lmul2:
    BENCH_KERNEL(gemm.template gemm_mippv2_panel_x100<2>(A, B, C));
  case UKernelType::mippv2_panel_x100_lmul4:
    BENCH_KERNEL(gemm.template gemm_mippv2_panel_x100<4>(A, B, C));
  case UKernelType::mippv2_panel_x100_lmul8:
    BENCH_KERNEL(gemm.template gemm_mippv2_panel_x100<8>(A, B, C));

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

  if (!csv_mode) {
    std::cout << "M: " << cfg.M << "\n"
              << "N: " << cfg.N << "\n"
              << "K: " << cfg.K << "\n"
              << "Time(s): " << std::setprecision(12) << time_s << "\n"
              << "GFLOP/s: " << gflops << "\n";
  } else {
    std::cout << name_buff << "," << cfg.M << "," << cfg.N << "," << cfg.K
              << "," << time_s << "," << gflops << "\n";
  }
}

template <typename T, typename APacked, typename BPacked, typename CPacked>
void runBenchmark(const BenchConfig &cfg) {
  GemmUKernel<T> gemm;

  SimdAlloc<T> *alloc_ptr;
  if (isKernelTileSize66(cfg.kernel)) {
    alloc_ptr = new SimdAlloc<T>(cfg.packet_size, cfg.lmul, cfg.alignment,
                                 cfg.seed, 66);
  } else {
    alloc_ptr = new SimdAlloc<T>(cfg.packet_size, cfg.lmul, cfg.alignment,
                                 cfg.seed, 64);
  }
  auto &alloc = *alloc_ptr;

  auto A =
      alloc.template allocatePacked<APacked>(cfg.M, cfg.K, InitMode::Random);

  auto B =
      alloc.template allocatePacked<BPacked>(cfg.K, cfg.N, InitMode::Random);

  auto C = alloc.template allocatePacked<CPacked>(cfg.M, cfg.N, InitMode::Zero);

  if (!csv_mode) {
    std::cout << "A.ld = " << A.ld << "\n";
    std::cout << "B.ld = " << B.ld << "\n";
    std::cout << "C.ld = " << C.ld << "\n";
  }

  const auto time = run<T>(cfg, gemm, A, B, C);

  printResults<T>(cfg, time);

  alloc_ptr->freePacked(A);
  alloc_ptr->freePacked(B);
  alloc_ptr->freePacked(C);
  delete alloc_ptr;
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
  auto cfg = parseArgs(argc, argv);

  dispatchBenchmark(cfg);

  return 0;
}