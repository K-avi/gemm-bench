#pragma once

#include "Alloc.h"
#include "Epilogue.hpp"
#include "StorageType.h"
#include "mipp.hpp"
#include "mipp_custom.h"

#include <algorithm>
#include <cassert>
#include <cstddef>

namespace GEMMBench {

enum class UKernelType {
  // Production / Optimized kernels (always compiled)
  IJK,
  blocked_register_blocked,

  // Skylake kernels
  mippv2_skylake_register_blocked,
  mippv2_skylake_lmul_register_blocked,
  mippv2_skylake_lmul2_register_blocked,
  mippv2_skylake_lmul4_register_blocked,

  mippv2_skylake_panel,
  mippv2_skylake_panel_lmul,
  mippv2_skylake_panel_lmul2,
  mippv2_skylake_panel_lmul4,

  // X100 RVV kernels
  mippv2_x100_register_blocked_lmul1,
  mippv2_x100_register_blocked_lmul2,
  mippv2_x100_register_blocked_lmul4,
  mippv2_x100_register_blocked_apack4,

  mippv2_panel_x100_lmul1,
  mippv2_panel_x100_lmul2,
  mippv2_panel_x100_lmul4,

#ifdef GEMMBENCH_ENABLE_EXPLO
  // Exploratory kernels (guarded by GEMMBENCH_ENABLE_EXPLO)
  IKJ,
  IJK_RC,
  IKJ_RC,

  blocked,
  blocked_unroll2,
  blocked_unroll4,

  mippv2,
  mippv2_lmul2,
  mippv2_lmul4,
  mippv2_lmul8,

  mippv2_blocked,
  mippv2_blocked_lmul2,
  mippv2_blocked_lmul4,
  mippv2_blocked_lmul8,

  mippv2_blocked_unroll2,
  mippv2_blocked_unroll4,
  mippv2_blocked_unroll_jam4,

  mippv2_blocked_register_blocked,
  mippv2_register_blocked_hoh,

  mippv2_skylake_lmul8_register_blocked,
  mippv2_skylake_panel_lmul8,

  mippv2_dot,

  mippv2_x100_lmul1,
  mippv2_x100_lmul2,
  mippv2_x100_lmul4,
  mippv2_x100_lmul8,

  mippv2_x100_register_blocked_lmul8,
  mippv2_panel_x100_lmul8,
#endif
};

struct KernelDescriptor {
  UKernelType type;
  const char *name;
  KernelPacking defaultPacking;
};

constexpr KernelPacking RRR = {PLayout::Row, PLayout::Row, PLayout::Row};
constexpr KernelPacking RCR = {PLayout::Row, PLayout::Col, PLayout::Row};
constexpr KernelPacking CRR = {PLayout::Col, PLayout::Row, PLayout::Row};

inline constexpr KernelDescriptor kernelTable[] = {
    // Production / Optimized kernels
    {UKernelType::IJK, "ijk", RRR},
    {UKernelType::blocked_register_blocked, "blocked_register_blocked", RRR},

    // Skylake register blocked
    {UKernelType::mippv2_skylake_register_blocked, "mippv2_skylake_register_blocked", RRR},
    {UKernelType::mippv2_skylake_lmul_register_blocked, "mippv2_skylake_lmul_register_blocked", RRR},
    {UKernelType::mippv2_skylake_lmul2_register_blocked, "mippv2_skylake_lmul2_register_blocked", RRR},
    {UKernelType::mippv2_skylake_lmul4_register_blocked, "mippv2_skylake_lmul4_register_blocked", RRR},

    // Skylake panel
    {UKernelType::mippv2_skylake_panel, "mippv2_skylake_panel", CRR},
    {UKernelType::mippv2_skylake_panel_lmul, "mippv2_skylake_panel_lmul", CRR},
    {UKernelType::mippv2_skylake_panel_lmul2, "mippv2_skylake_panel_lmul2", CRR},
    {UKernelType::mippv2_skylake_panel_lmul4, "mippv2_skylake_panel_lmul4", CRR},

    // X100 RVV register blocked
    {UKernelType::mippv2_x100_register_blocked_lmul1, "mippv2_x100_register_blocked_lmul1", RRR},
    {UKernelType::mippv2_x100_register_blocked_lmul2, "mippv2_x100_register_blocked_lmul2", RRR},
    {UKernelType::mippv2_x100_register_blocked_lmul4, "mippv2_x100_register_blocked_lmul4", RRR},
    {UKernelType::mippv2_x100_register_blocked_apack4, "mippv2_x100_register_blocked_apack4", RRR},

    // X100 RVV panel
    {UKernelType::mippv2_panel_x100_lmul1, "mippv2_panel_x100_lmul1", CRR},
    {UKernelType::mippv2_panel_x100_lmul2, "mippv2_panel_x100_lmul2", CRR},
    {UKernelType::mippv2_panel_x100_lmul4, "mippv2_panel_x100_lmul4", CRR},

#ifdef GEMMBENCH_ENABLE_EXPLO
    // Exploratory kernels
    {UKernelType::IKJ, "ikj", RRR},
    {UKernelType::IJK_RC, "ijk_rc", RCR},
    {UKernelType::IKJ_RC, "ikj_rc", RCR},

    {UKernelType::blocked, "blocked", RRR},
    {UKernelType::blocked_unroll2, "blocked_unroll2", RRR},
    {UKernelType::blocked_unroll4, "blocked_unroll4", RRR},

    {UKernelType::mippv2, "mippv2", RRR},
    {UKernelType::mippv2_lmul2, "mippv2_lmul2", RRR},
    {UKernelType::mippv2_lmul4, "mippv2_lmul4", RRR},
    {UKernelType::mippv2_lmul8, "mippv2_lmul8", RRR},

    {UKernelType::mippv2_blocked, "mippv2_blocked", RRR},
    {UKernelType::mippv2_blocked_lmul2, "mippv2_blocked_lmul2", RRR},
    {UKernelType::mippv2_blocked_lmul4, "mippv2_blocked_lmul4", RRR},
    {UKernelType::mippv2_blocked_lmul8, "mippv2_blocked_lmul8", RRR},

    {UKernelType::mippv2_blocked_unroll2, "mippv2_blocked_unroll2", RRR},
    {UKernelType::mippv2_blocked_unroll4, "mippv2_blocked_unroll4", RRR},
    {UKernelType::mippv2_blocked_unroll_jam4, "mippv2_blocked_unroll_jam4", RRR},

    {UKernelType::mippv2_blocked_register_blocked, "mippv2_blocked_register_blocked", RRR},
    {UKernelType::mippv2_register_blocked_hoh, "mippv2_register_blocked_hoh", RRR},

    {UKernelType::mippv2_skylake_lmul8_register_blocked, "mippv2_skylake_lmul8_register_blocked", RRR},
    {UKernelType::mippv2_skylake_panel_lmul8, "mippv2_skylake_panel_lmul8", CRR},

    {UKernelType::mippv2_dot, "mippv2_dot", RCR},

    {UKernelType::mippv2_x100_lmul1, "mippv2_x100_lmul1", RRR},
    {UKernelType::mippv2_x100_lmul2, "mippv2_x100_lmul2", RRR},
    {UKernelType::mippv2_x100_lmul4, "mippv2_x100_lmul4", RRR},
    {UKernelType::mippv2_x100_lmul8, "mippv2_x100_lmul8", RRR},

    {UKernelType::mippv2_x100_register_blocked_lmul8, "mippv2_x100_register_blocked_lmul8", RRR},
    {UKernelType::mippv2_panel_x100_lmul8, "mippv2_panel_x100_lmul8", CRR},
#endif
};

inline const KernelDescriptor &getKernelDescriptor(UKernelType type) {
  for (const auto &k : kernelTable) {
    if (k.type == type)
      return k;
  }

  throw std::runtime_error("Unknown kernel");
}

template <typename T> class GemmUKernel {
public:
  static void checkDimensions(const PackedRowMajor<T> &A,
                              const PackedRowMajor<T> &B,
                              const PackedRowMajor<T> &C) {
    assert(A.cols == B.rows);
    assert(C.rows == A.rows);
    assert(C.cols == B.cols);
  }

#include "GemmUKernel_opti.hpp"

#ifdef GEMMBENCH_ENABLE_EXPLO
#include "GemmUKernel_explo.hpp"
#endif
};

} // namespace GEMMBench
