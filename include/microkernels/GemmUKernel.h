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
  // Production / Champion kernels (always compiled)
  IJK,

  // Champions
  mippv2_meteorlake_mr4_nr3,
  mippv2_a76_mr6_nr3,
  mippv2_zen4_mr4_nr4_fmaddi,
  mippv2_firestorm_mr4_nr4_fmaddi,
  mippv2_firestorm_mr6_nr4_fmaddi,
  mippv2_x60_mr6_nr4_fmaddi,
  mippv2_x100_register_blocked_apack4,
  mippv2_a100_mr7_nr4_pipe,
  mippv2_a100_mr7_nr2_lmul2_pipe,

#ifdef GEMMBENCH_ENABLE_EXPLO
  // Exploratory kernels (guarded by GEMMBENCH_ENABLE_EXPLO)
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

  // Firestorm exploratory
  mippv2_firestorm_mr4_nr4,
  mippv2_firestorm_mr6_nr4,

  // Zen 4 exploratory
  mippv2_zen4_mr4_nr4,

  // X100 RVV exploratory
  mippv2_x100_register_blocked_lmul1,
  mippv2_x100_register_blocked_lmul2,
  mippv2_x100_register_blocked_lmul4,

  mippv2_panel_x100_lmul1,
  mippv2_panel_x100_lmul2,
  mippv2_panel_x100_lmul4,

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
  size_t defaultTileSize = 64;
};

constexpr KernelPacking RRR = {PLayout::Row, PLayout::Row, PLayout::Row};
constexpr KernelPacking RCR = {PLayout::Row, PLayout::Col, PLayout::Row};
constexpr KernelPacking CRR = {PLayout::Col, PLayout::Row, PLayout::Row};

inline constexpr KernelDescriptor kernelTable[] = {
    // Production / Champion kernels

    // Scalar baseline
    {UKernelType::IJK, "ijk", RRR, 64},

    // Champions
    {UKernelType::mippv2_meteorlake_mr4_nr3, "mippv2_meteorlake_mr4_nr3", RRR, 64},
    {UKernelType::mippv2_a76_mr6_nr3, "mippv2_a76_mr6_nr3", RRR, 64},
    {UKernelType::mippv2_zen4_mr4_nr4_fmaddi, "mippv2_zen4_mr4_nr4_fmaddi", RRR, 64},
    {UKernelType::mippv2_firestorm_mr4_nr4_fmaddi, "mippv2_firestorm_mr4_nr4_fmaddi", RRR, 64},
    {UKernelType::mippv2_firestorm_mr6_nr4_fmaddi, "mippv2_firestorm_mr6_nr4_fmaddi", RRR, 64},
    {UKernelType::mippv2_x60_mr6_nr4_fmaddi, "mippv2_x60_mr6_nr4_fmaddi", RRR, 64},
    {UKernelType::mippv2_x100_register_blocked_apack4, "mippv2_x100_register_blocked_apack4", RRR, 66},
    {UKernelType::mippv2_a100_mr7_nr4_pipe, "mippv2_a100_mr7_nr4_pipe", RRR, 64},
    {UKernelType::mippv2_a100_mr7_nr2_lmul2_pipe, "mippv2_a100_mr7_nr2_lmul2_pipe", RRR, 64},

#ifdef GEMMBENCH_ENABLE_EXPLO
    // Exploratory kernels
    {UKernelType::blocked_register_blocked, "blocked_register_blocked", RRR, 64},

    // Skylake register blocked
    {UKernelType::mippv2_skylake_register_blocked, "mippv2_skylake_register_blocked", RRR, 64},
    {UKernelType::mippv2_skylake_lmul_register_blocked, "mippv2_skylake_lmul_register_blocked", RRR, 64},
    {UKernelType::mippv2_skylake_lmul2_register_blocked, "mippv2_skylake_lmul2_register_blocked", RRR, 64},
    {UKernelType::mippv2_skylake_lmul4_register_blocked, "mippv2_skylake_lmul4_register_blocked", RRR, 64},

    // Skylake panel
    {UKernelType::mippv2_skylake_panel, "mippv2_skylake_panel", CRR, 64},
    {UKernelType::mippv2_skylake_panel_lmul, "mippv2_skylake_panel_lmul", CRR, 64},
    {UKernelType::mippv2_skylake_panel_lmul2, "mippv2_skylake_panel_lmul2", CRR, 64},
    {UKernelType::mippv2_skylake_panel_lmul4, "mippv2_skylake_panel_lmul4", CRR, 64},

    // Firestorm exploratory
    {UKernelType::mippv2_firestorm_mr4_nr4, "mippv2_firestorm_mr4_nr4", RRR, 64},
    {UKernelType::mippv2_firestorm_mr6_nr4, "mippv2_firestorm_mr6_nr4", RRR, 64},

    // Zen 4 exploratory
    {UKernelType::mippv2_zen4_mr4_nr4, "mippv2_zen4_mr4_nr4", RRR, 64},

    // X100 RVV register blocked exploratory
    {UKernelType::mippv2_x100_register_blocked_lmul1, "mippv2_x100_register_blocked_lmul1", RRR, 66},
    {UKernelType::mippv2_x100_register_blocked_lmul2, "mippv2_x100_register_blocked_lmul2", RRR, 66},
    {UKernelType::mippv2_x100_register_blocked_lmul4, "mippv2_x100_register_blocked_lmul4", RRR, 66},

    // X100 RVV panel
    {UKernelType::mippv2_panel_x100_lmul1, "mippv2_panel_x100_lmul1", CRR, 66},
    {UKernelType::mippv2_panel_x100_lmul2, "mippv2_panel_x100_lmul2", CRR, 66},
    {UKernelType::mippv2_panel_x100_lmul4, "mippv2_panel_x100_lmul4", CRR, 66},

    {UKernelType::IKJ, "ikj", RRR, 64},
    {UKernelType::IJK_RC, "ijk_rc", RCR, 64},
    {UKernelType::IKJ_RC, "ikj_rc", RCR, 64},

    {UKernelType::blocked, "blocked", RRR, 64},
    {UKernelType::blocked_unroll2, "blocked_unroll2", RRR, 64},
    {UKernelType::blocked_unroll4, "blocked_unroll4", RRR, 64},

    {UKernelType::mippv2, "mippv2", RRR, 64},
    {UKernelType::mippv2_lmul2, "mippv2_lmul2", RRR, 64},
    {UKernelType::mippv2_lmul4, "mippv2_lmul4", RRR, 64},
    {UKernelType::mippv2_lmul8, "mippv2_lmul8", RRR, 64},

    {UKernelType::mippv2_blocked, "mippv2_blocked", RRR, 64},
    {UKernelType::mippv2_blocked_lmul2, "mippv2_blocked_lmul2", RRR, 64},
    {UKernelType::mippv2_blocked_lmul4, "mippv2_blocked_lmul4", RRR, 64},
    {UKernelType::mippv2_blocked_lmul8, "mippv2_blocked_lmul8", RRR, 64},

    {UKernelType::mippv2_blocked_unroll2, "mippv2_blocked_unroll2", RRR, 64},
    {UKernelType::mippv2_blocked_unroll4, "mippv2_blocked_unroll4", RRR, 64},
    {UKernelType::mippv2_blocked_unroll_jam4, "mippv2_blocked_unroll_jam4", RRR, 64},

    {UKernelType::mippv2_blocked_register_blocked, "mippv2_blocked_register_blocked", RRR, 64},
    {UKernelType::mippv2_register_blocked_hoh, "mippv2_register_blocked_hoh", RRR, 64},

    {UKernelType::mippv2_skylake_lmul8_register_blocked, "mippv2_skylake_lmul8_register_blocked", RRR, 64},
    {UKernelType::mippv2_skylake_panel_lmul8, "mippv2_skylake_panel_lmul8", CRR, 64},

    {UKernelType::mippv2_dot, "mippv2_dot", RCR, 64},

    {UKernelType::mippv2_x100_lmul1, "mippv2_x100_lmul1", RRR, 64},
    {UKernelType::mippv2_x100_lmul2, "mippv2_x100_lmul2", RRR, 64},
    {UKernelType::mippv2_x100_lmul4, "mippv2_x100_lmul4", RRR, 64},
    {UKernelType::mippv2_x100_lmul8, "mippv2_x100_lmul8", RRR, 64},

    {UKernelType::mippv2_x100_register_blocked_lmul8, "mippv2_x100_register_blocked_lmul8", RRR, 66},
    {UKernelType::mippv2_panel_x100_lmul8, "mippv2_panel_x100_lmul8", CRR, 66},
#endif
};

inline const KernelDescriptor &getKernelDescriptor(UKernelType type) {
  for (const auto &k : kernelTable) {
    if (k.type == type)
      return k;
  }

  throw std::runtime_error("Unknown kernel");
}

inline const KernelDescriptor &getKernelDescriptor(const std::string &name) {
  for (const auto &k : kernelTable) {
    if (k.name == name)
      return k;
  }

  throw std::runtime_error("Unknown GEMM kernel: " + name);
}

template <typename T> class GemmUKernel {
public:
#include "GemmUKernel_opti.hpp"

#ifdef GEMMBENCH_ENABLE_EXPLO
#include "GemmUKernel_explo.hpp"
#endif
};

} // namespace GEMMBench
