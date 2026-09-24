#pragma once

#include <cstddef>

namespace GEMMBench {

/**
 * @brief Enum of the 9 target CPU microarchitectures supported by GEMMBench.
 */
enum class TargetUArch {
  Skylake,       // Intel Skylake (AVX2)
  MeteorLake,    // Intel Meteor Lake Redwood Cove P-Core (AVX2)
  Zen4,          // AMD Zen 4 (AVX-512)
  Zen5,          // AMD Zen 5 Strix Point (AVX-512, double-pumped 256-bit FPU)
  AppleM1,       // Apple M1 Firestorm P-Core (NEON)
  CortexA76,     // ARM Cortex-A76 / Raspberry Pi 5 (NEON)
  SpacemiT_X100, // SpacemiT X100 on K3 SoC (RVV 1.0, VLEN=256 bits)
  SpacemiT_A100, // SpacemiT A100 AI Cores on K3 SoC (RVV 1.0, VLEN=1024 bits)
  SpacemiT_X60   // SpacemiT X60 on BPI-F3 SoC (RVV 1.0, VLEN=256 bits)
};

/**
 * @brief Primary template for microarchitectural cache and macro-kernel tiling
 * presets.
 *
 * Specializations define:
 * - Hardware cache characteristics (L1D, L2, L3, cluster topology)
 * - Champion microkernel name and optimal register tile dimensions (MR, NR)
 * - Default MacroKernel tiling dimensions (MC, KC, NC) respecting divisibility
 */
template <TargetUArch Arch> struct UArchMacroTraits;

// =============================================================================
// Intel Architectures (AVX2)
// =============================================================================

// Intel Skylake (e.g. Core i5-6200U)
// Cache: 32 KB L1D, 256 KB L2, 3 MB shared L3
template <> struct UArchMacroTraits<TargetUArch::Skylake> {
  static constexpr const char *name = "Intel Skylake";
  static constexpr const char *champion_kernel = "mippv2_meteorlake_mr4_nr3";
  static constexpr bool has_l3 = true;
  static constexpr size_t l1d_kb = 32;
  static constexpr size_t l2_kb = 256;
  static constexpr size_t l3_kb = 3072;

  // Champion register tile (mippv2_meteorlake_mr4_nr3: MR=4, NR=12 = 3 * VL, AVX2 VL=4)
  static constexpr size_t MR = 4;
  static constexpr size_t NR = 12;

  // Tiling presets: A in L2 (256 KB), B in L3 (3 MB)
  static constexpr size_t MC = 128; // 32 * MR, 128 * 256 * 8 = 256 KB
  static constexpr size_t KC = 256;
  static constexpr size_t NC = 1008; // 84 * NR, 256 * 1008 * 8 = 2016 KB <= 3072 KB
};

// Intel Meteor Lake (Redwood Cove P-core)
// Cache: 48 KB L1D, 2 MB L2, 24 MB shared L3
template <> struct UArchMacroTraits<TargetUArch::MeteorLake> {
  static constexpr const char *name = "Intel Meteor Lake (Redwood Cove)";
  static constexpr const char *champion_kernel = "mippv2_meteorlake_mr4_nr3";
  static constexpr bool has_l3 = true;
  static constexpr size_t l1d_kb = 48;
  static constexpr size_t l2_kb = 2048;
  static constexpr size_t l3_kb = 24576;

  // Champion register tile (mippv2_meteorlake_mr4_nr3: MR=4, NR=12 = 3 * VL, AVX2 VL=4)
  static constexpr size_t MR = 4;
  static constexpr size_t NR = 12;

  // Tiling presets: generous L2 allows larger MC and KC, B in L3 (24 MB)
  static constexpr size_t MC = 256; // 64 * MR
  static constexpr size_t KC = 512;
  static constexpr size_t NC = 2016; // 168 * NR, 512 * 2016 * 8 = 8064 KB <= 24 MB
};

// =============================================================================
// AMD Architectures (AVX-512)
// =============================================================================

// AMD Zen 4 (e.g. Ryzen 9 7900X on az4-a7900-3)
// Cache: 32 KB L1D, 1 MB L2, 32 MB local L3 per CCD (2x 32 MB = 64 MB package total)
template <> struct UArchMacroTraits<TargetUArch::Zen4> {
  static constexpr const char *name = "AMD Zen 4";
  static constexpr const char *champion_kernel = "mippv2_firestorm_mr4_nr4_fmaddi";
  static constexpr bool has_l3 = true;
  static constexpr size_t l1d_kb = 32;
  static constexpr size_t l2_kb = 1024;
  static constexpr size_t l3_kb = 32768; // Local L3 slice for CCD 0

  // Champion register tile (mippv2_firestorm_mr4_nr4_fmaddi: MR=4, NR=32 = 4 * VL, AVX-512 VL=8)
  static constexpr size_t MR = 4;
  static constexpr size_t NR = 32;

  // Tiling presets: A in L2 (1 MB), B in local L3 (32 MB)
  static constexpr size_t MC = 256; // 64 * MR, 256 * 512 * 8 = 1024 KB
  static constexpr size_t KC = 512;
  static constexpr size_t NC = 2048; // 64 * NR, 512 * 2048 * 8 = 8 MB <= 32 MB
};

// AMD Zen 5 Strix Point (Ryzen AI 9 HX 370 on az5-a890m-0)
// Cache: 48 KB L1D, 1 MB L2, 16 MB L3 local to Zen 5 cluster (8 MB separate for Zen 5c)
template <> struct UArchMacroTraits<TargetUArch::Zen5> {
  static constexpr const char *name = "AMD Zen 5 (Strix Point)";
  static constexpr const char *champion_kernel = "mippv2_firestorm_mr4_nr4_fmaddi";
  static constexpr bool has_l3 = true;
  static constexpr size_t l1d_kb = 48;
  static constexpr size_t l2_kb = 1024;
  static constexpr size_t l3_kb = 16384; // Local 16 MB L3 for Cores 0-3 (24 MB total package)

  // Champion register tile (mippv2_firestorm_mr4_nr4_fmaddi: MR=4, NR=32 = 4 * VL, AVX-512 VL=8)
  static constexpr size_t MR = 4;
  static constexpr size_t NR = 32;

  static constexpr size_t MC = 256; // 64 * MR
  static constexpr size_t KC = 512;
  static constexpr size_t NC = 2048; // 64 * NR, B_pack = 8 MB <= 16 MB local L3
};

// =============================================================================
// AArch64 Architectures (NEON)
// =============================================================================

// Apple M1 Firestorm P-core (M1 Ultra: 4 clusters de 4 cœurs P, L2 partagé 12 Mo, aucun L3)
// Les cœurs E Icestorm (L2 4 Mo) sont ignorés en HPC/GEMM.
template <> struct UArchMacroTraits<TargetUArch::AppleM1> {
  static constexpr const char *name = "Apple M1 Firestorm";
  static constexpr const char *champion_kernel = "mippv2_x60_mr6_nr4_fmaddi";
  static constexpr bool has_l3 = false;
  static constexpr size_t l1d_kb = 128;
  static constexpr size_t l2_cluster_kb = 12288; // 12 Mo partagé par cluster quad-core P
  static constexpr size_t cores_per_cluster = 4;
  static constexpr size_t l3_kb = 0;

  // Champion register tile (mippv2_x60_mr6_nr4_fmaddi: MR=6, NR=8 = 4 * VL, NEON VL=2)
  static constexpr size_t MR = 6;
  static constexpr size_t NR = 8;

  // Presets pour L2 de 12 Mo : A_pack (1.5 Mo) + B_pack (6 Mo) = 7.5 Mo <= 12 Mo
  static constexpr size_t MC = 384; // 64 * MR
  static constexpr size_t KC = 512;
  static constexpr size_t NC = 1536; // 192 * NR
};

// ARM Cortex-A76 (Raspberry Pi 5)
// Cache: 64 KB L1D, 512 KB private L2, 2 MB shared L3
template <> struct UArchMacroTraits<TargetUArch::CortexA76> {
  static constexpr const char *name = "ARM Cortex-A76 (RPi 5)";
  static constexpr const char *champion_kernel = "mippv2_a76_mr6_nr3";
  static constexpr bool has_l3 = true;
  static constexpr size_t l1d_kb = 64;
  static constexpr size_t l2_kb = 512;
  static constexpr size_t l3_kb = 2048;

  // Champion register tile (mippv2_a76_mr6_nr3: MR=6, NR=6 = 3 * VL, NEON VL=2)
  static constexpr size_t MR = 6;
  static constexpr size_t NR = 6;

  static constexpr size_t MC = 144; // 24 * MR
  static constexpr size_t KC = 256;
  static constexpr size_t NC = 768; // 128 * NR
};

// =============================================================================
// RISC-V Architectures (RVV 1.0 — No L3 Cache)
// =============================================================================

// SpacemiT X100 (K3 SoC: 2 clusters of 4x X100 cores)
// Cache: 64 KB L1D per core, 4 MiB (4096 KB) shared L2 per 4-core cluster, NO L3
template <> struct UArchMacroTraits<TargetUArch::SpacemiT_X100> {
  static constexpr const char *name = "SpacemiT X100 (K3)";
  static constexpr const char *champion_kernel = "mippv2_x100_register_blocked_apack4";
  static constexpr bool has_l3 = false;
  static constexpr size_t l1d_kb = 64;
  static constexpr size_t l2_cluster_kb = 4096; // 4 MiB shared cluster L2
  static constexpr size_t cores_per_cluster = 4;
  static constexpr size_t l3_kb = 0;

  // Champion register tile (mippv2_x100_register_blocked_apack4: MR=3, NR=8 = 2 * VL, RVV 256b VL=4)
  static constexpr size_t MR = 3;
  static constexpr size_t NR = 8;

  // Single-thread presets: A_pack (258 KB) + B_pack (2048 KB) = 2.3 MB in 4 MB L2
  static constexpr size_t MC = 126; // 42 * MR
  static constexpr size_t KC = 256;
  static constexpr size_t NC = 1024; // 128 * NR
};

// SpacemiT A100 (K3 SoC: 2 clusters of 4x A100 AI cores)
// Cache: 32 KB L1D per core, 1 MiB (1024 KB) shared L2 per 4-core cluster + 1.5 MiB TCM, NO L3
template <> struct UArchMacroTraits<TargetUArch::SpacemiT_A100> {
  static constexpr const char *name = "SpacemiT A100 (K3 AI)";
  static constexpr const char *champion_kernel = "mippv2_a100_mr7_nr2_lmul2_pipe";
  static constexpr bool has_l3 = false;
  static constexpr size_t l1d_kb = 32;
  static constexpr size_t l2_cluster_kb = 1024; // 1 MiB hardware L2 cache
  static constexpr size_t tcm_kb = 1536;        // 1.5 MiB software-managed TCM SRAM
  static constexpr size_t cores_per_cluster = 4;
  static constexpr size_t l3_kb = 0;

  // Champion register tile (mippv2_a100_mr7_nr2_lmul2_pipe: MR=7, NR=64 = 2 * VL with lmul=2, RVV 1024b VL=32)
  static constexpr size_t MR = 7;
  static constexpr size_t NR = 64;

  // Single-thread presets: A_pack (115 KB) + B_pack (524 KB) = 639 KB <= 1024 KB L2
  static constexpr size_t MC = 56;  // 8 * MR
  static constexpr size_t KC = 256;
  static constexpr size_t NC = 256; // 4 * NR
};

// SpacemiT X60 (Banana Pi BPI-F3)
// Cache: 32 KB L1D per core, 512 KB shared L2 per 4-core cluster, NO L3
template <> struct UArchMacroTraits<TargetUArch::SpacemiT_X60> {
  static constexpr const char *name = "SpacemiT X60 (BPI-F3)";
  static constexpr const char *champion_kernel = "mippv2_a100_mr7_nr4_pipe";
  static constexpr bool has_l3 = false;
  static constexpr size_t l1d_kb = 32;
  static constexpr size_t l2_cluster_kb = 512;
  static constexpr size_t cores_per_cluster = 4;
  static constexpr size_t l3_kb = 0;

  // Champion register tile (mippv2_a100_mr7_nr4_pipe: MR=7, NR=16 = 4 * VL, RVV 256b VL=4)
  static constexpr size_t MR = 7;
  static constexpr size_t NR = 16;

  // Tiling presets constrained to 512 KB L2:
  // A_pack: 98 * 128 * 8 = 100 KB
  // B_pack: 128 * 256 * 8 = 256 KB
  // Total = 356 KB <= 512 KB
  static constexpr size_t MC = 98;  // 14 * MR
  static constexpr size_t KC = 128;
  static constexpr size_t NC = 256; // 16 * NR
};

} // namespace GEMMBench
