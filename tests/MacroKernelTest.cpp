#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include "macrokernel/MacroKernel.h"
#include "macrokernel/MicroKernelAdapter.h"
#include "macrokernel/UArchMacroTraits.h"

using namespace GEMMBench;

template <TargetUArch Arch>
void verify_uarch_traits() {
    using T = UArchMacroTraits<Arch>;
    STATIC_REQUIRE(T::MR > 0);
    STATIC_REQUIRE(T::NR > 0);
    STATIC_REQUIRE(T::MC > 0);
    STATIC_REQUIRE(T::KC > 0);
    STATIC_REQUIRE(T::NC > 0);
    // Strict divisibility required for MacroKernel tiling
    STATIC_REQUIRE(T::MC % T::MR == 0);
    STATIC_REQUIRE(T::NC % T::NR == 0);
}

TEST_CASE("MacroKernel - Champion Traits & Presets Divisibility Test", "[macro][compilation]") {
    // 1. Verify divisibility across all 9 target architectures
    verify_uarch_traits<TargetUArch::Skylake>();
    verify_uarch_traits<TargetUArch::MeteorLake>();
    verify_uarch_traits<TargetUArch::Zen4>();
    verify_uarch_traits<TargetUArch::Zen5>();
    verify_uarch_traits<TargetUArch::AppleM1>();
    verify_uarch_traits<TargetUArch::CortexA76>();
    verify_uarch_traits<TargetUArch::SpacemiT_X100>();
    verify_uarch_traits<TargetUArch::SpacemiT_A100>();
    verify_uarch_traits<TargetUArch::SpacemiT_X60>();

    // 2. Exact check on Skylake (Champion: mippv2_meteorlake_mr4_nr3)
    using SkylakeTraits = UArchMacroTraits<TargetUArch::Skylake>;
    STATIC_REQUIRE(SkylakeTraits::MR == 4);
    STATIC_REQUIRE(SkylakeTraits::NR == 12);
    STATIC_REQUIRE(SkylakeTraits::MC == 128);
    STATIC_REQUIRE(SkylakeTraits::NC == 1008);
    REQUIRE(std::string(SkylakeTraits::champion_kernel) == "mippv2_meteorlake_mr4_nr3");

    // 3. Exact check on Meteor Lake (Champion: mippv2_meteorlake_mr4_nr3)
    using MeteorTraits = UArchMacroTraits<TargetUArch::MeteorLake>;
    STATIC_REQUIRE(MeteorTraits::MR == 4);
    STATIC_REQUIRE(MeteorTraits::NR == 12);
    STATIC_REQUIRE(MeteorTraits::MC == 256);
    STATIC_REQUIRE(MeteorTraits::NC == 2016);
    REQUIRE(std::string(MeteorTraits::champion_kernel) == "mippv2_meteorlake_mr4_nr3");

    // 4. Exact check on Zen 4 (Champion: mippv2_firestorm_mr4_nr4_fmaddi)
    using Zen4Traits = UArchMacroTraits<TargetUArch::Zen4>;
    STATIC_REQUIRE(Zen4Traits::has_l3 == true);
    STATIC_REQUIRE(Zen4Traits::MR == 4);
    STATIC_REQUIRE(Zen4Traits::NR == 32); // 4 * VL with AVX-512 VL=8
    STATIC_REQUIRE(Zen4Traits::MC == 256);
    STATIC_REQUIRE(Zen4Traits::NC == 2048);
    REQUIRE(std::string(Zen4Traits::champion_kernel) == "mippv2_firestorm_mr4_nr4_fmaddi");

    // 5. Exact check on Zen 5 (Champion: mippv2_firestorm_mr4_nr4_fmaddi)
    using Zen5Traits = UArchMacroTraits<TargetUArch::Zen5>;
    STATIC_REQUIRE(Zen5Traits::MR == 4);
    STATIC_REQUIRE(Zen5Traits::NR == 32);
    STATIC_REQUIRE(Zen5Traits::l3_kb == 16384);
    REQUIRE(std::string(Zen5Traits::champion_kernel) == "mippv2_firestorm_mr4_nr4_fmaddi");

    // 6. Exact check on Apple M1 (Champion: mippv2_x60_mr6_nr4_fmaddi)
    using M1Traits = UArchMacroTraits<TargetUArch::AppleM1>;
    STATIC_REQUIRE(M1Traits::has_l3 == false);
    STATIC_REQUIRE(M1Traits::MR == 6);
    STATIC_REQUIRE(M1Traits::NR == 8);
    STATIC_REQUIRE(M1Traits::MC == 384);
    STATIC_REQUIRE(M1Traits::NC == 1536);
    REQUIRE(std::string(M1Traits::champion_kernel) == "mippv2_x60_mr6_nr4_fmaddi");

    // 7. Exact check on Cortex-A76 (Champion: mippv2_a76_mr6_nr3)
    using A76Traits = UArchMacroTraits<TargetUArch::CortexA76>;
    STATIC_REQUIRE(A76Traits::MR == 6);
    STATIC_REQUIRE(A76Traits::NR == 6);
    STATIC_REQUIRE(A76Traits::MC == 144);
    STATIC_REQUIRE(A76Traits::NC == 768);
    REQUIRE(std::string(A76Traits::champion_kernel) == "mippv2_a76_mr6_nr3");

    // 8. Exact check on SpacemiT X100 (Champion: mippv2_x100_register_blocked_apack4)
    using X100Traits = UArchMacroTraits<TargetUArch::SpacemiT_X100>;
    STATIC_REQUIRE(X100Traits::has_l3 == false);
    STATIC_REQUIRE(X100Traits::l1d_kb == 64);
    STATIC_REQUIRE(X100Traits::l2_cluster_kb == 4096);
    STATIC_REQUIRE(X100Traits::MR == 3);
    STATIC_REQUIRE(X100Traits::NR == 8);
    STATIC_REQUIRE(X100Traits::MC == 126); // 42 * 3
    STATIC_REQUIRE(X100Traits::NC == 1024);
    REQUIRE(std::string(X100Traits::champion_kernel) == "mippv2_x100_register_blocked_apack4");

    // 9. Exact check on SpacemiT A100 (Champion: mippv2_a100_mr7_nr2_lmul2_pipe)
    using A100Traits = UArchMacroTraits<TargetUArch::SpacemiT_A100>;
    STATIC_REQUIRE(A100Traits::has_l3 == false);
    STATIC_REQUIRE(A100Traits::l2_cluster_kb == 1024);
    STATIC_REQUIRE(A100Traits::tcm_kb == 1536);
    STATIC_REQUIRE(A100Traits::MR == 7);
    STATIC_REQUIRE(A100Traits::NR == 64); // 2 * VL with lmul=2, VL=32
    STATIC_REQUIRE(A100Traits::MC == 56);  // 8 * 7
    STATIC_REQUIRE(A100Traits::NC == 256); // 4 * 64
    REQUIRE(std::string(A100Traits::champion_kernel) == "mippv2_a100_mr7_nr2_lmul2_pipe");

    // 10. Exact check on SpacemiT X60 (Champion: mippv2_a100_mr7_nr4_pipe)
    using X60Traits = UArchMacroTraits<TargetUArch::SpacemiT_X60>;
    STATIC_REQUIRE(X60Traits::has_l3 == false);
    STATIC_REQUIRE(X60Traits::l2_cluster_kb == 512);
    STATIC_REQUIRE(X60Traits::MR == 7);
    STATIC_REQUIRE(X60Traits::NR == 16); // 4 * VL with VL=4
    STATIC_REQUIRE(X60Traits::MC == 98);  // 14 * 7
    STATIC_REQUIRE(X60Traits::NC == 256); // 16 * 16
    REQUIRE(std::string(X60Traits::champion_kernel) == "mippv2_a100_mr7_nr4_pipe");

    // Check concept compliance on scalar policy
    STATIC_REQUIRE(MicroKernelPolicy<Scalar_IJK_Policy<double, 4, 4>, double>);

    // Verify concept compliance and exact MR/NR alignment across all 9 champion policies
    STATIC_REQUIRE(MicroKernelPolicy<UArchChampionPolicy_t<TargetUArch::Skylake, double>, double>);
    STATIC_REQUIRE(UArchChampionPolicy_t<TargetUArch::Skylake, double>::MR == SkylakeTraits::MR);
    STATIC_REQUIRE(UArchChampionPolicy_t<TargetUArch::Skylake, double>::NR == SkylakeTraits::NR);

    STATIC_REQUIRE(MicroKernelPolicy<UArchChampionPolicy_t<TargetUArch::MeteorLake, double>, double>);
    STATIC_REQUIRE(UArchChampionPolicy_t<TargetUArch::MeteorLake, double>::MR == MeteorTraits::MR);
    STATIC_REQUIRE(UArchChampionPolicy_t<TargetUArch::MeteorLake, double>::NR == MeteorTraits::NR);

    STATIC_REQUIRE(MicroKernelPolicy<UArchChampionPolicy_t<TargetUArch::Zen4, double>, double>);
    STATIC_REQUIRE(UArchChampionPolicy_t<TargetUArch::Zen4, double>::MR == Zen4Traits::MR);
    STATIC_REQUIRE(UArchChampionPolicy_t<TargetUArch::Zen4, double>::NR == Zen4Traits::NR);

    STATIC_REQUIRE(MicroKernelPolicy<UArchChampionPolicy_t<TargetUArch::Zen5, double>, double>);
    STATIC_REQUIRE(UArchChampionPolicy_t<TargetUArch::Zen5, double>::MR == Zen5Traits::MR);
    STATIC_REQUIRE(UArchChampionPolicy_t<TargetUArch::Zen5, double>::NR == Zen5Traits::NR);

    STATIC_REQUIRE(MicroKernelPolicy<UArchChampionPolicy_t<TargetUArch::AppleM1, double>, double>);
    STATIC_REQUIRE(UArchChampionPolicy_t<TargetUArch::AppleM1, double>::MR == M1Traits::MR);
    STATIC_REQUIRE(UArchChampionPolicy_t<TargetUArch::AppleM1, double>::NR == M1Traits::NR);

    STATIC_REQUIRE(MicroKernelPolicy<UArchChampionPolicy_t<TargetUArch::CortexA76, double>, double>);
    STATIC_REQUIRE(UArchChampionPolicy_t<TargetUArch::CortexA76, double>::MR == A76Traits::MR);
    STATIC_REQUIRE(UArchChampionPolicy_t<TargetUArch::CortexA76, double>::NR == A76Traits::NR);

    STATIC_REQUIRE(MicroKernelPolicy<UArchChampionPolicy_t<TargetUArch::SpacemiT_X100, double>, double>);
    STATIC_REQUIRE(UArchChampionPolicy_t<TargetUArch::SpacemiT_X100, double>::MR == X100Traits::MR);
    STATIC_REQUIRE(UArchChampionPolicy_t<TargetUArch::SpacemiT_X100, double>::NR == X100Traits::NR);

    STATIC_REQUIRE(MicroKernelPolicy<UArchChampionPolicy_t<TargetUArch::SpacemiT_A100, double>, double>);
    STATIC_REQUIRE(UArchChampionPolicy_t<TargetUArch::SpacemiT_A100, double>::MR == A100Traits::MR);
    STATIC_REQUIRE(UArchChampionPolicy_t<TargetUArch::SpacemiT_A100, double>::NR == A100Traits::NR);

    STATIC_REQUIRE(MicroKernelPolicy<UArchChampionPolicy_t<TargetUArch::SpacemiT_X60, double>, double>);
    STATIC_REQUIRE(UArchChampionPolicy_t<TargetUArch::SpacemiT_X60, double>::MR == X60Traits::MR);
    STATIC_REQUIRE(UArchChampionPolicy_t<TargetUArch::SpacemiT_X60, double>::NR == X60Traits::NR);

    // Instantiate MacroKernel with scalar policy
    using TestMacro = MacroKernel<double, 16, 16, 16, Scalar_IJK_Policy<double, 4, 4>>;
    STATIC_REQUIRE(TestMacro::MR == 4);
    STATIC_REQUIRE(TestMacro::NR == 4);

    // Verify default UArchMacroKernel alias without passing explicit policy
    using DefaultZen4 = UArchMacroKernel<TargetUArch::Zen4, double>;
    STATIC_REQUIRE(DefaultZen4::MR == 4);
    STATIC_REQUIRE(DefaultZen4::NR == 32);

    // Call stub gemm (should execute without crash)
    double A[16 * 16] = {0};
    double B[16 * 16] = {0};
    double C[16 * 16] = {0};

    TestMacro::gemm(16, 16, 16, 1.0, A, 16, B, 16, 0.0, C, 16);
    REQUIRE(true);
}
