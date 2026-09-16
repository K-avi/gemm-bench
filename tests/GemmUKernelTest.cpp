#include <catch2/catch_test_macros.hpp>

#include "Alloc.h"
#include "GemmUKernel.h"
#include "BenchConfig.h"

#include <cmath>

using namespace GEMMBench;

#define TEST_KERNEL(NAME, CALL)                                        \
    SECTION(NAME)                                                      \
    {                                                                  \
        std::fill_n(C_test.data,                                       \
                    C_test.padded_rows * C_test.padded_cols,           \
                    T{});                                               \
                                                                       \
        CALL;                                                          \
                                                                       \
        checkMatrixEqual(C_ref, C_test);                               \
    }

template<typename MA, typename MB>
static void checkMatrixEqual(const MA& A,
                             const MB& B,
                             typename MA::value_type eps =
                                 typename MA::value_type(1e-5))
{
    using T = typename MA::value_type;

    static_assert(std::is_same_v<T, typename MB::value_type>);

    REQUIRE(A.rows == B.rows);
    REQUIRE(A.cols == B.cols);

    for(size_t i = 0; i < A.rows; ++i)
    {
        for(size_t j = 0; j < A.cols; ++j)
        {
            const T a = A(i, j);
            const T b = B(i, j);

            const T diff = std::abs(a - b);

            const bool close =
                diff <= eps ||
                diff <= eps * std::abs(b);

            INFO("Mismatch at (" << i << ", " << j << ") "
                 << "expected=" << b
                 << " got=" << a
                 << " diff=" << diff);

            REQUIRE(close);
        }
    }
}


template<
    typename T,
    class AMatrix,
    class BMatrix,
    class CMatrix>
static void testGemmSuite()
{

    constexpr size_t M = 32;
    constexpr size_t N = 32;
    constexpr size_t K = 32;

    auto& cfg = GEMMBench::config();

    SimdAlloc<T> alloc(
        cfg.packet_size,
        cfg.lmul,
        cfg.alignment,
        cfg.seed);

    auto A = alloc.template allocatePacked<AMatrix>(
        M,
        K,
        InitMode::Random);

    auto B = alloc.template allocatePacked<BMatrix>(
        K,
        N,
        InitMode::Random);

    auto C_ref = alloc.template allocatePacked<CMatrix>(
        M,
        N,
        InitMode::Zero);

    auto C_test = alloc.template allocatePacked<CMatrix>(
        M,
        N,
        InitMode::Zero);

    GemmUKernel<T> gemm;

    /*
     * Reference implementation.
     *
     * The naive ijk kernel is intentionally used
     * as the correctness oracle.
     */
    gemm.gemm_ijk(A, B, C_ref);

#define TEST_KERNEL(NAME, CALL)                                        \
    SECTION(NAME)                                                       \
    {                                                                   \
        std::fill_n(C_test.data,                                        \
                    C_test.padded_rows * C_test.padded_cols,            \
                    T{});                                                \
                                                                        \
        CALL;                                                           \
                                                                        \
        checkMatrixEqual(C_ref, C_test);                                \
    }

    TEST_KERNEL(
        "IKJ scalar GEMM",
        gemm.gemm_ikj(A, B, C_test));

    TEST_KERNEL(
        "Blocked scalar GEMM",
        gemm.gemm_blocked(A, B, C_test));

    TEST_KERNEL(
        "Blocked scalar GEMM unroll2",
        gemm.gemm_blocked_unroll2(A, B, C_test));

    TEST_KERNEL(
        "Blocked scalar GEMM unroll4",
        gemm.gemm_blocked_unroll4(A, B, C_test));

    TEST_KERNEL(
        "Blocked register blocked scalar GEMM",
        gemm.gemm_blocked_register_blocked(A, B, C_test));

    TEST_KERNEL(
        "MIPPv2 GEMM",
        gemm.gemm_mippv2(A, B, C_test));

    TEST_KERNEL(
        "MIPPv2 GEMM lmul=2",
        gemm.template gemm_mippv2<2>(A, B, C_test));

    TEST_KERNEL(
        "MIPPv2 GEMM lmul=4",
        gemm.template gemm_mippv2<4>(A, B, C_test));

    TEST_KERNEL(
        "MIPPv2 GEMM lmul=8",
        gemm.template gemm_mippv2<8>(A, B, C_test));

    TEST_KERNEL(
        "Blocked MIPPv2 GEMM",
        gemm.gemm_mippv2_blocked(A, B, C_test));

    TEST_KERNEL(
        "Blocked MIPPv2 GEMM lmul=2",
        gemm.template gemm_mippv2_blocked<2>(A, B, C_test));

    TEST_KERNEL(
        "Blocked MIPPv2 GEMM lmul=4",
        gemm.template gemm_mippv2_blocked<4>(A, B, C_test));

    TEST_KERNEL(
        "Blocked MIPPv2 GEMM lmul=8",
        gemm.template gemm_mippv2_blocked<8>(A, B, C_test));

    TEST_KERNEL(
        "Blocked MIPPv2 GEMM unroll2",
        gemm.gemm_mippv2_blocked_unroll2(A, B, C_test));

    TEST_KERNEL(
        "Blocked MIPPv2 GEMM unroll4",
        gemm.gemm_mippv2_blocked_unroll4(A, B, C_test));

    TEST_KERNEL(
        "Blocked MIPPv2 GEMM unroll jam4",
        gemm.gemm_mippv2_blocked_unroll_jam4(A, B, C_test));

    TEST_KERNEL(
        "MIPPv2 Skylake register blocked",
        gemm.gemm_mippv2_skylake_register_blocked(A, B, C_test));

    TEST_KERNEL(
        "MIPPv2 blocked register blocked",
        gemm.gemm_mippv2_blocked_register_blocked(A, B, C_test));

    // TEST_KERNEL(
    //     "MIPPv2 register blocked HOH",
    //     gemm.gemm_mippv2_register_blocked_hoh(A, B, C_test));
    
    TEST_KERNEL(
        "MIPPv2 Skylake LMUL register blocked",
        gemm.template gemm_mippv2_skylake_lmul_register_blocked<1>(A, B, C_test));

    TEST_KERNEL(
        "MIPPv2 Skylake LMUL2 register blocked",
        gemm.template gemm_mippv2_skylake_lmul_register_blocked<2>(A, B, C_test));

    TEST_KERNEL(
        "MIPPv2 Skylake LMUL4 register blocked",
        gemm.template gemm_mippv2_skylake_lmul_register_blocked<4>(A, B, C_test));



    // test x100 register blocked kernels
    TEST_KERNEL(
        "MIPPv2 x100 register blocked LMUL1",
        gemm.template gemm_mippv2_x100_register_blocked<1>(A, B, C_test));
    TEST_KERNEL(
        "MIPPv2 x100 register blocked LMUL2",
        gemm.template gemm_mippv2_x100_register_blocked<2>(A, B, C_test));
    TEST_KERNEL(
        "MIPPv2 x100 register blocked LMUL4",
        gemm.template gemm_mippv2_x100_register_blocked<4>(A, B, C_test));
    TEST_KERNEL(
        "MIPPv2 x100 register blocked LMUL8",
        gemm.template gemm_mippv2_x100_register_blocked<8>(A, B, C_test));
    
    TEST_KERNEL(
        "MIPPv2 x100 register blocked apack4",
        gemm.gemm_mippv2_x100_register_blocked_apack4(A, B, C_test));
#undef TEST_KERNEL

    alloc.freePacked(A);
    alloc.freePacked(B);
    alloc.freePacked(C_ref);
    alloc.freePacked(C_test);
}

template<typename T>
static void testGemmSuiteCRR()
{
    constexpr size_t M = 13;
    constexpr size_t N = 17;
    constexpr size_t K = 11;

    auto& cfg = GEMMBench::config();

    SimdAlloc<T> alloc(
        cfg.packet_size,
        cfg.lmul,
        cfg.alignment,
        cfg.seed);

    //
    // CRR matrices
    //
    auto A = alloc.template allocatePacked<PackedColMajor<T>>(
        M, K, InitMode::Random);

    auto B = alloc.template allocatePacked<PackedRowMajor<T>>(
        K, N, InitMode::Random);

    auto C_test = alloc.template allocatePacked<PackedRowMajor<T>>(
        M, N, InitMode::Zero);

    auto C_ref = alloc.template allocatePacked<PackedRowMajor<T>>(
        M, N, InitMode::Zero);

    //
    // Reference matrices (RRR)
    //
    auto A_ref = alloc.template allocatePacked<PackedRowMajor<T>>(
        M, K, InitMode::Zero);

    // Copy A (CR) -> A_ref (RR)
    for (size_t i = 0; i < M; ++i)
        for (size_t k = 0; k < K; ++k)
            A_ref(i, k) = A(i, k);

    GemmUKernel<T> gemm;

    //
    // Scalar reference.
    //
    gemm.gemm_ijk(A_ref, B, C_ref);

#define TEST_KERNEL(NAME, CALL)                                        \
    SECTION(NAME)                                                       \
    {                                                                   \
        std::fill_n(C_test.data,                                        \
                    C_test.padded_rows * C_test.padded_cols,            \
                    T{});                                                \
                                                                        \
        CALL;                                                           \
                                                                        \
        checkMatrixEqual(C_test, C_ref);                                \
    }

    TEST_KERNEL(
        "MIPPv2 Skylake panel",
        gemm.gemm_mippv2_skylake_panel(A, B, C_test));

    TEST_KERNEL(
        "MIPPv2 Skylake panel LMUL1",
        gemm.template gemm_mippv2_skylake_panel_lmul<1>(A, B, C_test));

    TEST_KERNEL(
        "MIPPv2 Skylake panel LMUL2",
        gemm.template gemm_mippv2_skylake_panel_lmul<2>(A, B, C_test));

    TEST_KERNEL(
        "MIPPv2 Skylake panel LMUL4",
        gemm.template gemm_mippv2_skylake_panel_lmul<4>(A, B, C_test));

    TEST_KERNEL(
        "MIPPv2 Skylake panel LMUL8",
        gemm.template gemm_mippv2_skylake_panel_lmul<8>(A, B, C_test));

#undef TEST_KERNEL

    alloc.freePacked(A);
    alloc.freePacked(B);
    alloc.freePacked(C_test);
    alloc.freePacked(C_ref);
    alloc.freePacked(A_ref);
}

TEST_CASE("GEMM RRR kernels", "[gemm]")
{
    testGemmSuite<
        double,
        PackedRowMajor<double>,
        PackedRowMajor<double>,
        PackedRowMajor<double>>();
}

TEST_CASE("GEMM CRR kernels", "[gemm]")
{
    testGemmSuiteCRR<double>();
}