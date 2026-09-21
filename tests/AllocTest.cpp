#include <catch2/catch_test_macros.hpp>

#include "Alloc.h"

#include <cstdint>


using namespace GEMMBench;


TEST_CASE("SIMD allocations are correctly aligned",
          "[alloc]")
{
    using T = float;


    SimdAlloc<T> alloc(
        8,      // AVX2 float packet
        1,
        32
    );


    auto v = alloc.allocateVector(
        13,
        InitMode::Uninitialized
    );


    const auto address =
        reinterpret_cast<uintptr_t>(v);


    REQUIRE(address % 32 == 0);


    alloc.freeVector(v);
}

TEST_CASE("Vector allocation respects LMUL padding",
          "[alloc]")
{
    using T = float;

    constexpr size_t packet_size = 8;
    constexpr size_t lmul = 4;


    SimdAlloc<T> alloc(
        packet_size,
        lmul,
        32
    );


    constexpr size_t size = 13;


    auto v = alloc.allocateVector(
        size,
        InitMode::Uninitialized
    );


    /*
     * Expected:
     *
     * packet_size*lmul = 32 elements
     *
     * round_up(13,32) = 32
     */
    constexpr size_t expected = 32;


    REQUIRE(
        alloc.paddedVectorSize(size) == expected
    );


    REQUIRE(
        alloc.simdWidth() == packet_size * lmul
    );


    alloc.freeVector(v);
}

TEST_CASE("Matrix allocation respects tile and SIMD padding",
          "[alloc]")
{
    using T = float;

    constexpr size_t packet_size = 8;
    constexpr size_t lmul = 2;
    constexpr size_t tile = 64;


    SimdAlloc<T> alloc(
        packet_size,
        lmul,
        32,
        0,
        tile
    );


    auto m = alloc.allocatePacked(
        70,
        70,
        InitMode::Random
    );


    REQUIRE(m.rows == 70);
    REQUIRE(m.cols == 70);


    // Rows are padded to tile size
    REQUIRE(m.padded_rows == 128);


    // Columns are padded to lcm(tile, SIMD width)
    //
    // SIMD width = 8*2 = 16
    // lcm(64,16)=64
    //
    REQUIRE(m.padded_cols == 128);


    REQUIRE(m.ld == m.padded_cols);


    alloc.freePacked(m);
}

TEST_CASE("Matrix padding is initialized to zero",
          "[alloc]")
{
    using T = float;

    SimdAlloc<T> alloc(
        8,
        1,
        32
    );


    auto m = alloc.allocatePacked(
        5,
        7,
        InitMode::Random
    );


    for(size_t i = 0; i < m.padded_rows; ++i)
    {
        for(size_t j = 0; j < m.padded_cols; ++j)
        {
            if(i >= m.rows || j >= m.cols)
            {
                REQUIRE(m[i][j] == T{});
            }
        }
    }


    alloc.freePacked(m);
}


TEST_CASE("Matrix leading dimension respects padding",
          "[alloc]")
{
    using T = double;


    SimdAlloc<T> alloc(
        8,
        2,
        64
    );


    constexpr size_t rows = 17;
    constexpr size_t cols = 19;


    auto m = alloc.allocatePacked(
        rows,
        cols,
        InitMode::Zero
    );


    /*
     * SIMD width:
     *
     * packet_size*lmul = 8*2 = 16 doubles
     *
     * GEMM tile:
     *
     * TileSize = 64
     *
     * lcm(64,16)=64
     *
     * round_up(19,64)=64
     */
    REQUIRE(m.ld == 64);

    REQUIRE(m.rows == rows);
    REQUIRE(m.cols == cols);

    REQUIRE(m.padded_cols == 64);
    REQUIRE(m.padded_rows == 64);


    const auto address =
        reinterpret_cast<uintptr_t>(m.data);


    REQUIRE(address % 64 == 0);


    alloc.freePacked(m);
}

TEST_CASE("ColMajor matrix allocation respects padding and layout",
          "[alloc]")
{
    using T = double;

    SimdAlloc<T> alloc(
        8,
        1,
        64
    );

    constexpr size_t rows = 15;
    constexpr size_t cols = 23;

    auto m = alloc.allocatePacked<PackedColMajor<T>>(
        rows,
        cols,
        InitMode::Random
    );

    REQUIRE(m.rows == rows);
    REQUIRE(m.cols == cols);
    REQUIRE(m.padded_rows == 64);
    REQUIRE(m.padded_cols == 64);
    REQUIRE(m.ld == m.padded_rows);

    const auto address = reinterpret_cast<uintptr_t>(m.data);
    REQUIRE(address % 64 == 0);

    m(0, 0) = 42.0;
    m(14, 22) = 84.0;
    REQUIRE(m(0, 0) == 42.0);
    REQUIRE(m(14, 22) == 84.0);
    REQUIRE(m[0][0] == 42.0);
    REQUIRE(m[22][14] == 84.0);

    alloc.freePacked(m);
}