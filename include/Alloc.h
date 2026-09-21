#pragma once

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <random>
#include <type_traits>

#include "mipp.hpp"
#include "StorageType.h"

namespace GEMMBench
{


template<typename T>
struct PackedMatBase
{
    using value_type = T;

    T* data = nullptr;

    // Logical dimensions
    size_t rows = 0;
    size_t cols = 0;

    // Physical dimensions
    size_t padded_rows = 0;
    size_t padded_cols = 0;

    // Leading dimension
    size_t ld = 0;
};

template<typename T>
struct PackedRowMajor : PackedMatBase<T>
{
    using Base = PackedMatBase<T>;

    using Base::data;
    using Base::rows;
    using Base::cols;
    using Base::padded_rows;
    using Base::padded_cols;
    using Base::ld;

    static constexpr PLayout order = PLayout::Row;

    T* operator[](size_t i)
    {
        return data + i * ld;
    }

    const T* operator[](size_t i) const
    {
        return data + i * ld;
    }

    T& operator()(size_t i, size_t j)
    {
        return data[i * ld + j];
    }

    const T& operator()(size_t i, size_t j) const
    {
        return data[i * ld + j];
    }

    T* ptr(size_t i = 0, size_t j = 0)
    {
        return data + i * ld + j;
    }

    const T* ptr(size_t i = 0, size_t j = 0) const
    {
        return data + i * ld + j;
    }
};
template<typename T>
struct PackedColMajor : PackedMatBase<T>
{
    using Base = PackedMatBase<T>;

    using Base::data;
    using Base::rows;
    using Base::cols;
    using Base::padded_rows;
    using Base::padded_cols;
    using Base::ld;

    static constexpr PLayout order = PLayout::Col;

    T* operator[](size_t j)
    {
        return data + j * ld;
    }

    const T* operator[](size_t j) const
    {
        return data + j * ld;
    }

    T& operator()(size_t i, size_t j)
    {
        return data[j * ld + i];
    }

    const T& operator()(size_t i, size_t j) const
    {
        return data[j * ld + i];
    }

    T* ptr(size_t i = 0, size_t j = 0)
    {
        return data + j * ld + i;
    }

    const T* ptr(size_t i = 0, size_t j = 0) const
    {
        return data + j * ld + i;
    }
};
enum class InitMode
{
    Uninitialized,
    Zero,
    Random
};



template<typename T>
class SimdAlloc
{
public:

    SimdAlloc(size_t packet_size,
              size_t lmul = 1,
              size_t alignment = 64,
              uint32_t seed = 0,
              size_t TileSize = 64)
        : packet_size_(packet_size),
          lmul_(lmul),
          alignment_(alignment),
          rng_(seed),
          TileSize(TileSize)
    {
        assert(packet_size_ > 0);
        assert(lmul_ > 0);
        // Alignment must be compatible with SIMD packet size (e.g. 64-byte cache line with 32-byte AVX2 register)
        assert(alignment_ % (packet_size_ * sizeof(T)) == 0 ||
               (packet_size_ * sizeof(T)) % alignment_ == 0);
    }

    size_t tileSize() const
    {
        return TileSize;
    }

    size_t packetSize() const
    {
        return packet_size_;
    }

    size_t lmul() const
    {
        return lmul_;
    }

    size_t simdWidth() const
    {
        return packet_size_ * lmul_;
    }

    size_t alignment() const
    {
        return alignment_;
    }

    size_t paddedVectorSize(size_t n) const
    {
        return roundUp(n, simdWidth());
    }

    size_t paddedRows(size_t rows) const
    {
        return roundUp(rows, TileSize);
    }

    size_t paddedCols(size_t cols) const
    {
        return roundUp(cols,
                       std::lcm(TileSize, simdWidth()));
    }

    T* allocateVector(size_t size,
                      InitMode init = InitMode::Random)
    {
        const size_t padded = paddedVectorSize(size);

        T* ptr = allocate(padded);

        initialize(ptr, padded, init);

        return ptr;
    }

    template<class M = PackedRowMajor<T>>
    M allocatePacked(size_t rows,
                     size_t cols,
                     InitMode init = InitMode::Random)
    {
        M m;

        m.rows = rows;
        m.cols = cols;

        m.padded_rows = paddedRows(rows);
        m.padded_cols = paddedCols(cols);

        if constexpr (M::order == PLayout::Row)
            m.ld = m.padded_cols;
        else
            m.ld = m.padded_rows;

        const size_t count =
            m.padded_rows * m.padded_cols;

        m.data = allocate(count);

        // Zero the entire physical allocation.
        std::fill_n(m.data, count, T{});

        initializeLogical(m, init);

        return m;
    }

    void freeVector(T* ptr) const
    {
        std::free(ptr);
    }

    template<class M>
    void freePacked(M& m) const
    {
        std::free(m.data);

        m.data = nullptr;

        m.rows = 0;
        m.cols = 0;

        m.padded_rows = 0;
        m.padded_cols = 0;

        m.ld = 0;
    }

    void setSeed(uint32_t seed)
    {
        rng_.seed(seed);
    }


private:

    static size_t roundUp(size_t value,
                          size_t multiple)
    {
        return ((value + multiple - 1) / multiple) * multiple;
    }

    T* allocate(size_t count) const
    {
        const size_t bytes = roundUp(count * sizeof(T), alignment_);

        T* ptr =
            static_cast<T*>(
                std::aligned_alloc(alignment_, bytes));

        assert(ptr != nullptr);

        return ptr;
    }

    template<class M>
    void initializeLogical(M& m,
                           InitMode mode)
    {
        switch(mode)
        {
        case InitMode::Uninitialized:
            return;

        case InitMode::Zero:
            // Already zeroed.
            return;

        case InitMode::Random:

            for(size_t i = 0; i < m.rows; ++i)
            {
                for(size_t j = 0; j < m.cols; ++j)
                {
                    m(i, j) = randomValue();
                }
            }

            return;
        }
    }

    void initialize(T* ptr,
                    size_t count,
                    InitMode mode)
    {
        switch(mode)
        {
        case InitMode::Uninitialized:
            return;

        case InitMode::Zero:
            std::fill_n(ptr, count, T{});
            return;

        case InitMode::Random:

            for(size_t i = 0; i < count; ++i)
            {
                ptr[i] = randomValue();
            }

            return;
        }
    }

    T randomValue()
    {
        if constexpr(std::floating_point<T>)
        {
            std::uniform_real_distribution<T> dist(T(-1), T(1));
            return dist(rng_);
        }
        else if constexpr(std::integral<T>)
        {
            std::uniform_int_distribution<T> dist(-100, 100);
            return dist(rng_);
        }
        else
        {
            static_assert(std::is_arithmetic_v<T>,
                          "Unsupported element type");
        }
    }


    size_t packet_size_;
    size_t lmul_;
    size_t alignment_;
    size_t TileSize = 64;

    std::mt19937 rng_;
};


} // namespace GEMMBench
