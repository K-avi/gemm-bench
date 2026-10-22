#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

namespace GEMMBench {

template <typename T>
class ColdCPool {
public:
  ColdCPool(size_t tile_bytes, size_t pool_mib, const T *initial_tile = nullptr, double beta = 0.0) {
    stride_bytes_ = ((tile_bytes + 63) / 64) * 64 + 64;
    const size_t pool_bytes = pool_mib * 1024 * 1024;
    const size_t req_count = (pool_bytes + stride_bytes_ - 1) / stride_bytes_;
    P_ = std::bit_ceil(std::max<size_t>(2, req_count));
    total_bytes_ = ((P_ * stride_bytes_ + 63) / 64) * 64;

    buffer_ = static_cast<char *>(std::aligned_alloc(64, total_bytes_));
    if (!buffer_) {
      throw std::bad_alloc();
    }

    if (beta == 0.0 || !initial_tile) {
      std::memset(buffer_, 0, total_bytes_);
    } else {
      for (size_t p = 0; p < P_; ++p) {
        char *dest = buffer_ + p * stride_bytes_;
        std::memcpy(dest, initial_tile, tile_bytes);
      }
    }

    pool_.resize(P_);
    for (size_t p = 0; p < P_; ++p) {
      pool_[p] = reinterpret_cast<T *>(buffer_ + p * stride_bytes_);
    }
  }

  ~ColdCPool() {
    if (buffer_) {
      std::free(buffer_);
    }
  }

  ColdCPool(const ColdCPool &) = delete;
  ColdCPool &operator=(const ColdCPool &) = delete;

  ColdCPool(ColdCPool &&o) noexcept
      : buffer_(o.buffer_), pool_(std::move(o.pool_)), P_(o.P_),
        stride_bytes_(o.stride_bytes_), total_bytes_(o.total_bytes_) {
    o.buffer_ = nullptr;
  }

  ColdCPool &operator=(ColdCPool &&o) noexcept {
    if (this != &o) {
      if (buffer_) std::free(buffer_);
      buffer_ = o.buffer_;
      pool_ = std::move(o.pool_);
      P_ = o.P_;
      stride_bytes_ = o.stride_bytes_;
      total_bytes_ = o.total_bytes_;
      o.buffer_ = nullptr;
    }
    return *this;
  }

  const std::vector<T *> &pool() const noexcept { return pool_; }
  size_t P() const noexcept { return P_; }
  size_t stride_bytes() const noexcept { return stride_bytes_; }
  size_t total_bytes() const noexcept { return total_bytes_; }

private:
  char *buffer_ = nullptr;
  std::vector<T *> pool_;
  size_t P_ = 0;
  size_t stride_bytes_ = 0;
  size_t total_bytes_ = 0;
};

} // namespace GEMMBench

