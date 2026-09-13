#pragma once

#include "Alloc.h"
#include "StorageType.h"
#include "mipp.hpp"
#include "mipp_custom.h"

#include <algorithm>
#include <cassert>
#include <cstddef>

namespace GEMMBench {

enum class UKernelType {
  IJK,
  IKJ,
  IJK_RC,
  IKJ_RC,

  blocked,
  blocked_unroll2,
  blocked_unroll4,
  blocked_register_blocked,

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

  mippv2_register_blocked,
  mippv2_blocked_register_blocked,
  mippv2_register_blocked_hoh,

  mippv2_lmul_register_blocked,
  mippv2_lmul2_register_blocked,
  mippv2_lmul4_register_blocked,
  mippv2_lmul8_register_blocked,

  mippv2_panel,
  mippv2_panel_lmul,
  mippv2_panel_lmul2,
  mippv2_panel_lmul4,
  mippv2_panel_lmul8,

  mippv2_dot,

  mippv2_x100_lmul1,
  mippv2_x100_lmul2,
  mippv2_x100_lmul4,
  mippv2_x100_lmul8,

  mippv2_x100_register_blocked_lmul1,
  mippv2_x100_register_blocked_lmul2,
  mippv2_x100_register_blocked_lmul4,
  mippv2_x100_register_blocked_lmul8,

  mippv2_x100_register_blocked_apack4,

  mippv2_panel_x100_lmul1,
  mippv2_panel_x100_lmul2,
  mippv2_panel_x100_lmul4,
  mippv2_panel_x100_lmul8,
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
    // Scalar reference kernels

    {UKernelType::IJK, "ijk", RRR},

    {UKernelType::IKJ, "ikj", RRR},

    {UKernelType::IJK_RC, "ijk_rc", RCR},

    {UKernelType::IKJ_RC, "ikj_rc", RCR},

    // Scalar blocked kernels

    {UKernelType::blocked, "blocked", RRR},

    {UKernelType::blocked_unroll2, "blocked_unroll2", RRR},

    {UKernelType::blocked_unroll4, "blocked_unroll4", RRR},

    {UKernelType::blocked_register_blocked, "blocked_register_blocked", RRR},

    // MIPPv2 basic kernels

    {UKernelType::mippv2, "mippv2", RRR},

    {UKernelType::mippv2_lmul2, "mippv2_lmul2", RRR},

    {UKernelType::mippv2_lmul4, "mippv2_lmul4", RRR},

    {UKernelType::mippv2_lmul8, "mippv2_lmul8", RRR},

    // MIPPv2 blocked kernels

    {UKernelType::mippv2_blocked, "mippv2_blocked", RRR},

    {UKernelType::mippv2_blocked_lmul2, "mippv2_blocked_lmul2", RRR},

    {UKernelType::mippv2_blocked_lmul4, "mippv2_blocked_lmul4", RRR},

    {UKernelType::mippv2_blocked_lmul8, "mippv2_blocked_lmul8", RRR},

    // MIPPv2 blocked unrolled

    {UKernelType::mippv2_blocked_unroll2, "mippv2_blocked_unroll2", RRR},

    {UKernelType::mippv2_blocked_unroll4, "mippv2_blocked_unroll4", RRR},

    {UKernelType::mippv2_blocked_unroll_jam4, "mippv2_blocked_unroll_jam4",
     RRR},

    // MIPPv2 register blocked

    {UKernelType::mippv2_register_blocked, "mippv2_register_blocked", RRR},

    {UKernelType::mippv2_blocked_register_blocked,
     "mippv2_blocked_register_blocked", RRR},

    {UKernelType::mippv2_register_blocked_hoh, "mippv2_register_blocked_hoh",
     RRR},

    // LMUL register blocked

    {UKernelType::mippv2_lmul_register_blocked, "mippv2_lmul_register_blocked",
     RRR},

    {UKernelType::mippv2_lmul2_register_blocked,
     "mippv2_lmul2_register_blocked", RRR},

    {UKernelType::mippv2_lmul4_register_blocked,
     "mippv2_lmul4_register_blocked", RRR},

    {UKernelType::mippv2_lmul8_register_blocked,
     "mippv2_lmul8_register_blocked", RRR},

    {UKernelType::mippv2_panel, "mippv2_panel", CRR},

    {UKernelType::mippv2_panel_lmul, "mippv2_panel_lmul", CRR},

    {UKernelType::mippv2_panel_lmul2, "mippv2_panel_lmul2", CRR},

    {UKernelType::mippv2_panel_lmul4, "mippv2_panel_lmul4", CRR},

    {UKernelType::mippv2_panel_lmul8, "mippv2_panel_lmul8", CRR},

    {UKernelType::mippv2_dot, "mippv2_dot", RCR},

    {UKernelType::mippv2_x100_lmul1, "mippv2_x100_lmul1", RRR},

    {UKernelType::mippv2_x100_lmul2, "mippv2_x100_lmul2", RRR},

    {UKernelType::mippv2_x100_lmul4, "mippv2_x100_lmul4", RRR},

    {UKernelType::mippv2_x100_lmul8, "mippv2_x100_lmul8", RRR},

    {UKernelType::mippv2_x100_register_blocked_lmul1,
     "mippv2_x100_register_blocked_lmul1", RRR},

    {UKernelType::mippv2_x100_register_blocked_lmul2,
     "mippv2_x100_register_blocked_lmul2", RRR},

    {UKernelType::mippv2_x100_register_blocked_lmul4,
     "mippv2_x100_register_blocked_lmul4", RRR},

    {UKernelType::mippv2_x100_register_blocked_lmul8,
     "mippv2_x100_register_blocked_lmul8", RRR},

    {UKernelType::mippv2_x100_register_blocked_apack4,
     "mippv2_x100_register_blocked_apack4", RRR},

    {UKernelType::mippv2_panel_x100_lmul1, "mippv2_panel_x100_lmul1", CRR},

    {UKernelType::mippv2_panel_x100_lmul2, "mippv2_panel_x100_lmul2", CRR},

    {UKernelType::mippv2_panel_x100_lmul4, "mippv2_panel_x100_lmul4", CRR},

    {UKernelType::mippv2_panel_x100_lmul8, "mippv2_panel_x100_lmul8", CRR}

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

  static void gemm_ijk(const PackedRowMajor<T> &A, const PackedRowMajor<T> &B,
                       PackedRowMajor<T> &C) {
    for (size_t i = 0; i < A.rows; ++i) {
      for (size_t j = 0; j < B.cols; ++j) {
        T sum = T{};

        for (size_t k = 0; k < A.cols; ++k) {
          sum += A[i][k] * B[k][j];
        }

        C[i][j] = sum;
      }
    }
  }

  static void gemm_ijk_rc(const PackedRowMajor<T> &A,
                          const PackedColMajor<T> &B, PackedRowMajor<T> &C) {
    for (size_t i = 0; i < A.rows; ++i) {
      for (size_t j = 0; j < B.cols; ++j) {
        T sum = T{};

        for (size_t k = 0; k < A.cols; ++k) {
          sum += A(i, k) * B(k, j);
        }

        C(i, j) = sum;
      }
    }
  }

  static inline void gemm_ikj(const PackedRowMajor<T> &A,
                              const PackedRowMajor<T> &B,
                              PackedRowMajor<T> &C) {

    for (size_t i = 0; i < A.rows; ++i) {
      for (size_t k = 0; k < A.cols; ++k) {
        const T aik = A[i][k];
        for (size_t j = 0; j < B.cols; ++j) {
          C[i][j] += aik * B[k][j];
        }
      }
    }
  }

  static inline void gemm_ikj_rc(const PackedRowMajor<T> &A,
                                 const PackedColMajor<T> &B,
                                 PackedRowMajor<T> &C) {
    for (size_t i = 0; i < A.rows; ++i) {
      for (size_t k = 0; k < A.cols; ++k) {
        const T aik = A[i][k];

        for (size_t j = 0; j < B.cols; ++j) {
          C[i][j] += aik * B(k, j);
        }
      }
    }
  }

  static inline void gemm_blocked(const PackedRowMajor<T> &A,
                                  const PackedRowMajor<T> &B,
                                  PackedRowMajor<T> &C) {
    constexpr size_t BS = 64;

    // Initialize C
    for (size_t i = 0; i < C.rows; ++i) {
      for (size_t j = 0; j < C.cols; ++j) {
        C[i][j] = T{};
      }
    }

    for (size_t ii = 0; ii < A.rows; ii += BS) {
      const size_t i_end = std::min(ii + BS, A.rows);

      for (size_t kk = 0; kk < A.cols; kk += BS) {
        const size_t k_end = std::min(kk + BS, A.cols);

        for (size_t jj = 0; jj < B.cols; jj += BS) {
          const size_t j_end = std::min(jj + BS, B.cols);

          for (size_t i = ii; i < i_end; ++i) {
            for (size_t k = kk; k < k_end; ++k) {
              const T aik = A[i][k];

              for (size_t j = jj; j < j_end; ++j) {
                C[i][j] += aik * B[k][j];
              }
            }
          }
        }
      }
    }
  }

  static inline void
  gemm_blocked_register_blocked(const PackedRowMajor<T> &__restrict A,
                                const PackedRowMajor<T> &__restrict B,
                                PackedRowMajor<T> &__restrict C) {
    constexpr size_t MR = 4;
    constexpr size_t NR = 2;

    for (size_t i = 0; i < A.rows; i += MR) {
      for (size_t j = 0; j < B.cols; j += NR) {
        /*
         * 4x2 scalar tile
         *
         * c00 c01
         * c10 c11
         * c20 c21
         * c30 c31
         */
        T c00 = 0;
        T c01 = 0;

        T c10 = 0;
        T c11 = 0;

        T c20 = 0;
        T c21 = 0;

        T c30 = 0;
        T c31 = 0;

        const auto a0_ptr = A[i + 0];
        const auto a1_ptr = A[i + 1];

        const auto a2_ptr = A[i + 2];
        const auto a3_ptr = A[i + 3];

        for (size_t k = 0; k < A.cols; ++k) {
          T b0 = B[k][j];
          T b1 = B[k][j + 1];

          T a0 = a0_ptr[k];
          T a1 = a1_ptr[k];
          T a2 = a2_ptr[k];
          T a3 = a3_ptr[k];

          c00 += a0 * b0;
          c01 += a0 * b1;

          c10 += a1 * b0;
          c11 += a1 * b1;

          c20 += a2 * b0;
          c21 += a2 * b1;

          c30 += a3 * b0;
          c31 += a3 * b1;
        }

        const auto c0_ptr = C[i + 0] + j;
        const auto c1_ptr = C[i + 1] + j;
        const auto c2_ptr = C[i + 2] + j;
        const auto c3_ptr = C[i + 3] + j;

        *c0_ptr = c00;
        *(c0_ptr + 1) = c01;

        *c1_ptr = c10;
        *(c1_ptr + 1) = c11;

        *c2_ptr = c20;
        *(c2_ptr + 1) = c21;

        *c3_ptr = c30;
        *(c3_ptr + 1) = c31;
      }
    }
  }

  static inline void gemm_blocked_unroll2(const PackedRowMajor<T> &A,
                                          const PackedRowMajor<T> &B,
                                          PackedRowMajor<T> &C) {
    constexpr size_t BS = 64;

    // Initialize C
    for (size_t i = 0; i < C.rows; ++i) {
      for (size_t j = 0; j < C.cols; ++j) {
        C[i][j] = T{};
      }
    }

    for (size_t ii = 0; ii < A.rows; ii += BS) {
      const size_t i_end = std::min(ii + BS, A.rows);

      for (size_t kk = 0; kk < A.cols; kk += BS) {
        const size_t k_end = std::min(kk + BS, A.cols);

        for (size_t jj = 0; jj < B.cols; jj += BS) {
          const size_t j_end = std::min(jj + BS, B.cols);

          for (size_t i = ii; i < i_end; ++i) {
            for (size_t k = kk; k < k_end; ++k) {
              const T aik = A[i][k];

              size_t j = jj;

              // j-unroll x2
              for (; j + 1 < j_end; j += 2) {
                C[i][j] += aik * B[k][j];
                C[i][j + 1] += aik * B[k][j + 1];
              }

              // remainder
              for (; j < j_end; ++j) {
                C[i][j] += aik * B[k][j];
              }
            }
          }
        }
      }
    }
  }

  static inline void gemm_blocked_unroll4(const PackedRowMajor<T> &A,
                                          const PackedRowMajor<T> &B,
                                          PackedRowMajor<T> &C) {
    constexpr size_t BS = 64;

    // Initialize C
    for (size_t i = 0; i < C.rows; ++i) {
      for (size_t j = 0; j < C.cols; ++j) {
        C[i][j] = T{};
      }
    }

    for (size_t ii = 0; ii < A.rows; ii += BS) {
      const size_t i_end = std::min(ii + BS, A.rows);

      for (size_t kk = 0; kk < A.cols; kk += BS) {
        const size_t k_end = std::min(kk + BS, A.cols);

        for (size_t jj = 0; jj < B.cols; jj += BS) {
          const size_t j_end = std::min(jj + BS, B.cols);

          for (size_t i = ii; i < i_end; ++i) {
            for (size_t k = kk; k < k_end; ++k) {
              const T aik = A[i][k];

              size_t j = jj;

              // j-unroll x4
              for (; j + 3 < j_end; j += 4) {
                C[i][j] += aik * B[k][j];
                C[i][j + 1] += aik * B[k][j + 1];
                C[i][j + 2] += aik * B[k][j + 2];
                C[i][j + 3] += aik * B[k][j + 3];
              }

              // remainder
              for (; j < j_end; ++j) {
                C[i][j] += aik * B[k][j];
              }
            }
          }
        }
      }
    }
  }

  template <int lmul = 1>
  static inline void gemm_mippv2_naive(const PackedRowMajor<T> &A,
                                       const PackedRowMajor<T> &B,
                                       PackedRowMajor<T> &C) {
    using namespace mipp;

    constexpr size_t BS = N<T, lmul>();

    for (size_t i = 0; i < A.rows; ++i) {
      for (size_t k = 0; k < A.cols; ++k) {
        const auto a = set1<T, lmul>(A[i][k]);

        for (size_t j = 0; j < B.cols; j += BS) {
          auto c = load<T, lmul>(C[i] + j);
          auto b = load<T, lmul>(B[k] + j);

          c = fmadd(a, b, c);

          store(C[i] + j, c);
        }
      }
    }
  }

  template <int lmul = 1>
  static inline void gemm_mippv2(const PackedRowMajor<T> &A,
                                 const PackedRowMajor<T> &B,
                                 PackedRowMajor<T> &C) {
    using namespace mipp;

    constexpr size_t BS = N<T, lmul>();

    for (size_t i = 0; i < A.rows; ++i) {
      for (size_t j = 0; j < B.cols; j += BS) {
        auto c = set0<T, lmul>();

        for (size_t k = 0; k < A.cols; ++k) {
          const auto a = set1<T, lmul>(A[i][k]);
          const auto b = load<T, lmul>(B[k] + j);

          c = fmadd(a, b, c);
        }

        store(C[i] + j, c);
      }
    }
  }

  template <int lmul = 1, size_t BS = 64>
  static inline void gemm_mippv2_blocked(const PackedRowMajor<T> &A,
                                         const PackedRowMajor<T> &B,
                                         PackedRowMajor<T> &C) {
    size_t counter = 0;
    using namespace mipp;

    constexpr size_t SIMD_SIZE = N<T, lmul>();

    // Initialize C
    for (size_t i = 0; i < C.rows; ++i) {
      for (size_t j = 0; j < C.cols; j += SIMD_SIZE) {
        store(C[i] + j, set0<T, lmul>());
      }
    }

    for (size_t ii = 0; ii < A.rows; ii += BS) {
      const size_t i_end = std::min(ii + BS, A.rows);
      for (size_t jj = 0; jj < B.cols; jj += BS) {
        const size_t j_end = std::min(jj + BS, B.cols);
        for (size_t kk = 0; kk < A.cols; kk += BS) {
          const size_t k_end = std::min(kk + BS, A.cols);
          for (size_t i = ii; i < i_end; ++i) {
            for (size_t j = jj; j < j_end; j += SIMD_SIZE) {
              auto c = load<T, lmul>(C[i] + j);

              for (size_t k = kk; k < k_end; ++k) {
                const auto a = set1<T, lmul>(A[i][k]);
                const auto b = load<T, lmul>(B[k] + j);

                c = fmadd(a, b, c);
              }

              store(C[i] + j, c);
            }
          }
        }
      }
    }
  }

  template <int lmul = 1, size_t BS = 64>
  static inline void gemm_mippv2_blocked_unroll2(const PackedRowMajor<T> &A,
                                                 const PackedRowMajor<T> &B,
                                                 PackedRowMajor<T> &C) {
    using namespace mipp;

    constexpr size_t SIMD_SIZE = N<T, lmul>();

    // Initialize C
    for (size_t i = 0; i < C.rows; ++i) {
      for (size_t j = 0; j < C.cols; j += SIMD_SIZE) {
        store(C[i] + j, set0<T, lmul>());
      }
    }

    for (size_t ii = 0; ii < A.rows; ii += BS) {
      const size_t i_end = std::min(ii + BS, A.rows);

      for (size_t jj = 0; jj < B.cols; jj += BS) {
        const size_t j_end = std::min(jj + BS, B.cols);

        for (size_t kk = 0; kk < A.cols; kk += BS) {
          const size_t k_end = std::min(kk + BS, A.cols);

          for (size_t i = ii; i < i_end; ++i) {
            for (size_t j = jj; j < j_end; j += SIMD_SIZE) {
              auto c = load<T, lmul>(C[i] + j);

              size_t k = kk;

              for (; k + 1 < k_end; k += 2) {
                const auto a0 = set1<T, lmul>(A[i][k]);
                const auto b0 = load<T, lmul>(B[k] + j);

                c = fmadd(a0, b0, c);

                const auto a1 = set1<T, lmul>(A[i][k + 1]);
                const auto b1 = load<T, lmul>(B[k + 1] + j);

                c = fmadd(a1, b1, c);
              }

              // remainder
              for (; k < k_end; ++k) {
                const auto a = set1<T, lmul>(A[i][k]);
                const auto b = load<T, lmul>(B[k] + j);

                c = fmadd(a, b, c);
              }

              store(C[i] + j, c);
            }
          }
        }
      }
    }
  }
  template <int lmul = 1, size_t BS = 64>
  static inline void gemm_mippv2_blocked_unroll4(const PackedRowMajor<T> &A,
                                                 const PackedRowMajor<T> &B,
                                                 PackedRowMajor<T> &C) {
    using namespace mipp;

    constexpr size_t SIMD_SIZE = N<T, lmul>();

    // Initialize C
    for (size_t i = 0; i < C.rows; ++i) {
      for (size_t j = 0; j < C.cols; j += SIMD_SIZE) {
        store(C[i] + j, set0<T, lmul>());
      }
    }

    for (size_t ii = 0; ii < A.rows; ii += BS) {
      const size_t i_end = std::min(ii + BS, A.rows);

      for (size_t jj = 0; jj < B.cols; jj += BS) {
        const size_t j_end = std::min(jj + BS, B.cols);

        for (size_t kk = 0; kk < A.cols; kk += BS) {
          const size_t k_end = std::min(kk + BS, A.cols);

          for (size_t i = ii; i < i_end; ++i) {
            for (size_t j = jj; j < j_end; j += SIMD_SIZE) {
              auto c = load<T, lmul>(C[i] + j);

              size_t k = kk;

              for (; k + 3 < k_end; k += 4) {
                const auto a0 = set1<T, lmul>(A[i][k]);
                const auto b0 = load<T, lmul>(B[k] + j);
                c = fmadd(a0, b0, c);

                const auto a1 = set1<T, lmul>(A[i][k + 1]);
                const auto b1 = load<T, lmul>(B[k + 1] + j);
                c = fmadd(a1, b1, c);

                const auto a2 = set1<T, lmul>(A[i][k + 2]);
                const auto b2 = load<T, lmul>(B[k + 2] + j);
                c = fmadd(a2, b2, c);

                const auto a3 = set1<T, lmul>(A[i][k + 3]);
                const auto b3 = load<T, lmul>(B[k + 3] + j);
                c = fmadd(a3, b3, c);
              }

              // remainder
              for (; k < k_end; ++k) {
                const auto a = set1<T, lmul>(A[i][k]);
                const auto b = load<T, lmul>(B[k] + j);

                c = fmadd(a, b, c);
              }

              store(C[i] + j, c);
            }
          }
        }
      }
    }
  }

  template <int lmul = 1, size_t BS = 64>
  static inline void gemm_mippv2_blocked_unroll_jam4(const PackedRowMajor<T> &A,
                                                     const PackedRowMajor<T> &B,
                                                     PackedRowMajor<T> &C) {
    using namespace mipp;

    constexpr size_t SIMD_SIZE = N<T, lmul>();
    constexpr size_t NJ = 4;

    // Initialize C
    for (size_t i = 0; i < C.rows; ++i) {
      for (size_t j = 0; j < C.cols; j += SIMD_SIZE) {
        store(C[i] + j, set0<T, lmul>());
      }
    }

    for (size_t ii = 0; ii < A.rows; ii += BS) {
      const size_t i_end = std::min(ii + BS, A.rows);

      for (size_t jj = 0; jj < B.cols; jj += BS) {
        const size_t j_end = std::min(jj + BS, B.cols);

        for (size_t kk = 0; kk < A.cols; kk += BS) {
          const size_t k_end = std::min(kk + BS, A.cols);

          for (size_t i = ii; i < i_end; ++i) {
            size_t j = jj;

            // J unroll & jam factor 4
            for (; j + NJ * SIMD_SIZE <= j_end; j += NJ * SIMD_SIZE) {
              auto c0 = load<T, lmul>(C[i] + j);
              auto c1 = load<T, lmul>(C[i] + j + SIMD_SIZE);
              auto c2 = load<T, lmul>(C[i] + j + 2 * SIMD_SIZE);
              auto c3 = load<T, lmul>(C[i] + j + 3 * SIMD_SIZE);

              for (size_t k = kk; k < k_end; ++k) {
                const auto a = set1<T, lmul>(A[i][k]);

                const auto b0 = load<T, lmul>(B[k] + j);
                const auto b1 = load<T, lmul>(B[k] + j + SIMD_SIZE);
                const auto b2 = load<T, lmul>(B[k] + j + 2 * SIMD_SIZE);
                const auto b3 = load<T, lmul>(B[k] + j + 3 * SIMD_SIZE);

                c0 = fmadd(a, b0, c0);
                c1 = fmadd(a, b1, c1);
                c2 = fmadd(a, b2, c2);
                c3 = fmadd(a, b3, c3);
              }

              store(C[i] + j, c0);
              store(C[i] + j + SIMD_SIZE, c1);
              store(C[i] + j + 2 * SIMD_SIZE, c2);
              store(C[i] + j + 3 * SIMD_SIZE, c3);
            }

            // Remainder
            for (; j < j_end; j += SIMD_SIZE) {
              auto c = load<T, lmul>(C[i] + j);

              for (size_t k = kk; k < k_end; ++k) {
                const auto a = set1<T, lmul>(A[i][k]);
                const auto b = load<T, lmul>(B[k] + j);

                c = fmadd(a, b, c);
              }

              store(C[i] + j, c);
            }
          }
        }
      }
    }
  }

  template <int lmul = 1, size_t BS = 64>
  static inline void
  gemm_mippv2_blocked_register_blocked(const PackedRowMajor<T> &A,
                                       const PackedRowMajor<T> &B,
                                       PackedRowMajor<T> &C) {
    using namespace mipp;

    constexpr size_t SIMD_SIZE = N<T, lmul>();
    constexpr size_t NR = 4;
    constexpr size_t BLOCK_J = NR * SIMD_SIZE;

    // Initialize C
    for (size_t i = 0; i < C.rows; ++i) {
      for (size_t j = 0; j < C.cols; j += SIMD_SIZE) {
        store(C[i] + j, set0<T, lmul>());
      }
    }

    for (size_t ii = 0; ii < A.rows; ii += BS) {
      const size_t i_end = std::min(ii + BS, A.rows);

      for (size_t jj = 0; jj < B.cols; jj += BS) {
        const size_t j_end = std::min(jj + BS, B.cols);

        for (size_t kk = 0; kk < A.cols; kk += BS) {
          const size_t k_end = std::min(kk + BS, A.cols);

          for (size_t i = ii; i < i_end; ++i) {
            const T *a_row = A[i];
            T *c_row = C[i];

            for (size_t j = jj; j < j_end; j += BLOCK_J) {
              auto c0 = load<T, lmul>(c_row + j);
              auto c1 = load<T, lmul>(c_row + j + SIMD_SIZE);
              auto c2 = load<T, lmul>(c_row + j + 2 * SIMD_SIZE);
              auto c3 = load<T, lmul>(c_row + j + 3 * SIMD_SIZE);

              for (size_t k = kk; k < k_end; ++k) {
                const auto a = set1<T, lmul>(a_row[k]);

                const T *b_row = B[k];

                const auto b0 = load<T, lmul>(b_row + j);
                const auto b1 = load<T, lmul>(b_row + j + SIMD_SIZE);
                const auto b2 = load<T, lmul>(b_row + j + 2 * SIMD_SIZE);
                const auto b3 = load<T, lmul>(b_row + j + 3 * SIMD_SIZE);

                c0 = fmadd(a, b0, c0);
                c1 = fmadd(a, b1, c1);
                c2 = fmadd(a, b2, c2);
                c3 = fmadd(a, b3, c3);
              }

              store(c_row + j, c0);
              store(c_row + j + SIMD_SIZE, c1);
              store(c_row + j + 2 * SIMD_SIZE, c2);
              store(c_row + j + 3 * SIMD_SIZE, c3);
            }
          }
        }
      }
    }
  }

  template <int lmul = 1>
  static inline void
  gemm_mippv2_register_blocked(const PackedRowMajor<T> &__restrict A,
                               const PackedRowMajor<T> &__restrict B,
                               PackedRowMajor<T> &__restrict C) {
    using namespace mipp;

    constexpr size_t VL = N<T, lmul>();

    constexpr size_t MR = 4;
    constexpr size_t NR = 2 * VL;

    for (size_t i = 0; i < A.rows; i += MR) {
      for (size_t j = 0; j < B.cols; j += NR) {
        /*
         * 4x2VL register tile
         *
         * c00 c01
         * c10 c11
         * c20 c21
         * c30 c31
         */

        auto c00 = set0<T, lmul>();
        auto c01 = set0<T, lmul>();

        auto c10 = set0<T, lmul>();
        auto c11 = set0<T, lmul>();

        auto c20 = set0<T, lmul>();
        auto c21 = set0<T, lmul>();

        auto c30 = set0<T, lmul>();
        auto c31 = set0<T, lmul>();

        const auto a0_ptr = A[i + 0];
        const auto a1_ptr = A[i + 1];

        const auto a2_ptr = A[i + 2];
        const auto a3_ptr = A[i + 3];

        for (size_t k = 0; k < A.cols; ++k) {
          const auto b0 = load<T, lmul>(B[k] + j);
          const auto b1 = load<T, lmul>(B[k] + j + VL);

          const auto a0 = set1<T, lmul>(a0_ptr[k]);
          const auto a1 = set1<T, lmul>(a1_ptr[k]);
          const auto a2 = set1<T, lmul>(a2_ptr[k]);
          const auto a3 = set1<T, lmul>(a3_ptr[k]);

          c00 = fmadd(a0, b0, c00);
          c01 = fmadd(a0, b1, c01);

          c10 = fmadd(a1, b0, c10);
          c11 = fmadd(a1, b1, c11);

          c20 = fmadd(a2, b0, c20);
          c21 = fmadd(a2, b1, c21);

          c30 = fmadd(a3, b0, c30);
          c31 = fmadd(a3, b1, c31);
        }

        const auto c0_ptr = C[i + 0] + j;
        const auto c1_ptr = C[i + 1] + j;
        const auto c2_ptr = C[i + 2] + j;
        const auto c3_ptr = C[i + 3] + j;
        store(c0_ptr, c00);
        store(c0_ptr + VL, c01);

        store(c1_ptr, c10);
        store(c1_ptr + VL, c11);

        store(c2_ptr, c20);
        store(c2_ptr + VL, c21);

        store(c3_ptr, c30);
        store(c3_ptr + VL, c31);
      }
    }
  }

  template <int lmul = 4>
  static inline void
  gemm_mippv2_lmul_register_blocked(const PackedRowMajor<T> &__restrict A,
                                    const PackedRowMajor<T> &__restrict B,
                                    PackedRowMajor<T> &__restrict C) {
    using namespace mipp;

    // constexpr int lmul = 4;

    constexpr size_t VL = N<T, lmul>();

    constexpr int MR = 2;
    constexpr int NR = VL;

    for (size_t i = 0; i < A.rows; i += MR) {
      for (size_t j = 0; j < B.cols; j += NR) {

        //
        // Register tile:
        //
        // c00
        // c10
        //

        auto c00 = set0<T, lmul>();
        auto c10 = set0<T, lmul>();

        const auto a0_ptr = A[i + 0];
        const auto a1_ptr = A[i + 1];

        for (size_t k = 0; k < A.cols; ++k) {
          const auto b0 = load<T, lmul>(B[k] + j);

          const auto a0 = set1<T, lmul>(a0_ptr[k]);
          const auto a1 = set1<T, lmul>(a1_ptr[k]);

          c00 = fmadd(a0, b0, c00);
          c10 = fmadd(a1, b0, c10);
        }

        store(C[i + 0] + j, c00);
        store(C[i + 1] + j, c10);
      }
    }
  }

  static inline void gemm_mippv2_panel(const PackedColMajor<T> &A,
                                       const PackedRowMajor<T> &B,
                                       PackedRowMajor<T> &C) {
    using namespace mipp;

    constexpr size_t MR = 4;
    constexpr size_t NR = 8;

    for (size_t i = 0; i < A.rows; i += MR) {
      for (size_t j = 0; j < B.cols; j += NR) {
        auto c00 = set0<T>();
        auto c01 = set0<T>();

        auto c10 = set0<T>();
        auto c11 = set0<T>();

        auto c20 = set0<T>();
        auto c21 = set0<T>();

        auto c30 = set0<T>();
        auto c31 = set0<T>();

        // #pragma unroll
        for (size_t k = 0; k < A.cols; ++k) {
          auto b0 = loadu<T>(&B(k, j + 0));
          auto b1 = loadu<T>(&B(k, j + 4));

          auto a0 = set1<T>(A(i + 0, k));
          auto a1 = set1<T>(A(i + 1, k));
          auto a2 = set1<T>(A(i + 2, k));
          auto a3 = set1<T>(A(i + 3, k));

          c00 = fmadd(a0, b0, c00);
          c01 = fmadd(a0, b1, c01);

          c10 = fmadd(a1, b0, c10);
          c11 = fmadd(a1, b1, c11);

          c20 = fmadd(a2, b0, c20);
          c21 = fmadd(a2, b1, c21);

          c30 = fmadd(a3, b0, c30);
          c31 = fmadd(a3, b1, c31);
        }

        storeu(&C(i + 0, j + 0), c00);
        storeu(&C(i + 0, j + 4), c01);

        storeu(&C(i + 1, j + 0), c10);
        storeu(&C(i + 1, j + 4), c11);

        storeu(&C(i + 2, j + 0), c20);
        storeu(&C(i + 2, j + 4), c21);

        storeu(&C(i + 3, j + 0), c30);
        storeu(&C(i + 3, j + 4), c31);
      }
    }
  }

  template <int lmul = 1, int MR = 4, int NR_PACKETS = 1>
  static inline void gemm_mippv2_panel_lmul(const PackedColMajor<T> &A,
                                            const PackedRowMajor<T> &B,
                                            PackedRowMajor<T> &C) {
    using namespace mipp;

    using VType = decltype(set0<T, lmul>());

    constexpr size_t VL = N<T, lmul>();
    constexpr size_t NR = NR_PACKETS * VL;

    for (size_t i = 0; i < A.rows; i += MR) {
      for (size_t j = 0; j < B.cols; j += NR) {
        std::array<std::array<VType, NR_PACKETS>, MR> c{};

#pragma unroll
        for (size_t r = 0; r < MR; ++r)
#pragma unroll
          for (size_t p = 0; p < NR_PACKETS; ++p)
            c[r][p] = set0<T, lmul>();

        for (size_t k = 0; k < A.cols; ++k) {
          std::array<VType, NR_PACKETS> b;

#pragma unroll
          for (size_t p = 0; p < NR_PACKETS; ++p)
            b[p] = load<T, lmul>(&B(k, j + p * VL));

#pragma unroll
          for (size_t r = 0; r < MR; ++r) {
            auto a = set1<T, lmul>(A(i + r, k));

#pragma unroll
            for (size_t p = 0; p < NR_PACKETS; ++p)
              c[r][p] = fmadd(a, b[p], c[r][p]);
          }
        }

#pragma unroll
        for (size_t r = 0; r < MR; ++r)
#pragma unroll
          for (size_t p = 0; p < NR_PACKETS; ++p)
            store(&C(i + r, j + p * VL), c[r][p]);
      }
    }
  }

  static inline void gemm_mippv2_dot(const PackedRowMajor<T> &A,
                                     const PackedColMajor<T> &B,
                                     PackedRowMajor<T> &C) {
    using namespace mipp;

    constexpr int lmul = 2;
    constexpr size_t VL = N<T, lmul>();

    constexpr size_t MR = 4;
    constexpr size_t NR = 2;

    for (size_t i = 0; i < A.rows; i += MR) {
      for (size_t j = 0; j < B.cols; j += NR) {
        auto c00 = set0<T, lmul>();
        auto c01 = set0<T, lmul>();

        auto c10 = set0<T, lmul>();
        auto c11 = set0<T, lmul>();

        auto c20 = set0<T, lmul>();
        auto c21 = set0<T, lmul>();

        auto c30 = set0<T, lmul>();
        auto c31 = set0<T, lmul>();

#pragma unroll
        for (size_t k = 0; k < A.cols; k += 2 * VL) {
          //
          // k
          //
          auto a0 = load<T, lmul>(&A(i + 0, k));
          auto a1 = load<T, lmul>(&A(i + 1, k));
          auto a2 = load<T, lmul>(&A(i + 2, k));
          auto a3 = load<T, lmul>(&A(i + 3, k));

          auto b0 = load<T, lmul>(&B(k, j + 0));
          auto b1 = load<T, lmul>(&B(k, j + 1));

          c00 = fmadd(a0, b0, c00);
          c01 = fmadd(a0, b1, c01);

          c10 = fmadd(a1, b0, c10);
          c11 = fmadd(a1, b1, c11);

          c20 = fmadd(a2, b0, c20);
          c21 = fmadd(a2, b1, c21);

          c30 = fmadd(a3, b0, c30);
          c31 = fmadd(a3, b1, c31);

          //
          // k + VL
          //
          a0 = load<T, lmul>(&A(i + 0, k + VL));
          a1 = load<T, lmul>(&A(i + 1, k + VL));
          a2 = load<T, lmul>(&A(i + 2, k + VL));
          a3 = load<T, lmul>(&A(i + 3, k + VL));

          b0 = load<T, lmul>(&B(k + VL, j + 0));
          b1 = load<T, lmul>(&B(k + VL, j + 1));

          c00 = fmadd(a0, b0, c00);
          c01 = fmadd(a0, b1, c01);

          c10 = fmadd(a1, b0, c10);
          c11 = fmadd(a1, b1, c11);

          c20 = fmadd(a2, b0, c20);
          c21 = fmadd(a2, b1, c21);

          c30 = fmadd(a3, b0, c30);
          c31 = fmadd(a3, b1, c31);
        }

        C(i + 0, j + 0) = hadd(c00);
        C(i + 0, j + 1) = hadd(c01);

        C(i + 1, j + 0) = hadd(c10);
        C(i + 1, j + 1) = hadd(c11);

        C(i + 2, j + 0) = hadd(c20);
        C(i + 2, j + 1) = hadd(c21);

        C(i + 3, j + 0) = hadd(c30);
        C(i + 3, j + 1) = hadd(c31);
      }
    }
  }

  template <int lmul = 1>
  static inline void gemm_mippv2_x100(const PackedRowMajor<T> &A,
                                      const PackedRowMajor<T> &B,
                                      PackedRowMajor<T> &C) {
    using namespace mipp;

    constexpr size_t BS = N<T, lmul>();

    for (size_t i = 0; i < A.rows; ++i) {
      for (size_t j = 0; j < B.cols; j += BS) {
        auto c = set0<T, lmul>();

        for (size_t k = 0; k < A.cols; ++k) {
          const auto a = A[i][k];
          const auto b = load<T, lmul>(B[k] + j);

          c = fmaddi(b, a, c);
        }

        store(C[i] + j, c);
      }
    }
  }

  template <int lmul = 4>
  static inline void
  gemm_mippv2_x100_register_blocked(const PackedRowMajor<T> &__restrict A,
                                    const PackedRowMajor<T> &__restrict B,
                                    PackedRowMajor<T> &__restrict C) {
    using namespace mipp;

    constexpr size_t VL = N<T, lmul>();

    constexpr size_t MR = 3;
    constexpr size_t NR = 2 * VL;

    const size_t Mfull = (A.rows / MR) * MR;

    //
    // Main 3-row kernel
    //
    for (size_t i = 0; i < Mfull; i += MR) {
      const T *a0_ptr = A[i + 0];
      const T *a1_ptr = A[i + 1];
      const T *a2_ptr = A[i + 2];

      for (size_t j = 0; j < B.cols; j += NR) {
        auto c00 = set0<T, lmul>();
        auto c01 = set0<T, lmul>();

        auto c10 = set0<T, lmul>();
        auto c11 = set0<T, lmul>();

        auto c20 = set0<T, lmul>();
        auto c21 = set0<T, lmul>();

        for (size_t k = 0; k < A.cols; ++k) {
          auto b0 = load<T, lmul>(B[k] + j);
          auto b1 = load<T, lmul>(B[k] + j + VL);

          c00 = fmaddi(b0, a0_ptr[k], c00);
          c01 = fmaddi(b1, a0_ptr[k], c01);

          c10 = fmaddi(b0, a1_ptr[k], c10);
          c11 = fmaddi(b1, a1_ptr[k], c11);

          c20 = fmaddi(b0, a2_ptr[k], c20);
          c21 = fmaddi(b1, a2_ptr[k], c21);
        }

        store(C[i + 0] + j, c00);
        store(C[i + 0] + j + VL, c01);

        store(C[i + 1] + j, c10);
        store(C[i + 1] + j + VL, c11);

        store(C[i + 2] + j, c20);
        store(C[i + 2] + j + VL, c21);
      }
    }

    //
    // Tail (1 or 2 remaining rows)
    //
    for (size_t i = Mfull; i < A.rows; ++i) {
      const T *a_ptr = A[i];

      for (size_t j = 0; j < B.cols; j += NR) {
        auto c0 = set0<T, lmul>();
        auto c1 = set0<T, lmul>();

        for (size_t k = 0; k < A.cols; ++k) {
          auto b0 = load<T, lmul>(B[k] + j);
          auto b1 = load<T, lmul>(B[k] + j + VL);

          c0 = fmaddi(b0, a_ptr[k], c0);
          c1 = fmaddi(b1, a_ptr[k], c1);
        }

        store(C[i] + j, c0);
        store(C[i] + j + VL, c1);
      }
    }
  }

  struct ABlock4 {
    double d0;
    double d1;
    double d2;
    double d3;
  };
  template <int lmul = 2>
  static inline void gemm_mippv2_x100_register_blocked_apack4(
      const PackedRowMajor<T> &__restrict A,
      const PackedRowMajor<T> &__restrict B, PackedRowMajor<T> &__restrict C) {
    using namespace mipp;

    constexpr size_t VL = N<T, lmul>();

    constexpr size_t MR = 3;
    constexpr size_t NR = 2 * VL;

    const size_t Mfull = (A.rows / MR) * MR;

    //
    // Main 3-row kernel
    //
    for (size_t i = 0; i < Mful l; i += MR) {
      const auto *a0_ptr = reinterpret_cast<const ABlock4 *>(A[i + 0]);
      const auto *a1_ptr = reinterpret_cast<const ABlock4 *>(A[i + 1]);
      const auto *a2_ptr = reinterpret_cast<const ABlock4 *>(A[i + 2]);

      for (size_t j = 0; j < B.cols; j += NR) {
        auto c00 = set0<T, lmul>();
        auto c01 = set0<T, lmul>();

        auto c10 = set0<T, lmul>();
        auto c11 = set0<T, lmul>();

        auto c20 = set0<T, lmul>();
        auto c21 = set0<T, lmul>();

        const size_t K4 = A.cols / 4;

        for (size_t kk = 0; kk < K4; ++kk) {
          const ABlock4 a0 = a0_ptr[kk];
          const ABlock4 a1 = a1_ptr[kk];
          const ABlock4 a2 = a2_ptr[kk];

          auto b00 = load<T, lmul>(B[4 * kk + 0] + j);
          auto b01 = load<T, lmul>(B[4 * kk + 0] + j + VL);

          c00 = fmaddi(b00, a0.d0, c00);
          c01 = fmaddi(b01, a0.d0, c01);

          c10 = fmaddi(b00, a1.d0, c10);
          c11 = fmaddi(b01, a1.d0, c11);

          c20 = fmaddi(b00, a2.d0, c20);
          c21 = fmaddi(b01, a2.d0, c21);

          b00 = load<T, lmul>(B[4 * kk + 1] + j);
          b01 = load<T, lmul>(B[4 * kk + 1] + j + VL);

          c00 = fmaddi(b00, a0.d1, c00);
          c01 = fmaddi(b01, a0.d1, c01);

          c10 = fmaddi(b00, a1.d1, c10);
          c11 = fmaddi(b01, a1.d1, c11);

          c20 = fmaddi(b00, a2.d1, c20);
          c21 = fmaddi(b01, a2.d1, c21);

          b00 = load<T, lmul>(B[4 * kk + 2] + j);
          b01 = load<T, lmul>(B[4 * kk + 2] + j + VL);

          c00 = fmaddi(b00, a0.d2, c00);
          c01 = fmaddi(b01, a0.d2, c01);

          c10 = fmaddi(b00, a1.d2, c10);
          c11 = fmaddi(b01, a1.d2, c11);

          c20 = fmaddi(b00, a2.d2, c20);
          c21 = fmaddi(b01, a2.d2, c21);

          b00 = load<T, lmul>(B[4 * kk + 3] + j);
          b01 = load<T, lmul>(B[4 * kk + 3] + j + VL);

          c00 = fmaddi(b00, a0.d3, c00);
          c01 = fmaddi(b01, a0.d3, c01);

          c10 = fmaddi(b00, a1.d3, c10);
          c11 = fmaddi(b01, a1.d3, c11);

          c20 = fmaddi(b00, a2.d3, c20);
          c21 = fmaddi(b01, a2.d3, c21);
        }

        store(C[i + 0] + j, c00);
        store(C[i + 0] + j + VL, c01);

        store(C[i + 1] + j, c10);
        store(C[i + 1] + j + VL, c11);

        store(C[i + 2] + j, c20);
        store(C[i + 2] + j + VL, c21);
      }
    }

    //
    // Remaining 1 or 2 rows
    //
    for (size_t i = Mfull; i < A.rows; ++i) {
      const auto *a_ptr = reinterpret_cast<const ABlock4 *>(A[i]);

      for (size_t j = 0; j < B.cols; j += NR) {
        auto c0 = set0<T, lmul>();
        auto c1 = set0<T, lmul>();

        const size_t K4 = A.cols / 4;

        for (size_t kk = 0; kk < K4; ++kk) {
          const ABlock4 a = a_ptr[kk];

          auto b0 = load<T, lmul>(B[4 * kk + 0] + j);
          auto b1 = load<T, lmul>(B[4 * kk + 0] + j + VL);

          c0 = fmaddi(b0, a.d0, c0);
          c1 = fmaddi(b1, a.d0, c1);

          b0 = load<T, lmul>(B[4 * kk + 1] + j);
          b1 = load<T, lmul>(B[4 * kk + 1] + j + VL);

          c0 = fmaddi(b0, a.d1, c0);
          c1 = fmaddi(b1, a.d1, c1);

          b0 = load<T, lmul>(B[4 * kk + 2] + j);
          b1 = load<T, lmul>(B[4 * kk + 2] + j + VL);

          c0 = fmaddi(b0, a.d2, c0);
          c1 = fmaddi(b1, a.d2, c1);

          b0 = load<T, lmul>(B[4 * kk + 3] + j);
          b1 = load<T, lmul>(B[4 * kk + 3] + j + VL);

          c0 = fmaddi(b0, a.d3, c0);
          c1 = fmaddi(b1, a.d3, c1);
        }

        store(C[i] + j, c0);
        store(C[i] + j + VL, c1);
      }
    }
  }

  template <int lmul = 4>
  static inline void
  gemm_mippv2_panel_x100(const PackedColMajor<T> &__restrict A,
                         const PackedRowMajor<T> &__restrict B,
                         PackedRowMajor<T> &__restrict C) {
    using namespace mipp;

    constexpr size_t VL = N<T, lmul>();

    constexpr size_t MR = 3;
    constexpr size_t NR = 2 * VL;

    for (size_t i = 0; i < A.rows; i += MR) {
      for (size_t j = 0; j < B.cols; j += NR) {
        auto c00 = set0<T, lmul>();
        auto c01 = set0<T, lmul>();

        auto c10 = set0<T, lmul>();
        auto c11 = set0<T, lmul>();

        auto c20 = set0<T, lmul>();
        auto c21 = set0<T, lmul>();

        for (size_t k = 0; k < A.cols; ++k) {
          auto b0 = load<T, lmul>(&B(k, j));
          auto b1 = load<T, lmul>(&B(k, j + VL));

          c00 = fmaddi(b0, A(i + 0, k), c00);
          c01 = fmaddi(b1, A(i + 0, k), c01);

          c10 = fmaddi(b0, A(i + 1, k), c10);
          c11 = fmaddi(b1, A(i + 1, k), c11);

          c20 = fmaddi(b0, A(i + 2, k), c20);
          c21 = fmaddi(b1, A(i + 2, k), c21);
        }

        store(&C(i + 0, j), c00);
        store(&C(i + 0, j + VL), c01);

        store(&C(i + 1, j), c10);
        store(&C(i + 1, j + VL), c11);

        store(&C(i + 2, j), c20);
        store(&C(i + 2, j + VL), c21);
      }
    }
  }
};

} // namespace GEMMBench