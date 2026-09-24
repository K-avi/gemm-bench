static void gemm_ijk_rc(const PackedRowMajor<T> &A, const PackedColMajor<T> &B,
                        PackedRowMajor<T> &C) {
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
                            const PackedRowMajor<T> &B, PackedRowMajor<T> &C) {

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

// =========================================================================
// Moved from GemmUKernel_opti.hpp (Non-champion kernels)
// =========================================================================

  static inline void
  gemm_blocked_register_blocked(const PackedRowMajor<T> &__restrict A,
                                const PackedRowMajor<T> &__restrict B,
                                PackedRowMajor<T> &__restrict C,
                                T alpha = T{1}, T beta = T{0}) {
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

        Epilogue::store(c0_ptr + 0, c00, alpha, beta);
        Epilogue::store(c0_ptr + 1, c01, alpha, beta);

        Epilogue::store(c1_ptr + 0, c10, alpha, beta);
        Epilogue::store(c1_ptr + 1, c11, alpha, beta);

        Epilogue::store(c2_ptr + 0, c20, alpha, beta);
        Epilogue::store(c2_ptr + 1, c21, alpha, beta);

        Epilogue::store(c3_ptr + 0, c30, alpha, beta);
        Epilogue::store(c3_ptr + 1, c31, alpha, beta);
      }
    }
  }

  template <int lmul = 1, bool FastPath = false>
  static inline void
  gemm_mippv2_skylake_register_blocked_core(const PackedRowMajor<T> &__restrict A,
                                            const PackedRowMajor<T> &__restrict B,
                                            PackedRowMajor<T> &__restrict C,
                                            T alpha = T{1}, T beta = T{0}) {
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
        Epilogue::store<T, lmul, FastPath>(c0_ptr, c00, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c0_ptr + VL, c01, alpha, beta);

        Epilogue::store<T, lmul, FastPath>(c1_ptr, c10, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_ptr + VL, c11, alpha, beta);

        Epilogue::store<T, lmul, FastPath>(c2_ptr, c20, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c2_ptr + VL, c21, alpha, beta);

        Epilogue::store<T, lmul, FastPath>(c3_ptr, c30, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c3_ptr + VL, c31, alpha, beta);
      }
    }
  }

  template <int lmul = 1>
  static inline void
  gemm_mippv2_skylake_register_blocked(const PackedRowMajor<T> &__restrict A,
                                       const PackedRowMajor<T> &__restrict B,
                                       PackedRowMajor<T> &__restrict C) {
    gemm_mippv2_skylake_register_blocked_core<lmul, true>(A, B, C);
  }

  template <int lmul = 1>
  static inline void
  gemm_mippv2_skylake_register_blocked(const PackedRowMajor<T> &__restrict A,
                                       const PackedRowMajor<T> &__restrict B,
                                       PackedRowMajor<T> &__restrict C,
                                       T alpha, T beta) {
    if (__builtin_expect(alpha == T{1} && beta == T{0}, 1)) {
      gemm_mippv2_skylake_register_blocked_core<lmul, true>(A, B, C);
    } else {
      gemm_mippv2_skylake_register_blocked_core<lmul, false>(A, B, C, alpha, beta);
    }
  }

  template <int lmul = 4>
  static inline void
  gemm_mippv2_skylake_lmul_register_blocked(const PackedRowMajor<T> &__restrict A,
                                            const PackedRowMajor<T> &__restrict B,
                                            PackedRowMajor<T> &__restrict C,
                                            T alpha = T{1}, T beta = T{0}) {
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

        Epilogue::store<T, lmul>(C[i + 0] + j, c00, alpha, beta);
        Epilogue::store<T, lmul>(C[i + 1] + j, c10, alpha, beta);
      }
    }
  }

    // ===========================================================================
  // Meteor Lake P-Core Kernel (Redwood Cove: MR=4, NR=3*VL = 12 doubles)
  // Optimal 4x3 register tile: 12 accumulators + 3 B vectors + 1 A broadcast
  // 100% Pure MIPPv2: lexical block scoping for zero vector register spills.
  // ===========================================================================

  // ===========================================================================
  template <int lmul = 1, bool FastPath = false>
  static inline void
  gemm_mippv2_zen4_mr4_nr4_core(const PackedRowMajor<T> &__restrict A,
                                const PackedRowMajor<T> &__restrict B,
                                PackedRowMajor<T> &__restrict C,
                                T alpha = T{1}, T beta = T{0}) {
    using namespace mipp;

    constexpr size_t VL = N<T, lmul>();
    constexpr size_t MR = 4;
    constexpr size_t NR4 = 4 * VL; // 32 doubles in AVX-512

    const size_t b_ld = B.ld;
    const size_t Mfull = (A.rows / MR) * MR;

    for (size_t i = 0; i < Mfull; i += MR) {
      const T *a0_base = A[i + 0];
      const T *a1_base = A[i + 1];
      const T *a2_base = A[i + 2];
      const T *a3_base = A[i + 3];

      T *c0_base = C[i + 0];
      T *c1_base = C[i + 1];
      T *c2_base = C[i + 2];
      T *c3_base = C[i + 3];

      size_t j = 0;

      // 1. Main 4x4 Tiles (16 accumulators, 16 FMAs per step, 2.0 FMAs/access)
      for (; j + NR4 <= B.cols; j += NR4) {
        auto c00 = set0<T, lmul>();
        auto c01 = set0<T, lmul>();
        auto c02 = set0<T, lmul>();
        auto c03 = set0<T, lmul>();

        auto c10 = set0<T, lmul>();
        auto c11 = set0<T, lmul>();
        auto c12 = set0<T, lmul>();
        auto c13 = set0<T, lmul>();

        auto c20 = set0<T, lmul>();
        auto c21 = set0<T, lmul>();
        auto c22 = set0<T, lmul>();
        auto c23 = set0<T, lmul>();

        auto c30 = set0<T, lmul>();
        auto c31 = set0<T, lmul>();
        auto c32 = set0<T, lmul>();
        auto c33 = set0<T, lmul>();

        const T *__restrict a0_k = a0_base;
        const T *__restrict a1_k = a1_base;
        const T *__restrict a2_k = a2_base;
        const T *__restrict a3_k = a3_base;
        const T *__restrict b_k = B.data + j;

        #pragma GCC unroll 1
        for (size_t k = 0; k < A.cols; ++k) {
          const auto b0 = load<T, lmul>(b_k + 0 * VL);
          const auto b1 = load<T, lmul>(b_k + 1 * VL);
          const auto b2 = load<T, lmul>(b_k + 2 * VL);
          const auto b3 = load<T, lmul>(b_k + 3 * VL);
          b_k += b_ld;

          {
            const auto a0 = set1<T, lmul>(*a0_k++);
            c00 = fmadd(a0, b0, c00);
            c01 = fmadd(a0, b1, c01);
            c02 = fmadd(a0, b2, c02);
            c03 = fmadd(a0, b3, c03);
          }
          {
            const auto a1 = set1<T, lmul>(*a1_k++);
            c10 = fmadd(a1, b0, c10);
            c11 = fmadd(a1, b1, c11);
            c12 = fmadd(a1, b2, c12);
            c13 = fmadd(a1, b3, c13);
          }
          {
            const auto a2 = set1<T, lmul>(*a2_k++);
            c20 = fmadd(a2, b0, c20);
            c21 = fmadd(a2, b1, c21);
            c22 = fmadd(a2, b2, c22);
            c23 = fmadd(a2, b3, c23);
          }
          {
            const auto a3 = set1<T, lmul>(*a3_k++);
            c30 = fmadd(a3, b0, c30);
            c31 = fmadd(a3, b1, c31);
            c32 = fmadd(a3, b2, c32);
            c33 = fmadd(a3, b3, c33);
          }
        }

        const auto c0_ptr = c0_base + j;
        const auto c1_ptr = c1_base + j;
        const auto c2_ptr = c2_base + j;
        const auto c3_ptr = c3_base + j;

        Epilogue::store<T, lmul, FastPath>(c0_ptr + 0 * VL, c00, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c0_ptr + 1 * VL, c01, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c0_ptr + 2 * VL, c02, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c0_ptr + 3 * VL, c03, alpha, beta);

        Epilogue::store<T, lmul, FastPath>(c1_ptr + 0 * VL, c10, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_ptr + 1 * VL, c11, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_ptr + 2 * VL, c12, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_ptr + 3 * VL, c13, alpha, beta);

        Epilogue::store<T, lmul, FastPath>(c2_ptr + 0 * VL, c20, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c2_ptr + 1 * VL, c21, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c2_ptr + 2 * VL, c22, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c2_ptr + 3 * VL, c23, alpha, beta);

        Epilogue::store<T, lmul, FastPath>(c3_ptr + 0 * VL, c30, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c3_ptr + 1 * VL, c31, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c3_ptr + 2 * VL, c32, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c3_ptr + 3 * VL, c33, alpha, beta);
      }

      // 2. Cleanup 4x2 Tiles (16 columns)
      while (j + 2 * VL <= B.cols) {
        auto c00 = set0<T, lmul>();
        auto c01 = set0<T, lmul>();
        auto c10 = set0<T, lmul>();
        auto c11 = set0<T, lmul>();
        auto c20 = set0<T, lmul>();
        auto c21 = set0<T, lmul>();
        auto c30 = set0<T, lmul>();
        auto c31 = set0<T, lmul>();

        const T *__restrict a0_k = a0_base;
        const T *__restrict a1_k = a1_base;
        const T *__restrict a2_k = a2_base;
        const T *__restrict a3_k = a3_base;
        const T *__restrict b_k = B.data + j;

        #pragma GCC unroll 1
        for (size_t k = 0; k < A.cols; ++k) {
          const auto b0 = load<T, lmul>(b_k);
          const auto b1 = load<T, lmul>(b_k + VL);
          b_k += b_ld;

          const auto a0 = set1<T, lmul>(*a0_k++);
          const auto a1 = set1<T, lmul>(*a1_k++);
          const auto a2 = set1<T, lmul>(*a2_k++);
          const auto a3 = set1<T, lmul>(*a3_k++);

          c00 = fmadd(a0, b0, c00);
          c01 = fmadd(a0, b1, c01);
          c10 = fmadd(a1, b0, c10);
          c11 = fmadd(a1, b1, c11);
          c20 = fmadd(a2, b0, c20);
          c21 = fmadd(a2, b1, c21);
          c30 = fmadd(a3, b0, c30);
          c31 = fmadd(a3, b1, c31);
        }

        const auto c0_ptr = c0_base + j;
        const auto c1_ptr = c1_base + j;
        const auto c2_ptr = c2_base + j;
        const auto c3_ptr = c3_base + j;

        Epilogue::store<T, lmul, FastPath>(c0_ptr + 0 * VL, c00, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c0_ptr + 1 * VL, c01, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_ptr + 0 * VL, c10, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_ptr + 1 * VL, c11, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c2_ptr + 0 * VL, c20, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c2_ptr + 1 * VL, c21, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c3_ptr + 0 * VL, c30, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c3_ptr + 1 * VL, c31, alpha, beta);

        j += 2 * VL;
      }

      // 3. Cleanup 4x1 Tiles (8 columns)
      while (j + VL <= B.cols) {
        auto c0 = set0<T, lmul>();
        auto c1 = set0<T, lmul>();
        auto c2 = set0<T, lmul>();
        auto c3 = set0<T, lmul>();

        const T *__restrict a0_k = a0_base;
        const T *__restrict a1_k = a1_base;
        const T *__restrict a2_k = a2_base;
        const T *__restrict a3_k = a3_base;
        const T *__restrict b_k = B.data + j;

        for (size_t k = 0; k < A.cols; ++k) {
          const auto b0 = load<T, lmul>(b_k);
          b_k += b_ld;

          c0 = fmadd(set1<T, lmul>(*a0_k++), b0, c0);
          c1 = fmadd(set1<T, lmul>(*a1_k++), b0, c1);
          c2 = fmadd(set1<T, lmul>(*a2_k++), b0, c2);
          c3 = fmadd(set1<T, lmul>(*a3_k++), b0, c3);
        }

        Epilogue::store<T, lmul, FastPath>(c0_base + j, c0, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_base + j, c1, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c2_base + j, c2, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c3_base + j, c3, alpha, beta);

        j += VL;
      }
    }

    //
    // Tail rows (if A.rows % 4 != 0)
    //
    for (size_t i = Mfull; i < A.rows; ++i) {
      const T *a_ptr = A[i];
      size_t j = 0;

      while (j + NR4 <= B.cols) {
        auto c0 = set0<T, lmul>();
        auto c1 = set0<T, lmul>();
        auto c2 = set0<T, lmul>();
        auto c3 = set0<T, lmul>();

        for (size_t k = 0; k < A.cols; ++k) {
          const auto b0 = load<T, lmul>(B[k] + j + 0 * VL);
          const auto b1 = load<T, lmul>(B[k] + j + 1 * VL);
          const auto b2 = load<T, lmul>(B[k] + j + 2 * VL);
          const auto b3 = load<T, lmul>(B[k] + j + 3 * VL);

          const auto a = set1<T, lmul>(a_ptr[k]);
          c0 = fmadd(a, b0, c0);
          c1 = fmadd(a, b1, c1);
          c2 = fmadd(a, b2, c2);
          c3 = fmadd(a, b3, c3);
        }

        Epilogue::store<T, lmul, FastPath>(C[i] + j + 0 * VL, c0, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(C[i] + j + 1 * VL, c1, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(C[i] + j + 2 * VL, c2, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(C[i] + j + 3 * VL, c3, alpha, beta);

        j += NR4;
      }

      while (j + VL <= B.cols) {
        auto c0 = set0<T, lmul>();

        for (size_t k = 0; k < A.cols; ++k) {
          const auto b0 = load<T, lmul>(B[k] + j);
          c0 = fmadd(set1<T, lmul>(a_ptr[k]), b0, c0);
        }

        Epilogue::store<T, lmul, FastPath>(C[i] + j, c0, alpha, beta);
        j += VL;
      }
    }
  }

  template <int lmul = 1>
  static inline void
  gemm_mippv2_zen4_mr4_nr4(const PackedRowMajor<T> &__restrict A,
                           const PackedRowMajor<T> &__restrict B,
                           PackedRowMajor<T> &__restrict C) {
    gemm_mippv2_zen4_mr4_nr4_core<lmul, true>(A, B, C);
  }

  template <int lmul = 1>
  static inline void
  gemm_mippv2_zen4_mr4_nr4(const PackedRowMajor<T> &__restrict A,
                           const PackedRowMajor<T> &__restrict B,
                           PackedRowMajor<T> &__restrict C,
                           T alpha, T beta) {
    if (__builtin_expect(alpha == T{1} && beta == T{0}, 1)) {
      gemm_mippv2_zen4_mr4_nr4_core<lmul, true>(A, B, C);
    } else {
      gemm_mippv2_zen4_mr4_nr4_core<lmul, false>(A, B, C, alpha, beta);
    }
  }

  // ===========================================================================
  // AMD Zen 4 AVX-512 Kernel (MR=4, NR=4*VL = 32 doubles with VL=8) - fmaddi
  // 100% Pure MIPPv2: fmaddi leveraging vector-scalar operations.

  // ===========================================================================
  // Apple Silicon Firestorm Microkernels (M1 / M1 Pro / M1 Max / M1 Ultra)
  // - mippv2_firestorm_mr4_nr4: 16 vector accumulators (4x4 tile, 4-wide wave)
  // - mippv2_firestorm_mr4_nr4_fmaddi: 16 vector accumulators via vector-scalar fmaddi
  // - mippv2_firestorm_mr6_nr4: 24 vector accumulators (6x4 tile, 2.4 FMAs/load)
  // - mippv2_firestorm_mr6_nr4_fmaddi: 24 vector accumulators via fmaddi
  // ===========================================================================

  // 1. Firestorm MR=4 NR=4 (16 accumulators)
  template <int lmul = 1, bool FastPath = false>
  static inline void
  gemm_mippv2_firestorm_mr4_nr4_core(const PackedRowMajor<T> &__restrict A,
                                     const PackedRowMajor<T> &__restrict B,
                                     PackedRowMajor<T> &__restrict C,
                                     T alpha = T{1}, T beta = T{0}) {
    gemm_mippv2_zen4_mr4_nr4_core<lmul, FastPath>(A, B, C, alpha, beta);
  }

  template <int lmul = 1>
  static inline void
  gemm_mippv2_firestorm_mr4_nr4(const PackedRowMajor<T> &__restrict A,
                                const PackedRowMajor<T> &__restrict B,
                                PackedRowMajor<T> &__restrict C) {
    gemm_mippv2_firestorm_mr4_nr4_core<lmul, true>(A, B, C);
  }

  template <int lmul = 1>
  static inline void
  gemm_mippv2_firestorm_mr4_nr4(const PackedRowMajor<T> &__restrict A,
                                const PackedRowMajor<T> &__restrict B,
                                PackedRowMajor<T> &__restrict C,
                                T alpha, T beta) {
    if (__builtin_expect(alpha == T{1} && beta == T{0}, 1)) {
      gemm_mippv2_firestorm_mr4_nr4_core<lmul, true>(A, B, C);
    } else {
      gemm_mippv2_firestorm_mr4_nr4_core<lmul, false>(A, B, C, alpha, beta);
    }
  }


  // 3. Firestorm MR=6 NR=4 (24 accumulators, 2.4 FMAs/load)
  template <int lmul = 1, bool FastPath = false>
  static inline void
  gemm_mippv2_firestorm_mr6_nr4_core(const PackedRowMajor<T> &__restrict A,
                                     const PackedRowMajor<T> &__restrict B,
                                     PackedRowMajor<T> &__restrict C,
                                     T alpha = T{1}, T beta = T{0}) {
    using namespace mipp;

    constexpr size_t VL = N<T, lmul>();
    constexpr size_t MR = 6;
    constexpr size_t NR4 = 4 * VL;

    const size_t b_ld = B.ld;
    const size_t Mfull = (A.rows / MR) * MR;

    for (size_t i = 0; i < Mfull; i += MR) {
      const T *a0_base = A[i + 0];
      const T *a1_base = A[i + 1];
      const T *a2_base = A[i + 2];
      const T *a3_base = A[i + 3];
      const T *a4_base = A[i + 4];
      const T *a5_base = A[i + 5];

      T *c0_base = C[i + 0];
      T *c1_base = C[i + 1];
      T *c2_base = C[i + 2];
      T *c3_base = C[i + 3];
      T *c4_base = C[i + 4];
      T *c5_base = C[i + 5];

      size_t j = 0;

      // Main 6x4 Tiles (24 accumulators)
      for (; j + NR4 <= B.cols; j += NR4) {
        auto c00 = set0<T, lmul>(); auto c01 = set0<T, lmul>(); auto c02 = set0<T, lmul>(); auto c03 = set0<T, lmul>();
        auto c10 = set0<T, lmul>(); auto c11 = set0<T, lmul>(); auto c12 = set0<T, lmul>(); auto c13 = set0<T, lmul>();
        auto c20 = set0<T, lmul>(); auto c21 = set0<T, lmul>(); auto c22 = set0<T, lmul>(); auto c23 = set0<T, lmul>();
        auto c30 = set0<T, lmul>(); auto c31 = set0<T, lmul>(); auto c32 = set0<T, lmul>(); auto c33 = set0<T, lmul>();
        auto c40 = set0<T, lmul>(); auto c41 = set0<T, lmul>(); auto c42 = set0<T, lmul>(); auto c43 = set0<T, lmul>();
        auto c50 = set0<T, lmul>(); auto c51 = set0<T, lmul>(); auto c52 = set0<T, lmul>(); auto c53 = set0<T, lmul>();

        const T *__restrict a0_k = a0_base;
        const T *__restrict a1_k = a1_base;
        const T *__restrict a2_k = a2_base;
        const T *__restrict a3_k = a3_base;
        const T *__restrict a4_k = a4_base;
        const T *__restrict a5_k = a5_base;
        const T *__restrict b_k = B.data + j;

        #pragma GCC unroll 1
        for (size_t k = 0; k < A.cols; ++k) {
          const auto b0 = load<T, lmul>(b_k + 0 * VL);
          const auto b1 = load<T, lmul>(b_k + 1 * VL);
          const auto b2 = load<T, lmul>(b_k + 2 * VL);
          const auto b3 = load<T, lmul>(b_k + 3 * VL);
          b_k += b_ld;

          {
            const auto a0 = set1<T, lmul>(*a0_k++);
            c00 = fmadd(a0, b0, c00);
            c01 = fmadd(a0, b1, c01);
            c02 = fmadd(a0, b2, c02);
            c03 = fmadd(a0, b3, c03);
          }
          {
            const auto a1 = set1<T, lmul>(*a1_k++);
            c10 = fmadd(a1, b0, c10);
            c11 = fmadd(a1, b1, c11);
            c12 = fmadd(a1, b2, c12);
            c13 = fmadd(a1, b3, c13);
          }
          {
            const auto a2 = set1<T, lmul>(*a2_k++);
            c20 = fmadd(a2, b0, c20);
            c21 = fmadd(a2, b1, c21);
            c22 = fmadd(a2, b2, c22);
            c23 = fmadd(a2, b3, c23);
          }
          {
            const auto a3 = set1<T, lmul>(*a3_k++);
            c30 = fmadd(a3, b0, c30);
            c31 = fmadd(a3, b1, c31);
            c32 = fmadd(a3, b2, c32);
            c33 = fmadd(a3, b3, c33);
          }
          {
            const auto a4 = set1<T, lmul>(*a4_k++);
            c40 = fmadd(a4, b0, c40);
            c41 = fmadd(a4, b1, c41);
            c42 = fmadd(a4, b2, c42);
            c43 = fmadd(a4, b3, c43);
          }
          {
            const auto a5 = set1<T, lmul>(*a5_k++);
            c50 = fmadd(a5, b0, c50);
            c51 = fmadd(a5, b1, c51);
            c52 = fmadd(a5, b2, c52);
            c53 = fmadd(a5, b3, c53);
          }
        }

        const auto c0_ptr = c0_base + j;
        const auto c1_ptr = c1_base + j;
        const auto c2_ptr = c2_base + j;
        const auto c3_ptr = c3_base + j;
        const auto c4_ptr = c4_base + j;
        const auto c5_ptr = c5_base + j;

        Epilogue::store<T, lmul, FastPath>(c0_ptr + 0 * VL, c00, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c0_ptr + 1 * VL, c01, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c0_ptr + 2 * VL, c02, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c0_ptr + 3 * VL, c03, alpha, beta);

        Epilogue::store<T, lmul, FastPath>(c1_ptr + 0 * VL, c10, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_ptr + 1 * VL, c11, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_ptr + 2 * VL, c12, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_ptr + 3 * VL, c13, alpha, beta);

        Epilogue::store<T, lmul, FastPath>(c2_ptr + 0 * VL, c20, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c2_ptr + 1 * VL, c21, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c2_ptr + 2 * VL, c22, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c2_ptr + 3 * VL, c23, alpha, beta);

        Epilogue::store<T, lmul, FastPath>(c3_ptr + 0 * VL, c30, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c3_ptr + 1 * VL, c31, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c3_ptr + 2 * VL, c32, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c3_ptr + 3 * VL, c33, alpha, beta);

        Epilogue::store<T, lmul, FastPath>(c4_ptr + 0 * VL, c40, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c4_ptr + 1 * VL, c41, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c4_ptr + 2 * VL, c42, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c4_ptr + 3 * VL, c43, alpha, beta);

        Epilogue::store<T, lmul, FastPath>(c5_ptr + 0 * VL, c50, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c5_ptr + 1 * VL, c51, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c5_ptr + 2 * VL, c52, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c5_ptr + 3 * VL, c53, alpha, beta);
      }

      while (j + 2 * VL <= B.cols) {
        auto c00 = set0<T, lmul>(); auto c01 = set0<T, lmul>();
        auto c10 = set0<T, lmul>(); auto c11 = set0<T, lmul>();
        auto c20 = set0<T, lmul>(); auto c21 = set0<T, lmul>();
        auto c30 = set0<T, lmul>(); auto c31 = set0<T, lmul>();
        auto c40 = set0<T, lmul>(); auto c41 = set0<T, lmul>();
        auto c50 = set0<T, lmul>(); auto c51 = set0<T, lmul>();

        const T *__restrict a0_k = a0_base;
        const T *__restrict a1_k = a1_base;
        const T *__restrict a2_k = a2_base;
        const T *__restrict a3_k = a3_base;
        const T *__restrict a4_k = a4_base;
        const T *__restrict a5_k = a5_base;
        const T *__restrict b_k = B.data + j;

        for (size_t k = 0; k < A.cols; ++k) {
          const auto b0 = load<T, lmul>(b_k + 0 * VL);
          const auto b1 = load<T, lmul>(b_k + 1 * VL);
          b_k += b_ld;

          const auto a0 = set1<T, lmul>(*a0_k++);
          const auto a1 = set1<T, lmul>(*a1_k++);
          const auto a2 = set1<T, lmul>(*a2_k++);
          const auto a3 = set1<T, lmul>(*a3_k++);
          const auto a4 = set1<T, lmul>(*a4_k++);
          const auto a5 = set1<T, lmul>(*a5_k++);

          c00 = fmadd(a0, b0, c00); c01 = fmadd(a0, b1, c01);
          c10 = fmadd(a1, b0, c10); c11 = fmadd(a1, b1, c11);
          c20 = fmadd(a2, b0, c20); c21 = fmadd(a2, b1, c21);
          c30 = fmadd(a3, b0, c30); c31 = fmadd(a3, b1, c31);
          c40 = fmadd(a4, b0, c40); c41 = fmadd(a4, b1, c41);
          c50 = fmadd(a5, b0, c50); c51 = fmadd(a5, b1, c51);
        }

        const auto c0_ptr = c0_base + j;
        const auto c1_ptr = c1_base + j;
        const auto c2_ptr = c2_base + j;
        const auto c3_ptr = c3_base + j;
        const auto c4_ptr = c4_base + j;
        const auto c5_ptr = c5_base + j;

        Epilogue::store<T, lmul, FastPath>(c0_ptr + 0 * VL, c00, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c0_ptr + 1 * VL, c01, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_ptr + 0 * VL, c10, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_ptr + 1 * VL, c11, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c2_ptr + 0 * VL, c20, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c2_ptr + 1 * VL, c21, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c3_ptr + 0 * VL, c30, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c3_ptr + 1 * VL, c31, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c4_ptr + 0 * VL, c40, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c4_ptr + 1 * VL, c41, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c5_ptr + 0 * VL, c50, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c5_ptr + 1 * VL, c51, alpha, beta);

        j += 2 * VL;
      }

      while (j + VL <= B.cols) {
        auto c0 = set0<T, lmul>(); auto c1 = set0<T, lmul>();
        auto c2 = set0<T, lmul>(); auto c3 = set0<T, lmul>();
        auto c4 = set0<T, lmul>(); auto c5 = set0<T, lmul>();

        const T *__restrict a0_k = a0_base;
        const T *__restrict a1_k = a1_base;
        const T *__restrict a2_k = a2_base;
        const T *__restrict a3_k = a3_base;
        const T *__restrict a4_k = a4_base;
        const T *__restrict a5_k = a5_base;
        const T *__restrict b_k = B.data + j;

        for (size_t k = 0; k < A.cols; ++k) {
          const auto b0 = load<T, lmul>(b_k);
          b_k += b_ld;

          const auto a0 = set1<T, lmul>(*a0_k++);
          const auto a1 = set1<T, lmul>(*a1_k++);
          const auto a2 = set1<T, lmul>(*a2_k++);
          const auto a3 = set1<T, lmul>(*a3_k++);
          const auto a4 = set1<T, lmul>(*a4_k++);
          const auto a5 = set1<T, lmul>(*a5_k++);

          c0 = fmadd(a0, b0, c0);
          c1 = fmadd(a1, b0, c1);
          c2 = fmadd(a2, b0, c2);
          c3 = fmadd(a3, b0, c3);
          c4 = fmadd(a4, b0, c4);
          c5 = fmadd(a5, b0, c5);
        }

        Epilogue::store<T, lmul, FastPath>(c0_base + j, c0, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_base + j, c1, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c2_base + j, c2, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c3_base + j, c3, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c4_base + j, c4, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c5_base + j, c5, alpha, beta);

        j += VL;
      }
    }

    // Row cleanups (4 rows, 2 rows, 1 row)
    size_t i = Mfull;
    while (i + 4 <= A.rows) {
      const T *a0_base = A[i + 0]; const T *a1_base = A[i + 1];
      const T *a2_base = A[i + 2]; const T *a3_base = A[i + 3];

      T *c0_base = C[i + 0]; T *c1_base = C[i + 1];
      T *c2_base = C[i + 2]; T *c3_base = C[i + 3];

      size_t j = 0;
      for (; j + NR4 <= B.cols; j += NR4) {
        auto c00 = set0<T, lmul>(); auto c01 = set0<T, lmul>(); auto c02 = set0<T, lmul>(); auto c03 = set0<T, lmul>();
        auto c10 = set0<T, lmul>(); auto c11 = set0<T, lmul>(); auto c12 = set0<T, lmul>(); auto c13 = set0<T, lmul>();
        auto c20 = set0<T, lmul>(); auto c21 = set0<T, lmul>(); auto c22 = set0<T, lmul>(); auto c23 = set0<T, lmul>();
        auto c30 = set0<T, lmul>(); auto c31 = set0<T, lmul>(); auto c32 = set0<T, lmul>(); auto c33 = set0<T, lmul>();

        const T *__restrict a0_k = a0_base; const T *__restrict a1_k = a1_base;
        const T *__restrict a2_k = a2_base; const T *__restrict a3_k = a3_base;
        const T *__restrict b_k = B.data + j;

        for (size_t k = 0; k < A.cols; ++k) {
          const auto b0 = load<T, lmul>(b_k + 0 * VL);
          const auto b1 = load<T, lmul>(b_k + 1 * VL);
          const auto b2 = load<T, lmul>(b_k + 2 * VL);
          const auto b3 = load<T, lmul>(b_k + 3 * VL);
          b_k += b_ld;

          const auto a0 = set1<T, lmul>(*a0_k++);
          const auto a1 = set1<T, lmul>(*a1_k++);
          const auto a2 = set1<T, lmul>(*a2_k++);
          const auto a3 = set1<T, lmul>(*a3_k++);

          c00 = fmadd(a0, b0, c00); c01 = fmadd(a0, b1, c01); c02 = fmadd(a0, b2, c02); c03 = fmadd(a0, b3, c03);
          c10 = fmadd(a1, b0, c10); c11 = fmadd(a1, b1, c11); c12 = fmadd(a1, b2, c12); c13 = fmadd(a1, b3, c13);
          c20 = fmadd(a2, b0, c20); c21 = fmadd(a2, b1, c21); c22 = fmadd(a2, b2, c22); c23 = fmadd(a2, b3, c23);
          c30 = fmadd(a3, b0, c30); c31 = fmadd(a3, b1, c31); c32 = fmadd(a3, b2, c32); c33 = fmadd(a3, b3, c33);
        }

        Epilogue::store<T, lmul, FastPath>(c0_base + j + 0 * VL, c00, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c0_base + j + 1 * VL, c01, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c0_base + j + 2 * VL, c02, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c0_base + j + 3 * VL, c03, alpha, beta);

        Epilogue::store<T, lmul, FastPath>(c1_base + j + 0 * VL, c10, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_base + j + 1 * VL, c11, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_base + j + 2 * VL, c12, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_base + j + 3 * VL, c13, alpha, beta);

        Epilogue::store<T, lmul, FastPath>(c2_base + j + 0 * VL, c20, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c2_base + j + 1 * VL, c21, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c2_base + j + 2 * VL, c22, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c2_base + j + 3 * VL, c23, alpha, beta);

        Epilogue::store<T, lmul, FastPath>(c3_base + j + 0 * VL, c30, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c3_base + j + 1 * VL, c31, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c3_base + j + 2 * VL, c32, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c3_base + j + 3 * VL, c33, alpha, beta);
      }
      while (j + VL <= B.cols) {
        auto c0 = set0<T, lmul>(); auto c1 = set0<T, lmul>();
        auto c2 = set0<T, lmul>(); auto c3 = set0<T, lmul>();
        const T *__restrict a0_k = a0_base; const T *__restrict a1_k = a1_base;
        const T *__restrict a2_k = a2_base; const T *__restrict a3_k = a3_base;
        const T *__restrict b_k = B.data + j;
        for (size_t k = 0; k < A.cols; ++k) {
          const auto b0 = load<T, lmul>(b_k);
          b_k += b_ld;
          const auto a0 = set1<T, lmul>(*a0_k++);
          const auto a1 = set1<T, lmul>(*a1_k++);
          const auto a2 = set1<T, lmul>(*a2_k++);
          const auto a3 = set1<T, lmul>(*a3_k++);
          c0 = fmadd(a0, b0, c0); c1 = fmadd(a1, b0, c1);
          c2 = fmadd(a2, b0, c2); c3 = fmadd(a3, b0, c3);
        }
        Epilogue::store<T, lmul, FastPath>(c0_base + j, c0, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_base + j, c1, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c2_base + j, c2, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c3_base + j, c3, alpha, beta);
        j += VL;
      }
      i += 4;
    }

    while (i + 2 <= A.rows) {
      const T *a0_base = A[i + 0]; const T *a1_base = A[i + 1];
      T *c0_base = C[i + 0]; T *c1_base = C[i + 1];
      size_t j = 0;
      for (; j + NR4 <= B.cols; j += NR4) {
        auto c00 = set0<T, lmul>(); auto c01 = set0<T, lmul>(); auto c02 = set0<T, lmul>(); auto c03 = set0<T, lmul>();
        auto c10 = set0<T, lmul>(); auto c11 = set0<T, lmul>(); auto c12 = set0<T, lmul>(); auto c13 = set0<T, lmul>();

        const T *__restrict a0_k = a0_base; const T *__restrict a1_k = a1_base;
        const T *__restrict b_k = B.data + j;

        for (size_t k = 0; k < A.cols; ++k) {
          const auto b0 = load<T, lmul>(b_k + 0 * VL);
          const auto b1 = load<T, lmul>(b_k + 1 * VL);
          const auto b2 = load<T, lmul>(b_k + 2 * VL);
          const auto b3 = load<T, lmul>(b_k + 3 * VL);
          b_k += b_ld;

          const auto a0 = set1<T, lmul>(*a0_k++);
          const auto a1 = set1<T, lmul>(*a1_k++);

          c00 = fmadd(a0, b0, c00); c01 = fmadd(a0, b1, c01); c02 = fmadd(a0, b2, c02); c03 = fmadd(a0, b3, c03);
          c10 = fmadd(a1, b0, c10); c11 = fmadd(a1, b1, c11); c12 = fmadd(a1, b2, c12); c13 = fmadd(a1, b3, c13);
        }

        Epilogue::store<T, lmul, FastPath>(c0_base + j + 0 * VL, c00, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c0_base + j + 1 * VL, c01, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c0_base + j + 2 * VL, c02, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c0_base + j + 3 * VL, c03, alpha, beta);

        Epilogue::store<T, lmul, FastPath>(c1_base + j + 0 * VL, c10, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_base + j + 1 * VL, c11, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_base + j + 2 * VL, c12, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_base + j + 3 * VL, c13, alpha, beta);
      }
      while (j + VL <= B.cols) {
        auto c0 = set0<T, lmul>(); auto c1 = set0<T, lmul>();
        const T *__restrict a0_k = a0_base; const T *__restrict a1_k = a1_base;
        const T *__restrict b_k = B.data + j;
        for (size_t k = 0; k < A.cols; ++k) {
          const auto b0 = load<T, lmul>(b_k);
          b_k += b_ld;
          const auto a0 = set1<T, lmul>(*a0_k++);
          const auto a1 = set1<T, lmul>(*a1_k++);
          c0 = fmadd(a0, b0, c0); c1 = fmadd(a1, b0, c1);
        }
        Epilogue::store<T, lmul, FastPath>(c0_base + j, c0, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_base + j, c1, alpha, beta);
        j += VL;
      }
      i += 2;
    }

    while (i < A.rows) {
      const T *a0_base = A[i + 0];
      T *c0_base = C[i + 0];
      size_t j = 0;
      while (j + VL <= B.cols) {
        auto c0 = set0<T, lmul>();
        const T *__restrict a0_k = a0_base;
        const T *__restrict b_k = B.data + j;
        for (size_t k = 0; k < A.cols; ++k) {
          const auto b0 = load<T, lmul>(b_k);
          b_k += b_ld;
          const auto a0 = set1<T, lmul>(*a0_k++);
          c0 = fmadd(a0, b0, c0);
        }
        Epilogue::store<T, lmul, FastPath>(c0_base + j, c0, alpha, beta);
        j += VL;
      }
      i += 1;
    }
  }

  template <int lmul = 1>
  static inline void
  gemm_mippv2_firestorm_mr6_nr4(const PackedRowMajor<T> &__restrict A,
                                const PackedRowMajor<T> &__restrict B,
                                PackedRowMajor<T> &__restrict C) {
    gemm_mippv2_firestorm_mr6_nr4_core<lmul, true>(A, B, C);
  }

  template <int lmul = 1>
  static inline void
  gemm_mippv2_firestorm_mr6_nr4(const PackedRowMajor<T> &__restrict A,
                                const PackedRowMajor<T> &__restrict B,
                                PackedRowMajor<T> &__restrict C,
                                T alpha, T beta) {
    if (__builtin_expect(alpha == T{1} && beta == T{0}, 1)) {
      gemm_mippv2_firestorm_mr6_nr4_core<lmul, true>(A, B, C);
    } else {
      gemm_mippv2_firestorm_mr6_nr4_core<lmul, false>(A, B, C, alpha, beta);
    }
  }


  static inline void gemm_mippv2_skylake_panel(const PackedColMajor<T> &A,
                                                const PackedRowMajor<T> &B,
                                                PackedRowMajor<T> &C,
                                                T alpha = T{1}, T beta = T{0}) {
    using namespace mipp;

    constexpr size_t VL = N<T>();
    constexpr size_t MR = 4;
    constexpr size_t NR = 2 * VL;

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
          auto b0 = loadu<T>(&B(k, j + 0 * VL));
          auto b1 = loadu<T>(&B(k, j + 1 * VL));

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

        Epilogue::store<T, 1>(&C(i + 0, j + 0 * VL), c00, alpha, beta);
        Epilogue::store<T, 1>(&C(i + 0, j + 1 * VL), c01, alpha, beta);

        Epilogue::store<T, 1>(&C(i + 1, j + 0 * VL), c10, alpha, beta);
        Epilogue::store<T, 1>(&C(i + 1, j + 1 * VL), c11, alpha, beta);

        Epilogue::store<T, 1>(&C(i + 2, j + 0 * VL), c20, alpha, beta);
        Epilogue::store<T, 1>(&C(i + 2, j + 1 * VL), c21, alpha, beta);

        Epilogue::store<T, 1>(&C(i + 3, j + 0 * VL), c30, alpha, beta);
        Epilogue::store<T, 1>(&C(i + 3, j + 1 * VL), c31, alpha, beta);
      }
    }
  }

  template <int lmul = 1, int MR = 4, int NR_PACKETS = 1>
  static inline void gemm_mippv2_skylake_panel_lmul(const PackedColMajor<T> &A,
                                                     const PackedRowMajor<T> &B,
                                                     PackedRowMajor<T> &C,
                                                     T alpha = T{1}, T beta = T{0}) {
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
            Epilogue::store<T, lmul>(&C(i + r, j + p * VL), c[r][p], alpha, beta);
      }
    }
  }


  // =========================================================================
  // X100: gemm_mippv2_x100_register_blocked
  // =========================================================================

  // FastPath 3-argument version (alpha=1, beta=0): pure, no alpha/beta in signature/body, hoisted C pointers
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

      T *__restrict c0_base = C[i + 0];
      T *__restrict c1_base = C[i + 1];
      T *__restrict c2_base = C[i + 2];

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

        store(c0_base + j, c00);
        store(c0_base + j + VL, c01);

        store(c1_base + j, c10);
        store(c1_base + j + VL, c11);

        store(c2_base + j, c20);
        store(c2_base + j + VL, c21);
      }
    }

    //
    // Tail (1 or 2 remaining rows)
    //
    for (size_t i = Mfull; i < A.rows; ++i) {
      const T *a_ptr = A[i];
      T *__restrict c_base = C[i];

      for (size_t j = 0; j < B.cols; j += NR) {
        auto c0 = set0<T, lmul>();
        auto c1 = set0<T, lmul>();

        for (size_t k = 0; k < A.cols; ++k) {
          auto b0 = load<T, lmul>(B[k] + j);
          auto b1 = load<T, lmul>(B[k] + j + VL);

          c0 = fmaddi(b0, a_ptr[k], c0);
          c1 = fmaddi(b1, a_ptr[k], c1);
        }

        store(c_base + j, c0);
        store(c_base + j + VL, c1);
      }
    }
  }

  // General 5-argument version (arbitrary alpha, beta)
  template <int lmul = 4>
  static inline void
  gemm_mippv2_x100_register_blocked(const PackedRowMajor<T> &__restrict A,
                                    const PackedRowMajor<T> &__restrict B,
                                    PackedRowMajor<T> &__restrict C,
                                    T alpha, T beta) {
    if (__builtin_expect(alpha == T{1} && beta == T{0}, 1)) {
      gemm_mippv2_x100_register_blocked<lmul>(A, B, C);
      return;
    }

    using namespace mipp;

    constexpr size_t VL = N<T, lmul>();

    constexpr size_t MR = 3;
    constexpr size_t NR = 2 * VL;

    const size_t Mfull = (A.rows / MR) * MR;

    for (size_t i = 0; i < Mfull; i += MR) {
      const T *a0_ptr = A[i + 0];
      const T *a1_ptr = A[i + 1];
      const T *a2_ptr = A[i + 2];

      T *__restrict c0_base = C[i + 0];
      T *__restrict c1_base = C[i + 1];
      T *__restrict c2_base = C[i + 2];

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

        Epilogue::store<T, lmul, false>(c0_base + j, c00, alpha, beta);
        Epilogue::store<T, lmul, false>(c0_base + j + VL, c01, alpha, beta);

        Epilogue::store<T, lmul, false>(c1_base + j, c10, alpha, beta);
        Epilogue::store<T, lmul, false>(c1_base + j + VL, c11, alpha, beta);

        Epilogue::store<T, lmul, false>(c2_base + j, c20, alpha, beta);
        Epilogue::store<T, lmul, false>(c2_base + j + VL, c21, alpha, beta);
      }
    }

    for (size_t i = Mfull; i < A.rows; ++i) {
      const T *a_ptr = A[i];
      T *__restrict c_base = C[i];

      for (size_t j = 0; j < B.cols; j += NR) {
        auto c0 = set0<T, lmul>();
        auto c1 = set0<T, lmul>();

        for (size_t k = 0; k < A.cols; ++k) {
          auto b0 = load<T, lmul>(B[k] + j);
          auto b1 = load<T, lmul>(B[k] + j + VL);

          c0 = fmaddi(b0, a_ptr[k], c0);
          c1 = fmaddi(b1, a_ptr[k], c1);
        }

        Epilogue::store<T, lmul, false>(c_base + j, c0, alpha, beta);
        Epilogue::store<T, lmul, false>(c_base + j + VL, c1, alpha, beta);
      }
    }
  }


  // =========================================================================
  // X100: gemm_mippv2_panel_x100
  // =========================================================================

  // FastPath 3-argument version (alpha=1, beta=0)
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
      T *__restrict c0_base = C[i + 0];
      T *__restrict c1_base = C[i + 1];
      T *__restrict c2_base = C[i + 2];

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

        store(c0_base + j, c00);
        store(c0_base + j + VL, c01);

        store(c1_base + j, c10);
        store(c1_base + j + VL, c11);

        store(c2_base + j, c20);
        store(c2_base + j + VL, c21);
      }
    }
  }

  // General 5-argument version (arbitrary alpha, beta)
  template <int lmul = 4>
  static inline void
  gemm_mippv2_panel_x100(const PackedColMajor<T> &__restrict A,
                         const PackedRowMajor<T> &__restrict B,
                         PackedRowMajor<T> &__restrict C,
                         T alpha, T beta) {
    if (__builtin_expect(alpha == T{1} && beta == T{0}, 1)) {
      gemm_mippv2_panel_x100<lmul>(A, B, C);
      return;
    }

    using namespace mipp;

    constexpr size_t VL = N<T, lmul>();

    constexpr size_t MR = 3;
    constexpr size_t NR = 2 * VL;

    for (size_t i = 0; i < A.rows; i += MR) {
      T *__restrict c0_base = C[i + 0];
      T *__restrict c1_base = C[i + 1];
      T *__restrict c2_base = C[i + 2];

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

        Epilogue::store<T, lmul, false>(c0_base + j, c00, alpha, beta);
        Epilogue::store<T, lmul, false>(c0_base + j + VL, c01, alpha, beta);

        Epilogue::store<T, lmul, false>(c1_base + j, c10, alpha, beta);
        Epilogue::store<T, lmul, false>(c1_base + j + VL, c11, alpha, beta);

        Epilogue::store<T, lmul, false>(c2_base + j, c20, alpha, beta);
        Epilogue::store<T, lmul, false>(c2_base + j + VL, c21, alpha, beta);
      }
    }
  }

