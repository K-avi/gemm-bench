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
