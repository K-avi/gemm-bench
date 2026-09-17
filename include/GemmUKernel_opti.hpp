  static void gemm_ijk(const PackedRowMajor<T> &A, const PackedRowMajor<T> &B,
                       PackedRowMajor<T> &C, T alpha = T{1}, T beta = T{0}) {
    for (size_t i = 0; i < A.rows; ++i) {
      for (size_t j = 0; j < B.cols; ++j) {
        T sum = T{};

        for (size_t k = 0; k < A.cols; ++k) {
          sum += A[i][k] * B[k][j];
        }

        Epilogue::store(&C[i][j], sum, alpha, beta);
      }
    }
  }

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
  template <int lmul = 1, bool FastPath = false>
  static inline void
  gemm_mippv2_meteorlake_mr4_nr3_core(const PackedRowMajor<T> &__restrict A,
                                      const PackedRowMajor<T> &__restrict B,
                                      PackedRowMajor<T> &__restrict C,
                                      T alpha = T{1}, T beta = T{0}) {
    using namespace mipp;

    constexpr size_t VL = N<T, lmul>();
    constexpr size_t MR = 4;
    constexpr size_t NR3 = 3 * VL; // 12 doubles in AVX2

    const size_t b_ld = B.ld;

    size_t limit12 = B.cols;
    if (B.cols >= 16 && (B.cols % NR3) == VL) {
      limit12 = B.cols - 4 * VL;
    }

    for (size_t i = 0; i < A.rows; i += MR) {
      const T *a0_base = A[i + 0];
      const T *a1_base = A[i + 1];
      const T *a2_base = A[i + 2];
      const T *a3_base = A[i + 3];

      T *c0_base = C[i + 0];
      T *c1_base = C[i + 1];
      T *c2_base = C[i + 2];
      T *c3_base = C[i + 3];

      size_t j = 0;

      // 1. Main 4x3 Tiles (12 accumulators, 3 FMAs per broadcast)
      // For N=64: runs 4 tiles (48 cols = 75% of matrix)
      // For N=32: runs 2 tiles (24 cols = 75% of matrix)
      for (; j + NR3 <= limit12; j += NR3) {
        auto c00 = set0<T, lmul>();
        auto c01 = set0<T, lmul>();
        auto c02 = set0<T, lmul>();

        auto c10 = set0<T, lmul>();
        auto c11 = set0<T, lmul>();
        auto c12 = set0<T, lmul>();

        auto c20 = set0<T, lmul>();
        auto c21 = set0<T, lmul>();
        auto c22 = set0<T, lmul>();

        auto c30 = set0<T, lmul>();
        auto c31 = set0<T, lmul>();
        auto c32 = set0<T, lmul>();

        const T *__restrict a0_k = a0_base;
        const T *__restrict a1_k = a1_base;
        const T *__restrict a2_k = a2_base;
        const T *__restrict a3_k = a3_base;
        const T *__restrict b_k = B.data + j;

        #pragma GCC unroll 1
        for (size_t k = 0; k < A.cols; ++k) {
          const auto b0 = load<T, lmul>(b_k);
          const auto b1 = load<T, lmul>(b_k + VL);
          const auto b2 = load<T, lmul>(b_k + 2 * VL);
          b_k += b_ld;

          {
            const auto a0 = set1<T, lmul>(*a0_k++);
            c00 = fmadd(a0, b0, c00);
            c01 = fmadd(a0, b1, c01);
            c02 = fmadd(a0, b2, c02);
          }
          {
            const auto a1 = set1<T, lmul>(*a1_k++);
            c10 = fmadd(a1, b0, c10);
            c11 = fmadd(a1, b1, c11);
            c12 = fmadd(a1, b2, c12);
          }
          {
            const auto a2 = set1<T, lmul>(*a2_k++);
            c20 = fmadd(a2, b0, c20);
            c21 = fmadd(a2, b1, c21);
            c22 = fmadd(a2, b2, c22);
          }
          {
            const auto a3 = set1<T, lmul>(*a3_k++);
            c30 = fmadd(a3, b0, c30);
            c31 = fmadd(a3, b1, c31);
            c32 = fmadd(a3, b2, c32);
          }
        }

        const auto c0_ptr = c0_base + j;
        const auto c1_ptr = c1_base + j;
        const auto c2_ptr = c2_base + j;
        const auto c3_ptr = c3_base + j;

        Epilogue::store<T, lmul, FastPath>(c0_ptr + 0 * VL, c00, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c0_ptr + 1 * VL, c01, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c0_ptr + 2 * VL, c02, alpha, beta);

        Epilogue::store<T, lmul, FastPath>(c1_ptr + 0 * VL, c10, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_ptr + 1 * VL, c11, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_ptr + 2 * VL, c12, alpha, beta);

        Epilogue::store<T, lmul, FastPath>(c2_ptr + 0 * VL, c20, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c2_ptr + 1 * VL, c21, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c2_ptr + 2 * VL, c22, alpha, beta);

        Epilogue::store<T, lmul, FastPath>(c3_ptr + 0 * VL, c30, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c3_ptr + 1 * VL, c31, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c3_ptr + 2 * VL, c32, alpha, beta);
      }

      // 2. Cleanup 4x2 Tiles (8 columns, decoupled broadcasts)
      // For N=32: runs 1 tile for remaining 8 cols (25%)
      // For N=64: runs 2 tiles for remaining 16 cols (25%)
      // 8 accumulators + 2 loads + 4 broadcasts = 14 registers (zero spills!)
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

      // 3. Fallback 4x1 Tiles (4 columns)
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
  }

  template <int lmul = 1>
  static inline void
  gemm_mippv2_meteorlake_mr4_nr3(const PackedRowMajor<T> &__restrict A,
                                 const PackedRowMajor<T> &__restrict B,
                                 PackedRowMajor<T> &__restrict C) {
    gemm_mippv2_meteorlake_mr4_nr3_core<lmul, true>(A, B, C);
  }

  template <int lmul = 1>
  static inline void
  gemm_mippv2_meteorlake_mr4_nr3(const PackedRowMajor<T> &__restrict A,
                                 const PackedRowMajor<T> &__restrict B,
                                 PackedRowMajor<T> &__restrict C,
                                 T alpha, T beta) {
    if (__builtin_expect(alpha == T{1} && beta == T{0}, 1)) {
      gemm_mippv2_meteorlake_mr4_nr3_core<lmul, true>(A, B, C);
    } else {
      gemm_mippv2_meteorlake_mr4_nr3_core<lmul, false>(A, B, C, alpha, beta);
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

  template <int lmul = 4>
  static inline void
  gemm_mippv2_x100_register_blocked(const PackedRowMajor<T> &__restrict A,
                                    const PackedRowMajor<T> &__restrict B,
                                    PackedRowMajor<T> &__restrict C,
                                    T alpha = T{1}, T beta = T{0}) {
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

        Epilogue::store<T, lmul>(C[i + 0] + j, c00, alpha, beta);
        Epilogue::store<T, lmul>(C[i + 0] + j + VL, c01, alpha, beta);

        Epilogue::store<T, lmul>(C[i + 1] + j, c10, alpha, beta);
        Epilogue::store<T, lmul>(C[i + 1] + j + VL, c11, alpha, beta);

        Epilogue::store<T, lmul>(C[i + 2] + j, c20, alpha, beta);
        Epilogue::store<T, lmul>(C[i + 2] + j + VL, c21, alpha, beta);
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

        Epilogue::store<T, lmul>(C[i] + j, c0, alpha, beta);
        Epilogue::store<T, lmul>(C[i] + j + VL, c1, alpha, beta);
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
      const PackedRowMajor<T> &__restrict B, PackedRowMajor<T> &__restrict C,
      T alpha = T{1}, T beta = T{0}) {
    using namespace mipp;

    constexpr size_t VL = N<T, lmul>();

    constexpr size_t MR = 3;
    constexpr size_t NR = 2 * VL;

    const size_t Mfull = (A.rows / MR) * MR;

    //
    // Main 3-row kernel
    //
    for (size_t i = 0; i < Mfull; i += MR) {
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

        Epilogue::store<T, lmul>(C[i + 0] + j, c00, alpha, beta);
        Epilogue::store<T, lmul>(C[i + 0] + j + VL, c01, alpha, beta);

        Epilogue::store<T, lmul>(C[i + 1] + j, c10, alpha, beta);
        Epilogue::store<T, lmul>(C[i + 1] + j + VL, c11, alpha, beta);

        Epilogue::store<T, lmul>(C[i + 2] + j, c20, alpha, beta);
        Epilogue::store<T, lmul>(C[i + 2] + j + VL, c21, alpha, beta);
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

        Epilogue::store<T, lmul>(C[i] + j, c0, alpha, beta);
        Epilogue::store<T, lmul>(C[i] + j + VL, c1, alpha, beta);
      }
    }
  }

  template <int lmul = 4>
  static inline void
  gemm_mippv2_panel_x100(const PackedColMajor<T> &__restrict A,
                         const PackedRowMajor<T> &__restrict B,
                         PackedRowMajor<T> &__restrict C,
                         T alpha = T{1}, T beta = T{0}) {
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

        Epilogue::store<T, lmul>(&C(i + 0, j), c00, alpha, beta);
        Epilogue::store<T, lmul>(&C(i + 0, j + VL), c01, alpha, beta);

        Epilogue::store<T, lmul>(&C(i + 1, j), c10, alpha, beta);
        Epilogue::store<T, lmul>(&C(i + 1, j + VL), c11, alpha, beta);

        Epilogue::store<T, lmul>(&C(i + 2, j), c20, alpha, beta);
        Epilogue::store<T, lmul>(&C(i + 2, j + VL), c21, alpha, beta);
      }
    }
  }
