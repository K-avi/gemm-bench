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

  // ===========================================================================
  // ARM Cortex-A76 NEON Kernel (MR=6, NR=3*VL = 6 doubles with VL=2)
  // Optimal 6x3 register tile: 18 NEON accumulators + 3 B loads + 1 A broadcast
  // Uses 22 of the 32 vector registers (v0..v31), leaving 10 registers for OoO.
  // 100% Pure MIPPv2: fmadd with high broadcast reuse (3 vector FMAs per dup).
  // ===========================================================================

  template <int lmul = 1, bool FastPath = false>
  static inline void
  gemm_mippv2_a76_mr6_nr3_core(const PackedRowMajor<T> &__restrict A,
                               const PackedRowMajor<T> &__restrict B,
                               PackedRowMajor<T> &__restrict C,
                               T alpha = T{1}, T beta = T{0}) {
    using namespace mipp;

    constexpr size_t VL = N<T, lmul>();
    constexpr size_t MR = 6;
    constexpr size_t NR3 = 3 * VL;

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

      // 1. Main 6x3 Tiles (18 accumulators: 6 rows x 3 vector columns = 36 doubles)
      for (; j + NR3 <= B.cols; j += NR3) {
        auto c00 = set0<T, lmul>(); auto c01 = set0<T, lmul>(); auto c02 = set0<T, lmul>();
        auto c10 = set0<T, lmul>(); auto c11 = set0<T, lmul>(); auto c12 = set0<T, lmul>();
        auto c20 = set0<T, lmul>(); auto c21 = set0<T, lmul>(); auto c22 = set0<T, lmul>();
        auto c30 = set0<T, lmul>(); auto c31 = set0<T, lmul>(); auto c32 = set0<T, lmul>();
        auto c40 = set0<T, lmul>(); auto c41 = set0<T, lmul>(); auto c42 = set0<T, lmul>();
        auto c50 = set0<T, lmul>(); auto c51 = set0<T, lmul>(); auto c52 = set0<T, lmul>();

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
          b_k += b_ld;

          // Pair 0: Rows 0 & 1 (dual-pipe issue across columns)
          {
            const auto a0 = set1<T, lmul>(*a0_k++);
            const auto a1 = set1<T, lmul>(*a1_k++);

            c00 = fmadd(a0, b0, c00);
            c10 = fmadd(a1, b0, c10);

            c01 = fmadd(a0, b1, c01);
            c11 = fmadd(a1, b1, c11);

            c02 = fmadd(a0, b2, c02);
            c12 = fmadd(a1, b2, c12);
          }

          // Pair 1: Rows 2 & 3
          {
            const auto a2 = set1<T, lmul>(*a2_k++);
            const auto a3 = set1<T, lmul>(*a3_k++);

            c20 = fmadd(a2, b0, c20);
            c30 = fmadd(a3, b0, c30);

            c21 = fmadd(a2, b1, c21);
            c31 = fmadd(a3, b1, c31);

            c22 = fmadd(a2, b2, c22);
            c32 = fmadd(a3, b2, c32);
          }

          // Pair 2: Rows 4 & 5
          {
            const auto a4 = set1<T, lmul>(*a4_k++);
            const auto a5 = set1<T, lmul>(*a5_k++);

            c40 = fmadd(a4, b0, c40);
            c50 = fmadd(a5, b0, c50);

            c41 = fmadd(a4, b1, c41);
            c51 = fmadd(a5, b1, c51);

            c42 = fmadd(a4, b2, c42);
            c52 = fmadd(a5, b2, c52);
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

        Epilogue::store<T, lmul, FastPath>(c1_ptr + 0 * VL, c10, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_ptr + 1 * VL, c11, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_ptr + 2 * VL, c12, alpha, beta);

        Epilogue::store<T, lmul, FastPath>(c2_ptr + 0 * VL, c20, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c2_ptr + 1 * VL, c21, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c2_ptr + 2 * VL, c22, alpha, beta);

        Epilogue::store<T, lmul, FastPath>(c3_ptr + 0 * VL, c30, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c3_ptr + 1 * VL, c31, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c3_ptr + 2 * VL, c32, alpha, beta);

        Epilogue::store<T, lmul, FastPath>(c4_ptr + 0 * VL, c40, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c4_ptr + 1 * VL, c41, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c4_ptr + 2 * VL, c42, alpha, beta);

        Epilogue::store<T, lmul, FastPath>(c5_ptr + 0 * VL, c50, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c5_ptr + 1 * VL, c51, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c5_ptr + 2 * VL, c52, alpha, beta);
      }

      // 2. Column Cleanup 6x2 (12 accumulators, 4 columns)
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

        #pragma GCC unroll 1
        for (size_t k = 0; k < A.cols; ++k) {
          const auto b0 = load<T, lmul>(b_k + 0 * VL);
          const auto b1 = load<T, lmul>(b_k + 1 * VL);
          b_k += b_ld;

          const auto a0 = set1<T, lmul>(*a0_k++);
          const auto a1 = set1<T, lmul>(*a1_k++);

          c00 = fmadd(a0, b0, c00);
          c10 = fmadd(a1, b0, c10);

          c01 = fmadd(a0, b1, c01);
          c11 = fmadd(a1, b1, c11);

          const auto a2 = set1<T, lmul>(*a2_k++);
          const auto a3 = set1<T, lmul>(*a3_k++);

          c20 = fmadd(a2, b0, c20);
          c30 = fmadd(a3, b0, c30);

          c21 = fmadd(a2, b1, c21);
          c31 = fmadd(a3, b1, c31);

          const auto a4 = set1<T, lmul>(*a4_k++);
          const auto a5 = set1<T, lmul>(*a5_k++);

          c40 = fmadd(a4, b0, c40);
          c50 = fmadd(a5, b0, c50);

          c41 = fmadd(a4, b1, c41);
          c51 = fmadd(a5, b1, c51);
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

      // 3. Column Cleanup 6x1 (6 accumulators, 2 columns)
      while (j + VL <= B.cols) {
        auto c0 = set0<T, lmul>(); auto c1 = set0<T, lmul>(); auto c2 = set0<T, lmul>();
        auto c3 = set0<T, lmul>(); auto c4 = set0<T, lmul>(); auto c5 = set0<T, lmul>();

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
          c0 = fmadd(a0, b0, c0);
          c1 = fmadd(a1, b0, c1);

          const auto a2 = set1<T, lmul>(*a2_k++);
          const auto a3 = set1<T, lmul>(*a3_k++);
          c2 = fmadd(a2, b0, c2);
          c3 = fmadd(a3, b0, c3);

          const auto a4 = set1<T, lmul>(*a4_k++);
          const auto a5 = set1<T, lmul>(*a5_k++);
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

    // -------------------------------------------------------------------------
    // Remainder Rows (M % 6 != 0)
    // -------------------------------------------------------------------------
    size_t i = Mfull;

    // Sub-block of 4 rows (e.g. for M=64 where 64 = 10*6 + 4)
    if (i + 4 <= A.rows) {
      const T *a0_base = A[i + 0]; const T *a1_base = A[i + 1];
      const T *a2_base = A[i + 2]; const T *a3_base = A[i + 3];

      T *c0_base = C[i + 0]; T *c1_base = C[i + 1];
      T *c2_base = C[i + 2]; T *c3_base = C[i + 3];

      size_t j = 0;
      for (; j + NR3 <= B.cols; j += NR3) {
        auto c00 = set0<T, lmul>(); auto c01 = set0<T, lmul>(); auto c02 = set0<T, lmul>();
        auto c10 = set0<T, lmul>(); auto c11 = set0<T, lmul>(); auto c12 = set0<T, lmul>();
        auto c20 = set0<T, lmul>(); auto c21 = set0<T, lmul>(); auto c22 = set0<T, lmul>();
        auto c30 = set0<T, lmul>(); auto c31 = set0<T, lmul>(); auto c32 = set0<T, lmul>();

        const T *__restrict a0_k = a0_base; const T *__restrict a1_k = a1_base;
        const T *__restrict a2_k = a2_base; const T *__restrict a3_k = a3_base;
        const T *__restrict b_k = B.data + j;

        for (size_t k = 0; k < A.cols; ++k) {
          const auto b0 = load<T, lmul>(b_k + 0 * VL);
          const auto b1 = load<T, lmul>(b_k + 1 * VL);
          const auto b2 = load<T, lmul>(b_k + 2 * VL);
          b_k += b_ld;

          const auto a0 = set1<T, lmul>(*a0_k++);
          const auto a1 = set1<T, lmul>(*a1_k++);

          c00 = fmadd(a0, b0, c00);
          c10 = fmadd(a1, b0, c10);

          c01 = fmadd(a0, b1, c01);
          c11 = fmadd(a1, b1, c11);

          c02 = fmadd(a0, b2, c02);
          c12 = fmadd(a1, b2, c12);

          const auto a2 = set1<T, lmul>(*a2_k++);
          const auto a3 = set1<T, lmul>(*a3_k++);

          c20 = fmadd(a2, b0, c20);
          c30 = fmadd(a3, b0, c30);

          c21 = fmadd(a2, b1, c21);
          c31 = fmadd(a3, b1, c31);

          c22 = fmadd(a2, b2, c22);
          c32 = fmadd(a3, b2, c32);
        }

        Epilogue::store<T, lmul, FastPath>(c0_base + j + 0 * VL, c00, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c0_base + j + 1 * VL, c01, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c0_base + j + 2 * VL, c02, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_base + j + 0 * VL, c10, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_base + j + 1 * VL, c11, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_base + j + 2 * VL, c12, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c2_base + j + 0 * VL, c20, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c2_base + j + 1 * VL, c21, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c2_base + j + 2 * VL, c22, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c3_base + j + 0 * VL, c30, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c3_base + j + 1 * VL, c31, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c3_base + j + 2 * VL, c32, alpha, beta);
      }
      while (j + 2 * VL <= B.cols) {
        auto c00 = set0<T, lmul>(); auto c01 = set0<T, lmul>();
        auto c10 = set0<T, lmul>(); auto c11 = set0<T, lmul>();
        auto c20 = set0<T, lmul>(); auto c21 = set0<T, lmul>();
        auto c30 = set0<T, lmul>(); auto c31 = set0<T, lmul>();

        const T *__restrict a0_k = a0_base; const T *__restrict a1_k = a1_base;
        const T *__restrict a2_k = a2_base; const T *__restrict a3_k = a3_base;
        const T *__restrict b_k = B.data + j;

        for (size_t k = 0; k < A.cols; ++k) {
          const auto b0 = load<T, lmul>(b_k + 0 * VL);
          const auto b1 = load<T, lmul>(b_k + 1 * VL);
          b_k += b_ld;

          const auto a0 = set1<T, lmul>(*a0_k++);
          const auto a1 = set1<T, lmul>(*a1_k++);

          c00 = fmadd(a0, b0, c00);
          c10 = fmadd(a1, b0, c10);

          c01 = fmadd(a0, b1, c01);
          c11 = fmadd(a1, b1, c11);

          const auto a2 = set1<T, lmul>(*a2_k++);
          const auto a3 = set1<T, lmul>(*a3_k++);

          c20 = fmadd(a2, b0, c20);
          c30 = fmadd(a3, b0, c30);

          c21 = fmadd(a2, b1, c21);
          c31 = fmadd(a3, b1, c31);
        }

        Epilogue::store<T, lmul, FastPath>(c0_base + j + 0 * VL, c00, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c0_base + j + 1 * VL, c01, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_base + j + 0 * VL, c10, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_base + j + 1 * VL, c11, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c2_base + j + 0 * VL, c20, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c2_base + j + 1 * VL, c21, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c3_base + j + 0 * VL, c30, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c3_base + j + 1 * VL, c31, alpha, beta);
        j += 2 * VL;
      }
      while (j + VL <= B.cols) {
        auto c0 = set0<T, lmul>(); auto c1 = set0<T, lmul>();
        auto c2 = set0<T, lmul>(); auto c3 = set0<T, lmul>();
        const T *__restrict a0_k = a0_base; const T *__restrict a1_k = a1_base;
        const T *__restrict a2_k = a2_base; const T *__restrict a3_k = a3_base;
        const T *__restrict b_k = B.data + j;
        for (size_t k = 0; k < A.cols; ++k) {
          const auto b0 = load<T, lmul>(b_k); b_k += b_ld;
          const auto a0 = set1<T, lmul>(*a0_k++);
          const auto a1 = set1<T, lmul>(*a1_k++);
          c0 = fmadd(a0, b0, c0);
          c1 = fmadd(a1, b0, c1);

          const auto a2 = set1<T, lmul>(*a2_k++);
          const auto a3 = set1<T, lmul>(*a3_k++);
          c2 = fmadd(a2, b0, c2);
          c3 = fmadd(a3, b0, c3);
        }
        Epilogue::store<T, lmul, FastPath>(c0_base + j, c0, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_base + j, c1, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c2_base + j, c2, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c3_base + j, c3, alpha, beta);
        j += VL;
      }
      i += 4;
    }

    // Sub-block of 2 rows (e.g. for M=32 where 32 = 5*6 + 2, or M=128 where 128 = 21*6 + 2)
    if (i + 2 <= A.rows) {
      const T *a0_base = A[i + 0]; const T *a1_base = A[i + 1];
      T *c0_base = C[i + 0]; T *c1_base = C[i + 1];

      size_t j = 0;
      for (; j + NR3 <= B.cols; j += NR3) {
        auto c00 = set0<T, lmul>(); auto c01 = set0<T, lmul>(); auto c02 = set0<T, lmul>();
        auto c10 = set0<T, lmul>(); auto c11 = set0<T, lmul>(); auto c12 = set0<T, lmul>();

        const T *__restrict a0_k = a0_base; const T *__restrict a1_k = a1_base;
        const T *__restrict b_k = B.data + j;

        for (size_t k = 0; k < A.cols; ++k) {
          const auto b0 = load<T, lmul>(b_k + 0 * VL);
          const auto b1 = load<T, lmul>(b_k + 1 * VL);
          const auto b2 = load<T, lmul>(b_k + 2 * VL);
          b_k += b_ld;

          const auto a0 = set1<T, lmul>(*a0_k++);
          const auto a1 = set1<T, lmul>(*a1_k++);

          c00 = fmadd(a0, b0, c00);
          c10 = fmadd(a1, b0, c10);

          c01 = fmadd(a0, b1, c01);
          c11 = fmadd(a1, b1, c11);

          c02 = fmadd(a0, b2, c02);
          c12 = fmadd(a1, b2, c12);
        }

        Epilogue::store<T, lmul, FastPath>(c0_base + j + 0 * VL, c00, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c0_base + j + 1 * VL, c01, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c0_base + j + 2 * VL, c02, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_base + j + 0 * VL, c10, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_base + j + 1 * VL, c11, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_base + j + 2 * VL, c12, alpha, beta);
      }
      while (j + 2 * VL <= B.cols) {
        auto c00 = set0<T, lmul>(); auto c01 = set0<T, lmul>();
        auto c10 = set0<T, lmul>(); auto c11 = set0<T, lmul>();

        const T *__restrict a0_k = a0_base; const T *__restrict a1_k = a1_base;
        const T *__restrict b_k = B.data + j;

        for (size_t k = 0; k < A.cols; ++k) {
          const auto b0 = load<T, lmul>(b_k + 0 * VL);
          const auto b1 = load<T, lmul>(b_k + 1 * VL);
          b_k += b_ld;
          const auto a0 = set1<T, lmul>(*a0_k++);
          const auto a1 = set1<T, lmul>(*a1_k++);
          c00 = fmadd(a0, b0, c00);
          c10 = fmadd(a1, b0, c10);

          c01 = fmadd(a0, b1, c01);
          c11 = fmadd(a1, b1, c11);
        }

        Epilogue::store<T, lmul, FastPath>(c0_base + j + 0 * VL, c00, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c0_base + j + 1 * VL, c01, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_base + j + 0 * VL, c10, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_base + j + 1 * VL, c11, alpha, beta);
        j += 2 * VL;
      }
      while (j + VL <= B.cols) {
        auto c0 = set0<T, lmul>(); auto c1 = set0<T, lmul>();
        const T *__restrict a0_k = a0_base; const T *__restrict a1_k = a1_base;
        const T *__restrict b_k = B.data + j;
        for (size_t k = 0; k < A.cols; ++k) {
          const auto b0 = load<T, lmul>(b_k); b_k += b_ld;
          const auto a0 = set1<T, lmul>(*a0_k++);
          const auto a1 = set1<T, lmul>(*a1_k++);
          c0 = fmadd(a0, b0, c0);
          c1 = fmadd(a1, b0, c1);
        }
        Epilogue::store<T, lmul, FastPath>(c0_base + j, c0, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_base + j, c1, alpha, beta);
        j += VL;
      }
      i += 2;
    }

    // Single remainder row
    if (i < A.rows) {
      const T *a0_base = A[i + 0];
      T *c0_base = C[i + 0];
      size_t j = 0;
      while (j + VL <= B.cols) {
        auto c0 = set0<T, lmul>();
        const T *__restrict a0_k = a0_base;
        const T *__restrict b_k = B.data + j;
        for (size_t k = 0; k < A.cols; ++k) {
          const auto b0 = load<T, lmul>(b_k); b_k += b_ld;
          c0 = fmadd(set1<T, lmul>(*a0_k++), b0, c0);
        }
        Epilogue::store<T, lmul, FastPath>(c0_base + j, c0, alpha, beta);
        j += VL;
      }
    }
  }

  template <int lmul = 1>
  static inline void
  gemm_mippv2_a76_mr6_nr3(const PackedRowMajor<T> &__restrict A,
                          const PackedRowMajor<T> &__restrict B,
                          PackedRowMajor<T> &__restrict C) {
    gemm_mippv2_a76_mr6_nr3_core<lmul, true>(A, B, C);
  }

  template <int lmul = 1>
  static inline void
  gemm_mippv2_a76_mr6_nr3(const PackedRowMajor<T> &__restrict A,
                          const PackedRowMajor<T> &__restrict B,
                          PackedRowMajor<T> &__restrict C,
                          T alpha, T beta) {
    if (__builtin_expect(alpha == T{1} && beta == T{0}, 1)) {
      gemm_mippv2_a76_mr6_nr3_core<lmul, true>(A, B, C);
    } else {
      gemm_mippv2_a76_mr6_nr3_core<lmul, false>(A, B, C, alpha, beta);
    }
  }


  // ===========================================================================
  // AMD Zen 4 AVX-512 Kernel (MR=4, NR=4*VL = 32 doubles with VL=8)
  // Optimal 4x4 register tile: 16 ZMM accumulators + 4 B loads + 1 A broadcast
  // 100% Pure MIPPv2: fmadd with explicit per-row broadcast reuse.

  // ===========================================================================
  template <int lmul = 1, bool FastPath = false>
  static inline void
  gemm_mippv2_zen4_mr4_nr4_fmaddi_core(const PackedRowMajor<T> &__restrict A,
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

      // 1. Main 4x4 Tiles using mipp::fmaddi
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
            const T a0 = *a0_k++;
            c00 = fmaddi(b0, a0, c00);
            c01 = fmaddi(b1, a0, c01);
            c02 = fmaddi(b2, a0, c02);
            c03 = fmaddi(b3, a0, c03);
          }
          {
            const T a1 = *a1_k++;
            c10 = fmaddi(b0, a1, c10);
            c11 = fmaddi(b1, a1, c11);
            c12 = fmaddi(b2, a1, c12);
            c13 = fmaddi(b3, a1, c13);
          }
          {
            const T a2 = *a2_k++;
            c20 = fmaddi(b0, a2, c20);
            c21 = fmaddi(b1, a2, c21);
            c22 = fmaddi(b2, a2, c22);
            c23 = fmaddi(b3, a2, c23);
          }
          {
            const T a3 = *a3_k++;
            c30 = fmaddi(b0, a3, c30);
            c31 = fmaddi(b1, a3, c31);
            c32 = fmaddi(b2, a3, c32);
            c33 = fmaddi(b3, a3, c33);
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

          const T a0 = *a0_k++;
          const T a1 = *a1_k++;
          const T a2 = *a2_k++;
          const T a3 = *a3_k++;

          c00 = fmaddi(b0, a0, c00);
          c01 = fmaddi(b1, a0, c01);
          c10 = fmaddi(b0, a1, c10);
          c11 = fmaddi(b1, a1, c11);
          c20 = fmaddi(b0, a2, c20);
          c21 = fmaddi(b1, a2, c21);
          c30 = fmaddi(b0, a3, c30);
          c31 = fmaddi(b1, a3, c31);
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

          c0 = fmaddi(b0, *a0_k++, c0);
          c1 = fmaddi(b0, *a1_k++, c1);
          c2 = fmaddi(b0, *a2_k++, c2);
          c3 = fmaddi(b0, *a3_k++, c3);
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

          const T a = a_ptr[k];
          c0 = fmaddi(b0, a, c0);
          c1 = fmaddi(b1, a, c1);
          c2 = fmaddi(b2, a, c2);
          c3 = fmaddi(b3, a, c3);
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
          c0 = fmaddi(b0, a_ptr[k], c0);
        }

        Epilogue::store<T, lmul, FastPath>(C[i] + j, c0, alpha, beta);
        j += VL;
      }
    }
  }

  template <int lmul = 1>
  static inline void
  gemm_mippv2_zen4_mr4_nr4_fmaddi(const PackedRowMajor<T> &__restrict A,
                                  const PackedRowMajor<T> &__restrict B,
                                  PackedRowMajor<T> &__restrict C) {
    gemm_mippv2_zen4_mr4_nr4_fmaddi_core<lmul, true>(A, B, C);
  }

  template <int lmul = 1>
  static inline void
  gemm_mippv2_zen4_mr4_nr4_fmaddi(const PackedRowMajor<T> &__restrict A,
                                  const PackedRowMajor<T> &__restrict B,
                                  PackedRowMajor<T> &__restrict C,
                                  T alpha, T beta) {
    if (__builtin_expect(alpha == T{1} && beta == T{0}, 1)) {
      gemm_mippv2_zen4_mr4_nr4_fmaddi_core<lmul, true>(A, B, C);
    } else {
      gemm_mippv2_zen4_mr4_nr4_fmaddi_core<lmul, false>(A, B, C, alpha, beta);
    }
  }


  // 2. Firestorm MR=4 NR=4 fmaddi (16 accumulators)
  template <int lmul = 1, bool FastPath = false>
  static inline void
  gemm_mippv2_firestorm_mr4_nr4_fmaddi_core(const PackedRowMajor<T> &__restrict A,
                                            const PackedRowMajor<T> &__restrict B,
                                            PackedRowMajor<T> &__restrict C,
                                            T alpha = T{1}, T beta = T{0}) {
    gemm_mippv2_zen4_mr4_nr4_fmaddi_core<lmul, FastPath>(A, B, C, alpha, beta);
  }

  template <int lmul = 1>
  static inline void
  gemm_mippv2_firestorm_mr4_nr4_fmaddi(const PackedRowMajor<T> &__restrict A,
                                       const PackedRowMajor<T> &__restrict B,
                                       PackedRowMajor<T> &__restrict C) {
    gemm_mippv2_firestorm_mr4_nr4_fmaddi_core<lmul, true>(A, B, C);
  }

  template <int lmul = 1>
  static inline void
  gemm_mippv2_firestorm_mr4_nr4_fmaddi(const PackedRowMajor<T> &__restrict A,
                                       const PackedRowMajor<T> &__restrict B,
                                       PackedRowMajor<T> &__restrict C,
                                       T alpha, T beta) {
    if (__builtin_expect(alpha == T{1} && beta == T{0}, 1)) {
      gemm_mippv2_firestorm_mr4_nr4_fmaddi_core<lmul, true>(A, B, C);
    } else {
      gemm_mippv2_firestorm_mr4_nr4_fmaddi_core<lmul, false>(A, B, C, alpha, beta);
    }
  }


  // 4. Firestorm MR=6 NR=4 fmaddi (24 accumulators)
  template <int lmul = 1, bool FastPath = false>
  static inline void
  gemm_mippv2_firestorm_mr6_nr4_fmaddi_core(const PackedRowMajor<T> &__restrict A,
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
            const T a0 = *a0_k++;
            c00 = fmaddi(b0, a0, c00);
            c01 = fmaddi(b1, a0, c01);
            c02 = fmaddi(b2, a0, c02);
            c03 = fmaddi(b3, a0, c03);
          }
          {
            const T a1 = *a1_k++;
            c10 = fmaddi(b0, a1, c10);
            c11 = fmaddi(b1, a1, c11);
            c12 = fmaddi(b2, a1, c12);
            c13 = fmaddi(b3, a1, c13);
          }
          {
            const T a2 = *a2_k++;
            c20 = fmaddi(b0, a2, c20);
            c21 = fmaddi(b1, a2, c21);
            c22 = fmaddi(b2, a2, c22);
            c23 = fmaddi(b3, a2, c23);
          }
          {
            const T a3 = *a3_k++;
            c30 = fmaddi(b0, a3, c30);
            c31 = fmaddi(b1, a3, c31);
            c32 = fmaddi(b2, a3, c32);
            c33 = fmaddi(b3, a3, c33);
          }
          {
            const T a4 = *a4_k++;
            c40 = fmaddi(b0, a4, c40);
            c41 = fmaddi(b1, a4, c41);
            c42 = fmaddi(b2, a4, c42);
            c43 = fmaddi(b3, a4, c43);
          }
          {
            const T a5 = *a5_k++;
            c50 = fmaddi(b0, a5, c50);
            c51 = fmaddi(b1, a5, c51);
            c52 = fmaddi(b2, a5, c52);
            c53 = fmaddi(b3, a5, c53);
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

          const T a0 = *a0_k++; const T a1 = *a1_k++;
          const T a2 = *a2_k++; const T a3 = *a3_k++;
          const T a4 = *a4_k++; const T a5 = *a5_k++;

          c00 = fmaddi(b0, a0, c00); c01 = fmaddi(b1, a0, c01);
          c10 = fmaddi(b0, a1, c10); c11 = fmaddi(b1, a1, c11);
          c20 = fmaddi(b0, a2, c20); c21 = fmaddi(b1, a2, c21);
          c30 = fmaddi(b0, a3, c30); c31 = fmaddi(b1, a3, c31);
          c40 = fmaddi(b0, a4, c40); c41 = fmaddi(b1, a4, c41);
          c50 = fmaddi(b0, a5, c50); c51 = fmaddi(b1, a5, c51);
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

          const T a0 = *a0_k++; const T a1 = *a1_k++;
          const T a2 = *a2_k++; const T a3 = *a3_k++;
          const T a4 = *a4_k++; const T a5 = *a5_k++;

          c0 = fmaddi(b0, a0, c0);
          c1 = fmaddi(b0, a1, c1);
          c2 = fmaddi(b0, a2, c2);
          c3 = fmaddi(b0, a3, c3);
          c4 = fmaddi(b0, a4, c4);
          c5 = fmaddi(b0, a5, c5);
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

    // Row cleanups
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

          const T a0 = *a0_k++; const T a1 = *a1_k++;
          const T a2 = *a2_k++; const T a3 = *a3_k++;

          c00 = fmaddi(b0, a0, c00); c01 = fmaddi(b1, a0, c01); c02 = fmaddi(b2, a0, c02); c03 = fmaddi(b3, a0, c03);
          c10 = fmaddi(b0, a1, c10); c11 = fmaddi(b1, a1, c11); c12 = fmaddi(b2, a1, c12); c13 = fmaddi(b3, a1, c13);
          c20 = fmaddi(b0, a2, c20); c21 = fmaddi(b1, a2, c21); c22 = fmaddi(b2, a2, c22); c23 = fmaddi(b3, a2, c23);
          c30 = fmaddi(b0, a3, c30); c31 = fmaddi(b1, a3, c31); c32 = fmaddi(b2, a3, c32); c33 = fmaddi(b3, a3, c33);
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
          const T a0 = *a0_k++; const T a1 = *a1_k++;
          const T a2 = *a2_k++; const T a3 = *a3_k++;
          c0 = fmaddi(b0, a0, c0); c1 = fmaddi(b0, a1, c1);
          c2 = fmaddi(b0, a2, c2); c3 = fmaddi(b0, a3, c3);
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

          const T a0 = *a0_k++; const T a1 = *a1_k++;

          c00 = fmaddi(b0, a0, c00); c01 = fmaddi(b1, a0, c01); c02 = fmaddi(b2, a0, c02); c03 = fmaddi(b3, a0, c03);
          c10 = fmaddi(b0, a1, c10); c11 = fmaddi(b1, a1, c11); c12 = fmaddi(b2, a1, c12); c13 = fmaddi(b3, a1, c13);
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
          const T a0 = *a0_k++; const T a1 = *a1_k++;
          c0 = fmaddi(b0, a0, c0); c1 = fmaddi(b0, a1, c1);
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
          const T a0 = *a0_k++;
          c0 = fmaddi(b0, a0, c0);
        }
        Epilogue::store<T, lmul, FastPath>(c0_base + j, c0, alpha, beta);
        j += VL;
      }
      i += 1;
    }
  }

  template <int lmul = 1>
  static inline void
  gemm_mippv2_firestorm_mr6_nr4_fmaddi(const PackedRowMajor<T> &__restrict A,
                                       const PackedRowMajor<T> &__restrict B,
                                       PackedRowMajor<T> &__restrict C) {
    gemm_mippv2_firestorm_mr6_nr4_fmaddi_core<lmul, true>(A, B, C);
  }

  template <int lmul = 1>
  static inline void
  gemm_mippv2_firestorm_mr6_nr4_fmaddi(const PackedRowMajor<T> &__restrict A,
                                       const PackedRowMajor<T> &__restrict B,
                                       PackedRowMajor<T> &__restrict C,
                                       T alpha, T beta) {
    if (__builtin_expect(alpha == T{1} && beta == T{0}, 1)) {
      gemm_mippv2_firestorm_mr6_nr4_fmaddi_core<lmul, true>(A, B, C);
    } else {
      gemm_mippv2_firestorm_mr6_nr4_fmaddi_core<lmul, false>(A, B, C, alpha, beta);
    }
  }


  // =========================================================================
  // X100: gemm_mippv2_x100_register_blocked_apack4
  // =========================================================================

  struct ABlock4 {
    double d0;
    double d1;
    double d2;
    double d3;
  };

  // FastPath 3-argument version (alpha=1, beta=0): identical to origin/main
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
 // General 5-argument version (arbitrary alpha, beta)
  template <int lmul = 2>
  static inline void gemm_mippv2_x100_register_blocked_apack4(
      const PackedRowMajor<T> &__restrict A,
      const PackedRowMajor<T> &__restrict B, PackedRowMajor<T> &__restrict C,
      T alpha, T beta) {
    if (__builtin_expect(alpha == T{1} && beta == T{0}, 1)) {
      gemm_mippv2_x100_register_blocked_apack4<lmul>(A, B, C);
      return;
    }

    using namespace mipp;

    constexpr size_t VL = N<T, lmul>();

    constexpr size_t MR = 3;
    constexpr size_t NR = 2 * VL;

    const size_t Mfull = (A.rows / MR) * MR;

    for (size_t i = 0; i < Mfull; i += MR) {
      const auto *a0_ptr = reinterpret_cast<const ABlock4 *>(A[i + 0]);
      const auto *a1_ptr = reinterpret_cast<const ABlock4 *>(A[i + 1]);
      const auto *a2_ptr = reinterpret_cast<const ABlock4 *>(A[i + 2]);

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

        Epilogue::store<T, lmul, false>(c0_base + j, c00, alpha, beta);
        Epilogue::store<T, lmul, false>(c0_base + j + VL, c01, alpha, beta);

        Epilogue::store<T, lmul, false>(c1_base + j, c10, alpha, beta);
        Epilogue::store<T, lmul, false>(c1_base + j + VL, c11, alpha, beta);

        Epilogue::store<T, lmul, false>(c2_base + j, c20, alpha, beta);
        Epilogue::store<T, lmul, false>(c2_base + j + VL, c21, alpha, beta);
      }
    }

    for (size_t i = Mfull; i < A.rows; ++i) {
      const auto *a_ptr = reinterpret_cast<const ABlock4 *>(A[i]);
      T *__restrict c_base = C[i];

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

        Epilogue::store<T, lmul, false>(c_base + j, c0, alpha, beta);
        Epilogue::store<T, lmul, false>(c_base + j + VL, c1, alpha, beta);
      }
    }
  }



  template <int lmul = 1>
  static inline void
  gemm_mippv2_x60_mr6_nr4_fmaddi(const PackedRowMajor<T> &__restrict A,
                                       const PackedRowMajor<T> &__restrict B,
                                       PackedRowMajor<T> &__restrict C) {
    gemm_mippv2_firestorm_mr6_nr4_fmaddi_core<lmul, true>(A, B, C);
  }

  template <int lmul = 1>
  static inline void
  gemm_mippv2_x60_mr6_nr4_fmaddi(const PackedRowMajor<T> &__restrict A,
                                       const PackedRowMajor<T> &__restrict B,
                                       PackedRowMajor<T> &__restrict C,
                                       T alpha, T beta) {
    if (__builtin_expect(alpha == T{1} && beta == T{0}, 1)) {
      gemm_mippv2_firestorm_mr6_nr4_fmaddi_core<lmul, true>(A, B, C);
    } else {
      gemm_mippv2_firestorm_mr6_nr4_fmaddi_core<lmul, false>(A, B, C, alpha, beta);
    }
  }


  // =========================================================================
  // SpacemiT A100 Dedicated DGEMM Microkernels
  // VLEN = 1024 bits (128 bytes = 16 FP64 doubles at LMUL=1)
  // 32 architectural vector registers (v0-v31)
  // =========================================================================

  // =========================================================================
  // 1. mippv2_a100_mr7_nr4_pipe (LMUL=1: VL=16 doubles)
  // MR=7, NR=4*VL = 64 doubles.
  // 28 accumulateurs vectoriels + 4 reg B = 32 registres vectoriels (100%).
  // 63 lignes sur 64 (98.44%) exécutées dans la tuile principale pour M=64.
  // Entièrement pipeliné sur les chargements A et B.
  // =========================================================================
  template <int lmul = 1, bool FastPath = false>
  static inline void
  gemm_mippv2_a100_mr7_nr4_pipe_core(const PackedRowMajor<T> &__restrict A,
                                     const PackedRowMajor<T> &__restrict B,
                                     PackedRowMajor<T> &__restrict C,
                                     T alpha = T{1}, T beta = T{0}) {
    using namespace mipp;

    constexpr size_t VL = N<T, lmul>();
    constexpr size_t MR = 7;
    constexpr size_t NR4 = 4 * VL;

    const size_t b_ld = B.ld;
    const size_t Mfull = (A.rows / MR) * MR;
    const size_t K = A.cols;

    for (size_t i = 0; i < Mfull; i += MR) {
      const T *a0_base = A[i + 0];
      const T *a1_base = A[i + 1];
      const T *a2_base = A[i + 2];
      const T *a3_base = A[i + 3];
      const T *a4_base = A[i + 4];
      const T *a5_base = A[i + 5];
      const T *a6_base = A[i + 6];

      T *c0_base = C[i + 0];
      T *c1_base = C[i + 1];
      T *c2_base = C[i + 2];
      T *c3_base = C[i + 3];
      T *c4_base = C[i + 4];
      T *c5_base = C[i + 5];
      T *c6_base = C[i + 6];

      size_t j = 0;

      for (; j + NR4 <= B.cols; j += NR4) {
        auto c00 = set0<T, lmul>(); auto c01 = set0<T, lmul>(); auto c02 = set0<T, lmul>(); auto c03 = set0<T, lmul>();
        auto c10 = set0<T, lmul>(); auto c11 = set0<T, lmul>(); auto c12 = set0<T, lmul>(); auto c13 = set0<T, lmul>();
        auto c20 = set0<T, lmul>(); auto c21 = set0<T, lmul>(); auto c22 = set0<T, lmul>(); auto c23 = set0<T, lmul>();
        auto c30 = set0<T, lmul>(); auto c31 = set0<T, lmul>(); auto c32 = set0<T, lmul>(); auto c33 = set0<T, lmul>();
        auto c40 = set0<T, lmul>(); auto c41 = set0<T, lmul>(); auto c42 = set0<T, lmul>(); auto c43 = set0<T, lmul>();
        auto c50 = set0<T, lmul>(); auto c51 = set0<T, lmul>(); auto c52 = set0<T, lmul>(); auto c53 = set0<T, lmul>();
        auto c60 = set0<T, lmul>(); auto c61 = set0<T, lmul>(); auto c62 = set0<T, lmul>(); auto c63 = set0<T, lmul>();

        const T *__restrict a0_k = a0_base;
        const T *__restrict a1_k = a1_base;
        const T *__restrict a2_k = a2_base;
        const T *__restrict a3_k = a3_base;
        const T *__restrict a4_k = a4_base;
        const T *__restrict a5_k = a5_base;
        const T *__restrict a6_k = a6_base;
        const T *__restrict b_k = B.data + j;

        if (__builtin_expect(K > 0, 1)) {
          auto b0 = load<T, lmul>(b_k + 0 * VL);
          auto b1 = load<T, lmul>(b_k + 1 * VL);
          auto b2 = load<T, lmul>(b_k + 2 * VL);
          auto b3 = load<T, lmul>(b_k + 3 * VL);
          b_k += b_ld;

          size_t k = 0;
          #pragma GCC unroll 1
          for (; k + 1 < K; ++k) {
            const T a0 = *a0_k++;
            const T a1 = *a1_k++;
            const T a2 = *a2_k++;
            const T a3 = *a3_k++;
            const T a4 = *a4_k++;
            const T a5 = *a5_k++;
            const T a6 = *a6_k++;

            c00 = fmaddi(b0, a0, c00);
            c10 = fmaddi(b0, a1, c10);
            c20 = fmaddi(b0, a2, c20);
            c30 = fmaddi(b0, a3, c30);
            c40 = fmaddi(b0, a4, c40);
            c50 = fmaddi(b0, a5, c50);
            c60 = fmaddi(b0, a6, c60);
            b0 = load<T, lmul>(b_k + 0 * VL);

            c01 = fmaddi(b1, a0, c01);
            c11 = fmaddi(b1, a1, c11);
            c21 = fmaddi(b1, a2, c21);
            c31 = fmaddi(b1, a3, c31);
            c41 = fmaddi(b1, a4, c41);
            c51 = fmaddi(b1, a5, c51);
            c61 = fmaddi(b1, a6, c61);
            b1 = load<T, lmul>(b_k + 1 * VL);

            c02 = fmaddi(b2, a0, c02);
            c12 = fmaddi(b2, a1, c12);
            c22 = fmaddi(b2, a2, c22);
            c32 = fmaddi(b2, a3, c32);
            c42 = fmaddi(b2, a4, c42);
            c52 = fmaddi(b2, a5, c52);
            c62 = fmaddi(b2, a6, c62);
            b2 = load<T, lmul>(b_k + 2 * VL);

            c03 = fmaddi(b3, a0, c03);
            c13 = fmaddi(b3, a1, c13);
            c23 = fmaddi(b3, a2, c23);
            c33 = fmaddi(b3, a3, c33);
            c43 = fmaddi(b3, a4, c43);
            c53 = fmaddi(b3, a5, c53);
            c63 = fmaddi(b3, a6, c63);
            b3 = load<T, lmul>(b_k + 3 * VL);

            b_k += b_ld;
          }

          const T a0 = *a0_k++;
          const T a1 = *a1_k++;
          const T a2 = *a2_k++;
          const T a3 = *a3_k++;
          const T a4 = *a4_k++;
          const T a5 = *a5_k++;
          const T a6 = *a6_k++;

          c00 = fmaddi(b0, a0, c00);
          c10 = fmaddi(b0, a1, c10);
          c20 = fmaddi(b0, a2, c20);
          c30 = fmaddi(b0, a3, c30);
          c40 = fmaddi(b0, a4, c40);
          c50 = fmaddi(b0, a5, c50);
          c60 = fmaddi(b0, a6, c60);

          c01 = fmaddi(b1, a0, c01);
          c11 = fmaddi(b1, a1, c11);
          c21 = fmaddi(b1, a2, c21);
          c31 = fmaddi(b1, a3, c31);
          c41 = fmaddi(b1, a4, c41);
          c51 = fmaddi(b1, a5, c51);
          c61 = fmaddi(b1, a6, c61);

          c02 = fmaddi(b2, a0, c02);
          c12 = fmaddi(b2, a1, c12);
          c22 = fmaddi(b2, a2, c22);
          c32 = fmaddi(b2, a3, c32);
          c42 = fmaddi(b2, a4, c42);
          c52 = fmaddi(b2, a5, c52);
          c62 = fmaddi(b2, a6, c62);

          c03 = fmaddi(b3, a0, c03);
          c13 = fmaddi(b3, a1, c13);
          c23 = fmaddi(b3, a2, c23);
          c33 = fmaddi(b3, a3, c33);
          c43 = fmaddi(b3, a4, c43);
          c53 = fmaddi(b3, a5, c53);
          c63 = fmaddi(b3, a6, c63);
        }

        const auto c0_ptr = c0_base + j;
        const auto c1_ptr = c1_base + j;
        const auto c2_ptr = c2_base + j;
        const auto c3_ptr = c3_base + j;
        const auto c4_ptr = c4_base + j;
        const auto c5_ptr = c5_base + j;
        const auto c6_ptr = c6_base + j;

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

        Epilogue::store<T, lmul, FastPath>(c6_ptr + 0 * VL, c60, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c6_ptr + 1 * VL, c61, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c6_ptr + 2 * VL, c62, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c6_ptr + 3 * VL, c63, alpha, beta);
      }

      while (j + 2 * VL <= B.cols) {
        auto c00 = set0<T, lmul>(); auto c01 = set0<T, lmul>();
        auto c10 = set0<T, lmul>(); auto c11 = set0<T, lmul>();
        auto c20 = set0<T, lmul>(); auto c21 = set0<T, lmul>();
        auto c30 = set0<T, lmul>(); auto c31 = set0<T, lmul>();
        auto c40 = set0<T, lmul>(); auto c41 = set0<T, lmul>();
        auto c50 = set0<T, lmul>(); auto c51 = set0<T, lmul>();
        auto c60 = set0<T, lmul>(); auto c61 = set0<T, lmul>();

        const T *__restrict a0_k = a0_base; const T *__restrict a1_k = a1_base;
        const T *__restrict a2_k = a2_base; const T *__restrict a3_k = a3_base;
        const T *__restrict a4_k = a4_base; const T *__restrict a5_k = a5_base;
        const T *__restrict a6_k = a6_base;
        const T *__restrict b_k = B.data + j;

        for (size_t k = 0; k < K; ++k) {
          const auto b0 = load<T, lmul>(b_k + 0 * VL);
          const auto b1 = load<T, lmul>(b_k + 1 * VL);
          b_k += b_ld;

          const T a0 = *a0_k++; const T a1 = *a1_k++;
          const T a2 = *a2_k++; const T a3 = *a3_k++;
          const T a4 = *a4_k++; const T a5 = *a5_k++;
          const T a6 = *a6_k++;

          c00 = fmaddi(b0, a0, c00); c01 = fmaddi(b1, a0, c01);
          c10 = fmaddi(b0, a1, c10); c11 = fmaddi(b1, a1, c11);
          c20 = fmaddi(b0, a2, c20); c21 = fmaddi(b1, a2, c21);
          c30 = fmaddi(b0, a3, c30); c31 = fmaddi(b1, a3, c31);
          c40 = fmaddi(b0, a4, c40); c41 = fmaddi(b1, a4, c41);
          c50 = fmaddi(b0, a5, c50); c51 = fmaddi(b1, a5, c51);
          c60 = fmaddi(b0, a6, c60); c61 = fmaddi(b1, a6, c61);
        }

        Epilogue::store<T, lmul, FastPath>(c0_base + j + 0 * VL, c00, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c0_base + j + 1 * VL, c01, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_base + j + 0 * VL, c10, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_base + j + 1 * VL, c11, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c2_base + j + 0 * VL, c20, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c2_base + j + 1 * VL, c21, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c3_base + j + 0 * VL, c30, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c3_base + j + 1 * VL, c31, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c4_base + j + 0 * VL, c40, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c4_base + j + 1 * VL, c41, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c5_base + j + 0 * VL, c50, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c5_base + j + 1 * VL, c51, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c6_base + j + 0 * VL, c60, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c6_base + j + 1 * VL, c61, alpha, beta);

        j += 2 * VL;
      }

      while (j + VL <= B.cols) {
        auto c0 = set0<T, lmul>(); auto c1 = set0<T, lmul>();
        auto c2 = set0<T, lmul>(); auto c3 = set0<T, lmul>();
        auto c4 = set0<T, lmul>(); auto c5 = set0<T, lmul>();
        auto c6 = set0<T, lmul>();

        const T *__restrict a0_k = a0_base; const T *__restrict a1_k = a1_base;
        const T *__restrict a2_k = a2_base; const T *__restrict a3_k = a3_base;
        const T *__restrict a4_k = a4_base; const T *__restrict a5_k = a5_base;
        const T *__restrict a6_k = a6_base;
        const T *__restrict b_k = B.data + j;

        for (size_t k = 0; k < K; ++k) {
          const auto b0 = load<T, lmul>(b_k);
          b_k += b_ld;

          const T a0 = *a0_k++; const T a1 = *a1_k++;
          const T a2 = *a2_k++; const T a3 = *a3_k++;
          const T a4 = *a4_k++; const T a5 = *a5_k++;
          const T a6 = *a6_k++;

          c0 = fmaddi(b0, a0, c0); c1 = fmaddi(b0, a1, c1);
          c2 = fmaddi(b0, a2, c2); c3 = fmaddi(b0, a3, c3);
          c4 = fmaddi(b0, a4, c4); c5 = fmaddi(b0, a5, c5);
          c6 = fmaddi(b0, a6, c6);
        }

        Epilogue::store<T, lmul, FastPath>(c0_base + j, c0, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_base + j, c1, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c2_base + j, c2, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c3_base + j, c3, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c4_base + j, c4, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c5_base + j, c5, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c6_base + j, c6, alpha, beta);

        j += VL;
      }

      for (; j < B.cols; ++j) {
        T c0 = T{0}, c1 = T{0}, c2 = T{0}, c3 = T{0}, c4 = T{0}, c5 = T{0}, c6 = T{0};
        for (size_t k = 0; k < K; ++k) {
          const T b_val = B.data[k * b_ld + j];
          c0 += a0_base[k] * b_val; c1 += a1_base[k] * b_val;
          c2 += a2_base[k] * b_val; c3 += a3_base[k] * b_val;
          c4 += a4_base[k] * b_val; c5 += a5_base[k] * b_val;
          c6 += a6_base[k] * b_val;
        }
        Epilogue::store(c0_base + j, c0, alpha, beta);
        Epilogue::store(c1_base + j, c1, alpha, beta);
        Epilogue::store(c2_base + j, c2, alpha, beta);
        Epilogue::store(c3_base + j, c3, alpha, beta);
        Epilogue::store(c4_base + j, c4, alpha, beta);
        Epilogue::store(c5_base + j, c5, alpha, beta);
        Epilogue::store(c6_base + j, c6, alpha, beta);
      }
    }

    // Row cleanups for rows beyond Mfull (for M=64, Mfull=63 -> exactly 1 remaining row)
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

        if (__builtin_expect(K > 0, 1)) {
          auto b0 = load<T, lmul>(b_k + 0 * VL);
          auto b1 = load<T, lmul>(b_k + 1 * VL);
          auto b2 = load<T, lmul>(b_k + 2 * VL);
          auto b3 = load<T, lmul>(b_k + 3 * VL);
          b_k += b_ld;

          size_t k = 0;
          #pragma GCC unroll 1
          for (; k + 1 < K; ++k) {
            const T a0 = *a0_k++; const T a1 = *a1_k++;
            const T a2 = *a2_k++; const T a3 = *a3_k++;

            c00 = fmaddi(b0, a0, c00); c10 = fmaddi(b0, a1, c10);
            c20 = fmaddi(b0, a2, c20); c30 = fmaddi(b0, a3, c30);
            b0 = load<T, lmul>(b_k + 0 * VL);

            c01 = fmaddi(b1, a0, c01); c11 = fmaddi(b1, a1, c11);
            c21 = fmaddi(b1, a2, c21); c31 = fmaddi(b1, a3, c31);
            b1 = load<T, lmul>(b_k + 1 * VL);

            c02 = fmaddi(b2, a0, c02); c12 = fmaddi(b2, a1, c12);
            c22 = fmaddi(b2, a2, c22); c32 = fmaddi(b2, a3, c32);
            b2 = load<T, lmul>(b_k + 2 * VL);

            c03 = fmaddi(b3, a0, c03); c13 = fmaddi(b3, a1, c13);
            c23 = fmaddi(b3, a2, c23); c33 = fmaddi(b3, a3, c33);
            b3 = load<T, lmul>(b_k + 3 * VL);

            b_k += b_ld;
          }

          const T a0 = *a0_k++; const T a1 = *a1_k++;
          const T a2 = *a2_k++; const T a3 = *a3_k++;
          c00 = fmaddi(b0, a0, c00); c10 = fmaddi(b0, a1, c10);
          c20 = fmaddi(b0, a2, c20); c30 = fmaddi(b0, a3, c30);

          c01 = fmaddi(b1, a0, c01); c11 = fmaddi(b1, a1, c11);
          c21 = fmaddi(b1, a2, c21); c31 = fmaddi(b1, a3, c31);

          c02 = fmaddi(b2, a0, c02); c12 = fmaddi(b2, a1, c12);
          c22 = fmaddi(b2, a2, c22); c32 = fmaddi(b2, a3, c32);

          c03 = fmaddi(b3, a0, c03); c13 = fmaddi(b3, a1, c13);
          c23 = fmaddi(b3, a2, c23); c33 = fmaddi(b3, a3, c33);
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
        for (size_t k = 0; k < K; ++k) {
          const auto b0 = load<T, lmul>(b_k);
          b_k += b_ld;
          const T a0 = *a0_k++; const T a1 = *a1_k++;
          const T a2 = *a2_k++; const T a3 = *a3_k++;
          c0 = fmaddi(b0, a0, c0); c1 = fmaddi(b0, a1, c1);
          c2 = fmaddi(b0, a2, c2); c3 = fmaddi(b0, a3, c3);
        }
        Epilogue::store<T, lmul, FastPath>(c0_base + j, c0, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_base + j, c1, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c2_base + j, c2, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c3_base + j, c3, alpha, beta);
        j += VL;
      }
      for (; j < B.cols; ++j) {
        T c0 = T{0}, c1 = T{0}, c2 = T{0}, c3 = T{0};
        for (size_t k = 0; k < K; ++k) {
          const T b_val = B.data[k * b_ld + j];
          c0 += a0_base[k] * b_val; c1 += a1_base[k] * b_val;
          c2 += a2_base[k] * b_val; c3 += a3_base[k] * b_val;
        }
        Epilogue::store(c0_base + j, c0, alpha, beta);
        Epilogue::store(c1_base + j, c1, alpha, beta);
        Epilogue::store(c2_base + j, c2, alpha, beta);
        Epilogue::store(c3_base + j, c3, alpha, beta);
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

        if (__builtin_expect(K > 0, 1)) {
          auto b0 = load<T, lmul>(b_k + 0 * VL);
          auto b1 = load<T, lmul>(b_k + 1 * VL);
          auto b2 = load<T, lmul>(b_k + 2 * VL);
          auto b3 = load<T, lmul>(b_k + 3 * VL);
          b_k += b_ld;

          size_t k = 0;
          #pragma GCC unroll 1
          for (; k + 1 < K; ++k) {
            const T a0 = *a0_k++; const T a1 = *a1_k++;
            c00 = fmaddi(b0, a0, c00); c10 = fmaddi(b0, a1, c10);
            b0 = load<T, lmul>(b_k + 0 * VL);
            c01 = fmaddi(b1, a0, c01); c11 = fmaddi(b1, a1, c11);
            b1 = load<T, lmul>(b_k + 1 * VL);
            c02 = fmaddi(b2, a0, c02); c12 = fmaddi(b2, a1, c12);
            b2 = load<T, lmul>(b_k + 2 * VL);
            c03 = fmaddi(b3, a0, c03); c13 = fmaddi(b3, a1, c13);
            b3 = load<T, lmul>(b_k + 3 * VL);
            b_k += b_ld;
          }
          const T a0 = *a0_k++; const T a1 = *a1_k++;
          c00 = fmaddi(b0, a0, c00); c10 = fmaddi(b0, a1, c10);
          c01 = fmaddi(b1, a0, c01); c11 = fmaddi(b1, a1, c11);
          c02 = fmaddi(b2, a0, c02); c12 = fmaddi(b2, a1, c12);
          c03 = fmaddi(b3, a0, c03); c13 = fmaddi(b3, a1, c13);
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
        for (size_t k = 0; k < K; ++k) {
          const auto b0 = load<T, lmul>(b_k);
          b_k += b_ld;
          const T a0 = *a0_k++; const T a1 = *a1_k++;
          c0 = fmaddi(b0, a0, c0); c1 = fmaddi(b0, a1, c1);
        }
        Epilogue::store<T, lmul, FastPath>(c0_base + j, c0, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_base + j, c1, alpha, beta);
        j += VL;
      }
      for (; j < B.cols; ++j) {
        T c0 = T{0}, c1 = T{0};
        for (size_t k = 0; k < K; ++k) {
          const T b_val = B.data[k * b_ld + j];
          c0 += a0_base[k] * b_val; c1 += a1_base[k] * b_val;
        }
        Epilogue::store(c0_base + j, c0, alpha, beta);
        Epilogue::store(c1_base + j, c1, alpha, beta);
      }
      i += 2;
    }

    while (i < A.rows) {
      const T *a0_base = A[i + 0];
      T *c0_base = C[i + 0];
      size_t j = 0;
      for (; j + NR4 <= B.cols; j += NR4) {
        auto c00 = set0<T, lmul>(); auto c01 = set0<T, lmul>(); auto c02 = set0<T, lmul>(); auto c03 = set0<T, lmul>();
        const T *__restrict a0_k = a0_base;
        const T *__restrict b_k = B.data + j;

        if (__builtin_expect(K > 0, 1)) {
          auto b0 = load<T, lmul>(b_k + 0 * VL);
          auto b1 = load<T, lmul>(b_k + 1 * VL);
          auto b2 = load<T, lmul>(b_k + 2 * VL);
          auto b3 = load<T, lmul>(b_k + 3 * VL);
          b_k += b_ld;

          size_t k = 0;
          #pragma GCC unroll 1
          for (; k + 1 < K; ++k) {
            const T a0 = *a0_k++;
            c00 = fmaddi(b0, a0, c00); b0 = load<T, lmul>(b_k + 0 * VL);
            c01 = fmaddi(b1, a0, c01); b1 = load<T, lmul>(b_k + 1 * VL);
            c02 = fmaddi(b2, a0, c02); b2 = load<T, lmul>(b_k + 2 * VL);
            c03 = fmaddi(b3, a0, c03); b3 = load<T, lmul>(b_k + 3 * VL);
            b_k += b_ld;
          }
          const T a0 = *a0_k++;
          c00 = fmaddi(b0, a0, c00); c01 = fmaddi(b1, a0, c01);
          c02 = fmaddi(b2, a0, c02); c03 = fmaddi(b3, a0, c03);
        }

        Epilogue::store<T, lmul, FastPath>(c0_base + j + 0 * VL, c00, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c0_base + j + 1 * VL, c01, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c0_base + j + 2 * VL, c02, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c0_base + j + 3 * VL, c03, alpha, beta);
      }
      while (j + VL <= B.cols) {
        auto c0 = set0<T, lmul>();
        const T *__restrict a0_k = a0_base;
        const T *__restrict b_k = B.data + j;
        for (size_t k = 0; k < K; ++k) {
          const auto b0 = load<T, lmul>(b_k);
          b_k += b_ld;
          const T a0 = *a0_k++;
          c0 = fmaddi(b0, a0, c0);
        }
        Epilogue::store<T, lmul, FastPath>(c0_base + j, c0, alpha, beta);
        j += VL;
      }
      for (; j < B.cols; ++j) {
        T c0 = T{0};
        for (size_t k = 0; k < K; ++k) {
          c0 += a0_base[k] * B.data[k * b_ld + j];
        }
        Epilogue::store(c0_base + j, c0, alpha, beta);
      }
      i += 1;
    }
  }

  template <int lmul = 1>
  static inline void
  gemm_mippv2_a100_mr7_nr4_pipe(const PackedRowMajor<T> &__restrict A,
                                const PackedRowMajor<T> &__restrict B,
                                PackedRowMajor<T> &__restrict C) {
    gemm_mippv2_a100_mr7_nr4_pipe_core<lmul, true>(A, B, C);
  }

  template <int lmul = 1>
  static inline void
  gemm_mippv2_a100_mr7_nr4_pipe(const PackedRowMajor<T> &__restrict A,
                                const PackedRowMajor<T> &__restrict B,
                                PackedRowMajor<T> &__restrict C,
                                T alpha, T beta) {
    if (__builtin_expect(alpha == T{1} && beta == T{0}, 1)) {
      gemm_mippv2_a100_mr7_nr4_pipe_core<lmul, true>(A, B, C);
    } else {
      gemm_mippv2_a100_mr7_nr4_pipe_core<lmul, false>(A, B, C, alpha, beta);
    }
  }

  // =========================================================================
  // 2. mippv2_a100_mr7_nr2_lmul2_pipe (LMUL=2: VL=32 doubles)
  // MR=7, NR=2*VL = 64 doubles.
  // 14 accumulateurs vectoriels (28 reg physiques) + 2 reg B (4 reg physiques) = 32 reg (100%).
  // Divise par 2 la pression du front-end d'instructions par rapport à LMUL=1 !
  // =========================================================================
  template <int lmul = 2, bool FastPath = false>
  static inline void
  gemm_mippv2_a100_mr7_nr2_lmul2_pipe_core(const PackedRowMajor<T> &__restrict A,
                                           const PackedRowMajor<T> &__restrict B,
                                           PackedRowMajor<T> &__restrict C,
                                           T alpha = T{1}, T beta = T{0}) {
    using namespace mipp;

    constexpr size_t VL = N<T, lmul>(); // 32 doubles à lmul=2
    constexpr size_t MR = 7;
    constexpr size_t NR2 = 2 * VL; // 64 doubles

    const size_t b_ld = B.ld;
    const size_t Mfull = (A.rows / MR) * MR;
    const size_t K = A.cols;

    for (size_t i = 0; i < Mfull; i += MR) {
      const T *a0_base = A[i + 0]; const T *a1_base = A[i + 1];
      const T *a2_base = A[i + 2]; const T *a3_base = A[i + 3];
      const T *a4_base = A[i + 4]; const T *a5_base = A[i + 5];
      const T *a6_base = A[i + 6];

      T *c0_base = C[i + 0]; T *c1_base = C[i + 1];
      T *c2_base = C[i + 2]; T *c3_base = C[i + 3];
      T *c4_base = C[i + 4]; T *c5_base = C[i + 5];
      T *c6_base = C[i + 6];

      size_t j = 0;
      for (; j + NR2 <= B.cols; j += NR2) {
        auto c00 = set0<T, lmul>(); auto c01 = set0<T, lmul>();
        auto c10 = set0<T, lmul>(); auto c11 = set0<T, lmul>();
        auto c20 = set0<T, lmul>(); auto c21 = set0<T, lmul>();
        auto c30 = set0<T, lmul>(); auto c31 = set0<T, lmul>();
        auto c40 = set0<T, lmul>(); auto c41 = set0<T, lmul>();
        auto c50 = set0<T, lmul>(); auto c51 = set0<T, lmul>();
        auto c60 = set0<T, lmul>(); auto c61 = set0<T, lmul>();

        const T *__restrict a0_k = a0_base; const T *__restrict a1_k = a1_base;
        const T *__restrict a2_k = a2_base; const T *__restrict a3_k = a3_base;
        const T *__restrict a4_k = a4_base; const T *__restrict a5_k = a5_base;
        const T *__restrict a6_k = a6_base;
        const T *__restrict b_k = B.data + j;

        if (__builtin_expect(K > 0, 1)) {
          auto b0 = load<T, lmul>(b_k + 0 * VL);
          auto b1 = load<T, lmul>(b_k + 1 * VL);
          b_k += b_ld;

          size_t k = 0;
          #pragma GCC unroll 1
          for (; k + 1 < K; ++k) {
            const T a0 = *a0_k++; const T a1 = *a1_k++;
            const T a2 = *a2_k++; const T a3 = *a3_k++;
            const T a4 = *a4_k++; const T a5 = *a5_k++;
            const T a6 = *a6_k++;

            c00 = fmaddi(b0, a0, c00); c10 = fmaddi(b0, a1, c10);
            c20 = fmaddi(b0, a2, c20); c30 = fmaddi(b0, a3, c30);
            c40 = fmaddi(b0, a4, c40); c50 = fmaddi(b0, a5, c50);
            c60 = fmaddi(b0, a6, c60);
            b0 = load<T, lmul>(b_k + 0 * VL);

            c01 = fmaddi(b1, a0, c01); c11 = fmaddi(b1, a1, c11);
            c21 = fmaddi(b1, a2, c21); c31 = fmaddi(b1, a3, c31);
            c41 = fmaddi(b1, a4, c41); c51 = fmaddi(b1, a5, c51);
            c61 = fmaddi(b1, a6, c61);
            b1 = load<T, lmul>(b_k + 1 * VL);

            b_k += b_ld;
          }

          const T a0 = *a0_k++; const T a1 = *a1_k++;
          const T a2 = *a2_k++; const T a3 = *a3_k++;
          const T a4 = *a4_k++; const T a5 = *a5_k++;
          const T a6 = *a6_k++;

          c00 = fmaddi(b0, a0, c00); c10 = fmaddi(b0, a1, c10);
          c20 = fmaddi(b0, a2, c20); c30 = fmaddi(b0, a3, c30);
          c40 = fmaddi(b0, a4, c40); c50 = fmaddi(b0, a5, c50);
          c60 = fmaddi(b0, a6, c60);

          c01 = fmaddi(b1, a0, c01); c11 = fmaddi(b1, a1, c11);
          c21 = fmaddi(b1, a2, c21); c31 = fmaddi(b1, a3, c31);
          c41 = fmaddi(b1, a4, c41); c51 = fmaddi(b1, a5, c51);
          c61 = fmaddi(b1, a6, c61);
        }

        const auto c0_ptr = c0_base + j; const auto c1_ptr = c1_base + j;
        const auto c2_ptr = c2_base + j; const auto c3_ptr = c3_base + j;
        const auto c4_ptr = c4_base + j; const auto c5_ptr = c5_base + j;
        const auto c6_ptr = c6_base + j;

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
        Epilogue::store<T, lmul, FastPath>(c6_ptr + 0 * VL, c60, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c6_ptr + 1 * VL, c61, alpha, beta);
      }

      while (j + VL <= B.cols) {
        auto c0 = set0<T, lmul>(); auto c1 = set0<T, lmul>();
        auto c2 = set0<T, lmul>(); auto c3 = set0<T, lmul>();
        auto c4 = set0<T, lmul>(); auto c5 = set0<T, lmul>();
        auto c6 = set0<T, lmul>();

        const T *__restrict a0_k = a0_base; const T *__restrict a1_k = a1_base;
        const T *__restrict a2_k = a2_base; const T *__restrict a3_k = a3_base;
        const T *__restrict a4_k = a4_base; const T *__restrict a5_k = a5_base;
        const T *__restrict a6_k = a6_base;
        const T *__restrict b_k = B.data + j;

        for (size_t k = 0; k < K; ++k) {
          const auto b0 = load<T, lmul>(b_k);
          b_k += b_ld;
          const T a0 = *a0_k++; const T a1 = *a1_k++;
          const T a2 = *a2_k++; const T a3 = *a3_k++;
          const T a4 = *a4_k++; const T a5 = *a5_k++;
          const T a6 = *a6_k++;

          c0 = fmaddi(b0, a0, c0); c1 = fmaddi(b0, a1, c1);
          c2 = fmaddi(b0, a2, c2); c3 = fmaddi(b0, a3, c3);
          c4 = fmaddi(b0, a4, c4); c5 = fmaddi(b0, a5, c5);
          c6 = fmaddi(b0, a6, c6);
        }

        Epilogue::store<T, lmul, FastPath>(c0_base + j, c0, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_base + j, c1, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c2_base + j, c2, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c3_base + j, c3, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c4_base + j, c4, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c5_base + j, c5, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c6_base + j, c6, alpha, beta);
        j += VL;
      }

      for (; j < B.cols; ++j) {
        T c0 = T{0}, c1 = T{0}, c2 = T{0}, c3 = T{0}, c4 = T{0}, c5 = T{0}, c6 = T{0};
        for (size_t k = 0; k < K; ++k) {
          const T b_val = B.data[k * b_ld + j];
          c0 += a0_base[k] * b_val; c1 += a1_base[k] * b_val;
          c2 += a2_base[k] * b_val; c3 += a3_base[k] * b_val;
          c4 += a4_base[k] * b_val; c5 += a5_base[k] * b_val;
          c6 += a6_base[k] * b_val;
        }
        Epilogue::store(c0_base + j, c0, alpha, beta); Epilogue::store(c1_base + j, c1, alpha, beta);
        Epilogue::store(c2_base + j, c2, alpha, beta); Epilogue::store(c3_base + j, c3, alpha, beta);
        Epilogue::store(c4_base + j, c4, alpha, beta); Epilogue::store(c5_base + j, c5, alpha, beta);
        Epilogue::store(c6_base + j, c6, alpha, beta);
      }
    }

    // Row cleanups
    size_t i = Mfull;
    while (i < A.rows) {
      const T *a0_base = A[i + 0];
      T *c0_base = C[i + 0];
      size_t j = 0;
      while (j + VL <= B.cols) {
        auto c0 = set0<T, lmul>();
        const T *__restrict a0_k = a0_base;
        const T *__restrict b_k = B.data + j;
        for (size_t k = 0; k < K; ++k) {
          const auto b0 = load<T, lmul>(b_k);
          b_k += b_ld;
          c0 = fmaddi(b0, *a0_k++, c0);
        }
        Epilogue::store<T, lmul, FastPath>(c0_base + j, c0, alpha, beta);
        j += VL;
      }
      for (; j < B.cols; ++j) {
        T c0 = T{0};
        for (size_t k = 0; k < K; ++k) c0 += a0_base[k] * B.data[k * b_ld + j];
        Epilogue::store(c0_base + j, c0, alpha, beta);
      }
      i += 1;
    }
  }

  template <int lmul = 2>
  static inline void
  gemm_mippv2_a100_mr7_nr2_lmul2_pipe(const PackedRowMajor<T> &__restrict A,
                                      const PackedRowMajor<T> &__restrict B,
                                      PackedRowMajor<T> &__restrict C) {
    gemm_mippv2_a100_mr7_nr2_lmul2_pipe_core<lmul, true>(A, B, C);
  }

  template <int lmul = 2>
  static inline void
  gemm_mippv2_a100_mr7_nr2_lmul2_pipe(const PackedRowMajor<T> &__restrict A,
                                      const PackedRowMajor<T> &__restrict B,
                                      PackedRowMajor<T> &__restrict C,
                                      T alpha, T beta) {
    if (__builtin_expect(alpha == T{1} && beta == T{0}, 1)) {
      gemm_mippv2_a100_mr7_nr2_lmul2_pipe_core<lmul, true>(A, B, C);
    } else {
      gemm_mippv2_a100_mr7_nr2_lmul2_pipe_core<lmul, false>(A, B, C, alpha, beta);
    }
  }

