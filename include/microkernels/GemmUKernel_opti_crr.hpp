  // ===========================================================================
  // Intel Meteor Lake & Skylake AVX2 / AVX-512 (CRR Variant)
  // Optimal 4x3 register tile: MR=4, NR=3*VL (12 DP in AVX2, 24 DP in AVX-512)
  // A is PackedColMajor<T>: 4 contiguous scalars A(i..i+3, k) per k iteration
  // B is PackedRowMajor<T>, C is PackedRowMajor<T>
  // ===========================================================================
  template <int lmul = 1, bool FastPath = false>
  static inline void
  gemm_mippv2_meteorlake_mr4_nr3_crr_core(const PackedColMajor<T> &__restrict A,
                                          const PackedRowMajor<T> &__restrict B,
                                          PackedRowMajor<T> &__restrict C,
                                          T alpha = T{1}, T beta = T{0}) {
    using namespace mipp;

    constexpr size_t VL = N<T, lmul>();
    constexpr size_t MR = 4;
    constexpr size_t NR3 = 3 * VL;

    const size_t a_ld = A.ld;
    const size_t b_ld = B.ld;
    const size_t K = A.cols;

    // Fast-path: single micro-tile (M == MR and N == NR3)
    if (__builtin_expect(A.rows == MR && B.cols == NR3, 1)) {
      const T *__restrict a_k = A.data;
      const T *__restrict b_k = B.data;
      const size_t c_ld = C.ld;

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

      const size_t K4 = (K / 4) * 4;
      for (size_t k = 0; k < K4; k += 4) {
        #pragma GCC unroll 4
        for (size_t s = 0; s < 4; ++s) {
          const auto b0 = load<T, lmul>(b_k + s * b_ld);
          const auto b1 = load<T, lmul>(b_k + s * b_ld + VL);
          const auto b2 = load<T, lmul>(b_k + s * b_ld + 2 * VL);

          const T *a_s = a_k + s * a_ld;
          {
            const auto a0 = set1<T, lmul>(a_s[0]);
            c00 = fmadd(a0, b0, c00);
            c01 = fmadd(a0, b1, c01);
            c02 = fmadd(a0, b2, c02);
          }
          {
            const auto a1 = set1<T, lmul>(a_s[1]);
            c10 = fmadd(a1, b0, c10);
            c11 = fmadd(a1, b1, c11);
            c12 = fmadd(a1, b2, c12);
          }
          {
            const auto a2 = set1<T, lmul>(a_s[2]);
            c20 = fmadd(a2, b0, c20);
            c21 = fmadd(a2, b1, c21);
            c22 = fmadd(a2, b2, c22);
          }
          {
            const auto a3 = set1<T, lmul>(a_s[3]);
            c30 = fmadd(a3, b0, c30);
            c31 = fmadd(a3, b1, c31);
            c32 = fmadd(a3, b2, c32);
          }
        }
        b_k += 4 * b_ld;
        a_k += 4 * a_ld;
      }

      for (size_t k = K4; k < K; ++k) {
        const auto b0 = load<T, lmul>(b_k);
        const auto b1 = load<T, lmul>(b_k + VL);
        const auto b2 = load<T, lmul>(b_k + 2 * VL);
        b_k += b_ld;

        {
          const auto a0 = set1<T, lmul>(a_k[0]);
          c00 = fmadd(a0, b0, c00);
          c01 = fmadd(a0, b1, c01);
          c02 = fmadd(a0, b2, c02);
        }
        {
          const auto a1 = set1<T, lmul>(a_k[1]);
          c10 = fmadd(a1, b0, c10);
          c11 = fmadd(a1, b1, c11);
          c12 = fmadd(a1, b2, c12);
        }
        {
          const auto a2 = set1<T, lmul>(a_k[2]);
          c20 = fmadd(a2, b0, c20);
          c21 = fmadd(a2, b1, c21);
          c22 = fmadd(a2, b2, c22);
        }
        {
          const auto a3 = set1<T, lmul>(a_k[3]);
          c30 = fmadd(a3, b0, c30);
          c31 = fmadd(a3, b1, c31);
          c32 = fmadd(a3, b2, c32);
        }
        a_k += a_ld;
      }

      T *c0 = C.data;
      T *c1 = c0 + c_ld;
      T *c2 = c1 + c_ld;
      T *c3 = c2 + c_ld;

      Epilogue::store<T, lmul, FastPath>(c0 + 0 * VL, c00, alpha, beta);
      Epilogue::store<T, lmul, FastPath>(c0 + 1 * VL, c01, alpha, beta);
      Epilogue::store<T, lmul, FastPath>(c0 + 2 * VL, c02, alpha, beta);
      Epilogue::store<T, lmul, FastPath>(c1 + 0 * VL, c10, alpha, beta);
      Epilogue::store<T, lmul, FastPath>(c1 + 1 * VL, c11, alpha, beta);
      Epilogue::store<T, lmul, FastPath>(c1 + 2 * VL, c12, alpha, beta);
      Epilogue::store<T, lmul, FastPath>(c2 + 0 * VL, c20, alpha, beta);
      Epilogue::store<T, lmul, FastPath>(c2 + 1 * VL, c21, alpha, beta);
      Epilogue::store<T, lmul, FastPath>(c2 + 2 * VL, c22, alpha, beta);
      Epilogue::store<T, lmul, FastPath>(c3 + 0 * VL, c30, alpha, beta);
      Epilogue::store<T, lmul, FastPath>(c3 + 1 * VL, c31, alpha, beta);
      Epilogue::store<T, lmul, FastPath>(c3 + 2 * VL, c32, alpha, beta);
      return;
    }

    const size_t Mfull = (A.rows / MR) * MR;

    size_t limit12 = B.cols;
    if (B.cols >= 4 * VL && (B.cols % NR3) == VL) {
      limit12 = B.cols - 4 * VL;
    }

    for (size_t i = 0; i < Mfull; i += MR) {
      const T *a_base = A.data + i;

      T *c0_base = C[i + 0];
      T *c1_base = C[i + 1];
      T *c2_base = C[i + 2];
      T *c3_base = C[i + 3];

      size_t j = 0;

      // 1. Main 4x3 Tiles (12 accumulators, 3 FMAs per broadcast)
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

        const T *__restrict a_k = a_base;
        const T *__restrict b_k = B.data + j;

        #pragma GCC unroll 1
        for (size_t k = 0; k < K; ++k) {
          const auto b0 = load<T, lmul>(b_k);
          const auto b1 = load<T, lmul>(b_k + VL);
          const auto b2 = load<T, lmul>(b_k + 2 * VL);
          b_k += b_ld;

          {
            const auto a0 = set1<T, lmul>(a_k[0]);
            c00 = fmadd(a0, b0, c00);
            c01 = fmadd(a0, b1, c01);
            c02 = fmadd(a0, b2, c02);
          }
          {
            const auto a1 = set1<T, lmul>(a_k[1]);
            c10 = fmadd(a1, b0, c10);
            c11 = fmadd(a1, b1, c11);
            c12 = fmadd(a1, b2, c12);
          }
          {
            const auto a2 = set1<T, lmul>(a_k[2]);
            c20 = fmadd(a2, b0, c20);
            c21 = fmadd(a2, b1, c21);
            c22 = fmadd(a2, b2, c22);
          }
          {
            const auto a3 = set1<T, lmul>(a_k[3]);
            c30 = fmadd(a3, b0, c30);
            c31 = fmadd(a3, b1, c31);
            c32 = fmadd(a3, b2, c32);
          }
          a_k += a_ld;
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

      // 2. Cleanup 4x2 Tiles
      while (j + 2 * VL <= B.cols) {
        auto c00 = set0<T, lmul>();
        auto c01 = set0<T, lmul>();
        auto c10 = set0<T, lmul>();
        auto c11 = set0<T, lmul>();
        auto c20 = set0<T, lmul>();
        auto c21 = set0<T, lmul>();
        auto c30 = set0<T, lmul>();
        auto c31 = set0<T, lmul>();

        const T *__restrict a_k = a_base;
        const T *__restrict b_k = B.data + j;

        #pragma GCC unroll 1
        for (size_t k = 0; k < K; ++k) {
          const auto b0 = load<T, lmul>(b_k);
          const auto b1 = load<T, lmul>(b_k + VL);
          b_k += b_ld;

          {
            const auto a0 = set1<T, lmul>(a_k[0]);
            c00 = fmadd(a0, b0, c00);
            c01 = fmadd(a0, b1, c01);
          }
          {
            const auto a1 = set1<T, lmul>(a_k[1]);
            c10 = fmadd(a1, b0, c10);
            c11 = fmadd(a1, b1, c11);
          }
          {
            const auto a2 = set1<T, lmul>(a_k[2]);
            c20 = fmadd(a2, b0, c20);
            c21 = fmadd(a2, b1, c21);
          }
          {
            const auto a3 = set1<T, lmul>(a_k[3]);
            c30 = fmadd(a3, b0, c30);
            c31 = fmadd(a3, b1, c31);
          }
          a_k += a_ld;
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

      // 3. Fallback 4x1 Tiles
      while (j + VL <= B.cols) {
        auto c0 = set0<T, lmul>();
        auto c1 = set0<T, lmul>();
        auto c2 = set0<T, lmul>();
        auto c3 = set0<T, lmul>();

        const T *__restrict a_k = a_base;
        const T *__restrict b_k = B.data + j;

        for (size_t k = 0; k < K; ++k) {
          const auto b0 = load<T, lmul>(b_k);
          b_k += b_ld;

          c0 = fmadd(set1<T, lmul>(a_k[0]), b0, c0);
          c1 = fmadd(set1<T, lmul>(a_k[1]), b0, c1);
          c2 = fmadd(set1<T, lmul>(a_k[2]), b0, c2);
          c3 = fmadd(set1<T, lmul>(a_k[3]), b0, c3);
          a_k += a_ld;
        }

        Epilogue::store<T, lmul, FastPath>(c0_base + j, c0, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_base + j, c1, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c2_base + j, c2, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c3_base + j, c3, alpha, beta);

        j += VL;
      }
    }

    // Tail rows (if A.rows % MR != 0)
    for (size_t i = Mfull; i < A.rows; ++i) {
      size_t j = 0;

      // NR3 tiles
      while (j + NR3 <= B.cols) {
        auto c0 = set0<T, lmul>(); auto c1 = set0<T, lmul>(); auto c2 = set0<T, lmul>();

        const T *__restrict a_k = A.data + i;
        const T *__restrict b_k = B.data + j;

        for (size_t k = 0; k < K; ++k) {
          const auto b0 = load<T, lmul>(b_k);
          const auto b1 = load<T, lmul>(b_k + VL);
          const auto b2 = load<T, lmul>(b_k + 2 * VL);
          b_k += b_ld;
          const T a = *a_k;
          a_k += a_ld;
          c0 = fmadd(set1<T, lmul>(a), b0, c0);
          c1 = fmadd(set1<T, lmul>(a), b1, c1);
          c2 = fmadd(set1<T, lmul>(a), b2, c2);
        }

        Epilogue::store<T, lmul, FastPath>(C[i] + j + 0 * VL, c0, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(C[i] + j + 1 * VL, c1, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(C[i] + j + 2 * VL, c2, alpha, beta);
        j += NR3;
      }

      // 2VL tile
      while (j + 2 * VL <= B.cols) {
        auto c0 = set0<T, lmul>(); auto c1 = set0<T, lmul>();

        const T *__restrict a_k = A.data + i;
        const T *__restrict b_k = B.data + j;

        for (size_t k = 0; k < K; ++k) {
          const auto b0 = load<T, lmul>(b_k);
          const auto b1 = load<T, lmul>(b_k + VL);
          b_k += b_ld;
          const T a = *a_k;
          a_k += a_ld;
          c0 = fmadd(set1<T, lmul>(a), b0, c0);
          c1 = fmadd(set1<T, lmul>(a), b1, c1);
        }

        Epilogue::store<T, lmul, FastPath>(C[i] + j + 0 * VL, c0, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(C[i] + j + 1 * VL, c1, alpha, beta);
        j += 2 * VL;
      }

      // 1VL tile
      while (j + VL <= B.cols) {
        auto c0 = set0<T, lmul>();

        const T *__restrict a_k = A.data + i;
        const T *__restrict b_k = B.data + j;

        for (size_t k = 0; k < K; ++k) {
          const auto b0 = load<T, lmul>(b_k);
          b_k += b_ld;
          const T a = *a_k;
          a_k += a_ld;
          c0 = fmadd(set1<T, lmul>(a), b0, c0);
        }

        Epilogue::store<T, lmul, FastPath>(C[i] + j, c0, alpha, beta);
        j += VL;
      }
    }
  }

  template <int lmul = 1>
  static inline void
  gemm_mippv2_meteorlake_mr4_nr3_crr(const PackedColMajor<T> &__restrict A,
                                     const PackedRowMajor<T> &__restrict B,
                                     PackedRowMajor<T> &__restrict C) {
    gemm_mippv2_meteorlake_mr4_nr3_crr_core<lmul, true>(A, B, C);
  }

  template <int lmul = 1>
  static inline void
  gemm_mippv2_meteorlake_mr4_nr3_crr(const PackedColMajor<T> &__restrict A,
                                     const PackedRowMajor<T> &__restrict B,
                                     PackedRowMajor<T> &__restrict C,
                                     T alpha, T beta) {
    if (__builtin_expect(alpha == T{1} && beta == T{0}, 1)) {
      gemm_mippv2_meteorlake_mr4_nr3_crr_core<lmul, true>(A, B, C);
    } else {
      gemm_mippv2_meteorlake_mr4_nr3_crr_core<lmul, false>(A, B, C, alpha, beta);
    }
  }

  template <int lmul = 1, bool FastPath = false>
  static inline void
  gemm_mippv2_zen4_mr4_nr4_fmaddi_crr_core(const PackedColMajor<T> &__restrict A,
                                           const PackedRowMajor<T> &__restrict B,
                                           PackedRowMajor<T> &__restrict C,
                                           T alpha = T{1}, T beta = T{0}) {
    using namespace mipp;

    constexpr size_t VL = N<T, lmul>();
    constexpr size_t MR = 4;
    constexpr size_t NR4 = 4 * VL;

    const size_t a_ld = A.ld;
    const size_t b_ld = B.ld;
    const size_t K = A.cols;

    // Fast-path: single micro-tile (M == MR and N == NR4)
    if (__builtin_expect(A.rows == MR && B.cols == NR4, 1)) {
      const T *__restrict a_k = A.data;
      const T *__restrict b_k = B.data;

      auto c00 = set0<T, lmul>(); auto c01 = set0<T, lmul>(); auto c02 = set0<T, lmul>(); auto c03 = set0<T, lmul>();
      auto c10 = set0<T, lmul>(); auto c11 = set0<T, lmul>(); auto c12 = set0<T, lmul>(); auto c13 = set0<T, lmul>();
      auto c20 = set0<T, lmul>(); auto c21 = set0<T, lmul>(); auto c22 = set0<T, lmul>(); auto c23 = set0<T, lmul>();
      auto c30 = set0<T, lmul>(); auto c31 = set0<T, lmul>(); auto c32 = set0<T, lmul>(); auto c33 = set0<T, lmul>();

      #pragma GCC unroll 1
      for (size_t k = 0; k < K; ++k) {
        const auto b0 = load<T, lmul>(b_k + 0 * VL);
        const auto b1 = load<T, lmul>(b_k + 1 * VL);
        const auto b2 = load<T, lmul>(b_k + 2 * VL);
        const auto b3 = load<T, lmul>(b_k + 3 * VL);
        b_k += b_ld;

        const auto a0 = set1<T, lmul>(a_k[0]);
        const auto a1 = set1<T, lmul>(a_k[1]);
        const auto a2 = set1<T, lmul>(a_k[2]);
        const auto a3 = set1<T, lmul>(a_k[3]);

        c00 = fmadd(b0, a0, c00);
        c01 = fmadd(b1, a0, c01);
        c02 = fmadd(b2, a0, c02);
        c03 = fmadd(b3, a0, c03);

        c10 = fmadd(b0, a1, c10);
        c11 = fmadd(b1, a1, c11);
        c12 = fmadd(b2, a1, c12);
        c13 = fmadd(b3, a1, c13);

        c20 = fmadd(b0, a2, c20);
        c21 = fmadd(b1, a2, c21);
        c22 = fmadd(b2, a2, c22);
        c23 = fmadd(b3, a2, c23);

        c30 = fmadd(b0, a3, c30);
        c31 = fmadd(b1, a3, c31);
        c32 = fmadd(b2, a3, c32);
        c33 = fmadd(b3, a3, c33);
        a_k += a_ld;
      }

      T *c0_ptr = C[0];
      T *c1_ptr = C[1];
      T *c2_ptr = C[2];
      T *c3_ptr = C[3];

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
      return;
    }

    const size_t Mfull = (A.rows / MR) * MR;

    for (size_t i = 0; i < Mfull; i += MR) {
      const T *a_base = A.data + i;

      T *c0_base = C[i + 0];
      T *c1_base = C[i + 1];
      T *c2_base = C[i + 2];
      T *c3_base = C[i + 3];

      size_t j = 0;

      // 1. Main 4x4 Tiles using mipp::fmaddi
      for (; j + NR4 <= B.cols; j += NR4) {
        auto c00 = set0<T, lmul>(); auto c01 = set0<T, lmul>(); auto c02 = set0<T, lmul>(); auto c03 = set0<T, lmul>();
        auto c10 = set0<T, lmul>(); auto c11 = set0<T, lmul>(); auto c12 = set0<T, lmul>(); auto c13 = set0<T, lmul>();
        auto c20 = set0<T, lmul>(); auto c21 = set0<T, lmul>(); auto c22 = set0<T, lmul>(); auto c23 = set0<T, lmul>();
        auto c30 = set0<T, lmul>(); auto c31 = set0<T, lmul>(); auto c32 = set0<T, lmul>(); auto c33 = set0<T, lmul>();

        const T *__restrict a_k = a_base;
        const T *__restrict b_k = B.data + j;

        #pragma GCC unroll 1
        for (size_t k = 0; k < K; ++k) {
          const auto b0 = load<T, lmul>(b_k + 0 * VL);
          const auto b1 = load<T, lmul>(b_k + 1 * VL);
          const auto b2 = load<T, lmul>(b_k + 2 * VL);
          const auto b3 = load<T, lmul>(b_k + 3 * VL);
          b_k += b_ld;

          const auto a0 = set1<T, lmul>(a_k[0]);
          const auto a1 = set1<T, lmul>(a_k[1]);
          const auto a2 = set1<T, lmul>(a_k[2]);
          const auto a3 = set1<T, lmul>(a_k[3]);

          c00 = fmadd(b0, a0, c00);
          c01 = fmadd(b1, a0, c01);
          c02 = fmadd(b2, a0, c02);
          c03 = fmadd(b3, a0, c03);

          c10 = fmadd(b0, a1, c10);
          c11 = fmadd(b1, a1, c11);
          c12 = fmadd(b2, a1, c12);
          c13 = fmadd(b3, a1, c13);

          c20 = fmadd(b0, a2, c20);
          c21 = fmadd(b1, a2, c21);
          c22 = fmadd(b2, a2, c22);
          c23 = fmadd(b3, a2, c23);

          c30 = fmadd(b0, a3, c30);
          c31 = fmadd(b1, a3, c31);
          c32 = fmadd(b2, a3, c32);
          c33 = fmadd(b3, a3, c33);
          a_k += a_ld;
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

      // 2. Cleanup 4x2 Tiles
      while (j + 2 * VL <= B.cols) {
        auto c00 = set0<T, lmul>(); auto c01 = set0<T, lmul>();
        auto c10 = set0<T, lmul>(); auto c11 = set0<T, lmul>();
        auto c20 = set0<T, lmul>(); auto c21 = set0<T, lmul>();
        auto c30 = set0<T, lmul>(); auto c31 = set0<T, lmul>();

        const T *__restrict a_k = a_base;
        const T *__restrict b_k = B.data + j;

        #pragma GCC unroll 1
        for (size_t k = 0; k < K; ++k) {
          const auto b0 = load<T, lmul>(b_k);
          const auto b1 = load<T, lmul>(b_k + VL);
          b_k += b_ld;

          {
            const T a0 = a_k[0];
            c00 = fmaddi(b0, a0, c00);
            c01 = fmaddi(b1, a0, c01);
          }
          {
            const T a1 = a_k[1];
            c10 = fmaddi(b0, a1, c10);
            c11 = fmaddi(b1, a1, c11);
          }
          {
            const T a2 = a_k[2];
            c20 = fmaddi(b0, a2, c20);
            c21 = fmaddi(b1, a2, c21);
          }
          {
            const T a3 = a_k[3];
            c30 = fmaddi(b0, a3, c30);
            c31 = fmaddi(b1, a3, c31);
          }
          a_k += a_ld;
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

      // 3. Cleanup 4x1 Tiles
      while (j + VL <= B.cols) {
        auto c0 = set0<T, lmul>(); auto c1 = set0<T, lmul>();
        auto c2 = set0<T, lmul>(); auto c3 = set0<T, lmul>();

        const T *__restrict a_k = a_base;
        const T *__restrict b_k = B.data + j;

        for (size_t k = 0; k < K; ++k) {
          const auto b0 = load<T, lmul>(b_k);
          b_k += b_ld;

          {
            const T a0 = a_k[0];
            c0 = fmaddi(b0, a0, c0);
          }
          {
            const T a1 = a_k[1];
            c1 = fmaddi(b0, a1, c1);
          }
          {
            const T a2 = a_k[2];
            c2 = fmaddi(b0, a2, c2);
          }
          {
            const T a3 = a_k[3];
            c3 = fmaddi(b0, a3, c3);
          }
          a_k += a_ld;
        }

        Epilogue::store<T, lmul, FastPath>(c0_base + j, c0, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c1_base + j, c1, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c2_base + j, c2, alpha, beta);
        Epilogue::store<T, lmul, FastPath>(c3_base + j, c3, alpha, beta);

        j += VL;
      }
    }

    // Tail rows (if A.rows % 4 != 0)
    for (size_t i = Mfull; i < A.rows; ++i) {
      size_t j = 0;

      while (j + NR4 <= B.cols) {
        auto c0 = set0<T, lmul>(); auto c1 = set0<T, lmul>();
        auto c2 = set0<T, lmul>(); auto c3 = set0<T, lmul>();

        const T *__restrict a_k = A.data + i;
        const T *__restrict b_k = B.data + j;

        for (size_t k = 0; k < K; ++k) {
          const auto b0 = load<T, lmul>(b_k + 0 * VL);
          const auto b1 = load<T, lmul>(b_k + 1 * VL);
          const auto b2 = load<T, lmul>(b_k + 2 * VL);
          const auto b3 = load<T, lmul>(b_k + 3 * VL);
          b_k += b_ld;

          const T a = *a_k;
          a_k += a_ld;

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
        const T *__restrict a_k = A.data + i;
        const T *__restrict b_k = B.data + j;

        for (size_t k = 0; k < K; ++k) {
          const auto b0 = load<T, lmul>(b_k);
          b_k += b_ld;
          const T a = *a_k;
          a_k += a_ld;
          c0 = fmaddi(b0, a, c0);
        }

        Epilogue::store<T, lmul, FastPath>(C[i] + j, c0, alpha, beta);
        j += VL;
      }
    }
  }

  template <int lmul = 1>
  static inline void
  gemm_mippv2_zen4_mr4_nr4_fmaddi_crr(const PackedColMajor<T> &__restrict A,
                                      const PackedRowMajor<T> &__restrict B,
                                      PackedRowMajor<T> &__restrict C) {
    gemm_mippv2_zen4_mr4_nr4_fmaddi_crr_core<lmul, true>(A, B, C);
  }

  template <int lmul = 1>
  static inline void
  gemm_mippv2_zen4_mr4_nr4_fmaddi_crr(const PackedColMajor<T> &__restrict A,
                                      const PackedRowMajor<T> &__restrict B,
                                      PackedRowMajor<T> &__restrict C,
                                      T alpha, T beta) {
    if (__builtin_expect(alpha == T{1} && beta == T{0}, 1)) {
      gemm_mippv2_zen4_mr4_nr4_fmaddi_crr_core<lmul, true>(A, B, C);
    } else {
      gemm_mippv2_zen4_mr4_nr4_fmaddi_crr_core<lmul, false>(A, B, C, alpha, beta);
    }
  }

  template <int lmul = 1, bool FastPath = false>
  static inline void
  gemm_mippv2_firestorm_mr4_nr4_fmaddi_crr_core(const PackedColMajor<T> &__restrict A,
                                                const PackedRowMajor<T> &__restrict B,
                                                PackedRowMajor<T> &__restrict C,
                                                T alpha = T{1}, T beta = T{0}) {
    gemm_mippv2_zen4_mr4_nr4_fmaddi_crr_core<lmul, FastPath>(A, B, C, alpha, beta);
  }

  template <int lmul = 1>
  static inline void
  gemm_mippv2_firestorm_mr4_nr4_fmaddi_crr(const PackedColMajor<T> &__restrict A,
                                           const PackedRowMajor<T> &__restrict B,
                                           PackedRowMajor<T> &__restrict C) {
    gemm_mippv2_firestorm_mr4_nr4_fmaddi_crr_core<lmul, true>(A, B, C);
  }

  template <int lmul = 1>
  static inline void
  gemm_mippv2_firestorm_mr4_nr4_fmaddi_crr(const PackedColMajor<T> &__restrict A,
                                           const PackedRowMajor<T> &__restrict B,
                                           PackedRowMajor<T> &__restrict C,
                                           T alpha, T beta) {
    if (__builtin_expect(alpha == T{1} && beta == T{0}, 1)) {
      gemm_mippv2_firestorm_mr4_nr4_fmaddi_crr_core<lmul, true>(A, B, C);
    } else {
      gemm_mippv2_firestorm_mr4_nr4_fmaddi_crr_core<lmul, false>(A, B, C, alpha, beta);
    }
  }

  // FastPath 3-argument version (alpha=1, beta=0) for CRR
  template <int lmul = 2>
  static inline void gemm_mippv2_x100_register_blocked_apack4_crr(
      const PackedColMajor<T> &__restrict A,
      const PackedRowMajor<T> &__restrict B, PackedRowMajor<T> &__restrict C) {
    using namespace mipp;

    constexpr size_t VL = N<T, lmul>();
    constexpr size_t MR = 3;
    constexpr size_t NR = 2 * VL;

    const size_t a_ld = A.ld;
    const size_t b_ld = B.ld;
    const size_t a_ld4 = 4 * a_ld;
    const size_t b_ld4 = 4 * b_ld;
    const size_t Mfull = (A.rows / MR) * MR;
    const size_t K4 = A.cols / 4;

    for (size_t i = 0; i < Mfull; i += MR) {
      const T *a_base = A.data + i;

      for (size_t j = 0; j < B.cols; j += NR) {
        auto c00 = set0<T, lmul>();
        auto c01 = set0<T, lmul>();
        auto c10 = set0<T, lmul>();
        auto c11 = set0<T, lmul>();
        auto c20 = set0<T, lmul>();
        auto c21 = set0<T, lmul>();

        const T *__restrict a_k = a_base;
        const T *__restrict b_k = B.data + j;

        for (size_t kk = 0; kk < K4; ++kk) {
          const T a0_k0 = a_k[0], a1_k0 = a_k[1], a2_k0 = a_k[2];
          const T *a_k1 = a_k + a_ld;
          const T a0_k1 = a_k1[0], a1_k1 = a_k1[1], a2_k1 = a_k1[2];
          const T *a_k2 = a_k1 + a_ld;
          const T a0_k2 = a_k2[0], a1_k2 = a_k2[1], a2_k2 = a_k2[2];
          const T *a_k3 = a_k2 + a_ld;
          const T a0_k3 = a_k3[0], a1_k3 = a_k3[1], a2_k3 = a_k3[2];
          a_k += a_ld4;

          const T *b_k1 = b_k + b_ld;
          const T *b_k2 = b_k1 + b_ld;
          const T *b_k3 = b_k2 + b_ld;

          // k = 0
          auto b00 = load<T, lmul>(b_k);
          auto b01 = load<T, lmul>(b_k + VL);

          c00 = fmaddi(b00, a0_k0, c00);
          c01 = fmaddi(b01, a0_k0, c01);
          c10 = fmaddi(b00, a1_k0, c10);
          c11 = fmaddi(b01, a1_k0, c11);
          c20 = fmaddi(b00, a2_k0, c20);
          c21 = fmaddi(b01, a2_k0, c21);

          // k = 1
          b00 = load<T, lmul>(b_k1);
          b01 = load<T, lmul>(b_k1 + VL);

          c00 = fmaddi(b00, a0_k1, c00);
          c01 = fmaddi(b01, a0_k1, c01);
          c10 = fmaddi(b00, a1_k1, c10);
          c11 = fmaddi(b01, a1_k1, c11);
          c20 = fmaddi(b00, a2_k1, c20);
          c21 = fmaddi(b01, a2_k1, c21);

          // k = 2
          b00 = load<T, lmul>(b_k2);
          b01 = load<T, lmul>(b_k2 + VL);

          c00 = fmaddi(b00, a0_k2, c00);
          c01 = fmaddi(b01, a0_k2, c01);
          c10 = fmaddi(b00, a1_k2, c10);
          c11 = fmaddi(b01, a1_k2, c11);
          c20 = fmaddi(b00, a2_k2, c20);
          c21 = fmaddi(b01, a2_k2, c21);

          // k = 3
          b00 = load<T, lmul>(b_k3);
          b01 = load<T, lmul>(b_k3 + VL);
          b_k += b_ld4;

          c00 = fmaddi(b00, a0_k3, c00);
          c01 = fmaddi(b01, a0_k3, c01);
          c10 = fmaddi(b00, a1_k3, c10);
          c11 = fmaddi(b01, a1_k3, c11);
          c20 = fmaddi(b00, a2_k3, c20);
          c21 = fmaddi(b01, a2_k3, c21);
        }

        // K remainder: handle K % 4 remaining k-steps
        for (size_t k = K4 * 4; k < A.cols; ++k) {
          const auto b0 = load<T, lmul>(b_k);
          const auto b1 = load<T, lmul>(b_k + VL);
          b_k += b_ld;
          c00 = fmaddi(b0, a_k[0], c00);
          c01 = fmaddi(b1, a_k[0], c01);
          c10 = fmaddi(b0, a_k[1], c10);
          c11 = fmaddi(b1, a_k[1], c11);
          c20 = fmaddi(b0, a_k[2], c20);
          c21 = fmaddi(b1, a_k[2], c21);
          a_k += a_ld;
        }

        store(C[i + 0] + j, c00);
        store(C[i + 0] + j + VL, c01);
        store(C[i + 1] + j, c10);
        store(C[i + 1] + j + VL, c11);
        store(C[i + 2] + j, c20);
        store(C[i + 2] + j + VL, c21);
      }
    }

    for (size_t i = Mfull; i < A.rows; ++i) {
      for (size_t j = 0; j < B.cols; j += NR) {
        auto c0 = set0<T, lmul>();
        auto c1 = set0<T, lmul>();

        const T *__restrict a_k = A.data + i;

        for (size_t kk = 0; kk < K4; ++kk) {
          c0 = fmaddi(load<T, lmul>(B[4 * kk + 0] + j), a_k[0], c0);
          c1 = fmaddi(load<T, lmul>(B[4 * kk + 0] + j + VL), a_k[0], c1);
          a_k += a_ld;

          c0 = fmaddi(load<T, lmul>(B[4 * kk + 1] + j), a_k[0], c0);
          c1 = fmaddi(load<T, lmul>(B[4 * kk + 1] + j + VL), a_k[0], c1);
          a_k += a_ld;

          c0 = fmaddi(load<T, lmul>(B[4 * kk + 2] + j), a_k[0], c0);
          c1 = fmaddi(load<T, lmul>(B[4 * kk + 2] + j + VL), a_k[0], c1);
          a_k += a_ld;

          c0 = fmaddi(load<T, lmul>(B[4 * kk + 3] + j), a_k[0], c0);
          c1 = fmaddi(load<T, lmul>(B[4 * kk + 3] + j + VL), a_k[0], c1);
          a_k += a_ld;
        }

        // K remainder
        for (size_t k = K4 * 4; k < A.cols; ++k) {
          c0 = fmaddi(load<T, lmul>(B[k] + j), a_k[0], c0);
          c1 = fmaddi(load<T, lmul>(B[k] + j + VL), a_k[0], c1);
          a_k += a_ld;
        }

        store(C[i] + j, c0);
        store(C[i] + j + VL, c1);
      }
    }
  }

  // General 5-argument version (arbitrary alpha, beta) for CRR
  template <int lmul = 2>
  static inline void gemm_mippv2_x100_register_blocked_apack4_crr(
      const PackedColMajor<T> &__restrict A,
      const PackedRowMajor<T> &__restrict B, PackedRowMajor<T> &__restrict C,
      T alpha, T beta) {
    if (__builtin_expect(alpha == T{1} && beta == T{0}, 1)) {
      gemm_mippv2_x100_register_blocked_apack4_crr<lmul>(A, B, C);
      return;
    }

    using namespace mipp;

    constexpr size_t VL = N<T, lmul>();
    constexpr size_t MR = 3;
    constexpr size_t NR = 2 * VL;

    const size_t a_ld = A.ld;
    const size_t b_ld = B.ld;
    const size_t a_ld4 = 4 * a_ld;
    const size_t b_ld4 = 4 * b_ld;
    const size_t Mfull = (A.rows / MR) * MR;
    const size_t K4 = A.cols / 4;

    for (size_t i = 0; i < Mfull; i += MR) {
      const T *a_base = A.data + i;

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

        const T *__restrict a_k = a_base;
        const T *__restrict b_k = B.data + j;

        for (size_t kk = 0; kk < K4; ++kk) {
          const T a0_k0 = a_k[0], a1_k0 = a_k[1], a2_k0 = a_k[2];
          const T *a_k1 = a_k + a_ld;
          const T a0_k1 = a_k1[0], a1_k1 = a_k1[1], a2_k1 = a_k1[2];
          const T *a_k2 = a_k1 + a_ld;
          const T a0_k2 = a_k2[0], a1_k2 = a_k2[1], a2_k2 = a_k2[2];
          const T *a_k3 = a_k2 + a_ld;
          const T a0_k3 = a_k3[0], a1_k3 = a_k3[1], a2_k3 = a_k3[2];
          a_k += a_ld4;

          const T *b_k1 = b_k + b_ld;
          const T *b_k2 = b_k1 + b_ld;
          const T *b_k3 = b_k2 + b_ld;

          // k = 0
          auto b00 = load<T, lmul>(b_k);
          auto b01 = load<T, lmul>(b_k + VL);

          c00 = fmaddi(b00, a0_k0, c00);
          c01 = fmaddi(b01, a0_k0, c01);
          c10 = fmaddi(b00, a1_k0, c10);
          c11 = fmaddi(b01, a1_k0, c11);
          c20 = fmaddi(b00, a2_k0, c20);
          c21 = fmaddi(b01, a2_k0, c21);

          // k = 1
          b00 = load<T, lmul>(b_k1);
          b01 = load<T, lmul>(b_k1 + VL);

          c00 = fmaddi(b00, a0_k1, c00);
          c01 = fmaddi(b01, a0_k1, c01);
          c10 = fmaddi(b00, a1_k1, c10);
          c11 = fmaddi(b01, a1_k1, c11);
          c20 = fmaddi(b00, a2_k1, c20);
          c21 = fmaddi(b01, a2_k1, c21);

          // k = 2
          b00 = load<T, lmul>(b_k2);
          b01 = load<T, lmul>(b_k2 + VL);

          c00 = fmaddi(b00, a0_k2, c00);
          c01 = fmaddi(b01, a0_k2, c01);
          c10 = fmaddi(b00, a1_k2, c10);
          c11 = fmaddi(b01, a1_k2, c11);
          c20 = fmaddi(b00, a2_k2, c20);
          c21 = fmaddi(b01, a2_k2, c21);

          // k = 3
          b00 = load<T, lmul>(b_k3);
          b01 = load<T, lmul>(b_k3 + VL);
          b_k += b_ld4;

          c00 = fmaddi(b00, a0_k3, c00);
          c01 = fmaddi(b01, a0_k3, c01);
          c10 = fmaddi(b00, a1_k3, c10);
          c11 = fmaddi(b01, a1_k3, c11);
          c20 = fmaddi(b00, a2_k3, c20);
          c21 = fmaddi(b01, a2_k3, c21);
        }

        // K remainder
        for (size_t k = K4 * 4; k < A.cols; ++k) {
          const auto b0 = load<T, lmul>(b_k);
          const auto b1 = load<T, lmul>(b_k + VL);
          b_k += b_ld;
          c00 = fmaddi(b0, a_k[0], c00);
          c01 = fmaddi(b1, a_k[0], c01);
          c10 = fmaddi(b0, a_k[1], c10);
          c11 = fmaddi(b1, a_k[1], c11);
          c20 = fmaddi(b0, a_k[2], c20);
          c21 = fmaddi(b1, a_k[2], c21);
          a_k += a_ld;
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
      T *__restrict c_base = C[i];
      for (size_t j = 0; j < B.cols; j += NR) {
        auto c0 = set0<T, lmul>();
        auto c1 = set0<T, lmul>();

        const T *__restrict a_k = A.data + i;

        for (size_t kk = 0; kk < K4; ++kk) {
          c0 = fmaddi(load<T, lmul>(B[4 * kk + 0] + j), a_k[0], c0);
          c1 = fmaddi(load<T, lmul>(B[4 * kk + 0] + j + VL), a_k[0], c1);
          a_k += a_ld;

          c0 = fmaddi(load<T, lmul>(B[4 * kk + 1] + j), a_k[0], c0);
          c1 = fmaddi(load<T, lmul>(B[4 * kk + 1] + j + VL), a_k[0], c1);
          a_k += a_ld;

          c0 = fmaddi(load<T, lmul>(B[4 * kk + 2] + j), a_k[0], c0);
          c1 = fmaddi(load<T, lmul>(B[4 * kk + 2] + j + VL), a_k[0], c1);
          a_k += a_ld;

          c0 = fmaddi(load<T, lmul>(B[4 * kk + 3] + j), a_k[0], c0);
          c1 = fmaddi(load<T, lmul>(B[4 * kk + 3] + j + VL), a_k[0], c1);
          a_k += a_ld;
        }

        Epilogue::store<T, lmul, false>(c_base + j, c0, alpha, beta);
        Epilogue::store<T, lmul, false>(c_base + j + VL, c1, alpha, beta);
      }
    }
  }

  template <int lmul = 1, bool FastPath = false>
  static inline void
  gemm_mippv2_firestorm_mr6_nr4_fmaddi_crr_core(const PackedColMajor<T> &__restrict A,
                                                const PackedRowMajor<T> &__restrict B,
                                                PackedRowMajor<T> &__restrict C,
                                                T alpha = T{1}, T beta = T{0}) {
    using namespace mipp;

    constexpr size_t VL = N<T, lmul>();
    constexpr size_t MR = 6;
    constexpr size_t NR4 = 4 * VL;

    const size_t a_ld = A.ld;
    const size_t b_ld = B.ld;
    const size_t K = A.cols;
    const size_t a_step = a_ld - MR;

    // Fast-path: single micro-tile (M == MR and N == NR4)
    if (__builtin_expect(A.rows == MR && B.cols == NR4, 1)) {
      const T *__restrict a_k = A.data;
      const T *__restrict b_k = B.data;

      auto c00 = set0<T, lmul>(); auto c01 = set0<T, lmul>(); auto c02 = set0<T, lmul>(); auto c03 = set0<T, lmul>();
      auto c10 = set0<T, lmul>(); auto c11 = set0<T, lmul>(); auto c12 = set0<T, lmul>(); auto c13 = set0<T, lmul>();
      auto c20 = set0<T, lmul>(); auto c21 = set0<T, lmul>(); auto c22 = set0<T, lmul>(); auto c23 = set0<T, lmul>();
      auto c30 = set0<T, lmul>(); auto c31 = set0<T, lmul>(); auto c32 = set0<T, lmul>(); auto c33 = set0<T, lmul>();
      auto c40 = set0<T, lmul>(); auto c41 = set0<T, lmul>(); auto c42 = set0<T, lmul>(); auto c43 = set0<T, lmul>();
      auto c50 = set0<T, lmul>(); auto c51 = set0<T, lmul>(); auto c52 = set0<T, lmul>(); auto c53 = set0<T, lmul>();

      #pragma GCC unroll 1
      for (size_t k = 0; k < K; ++k) {
        const auto b0 = load<T, lmul>(b_k + 0 * VL);
        const auto b1 = load<T, lmul>(b_k + 1 * VL);
        const auto b2 = load<T, lmul>(b_k + 2 * VL);
        const auto b3 = load<T, lmul>(b_k + 3 * VL);
        b_k += b_ld;

        {
          T a0 = *a_k++;
#if defined(__ARM_NEON)
          asm ("" : "+w"(a0));
#endif
          c00 = fmaddi(b0, a0, c00);
          c01 = fmaddi(b1, a0, c01);
          c02 = fmaddi(b2, a0, c02);
          c03 = fmaddi(b3, a0, c03);
        }
        {
          T a1 = *a_k++;
#if defined(__ARM_NEON)
          asm ("" : "+w"(a1));
#endif
          c10 = fmaddi(b0, a1, c10);
          c11 = fmaddi(b1, a1, c11);
          c12 = fmaddi(b2, a1, c12);
          c13 = fmaddi(b3, a1, c13);
        }
        {
          T a2 = *a_k++;
#if defined(__ARM_NEON)
          asm ("" : "+w"(a2));
#endif
          c20 = fmaddi(b0, a2, c20);
          c21 = fmaddi(b1, a2, c21);
          c22 = fmaddi(b2, a2, c22);
          c23 = fmaddi(b3, a2, c23);
        }
        {
          T a3 = *a_k++;
#if defined(__ARM_NEON)
          asm ("" : "+w"(a3));
#endif
          c30 = fmaddi(b0, a3, c30);
          c31 = fmaddi(b1, a3, c31);
          c32 = fmaddi(b2, a3, c32);
          c33 = fmaddi(b3, a3, c33);
        }
        {
          T a4 = *a_k++;
#if defined(__ARM_NEON)
          asm ("" : "+w"(a4));
#endif
          c40 = fmaddi(b0, a4, c40);
          c41 = fmaddi(b1, a4, c41);
          c42 = fmaddi(b2, a4, c42);
          c43 = fmaddi(b3, a4, c43);
        }
        {
          T a5 = *a_k++;
#if defined(__ARM_NEON)
          asm ("" : "+w"(a5));
#endif
          c50 = fmaddi(b0, a5, c50);
          c51 = fmaddi(b1, a5, c51);
          c52 = fmaddi(b2, a5, c52);
          c53 = fmaddi(b3, a5, c53);
        }
        a_k += a_step;
      }

      T *c0_ptr = C[0];
      T *c1_ptr = C[1];
      T *c2_ptr = C[2];
      T *c3_ptr = C[3];
      T *c4_ptr = C[4];
      T *c5_ptr = C[5];

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
      return;
    }

    const size_t Mfull = (A.rows / MR) * MR;

    for (size_t i = 0; i < Mfull; i += MR) {
      const T *a_base = A.data + i;

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

        const T *__restrict a_k = a_base;
        const T *__restrict b_k = B.data + j;

        #pragma GCC unroll 1
        for (size_t k = 0; k < K; ++k) {
          const auto b0 = load<T, lmul>(b_k + 0 * VL);
          const auto b1 = load<T, lmul>(b_k + 1 * VL);
          const auto b2 = load<T, lmul>(b_k + 2 * VL);
          const auto b3 = load<T, lmul>(b_k + 3 * VL);
          b_k += b_ld;

          {
            T a0 = *a_k++;
#if defined(__ARM_NEON)
            asm ("" : "+w"(a0));
#endif
            c00 = fmaddi(b0, a0, c00);
            c01 = fmaddi(b1, a0, c01);
            c02 = fmaddi(b2, a0, c02);
            c03 = fmaddi(b3, a0, c03);
          }
          {
            T a1 = *a_k++;
#if defined(__ARM_NEON)
            asm ("" : "+w"(a1));
#endif
            c10 = fmaddi(b0, a1, c10);
            c11 = fmaddi(b1, a1, c11);
            c12 = fmaddi(b2, a1, c12);
            c13 = fmaddi(b3, a1, c13);
          }
          {
            T a2 = *a_k++;
#if defined(__ARM_NEON)
            asm ("" : "+w"(a2));
#endif
            c20 = fmaddi(b0, a2, c20);
            c21 = fmaddi(b1, a2, c21);
            c22 = fmaddi(b2, a2, c22);
            c23 = fmaddi(b3, a2, c23);
          }
          {
            T a3 = *a_k++;
#if defined(__ARM_NEON)
            asm ("" : "+w"(a3));
#endif
            c30 = fmaddi(b0, a3, c30);
            c31 = fmaddi(b1, a3, c31);
            c32 = fmaddi(b2, a3, c32);
            c33 = fmaddi(b3, a3, c33);
          }
          {
            T a4 = *a_k++;
#if defined(__ARM_NEON)
            asm ("" : "+w"(a4));
#endif
            c40 = fmaddi(b0, a4, c40);
            c41 = fmaddi(b1, a4, c41);
            c42 = fmaddi(b2, a4, c42);
            c43 = fmaddi(b3, a4, c43);
          }
          {
            T a5 = *a_k++;
#if defined(__ARM_NEON)
            asm ("" : "+w"(a5));
#endif
            c50 = fmaddi(b0, a5, c50);
            c51 = fmaddi(b1, a5, c51);
            c52 = fmaddi(b2, a5, c52);
            c53 = fmaddi(b3, a5, c53);
          }
          a_k += a_step;
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

      // Cleanup 6x2
      while (j + 2 * VL <= B.cols) {
        auto c00 = set0<T, lmul>(); auto c01 = set0<T, lmul>();
        auto c10 = set0<T, lmul>(); auto c11 = set0<T, lmul>();
        auto c20 = set0<T, lmul>(); auto c21 = set0<T, lmul>();
        auto c30 = set0<T, lmul>(); auto c31 = set0<T, lmul>();
        auto c40 = set0<T, lmul>(); auto c41 = set0<T, lmul>();
        auto c50 = set0<T, lmul>(); auto c51 = set0<T, lmul>();

        const T *__restrict a_k = a_base;
        const T *__restrict b_k = B.data + j;

        for (size_t k = 0; k < K; ++k) {
          const auto b0 = load<T, lmul>(b_k + 0 * VL);
          const auto b1 = load<T, lmul>(b_k + 1 * VL);
          b_k += b_ld;

          {
            const T a0 = a_k[0];
            c00 = fmaddi(b0, a0, c00);
            c01 = fmaddi(b1, a0, c01);
          }
          {
            const T a1 = a_k[1];
            c10 = fmaddi(b0, a1, c10);
            c11 = fmaddi(b1, a1, c11);
          }
          {
            const T a2 = a_k[2];
            c20 = fmaddi(b0, a2, c20);
            c21 = fmaddi(b1, a2, c21);
          }
          {
            const T a3 = a_k[3];
            c30 = fmaddi(b0, a3, c30);
            c31 = fmaddi(b1, a3, c31);
          }
          {
            const T a4 = a_k[4];
            c40 = fmaddi(b0, a4, c40);
            c41 = fmaddi(b1, a4, c41);
          }
          {
            const T a5 = a_k[5];
            c50 = fmaddi(b0, a5, c50);
            c51 = fmaddi(b1, a5, c51);
          }
          a_k += a_ld;
        }

        const auto c0_ptr = c0_base + j; const auto c1_ptr = c1_base + j;
        const auto c2_ptr = c2_base + j; const auto c3_ptr = c3_base + j;
        const auto c4_ptr = c4_base + j; const auto c5_ptr = c5_base + j;

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

      // Cleanup 6x1
      while (j + VL <= B.cols) {
        auto c0 = set0<T, lmul>(); auto c1 = set0<T, lmul>();
        auto c2 = set0<T, lmul>(); auto c3 = set0<T, lmul>();
        auto c4 = set0<T, lmul>(); auto c5 = set0<T, lmul>();

        const T *__restrict a_k = a_base;
        const T *__restrict b_k = B.data + j;

        for (size_t k = 0; k < K; ++k) {
          const auto b0 = load<T, lmul>(b_k);
          b_k += b_ld;

          {
            const T a0 = a_k[0];
            c0 = fmaddi(b0, a0, c0);
          }
          {
            const T a1 = a_k[1];
            c1 = fmaddi(b0, a1, c1);
          }
          {
            const T a2 = a_k[2];
            c2 = fmaddi(b0, a2, c2);
          }
          {
            const T a3 = a_k[3];
            c3 = fmaddi(b0, a3, c3);
          }
          {
            const T a4 = a_k[4];
            c4 = fmaddi(b0, a4, c4);
          }
          {
            const T a5 = a_k[5];
            c5 = fmaddi(b0, a5, c5);
          }
          a_k += a_ld;
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

    // Tail rows
    for (size_t i = Mfull; i < A.rows; ++i) {
      size_t j = 0;
      while (j + NR4 <= B.cols) {
        auto c0 = set0<T, lmul>(); auto c1 = set0<T, lmul>();
        auto c2 = set0<T, lmul>(); auto c3 = set0<T, lmul>();

        const T *__restrict a_k = A.data + i;
        const T *__restrict b_k = B.data + j;

        for (size_t k = 0; k < K; ++k) {
          const auto b0 = load<T, lmul>(b_k + 0 * VL);
          const auto b1 = load<T, lmul>(b_k + 1 * VL);
          const auto b2 = load<T, lmul>(b_k + 2 * VL);
          const auto b3 = load<T, lmul>(b_k + 3 * VL);
          b_k += b_ld;

          const T a = *a_k;
          a_k += a_ld;

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
        const T *__restrict a_k = A.data + i;
        const T *__restrict b_k = B.data + j;

        for (size_t k = 0; k < K; ++k) {
          const auto b0 = load<T, lmul>(b_k);
          b_k += b_ld;
          const T a = *a_k;
          a_k += a_ld;
          c0 = fmaddi(b0, a, c0);
        }

        Epilogue::store<T, lmul, FastPath>(C[i] + j, c0, alpha, beta);
        j += VL;
      }
    }
  }

  template <int lmul = 1>
  static inline void
  gemm_mippv2_firestorm_mr6_nr4_fmaddi_crr(const PackedColMajor<T> &__restrict A,
                                           const PackedRowMajor<T> &__restrict B,
                                           PackedRowMajor<T> &__restrict C) {
    gemm_mippv2_firestorm_mr6_nr4_fmaddi_crr_core<lmul, true>(A, B, C);
  }

  template <int lmul = 1>
  static inline void
  gemm_mippv2_firestorm_mr6_nr4_fmaddi_crr(const PackedColMajor<T> &__restrict A,
                                           const PackedRowMajor<T> &__restrict B,
                                           PackedRowMajor<T> &__restrict C,
                                           T alpha, T beta) {
    if (__builtin_expect(alpha == T{1} && beta == T{0}, 1)) {
      gemm_mippv2_firestorm_mr6_nr4_fmaddi_crr_core<lmul, true>(A, B, C);
    } else {
      gemm_mippv2_firestorm_mr6_nr4_fmaddi_crr_core<lmul, false>(A, B, C, alpha, beta);
    }
  }

  template <int lmul = 1>
  static inline void
  gemm_mippv2_x60_mr6_nr4_fmaddi_crr(const PackedColMajor<T> &__restrict A,
                                     const PackedRowMajor<T> &__restrict B,
                                     PackedRowMajor<T> &__restrict C) {
    gemm_mippv2_firestorm_mr6_nr4_fmaddi_crr_core<lmul, true>(A, B, C);
  }

  template <int lmul = 1>
  static inline void
  gemm_mippv2_x60_mr6_nr4_fmaddi_crr(const PackedColMajor<T> &__restrict A,
                                     const PackedRowMajor<T> &__restrict B,
                                     PackedRowMajor<T> &__restrict C,
                                     T alpha, T beta) {
    if (__builtin_expect(alpha == T{1} && beta == T{0}, 1)) {
      gemm_mippv2_firestorm_mr6_nr4_fmaddi_crr_core<lmul, true>(A, B, C);
    } else {
      gemm_mippv2_firestorm_mr6_nr4_fmaddi_crr_core<lmul, false>(A, B, C, alpha, beta);
    }
  }

  template <int lmul = 1>
  static inline void
  gemm_mippv2_a76_mr6_nr4_fmaddi_crr(const PackedColMajor<T> &__restrict A,
                                     const PackedRowMajor<T> &__restrict B,
                                     PackedRowMajor<T> &__restrict C) {
    gemm_mippv2_firestorm_mr6_nr4_fmaddi_crr_core<lmul, true>(A, B, C);
  }

  template <int lmul = 1>
  static inline void
  gemm_mippv2_a76_mr6_nr4_fmaddi_crr(const PackedColMajor<T> &__restrict A,
                                     const PackedRowMajor<T> &__restrict B,
                                     PackedRowMajor<T> &__restrict C,
                                     T alpha, T beta) {
    if (__builtin_expect(alpha == T{1} && beta == T{0}, 1)) {
      gemm_mippv2_firestorm_mr6_nr4_fmaddi_crr_core<lmul, true>(A, B, C);
    } else {
      gemm_mippv2_firestorm_mr6_nr4_fmaddi_crr_core<lmul, false>(A, B, C, alpha, beta);
    }
  }

  // Alias for backward-compatibility
  template <int lmul = 1>
  static inline void
  gemm_mippv2_a76_mr6_nr3_crr(const PackedColMajor<T> &__restrict A,
                              const PackedRowMajor<T> &__restrict B,
                              PackedRowMajor<T> &__restrict C) {
    gemm_mippv2_a76_mr6_nr4_fmaddi_crr<lmul>(A, B, C);
  }

  template <int lmul = 1>
  static inline void
  gemm_mippv2_a76_mr6_nr3_crr(const PackedColMajor<T> &__restrict A,
                              const PackedRowMajor<T> &__restrict B,
                              PackedRowMajor<T> &__restrict C,
                              T alpha, T beta) {
    gemm_mippv2_a76_mr6_nr4_fmaddi_crr<lmul>(A, B, C, alpha, beta);
  }


  template <int lmul = 1, bool FastPath = false>
  static inline void
  gemm_mippv2_a100_mr7_nr4_pipe_crr_core(const PackedColMajor<T> &__restrict A,
                                         const PackedRowMajor<T> &__restrict B,
                                         PackedRowMajor<T> &__restrict C,
                                         T alpha = T{1}, T beta = T{0}) {
    using namespace mipp;

    constexpr size_t VL = N<T, lmul>();
    constexpr size_t MR = 7;
    constexpr size_t NR4 = 4 * VL;

    const size_t a_ld = A.ld;
    const size_t b_ld = B.ld;
    const size_t Mfull = (A.rows / MR) * MR;
    const size_t K = A.cols;

    for (size_t i = 0; i < Mfull; i += MR) {
      const T *a_base = A.data + i;

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

        const T *__restrict a_k = a_base;
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
            const T a0 = a_k[0];
            const T a1 = a_k[1];
            const T a2 = a_k[2];
            const T a3 = a_k[3];
            const T a4 = a_k[4];
            const T a5 = a_k[5];
            const T a6 = a_k[6];
            a_k += a_ld;

            c00 = fmaddi(b0, a0, c00);
            c10 = fmaddi(b0, a1, c10);
            c20 = fmaddi(b0, a2, c20);
            c30 = fmaddi(b0, a3, c30);
            c40 = fmaddi(b0, a4, c40);
            c50 = fmaddi(b0, a5, c50);
            c60 = fmaddi(b0, a6, c60);

            // b0 is now dead for step k -> reload for step k+1
            b0 = load<T, lmul>(b_k + 0 * VL);

            c01 = fmaddi(b1, a0, c01);
            c11 = fmaddi(b1, a1, c11);
            c21 = fmaddi(b1, a2, c21);
            c31 = fmaddi(b1, a3, c31);
            c41 = fmaddi(b1, a4, c41);
            c51 = fmaddi(b1, a5, c51);
            c61 = fmaddi(b1, a6, c61);

            // b1 is now dead for step k -> reload for step k+1
            b1 = load<T, lmul>(b_k + 1 * VL);

            c02 = fmaddi(b2, a0, c02);
            c12 = fmaddi(b2, a1, c12);
            c22 = fmaddi(b2, a2, c22);
            c32 = fmaddi(b2, a3, c32);
            c42 = fmaddi(b2, a4, c42);
            c52 = fmaddi(b2, a5, c52);
            c62 = fmaddi(b2, a6, c62);

            // b2 is now dead for step k -> reload for step k+1
            b2 = load<T, lmul>(b_k + 2 * VL);

            c03 = fmaddi(b3, a0, c03);
            c13 = fmaddi(b3, a1, c13);
            c23 = fmaddi(b3, a2, c23);
            c33 = fmaddi(b3, a3, c33);
            c43 = fmaddi(b3, a4, c43);
            c53 = fmaddi(b3, a5, c53);
            c63 = fmaddi(b3, a6, c63);

            // b3 is now dead for step k -> reload for step k+1
            b3 = load<T, lmul>(b_k + 3 * VL);
            b_k += b_ld;
          }

          const T a0 = a_k[0];
          const T a1 = a_k[1];
          const T a2 = a_k[2];
          const T a3 = a_k[3];
          const T a4 = a_k[4];
          const T a5 = a_k[5];
          const T a6 = a_k[6];

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

      // Cleanup 7x2
      while (j + 2 * VL <= B.cols) {
        auto c00 = set0<T, lmul>(); auto c01 = set0<T, lmul>();
        auto c10 = set0<T, lmul>(); auto c11 = set0<T, lmul>();
        auto c20 = set0<T, lmul>(); auto c21 = set0<T, lmul>();
        auto c30 = set0<T, lmul>(); auto c31 = set0<T, lmul>();
        auto c40 = set0<T, lmul>(); auto c41 = set0<T, lmul>();
        auto c50 = set0<T, lmul>(); auto c51 = set0<T, lmul>();
        auto c60 = set0<T, lmul>(); auto c61 = set0<T, lmul>();

        const T *__restrict a_k = a_base;
        const T *__restrict b_k = B.data + j;

        for (size_t k = 0; k < K; ++k) {
          const auto b0 = load<T, lmul>(b_k + 0 * VL);
          const auto b1 = load<T, lmul>(b_k + 1 * VL);
          b_k += b_ld;

          const T a0 = a_k[0]; const T a1 = a_k[1];
          const T a2 = a_k[2]; const T a3 = a_k[3];
          const T a4 = a_k[4]; const T a5 = a_k[5];
          const T a6 = a_k[6];
          a_k += a_ld;

          c00 = fmaddi(b0, a0, c00); c01 = fmaddi(b1, a0, c01);
          c10 = fmaddi(b0, a1, c10); c11 = fmaddi(b1, a1, c11);
          c20 = fmaddi(b0, a2, c20); c21 = fmaddi(b1, a2, c21);
          c30 = fmaddi(b0, a3, c30); c31 = fmaddi(b1, a3, c31);
          c40 = fmaddi(b0, a4, c40); c41 = fmaddi(b1, a4, c41);
          c50 = fmaddi(b0, a5, c50); c51 = fmaddi(b1, a5, c51);
          c60 = fmaddi(b0, a6, c60); c61 = fmaddi(b1, a6, c61);
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

        j += 2 * VL;
      }

      // Cleanup 7x1
      while (j + VL <= B.cols) {
        auto c0 = set0<T, lmul>(); auto c1 = set0<T, lmul>();
        auto c2 = set0<T, lmul>(); auto c3 = set0<T, lmul>();
        auto c4 = set0<T, lmul>(); auto c5 = set0<T, lmul>();
        auto c6 = set0<T, lmul>();

        const T *__restrict a_k = a_base;
        const T *__restrict b_k = B.data + j;

        for (size_t k = 0; k < K; ++k) {
          const auto b0 = load<T, lmul>(b_k);
          b_k += b_ld;

          const T a0 = a_k[0]; const T a1 = a_k[1];
          const T a2 = a_k[2]; const T a3 = a_k[3];
          const T a4 = a_k[4]; const T a5 = a_k[5];
          const T a6 = a_k[6];
          a_k += a_ld;

          c0 = fmaddi(b0, a0, c0);
          c1 = fmaddi(b0, a1, c1);
          c2 = fmaddi(b0, a2, c2);
          c3 = fmaddi(b0, a3, c3);
          c4 = fmaddi(b0, a4, c4);
          c5 = fmaddi(b0, a5, c5);
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
    }

    // Tail rows
    for (size_t i = Mfull; i < A.rows; ++i) {
      size_t j = 0;
      while (j + NR4 <= B.cols) {
        auto c0 = set0<T, lmul>(); auto c1 = set0<T, lmul>();
        auto c2 = set0<T, lmul>(); auto c3 = set0<T, lmul>();

        const T *__restrict a_k = A.data + i;
        const T *__restrict b_k = B.data + j;

        for (size_t k = 0; k < K; ++k) {
          const auto b0 = load<T, lmul>(b_k + 0 * VL);
          const auto b1 = load<T, lmul>(b_k + 1 * VL);
          const auto b2 = load<T, lmul>(b_k + 2 * VL);
          const auto b3 = load<T, lmul>(b_k + 3 * VL);
          b_k += b_ld;

          const T a = *a_k;
          a_k += a_ld;

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
        const T *__restrict a_k = A.data + i;
        const T *__restrict b_k = B.data + j;

        for (size_t k = 0; k < K; ++k) {
          const auto b0 = load<T, lmul>(b_k);
          b_k += b_ld;
          const T a = *a_k;
          a_k += a_ld;
          c0 = fmaddi(b0, a, c0);
        }

        Epilogue::store<T, lmul, FastPath>(C[i] + j, c0, alpha, beta);
        j += VL;
      }
    }
  }

  template <int lmul = 1>
  static inline void
  gemm_mippv2_a100_mr7_nr4_pipe_crr(const PackedColMajor<T> &__restrict A,
                                    const PackedRowMajor<T> &__restrict B,
                                    PackedRowMajor<T> &__restrict C) {
    gemm_mippv2_a100_mr7_nr4_pipe_crr_core<lmul, true>(A, B, C);
  }

  template <int lmul = 1>
  static inline void
  gemm_mippv2_a100_mr7_nr4_pipe_crr(const PackedColMajor<T> &__restrict A,
                                    const PackedRowMajor<T> &__restrict B,
                                    PackedRowMajor<T> &__restrict C,
                                    T alpha, T beta) {
    if (__builtin_expect(alpha == T{1} && beta == T{0}, 1)) {
      gemm_mippv2_a100_mr7_nr4_pipe_crr_core<lmul, true>(A, B, C);
    } else {
      gemm_mippv2_a100_mr7_nr4_pipe_crr_core<lmul, false>(A, B, C, alpha, beta);
    }
  }

  template <int lmul = 2, bool FastPath = false>
  static inline void
  gemm_mippv2_a100_mr7_nr2_lmul2_pipe_crr_core(const PackedColMajor<T> &__restrict A,
                                                const PackedRowMajor<T> &__restrict B,
                                                PackedRowMajor<T> &__restrict C,
                                                T alpha = T{1}, T beta = T{0}) {
    using namespace mipp;

    constexpr size_t VL = N<T, lmul>();
    constexpr size_t MR = 7;
    constexpr size_t NR2 = 2 * VL;

    const size_t a_ld = A.ld;
    const size_t b_ld = B.ld;
    const size_t Mfull = (A.rows / MR) * MR;
    const size_t K = A.cols;

    for (size_t i = 0; i < Mfull; i += MR) {
      const T *a_base = A.data + i;

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

        const T *__restrict a_k = a_base;
        const T *__restrict b_k = B.data + j;

        if (__builtin_expect(K > 0, 1)) {
          auto b0 = load<T, lmul>(b_k + 0 * VL);
          auto b1 = load<T, lmul>(b_k + 1 * VL);
          b_k += b_ld;

          size_t k = 0;
          #pragma GCC unroll 1
          for (; k + 1 < K; ++k) {
            const T a0 = a_k[0]; const T a1 = a_k[1];
            const T a2 = a_k[2]; const T a3 = a_k[3];
            const T a4 = a_k[4]; const T a5 = a_k[5];
            const T a6 = a_k[6];
            a_k += a_ld;

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

          const T a0 = a_k[0]; const T a1 = a_k[1];
          const T a2 = a_k[2]; const T a3 = a_k[3];
          const T a4 = a_k[4]; const T a5 = a_k[5];
          const T a6 = a_k[6];

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

        const T *__restrict a_k = a_base;
        const T *__restrict b_k = B.data + j;

        for (size_t k = 0; k < K; ++k) {
          const auto b0 = load<T, lmul>(b_k);
          b_k += b_ld;
          const T a0 = a_k[0]; const T a1 = a_k[1];
          const T a2 = a_k[2]; const T a3 = a_k[3];
          const T a4 = a_k[4]; const T a5 = a_k[5];
          const T a6 = a_k[6];
          a_k += a_ld;

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
          c0 += a_base[0 + k * a_ld] * b_val; c1 += a_base[1 + k * a_ld] * b_val;
          c2 += a_base[2 + k * a_ld] * b_val; c3 += a_base[3 + k * a_ld] * b_val;
          c4 += a_base[4 + k * a_ld] * b_val; c5 += a_base[5 + k * a_ld] * b_val;
          c6 += a_base[6 + k * a_ld] * b_val;
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
      T *c0_base = C[i + 0];
      size_t j = 0;
      while (j + VL <= B.cols) {
        auto c0 = set0<T, lmul>();
        const T *__restrict a_k = A.data + i;
        const T *__restrict b_k = B.data + j;
        for (size_t k = 0; k < K; ++k) {
          const auto b0 = load<T, lmul>(b_k);
          b_k += b_ld;
          c0 = fmaddi(b0, *a_k, c0);
          a_k += a_ld;
        }
        Epilogue::store<T, lmul, FastPath>(c0_base + j, c0, alpha, beta);
        j += VL;
      }
      for (; j < B.cols; ++j) {
        T c0 = T{0};
        for (size_t k = 0; k < K; ++k) c0 += A.data[i + k * a_ld] * B.data[k * b_ld + j];
        Epilogue::store(c0_base + j, c0, alpha, beta);
      }
      i += 1;
    }
  }

  template <int lmul = 2>
  static inline void
  gemm_mippv2_a100_mr7_nr2_lmul2_pipe_crr(const PackedColMajor<T> &__restrict A,
                                          const PackedRowMajor<T> &__restrict B,
                                          PackedRowMajor<T> &__restrict C) {
    gemm_mippv2_a100_mr7_nr2_lmul2_pipe_crr_core<lmul, true>(A, B, C);
  }

  template <int lmul = 2>
  static inline void
  gemm_mippv2_a100_mr7_nr2_lmul2_pipe_crr(const PackedColMajor<T> &__restrict A,
                                          const PackedRowMajor<T> &__restrict B,
                                          PackedRowMajor<T> &__restrict C,
                                          T alpha, T beta) {
    if (__builtin_expect(alpha == T{1} && beta == T{0}, 1)) {
      gemm_mippv2_a100_mr7_nr2_lmul2_pipe_crr_core<lmul, true>(A, B, C);
    } else {
      gemm_mippv2_a100_mr7_nr2_lmul2_pipe_crr_core<lmul, false>(A, B, C, alpha, beta);
    }
  }

