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

  template <int lmul = 1>
  static inline void
  gemm_mippv2_skylake_register_blocked(const PackedRowMajor<T> &__restrict A,
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
  gemm_mippv2_skylake_lmul_register_blocked(const PackedRowMajor<T> &__restrict A,
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

  static inline void gemm_mippv2_skylake_panel(const PackedColMajor<T> &A,
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
  static inline void gemm_mippv2_skylake_panel_lmul(const PackedColMajor<T> &A,
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
