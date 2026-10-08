#pragma once

#ifdef GEMMBENCH_HAS_BLIS

// CRR Variant: A is PackedColMajor, B is PackedRowMajor, C is PackedRowMajor
inline void gemm_blis_native(const PackedColMajor<T> &A,
                             const PackedRowMajor<T> &B,
                             PackedRowMajor<T> &C,
                             T alpha = T{1}, T beta = T{0}) const {
    if constexpr (std::is_same_v<T, double>) {
        const dim_t m = static_cast<dim_t>(A.rows);
        const dim_t n = static_cast<dim_t>(B.cols);
        const dim_t k = static_cast<dim_t>(A.cols);
        const inc_t rs_c = static_cast<inc_t>(C.ld);
        const inc_t cs_c = 1;
        auxinfo_t aux{};
        cntx_t cntx{};
        BLIS_DGEMM_NATIVE_UKR(m, n, k,
                              &alpha, A.data, B.data, &beta,
                              C.data, rs_c, cs_c,
                              &aux, &cntx);
    } else {
        throw std::runtime_error("blis_native is only implemented for double precision");
    }
}

// CRC Variant: A is PackedColMajor, B is PackedRowMajor, C is PackedColMajor
inline void gemm_blis_native(const PackedColMajor<T> &A,
                             const PackedRowMajor<T> &B,
                             PackedColMajor<T> &C,
                             T alpha = T{1}, T beta = T{0}) const {
    if constexpr (std::is_same_v<T, double>) {
        const dim_t m = static_cast<dim_t>(A.rows);
        const dim_t n = static_cast<dim_t>(B.cols);
        const dim_t k = static_cast<dim_t>(A.cols);
        const inc_t rs_c = 1;
        const inc_t cs_c = static_cast<inc_t>(C.ld);
        auxinfo_t aux{};
        cntx_t cntx{};
        BLIS_DGEMM_NATIVE_UKR(m, n, k,
                              &alpha, A.data, B.data, &beta,
                              C.data, rs_c, cs_c,
                              &aux, &cntx);
    } else {
        throw std::runtime_error("blis_native is only implemented for double precision");
    }
}

#endif // GEMMBENCH_HAS_BLIS
