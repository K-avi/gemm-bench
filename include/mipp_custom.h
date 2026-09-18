#pragma once

#include "mipp.hpp"

/*
mipp::fmaddi is a function that was added a proof of concept during
the gemm micro-kernel development. It was necessary to get good
performance on the SpacemiT K3 X100 core. It leverages vfmadd.v.f
instead of vfmadd.v.v

This makes it so the value used from A matrix during the
k loop of the ukernels is never explicitly broadcasted (vfmv.v.f instruction).

Which in term allows for better performance.
*/

namespace mipp {

#if defined(MIPP_RVV) && defined(__riscv_vector)

// =============================================================================
// Native RVV 1.0 Intrinsics (vfmacc.vf)
// Evaluates: c = (a * b) + c using single vector-scalar instruction
// =============================================================================

#if defined(__riscv_v_elen_fp) && __riscv_v_elen_fp >= 64
static inline rvd_rvv_float64_m1_t fmaddi(const rvd_rvv_float64_m1_t a,
                                          const float64_t b,
                                          const rvd_rvv_float64_m1_t c) {
  rvd_rvv_float64_m1_t res;
  res.r = __riscv_vfmacc_vf_f64m1(c.r, b, a.r, MIPP_RVV_N_FLOAT64_M1);
  return res;
}

static inline rvd_rvv_float64_m2_t fmaddi(const rvd_rvv_float64_m2_t a,
                                          const float64_t b,
                                          const rvd_rvv_float64_m2_t c) {
  rvd_rvv_float64_m2_t res;
  res.r = __riscv_vfmacc_vf_f64m2(c.r, b, a.r, MIPP_RVV_N_FLOAT64_M2);
  return res;
}

static inline rvd_rvv_float64_m4_t fmaddi(const rvd_rvv_float64_m4_t a,
                                          const float64_t b,
                                          const rvd_rvv_float64_m4_t c) {
  rvd_rvv_float64_m4_t res;
  res.r = __riscv_vfmacc_vf_f64m4(c.r, b, a.r, MIPP_RVV_N_FLOAT64_M4);
  return res;
}

static inline rvd_rvv_float64_m8_t fmaddi(const rvd_rvv_float64_m8_t a,
                                          const float64_t b,
                                          const rvd_rvv_float64_m8_t c) {
  rvd_rvv_float64_m8_t res;
  res.r = __riscv_vfmacc_vf_f64m8(c.r, b, a.r, MIPP_RVV_N_FLOAT64_M8);
  return res;
}
#endif // __riscv_v_elen_fp >= 64

static inline rvd_rvv_float32_m1_t fmaddi(const rvd_rvv_float32_m1_t a,
                                          const float32_t b,
                                          const rvd_rvv_float32_m1_t c) {
  rvd_rvv_float32_m1_t res;
  res.r = __riscv_vfmacc_vf_f32m1(c.r, b, a.r, MIPP_RVV_N_FLOAT32_M1);
  return res;
}

static inline rvd_rvv_float32_m2_t fmaddi(const rvd_rvv_float32_m2_t a,
                                          const float32_t b,
                                          const rvd_rvv_float32_m2_t c) {
  rvd_rvv_float32_m2_t res;
  res.r = __riscv_vfmacc_vf_f32m2(c.r, b, a.r, MIPP_RVV_N_FLOAT32_M2);
  return res;
}

static inline rvd_rvv_float32_m4_t fmaddi(const rvd_rvv_float32_m4_t a,
                                          const float32_t b,
                                          const rvd_rvv_float32_m4_t c) {
  rvd_rvv_float32_m4_t res;
  res.r = __riscv_vfmacc_vf_f32m4(c.r, b, a.r, MIPP_RVV_N_FLOAT32_M4);
  return res;
}

static inline rvd_rvv_float32_m8_t fmaddi(const rvd_rvv_float32_m8_t a,
                                          const float32_t b,
                                          const rvd_rvv_float32_m8_t c) {
  rvd_rvv_float32_m8_t res;
  res.r = __riscv_vfmacc_vf_f32m8(c.r, b, a.r, MIPP_RVV_N_FLOAT32_M8);
  return res;
}

#elif defined(MIPP_NEON) && defined(__ARM_NEON)

// =============================================================================
// Native ARM NEON Intrinsics (vfmaq_n_f64 / vfmaq_n_f32)
// Evaluates: c = (a * b) + c using single vector-scalar FMLA instruction (by lane)
// =============================================================================
#include <arm_neon.h>

#if defined(__aarch64__)
static inline rvd_neon_float64_m1_t fmaddi(const rvd_neon_float64_m1_t a,
                                           const float64_t b,
                                           const rvd_neon_float64_m1_t c) {
  rvd_neon_float64_m1_t res;
  res.r = vfmaq_n_f64(c.r, a.r, b);
  return res;
}

static inline rvd_neon_float64_m2_t fmaddi(const rvd_neon_float64_m2_t a,
                                           const float64_t b,
                                           const rvd_neon_float64_m2_t c) {
  rvd_neon_float64_m2_t res;
  res.r1 = fmaddi(a.r1, b, c.r1);
  res.r2 = fmaddi(a.r2, b, c.r2);
  return res;
}

static inline rvd_neon_float64_m4_t fmaddi(const rvd_neon_float64_m4_t a,
                                           const float64_t b,
                                           const rvd_neon_float64_m4_t c) {
  rvd_neon_float64_m4_t res;
  res.r1 = fmaddi(a.r1, b, c.r1);
  res.r2 = fmaddi(a.r2, b, c.r2);
  return res;
}

static inline rvd_neon_float64_m8_t fmaddi(const rvd_neon_float64_m8_t a,
                                           const float64_t b,
                                           const rvd_neon_float64_m8_t c) {
  rvd_neon_float64_m8_t res;
  res.r1 = fmaddi(a.r1, b, c.r1);
  res.r2 = fmaddi(a.r2, b, c.r2);
  return res;
}
#endif // __aarch64__

static inline rvd_neon_float32_m1_t fmaddi(const rvd_neon_float32_m1_t a,
                                           const float32_t b,
                                           const rvd_neon_float32_m1_t c) {
  rvd_neon_float32_m1_t res;
  res.r = vfmaq_n_f32(c.r, a.r, b);
  return res;
}

static inline rvd_neon_float32_m2_t fmaddi(const rvd_neon_float32_m2_t a,
                                           const float32_t b,
                                           const rvd_neon_float32_m2_t c) {
  rvd_neon_float32_m2_t res;
  res.r1 = fmaddi(a.r1, b, c.r1);
  res.r2 = fmaddi(a.r2, b, c.r2);
  return res;
}

static inline rvd_neon_float32_m4_t fmaddi(const rvd_neon_float32_m4_t a,
                                           const float32_t b,
                                           const rvd_neon_float32_m4_t c) {
  rvd_neon_float32_m4_t res;
  res.r1 = fmaddi(a.r1, b, c.r1);
  res.r2 = fmaddi(a.r2, b, c.r2);
  return res;
}

static inline rvd_neon_float32_m8_t fmaddi(const rvd_neon_float32_m8_t a,
                                           const float32_t b,
                                           const rvd_neon_float32_m8_t c) {
  rvd_neon_float32_m8_t res;
  res.r1 = fmaddi(a.r1, b, c.r1);
  res.r2 = fmaddi(a.r2, b, c.r2);
  return res;
}

#else

// =============================================================================
// Generic Overloads using MIPPv2 fmadd + set1
// Covers x86 (AVX, AVX2, AVX512, SSE), ARM Neon and scalar fallback
// =============================================================================

// float64_t overloads (LMUL 1, 2, 4, 8)
static inline rvd<float64_t, 1> fmaddi(const rvd<float64_t, 1> a,
                                       const float64_t b,
                                       const rvd<float64_t, 1> c) {
  return fmadd(a, set1<float64_t, 1>(b), c);
}
static inline rvd<float64_t, 2> fmaddi(const rvd<float64_t, 2> a,
                                       const float64_t b,
                                       const rvd<float64_t, 2> c) {
  return fmadd(a, set1<float64_t, 2>(b), c);
}
static inline rvd<float64_t, 4> fmaddi(const rvd<float64_t, 4> a,
                                       const float64_t b,
                                       const rvd<float64_t, 4> c) {
  return fmadd(a, set1<float64_t, 4>(b), c);
}
static inline rvd<float64_t, 8> fmaddi(const rvd<float64_t, 8> a,
                                       const float64_t b,
                                       const rvd<float64_t, 8> c) {
  return fmadd(a, set1<float64_t, 8>(b), c);
}

// float32_t overloads (LMUL 1, 2, 4, 8)
static inline rvd<float32_t, 1> fmaddi(const rvd<float32_t, 1> a,
                                       const float32_t b,
                                       const rvd<float32_t, 1> c) {
  return fmadd(a, set1<float32_t, 1>(b), c);
}
static inline rvd<float32_t, 2> fmaddi(const rvd<float32_t, 2> a,
                                       const float32_t b,
                                       const rvd<float32_t, 2> c) {
  return fmadd(a, set1<float32_t, 2>(b), c);
}
static inline rvd<float32_t, 4> fmaddi(const rvd<float32_t, 4> a,
                                       const float32_t b,
                                       const rvd<float32_t, 4> c) {
  return fmadd(a, set1<float32_t, 4>(b), c);
}
static inline rvd<float32_t, 8> fmaddi(const rvd<float32_t, 8> a,
                                       const float32_t b,
                                       const rvd<float32_t, 8> c) {
  return fmadd(a, set1<float32_t, 8>(b), c);
}

#endif

// Universal fallback template for unspecialized types or architectures
template <typename T, int LMUL = 1>
static inline rvd<T, LMUL> fmaddi(const rvd<T, LMUL> a, const T b,
                                  const rvd<T, LMUL> c) {
  return fmadd(a, set1<T, LMUL>(b), c);
}

} // namespace mipp
