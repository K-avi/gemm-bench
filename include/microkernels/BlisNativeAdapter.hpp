#pragma once

/**
 * BLIS Native Microkernel C++ Adapter
 *
 * Provides typed declarations and dispatching to standalone reference assembly
 * microkernels sourced from:
 *   - RISC-V Vector 1.0 (SpacemiT X60):
 *       Stepan Nassyr (https://github.com/stepannassyr/blis/tree/x60, commit 6ffc1d3)
 *       - bli_dgemm_x60_2vx14_2u (FP64 8x14)
 *       - bli_sgemm_x60_2vx14_2u (FP32 16x14)
 *   - x86-64 (AVX-512, AVX2) and AArch64 (NEON):
 *       Upstream FLAME BLIS (https://github.com/flame/blis, commit a8037cf)
 *       - bli_dgemm_skx_asm_16x14, bli_sgemm_skx_asm_32x12_l2
 *       - bli_dgemm_haswell_asm_6x8, bli_sgemm_haswell_asm_6x16
 *       - bli_dgemm_armv8a_asm_6x8, bli_sgemm_armv8a_asm_8x12
 */

#include "StorageType.h"
#include "Alloc.h"
#include <cstdint>

#ifdef GEMMBENCH_HAS_BLIS

extern "C" {
typedef int64_t dim_t;
typedef int64_t inc_t;

typedef struct auxinfo_s {
    const void* next_a;
    const void* next_b;
    uint64_t is;
} auxinfo_t;

typedef struct cntx_s {
    void* dummy;
} cntx_t;
}

#if defined(GEMMBENCH_BLIS_HASWELL)
extern "C" void bli_dgemm_haswell_asm_6x8(
    dim_t m, dim_t n, dim_t k,
    const void* alpha, const void* a, const void* b, const void* beta,
    void* c, inc_t rs_c, inc_t cs_c,
    const auxinfo_t* data, const cntx_t* cntx);
extern "C" void bli_sgemm_haswell_asm_6x16(
    dim_t m, dim_t n, dim_t k,
    const void* alpha, const void* a, const void* b, const void* beta,
    void* c, inc_t rs_c, inc_t cs_c,
    const auxinfo_t* data, const cntx_t* cntx);
#define BLIS_DGEMM_NATIVE_UKR bli_dgemm_haswell_asm_6x8
#define BLIS_SGEMM_NATIVE_UKR bli_sgemm_haswell_asm_6x16

#elif defined(GEMMBENCH_BLIS_SKX)
extern "C" void bli_dgemm_skx_asm_16x14(
    dim_t m, dim_t n, dim_t k,
    const void* alpha, const void* a, const void* b, const void* beta,
    void* c, inc_t rs_c, inc_t cs_c,
    const auxinfo_t* data, const cntx_t* cntx);
extern "C" void bli_sgemm_skx_asm_32x12_l2(
    dim_t m, dim_t n, dim_t k,
    const void* alpha, const void* a, const void* b, const void* beta,
    void* c, inc_t rs_c, inc_t cs_c,
    const auxinfo_t* data, const cntx_t* cntx);
#define BLIS_DGEMM_NATIVE_UKR bli_dgemm_skx_asm_16x14
#define BLIS_SGEMM_NATIVE_UKR bli_sgemm_skx_asm_32x12_l2

#elif defined(GEMMBENCH_BLIS_ARMV8A)
extern "C" void bli_dgemm_armv8a_asm_6x8(
    dim_t m, dim_t n, dim_t k,
    const void* alpha, const void* a, const void* b, const void* beta,
    void* c, inc_t rs_c, inc_t cs_c,
    const auxinfo_t* data, const cntx_t* cntx);
extern "C" void bli_sgemm_armv8a_asm_8x12(
    dim_t m, dim_t n, dim_t k,
    const void* alpha, const void* a, const void* b, const void* beta,
    void* c, inc_t rs_c, inc_t cs_c,
    const auxinfo_t* data, const cntx_t* cntx);
#define BLIS_DGEMM_NATIVE_UKR bli_dgemm_armv8a_asm_6x8
#define BLIS_SGEMM_NATIVE_UKR bli_sgemm_armv8a_asm_8x12

#elif defined(GEMMBENCH_BLIS_X60)
extern "C" void bli_dgemm_x60_2vx14_2u(
    dim_t m, dim_t n, dim_t k,
    const void* alpha, const void* a, const void* b, const void* beta,
    void* c, inc_t rs_c, inc_t cs_c,
    const auxinfo_t* data, const cntx_t* cntx);
extern "C" void bli_sgemm_x60_2vx14_2u(
    dim_t m, dim_t n, dim_t k,
    const void* alpha, const void* a, const void* b, const void* beta,
    void* c, inc_t rs_c, inc_t cs_c,
    const auxinfo_t* data, const cntx_t* cntx);
#define BLIS_DGEMM_NATIVE_UKR bli_dgemm_x60_2vx14_2u
#define BLIS_SGEMM_NATIVE_UKR bli_sgemm_x60_2vx14_2u

#endif

#endif // GEMMBENCH_HAS_BLIS
