#pragma once

#include "StorageType.h"
#include "Alloc.h"

#ifdef GEMMBENCH_HAS_BLIS

#include "blis.h"

#if defined(GEMMBENCH_BLIS_HASWELL)
extern "C" void bli_dgemm_haswell_asm_6x8(
    dim_t m, dim_t n, dim_t k,
    const void* alpha, const void* a, const void* b, const void* beta,
    void* c, inc_t rs_c, inc_t cs_c,
    const auxinfo_t* data, const cntx_t* cntx);
#define BLIS_DGEMM_NATIVE_UKR bli_dgemm_haswell_asm_6x8

#elif defined(GEMMBENCH_BLIS_SKX)
extern "C" void bli_dgemm_skx_asm_16x14(
    dim_t m, dim_t n, dim_t k,
    const void* alpha, const void* a, const void* b, const void* beta,
    void* c, inc_t rs_c, inc_t cs_c,
    const auxinfo_t* data, const cntx_t* cntx);
#define BLIS_DGEMM_NATIVE_UKR bli_dgemm_skx_asm_16x14

#elif defined(GEMMBENCH_BLIS_ARMV8A)
extern "C" void bli_dgemm_armv8a_asm_6x8(
    dim_t m, dim_t n, dim_t k,
    const void* alpha, const void* a, const void* b, const void* beta,
    void* c, inc_t rs_c, inc_t cs_c,
    const auxinfo_t* data, const cntx_t* cntx);
#define BLIS_DGEMM_NATIVE_UKR bli_dgemm_armv8a_asm_6x8

#elif defined(GEMMBENCH_BLIS_X60)
extern "C" void bli_dgemm_x60_2vx14_2u(
    dim_t m, dim_t n, dim_t k,
    const void* alpha, const void* a, const void* b, const void* beta,
    void* c, inc_t rs_c, inc_t cs_c,
    const auxinfo_t* data, const cntx_t* cntx);
#define BLIS_DGEMM_NATIVE_UKR bli_dgemm_x60_2vx14_2u

#endif

#endif // GEMMBENCH_HAS_BLIS
