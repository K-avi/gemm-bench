#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Standard BLIS basic types
typedef int64_t dim_t;
typedef int64_t inc_t;

// Minimal auxinfo_t for prefetching
typedef struct auxinfo_s {
    const void* next_a;
    const void* next_b;
    uint64_t is;
} auxinfo_t;

// Dummy context structure
typedef struct cntx_s {
    void* dummy;
} cntx_t;

#define restrict __restrict__

#define bli_auxinfo_next_a(data) ((data) ? ((const auxinfo_t*)(data))->next_a : NULL)
#define bli_auxinfo_next_b(data) ((data) ? ((const auxinfo_t*)(data))->next_b : NULL)
#define bli_cntx_get_blksz_def_dt(dt, bs, cntx) ((dim_t)8)
#define bli_rvv_get_vlen() ((uint64_t)32)

// Complex-transpose helpers (no-op when native tile layout matches)
#define GEMM_UKR_SETUP_CT_ANY(...)
#define GEMM_UKR_SETUP_CT(...)
#define GEMM_UKR_FLUSH_CT(...)

#ifdef __cplusplus
}
#endif
