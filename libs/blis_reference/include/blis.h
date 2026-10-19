#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Standard BLIS basic types
typedef int64_t dim_t;
typedef int64_t inc_t;

// Minimal auxinfo_t for prefetching
typedef struct auxinfo_s {
  const void *next_a;
  const void *next_b;
  uint64_t is;
} auxinfo_t;

// Dummy context structure
typedef struct cntx_s {
  void *dummy;
} cntx_t;

#ifndef __cplusplus
#define restrict __restrict__
#endif

#define bli_auxinfo_next_a(data)                                               \
  ((data) ? ((const auxinfo_t *)(data))->next_a : NULL)
#define bli_auxinfo_next_b(data)                                               \
  ((data) ? ((const auxinfo_t *)(data))->next_b : NULL)
#define BLIS_FLOAT 0
#define BLIS_SINGLE 0
#define BLIS_DOUBLE 2

#define BLIS_MR 0
#define BLIS_NR 1

#if defined(GEMMBENCH_BLIS_X60)
// On SpacemiT X60 (VLEN=256b = 32 bytes):
// For Double precision: MR = 2 * (32 / sizeof(double)) = 8
// For Single precision: MR = 2 * (32 / sizeof(float))  = 16
#define bli_cntx_get_blksz_def_dt(dt, bs, cntx)                                \
  (((dt) == BLIS_DOUBLE) ? (dim_t)8 : (dim_t)16)
#define bli_rvv_get_vlen() ((uint64_t)32)
#endif

// Complex-transpose helpers (no-op when native tile layout matches)
#define GEMM_UKR_SETUP_CT_ANY(...)
#define GEMM_UKR_SETUP_CT(...)
#define GEMM_UKR_FLUSH_CT(...)

#ifdef __cplusplus
}
#endif
