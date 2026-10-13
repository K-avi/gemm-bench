# BLIS Reference Microkernels in GEMMBench

This directory contains standalone reference assembly microkernels from the [BLIS (Basic Linear Algebra Subprograms)](https://github.com/flame/blis) framework and specialized community ports, embedded directly into `gemm_bench` for cycle-accurate baseline benchmarking.

All microkernels in this directory are distributed under the terms of the **BSD 3-Clause License** (see individual file headers and upstream project repositories).

---

## 1. Provenance and Attribution Matrix

| Source File | Target ISA / Architecture | Precision | Geometry ($M_R \times N_R$) | Upstream Repository & Commit | Author / Project | License |
| :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| [`dgemm_x60_2vx14.c`](src/dgemm_x60_2vx14.c) | RISC-V Vector 1.0 (SpacemiT X60) | FP64 | $2\text{v} \times 14$ ($8 \times 14$) | [stepannassyr/blis (`x60`)](https://github.com/stepannassyr/blis/tree/6ffc1d35fcafe74788b2aeaf00d5792d2618e0b9/kernels/x60) (`6ffc1d3`) | **Stepan Nassyr** | BSD-3-Clause |
| [`sgemm_x60_2vx14.c`](src/sgemm_x60_2vx14.c) | RISC-V Vector 1.0 (SpacemiT X60) | FP32 | $2\text{v} \times 14$ ($16 \times 14$) | [stepannassyr/blis (`x60`)](https://github.com/stepannassyr/blis/tree/6ffc1d35fcafe74788b2aeaf00d5792d2618e0b9/kernels/x60) (`6ffc1d3`) | **Stepan Nassyr** | BSD-3-Clause |
| [`skx_d16x14.c`](src/skx_d16x14.c) | x86-64 AVX-512 (Skylake-X, Zen 4, Zen 5) | FP64 | $16 \times 14$ | [flame/blis (upstream)](https://github.com/flame/blis/tree/a8037cfc4330030c726efaec1ffeb1f93d468aa3) (`a8037cf`) | FLAME Project, UT Austin | BSD-3-Clause |
| [`skx_s32x12.c`](src/skx_s32x12.c) | x86-64 AVX-512 (Skylake-X, Zen 4, Zen 5) | FP32 | $32 \times 12$ | [flame/blis (upstream)](https://github.com/flame/blis/tree/a8037cfc4330030c726efaec1ffeb1f93d468aa3) (`a8037cf`) | FLAME Project, UT Austin | BSD-3-Clause |
| [`haswell_d6x8.c`](src/haswell_d6x8.c) | x86-64 AVX2 / FMA3 (Haswell, Meteor Lake) | FP64 & FP32 | $6 \times 8$ (DP) / $6 \times 16$ (SP) | [flame/blis (upstream)](https://github.com/flame/blis/tree/a8037cfc4330030c726efaec1ffeb1f93d468aa3) (`a8037cf`) | FLAME Project, UT Austin, AMD | BSD-3-Clause |
| [`armv8a_d6x8.c`](src/armv8a_d6x8.c) | AArch64 NEON (Armv8-A, Apple M1, Cortex-A76) | FP64 & FP32 | $6 \times 8$ (DP) / $8 \times 12$ (SP) | [flame/blis (upstream)](https://github.com/flame/blis/tree/a8037cfc4330030c726efaec1ffeb1f93d468aa3) (`a8037cf`) | FLAME Project, UT Austin, Univ. of Tokyo | BSD-3-Clause |

---

## 2. Kernel Details & Architectural Context

### RISC-V Vector 1.0 (SpacemiT X60)
The microkernels for SpacemiT X60 (`dgemm_x60_2vx14.c` and `sgemm_x60_2vx14.c`) were developed and empirically tuned by **Stepan Nassyr** ([@stepannassyr](https://github.com/stepannassyr)) in his BLIS fork:
- **Repository:** [`https://github.com/stepannassyr/blis`](https://github.com/stepannassyr/blis)
- **Branch:** `x60`
- **Tracked Commit:** [`6ffc1d35fcafe74788b2aeaf00d5792d2618e0b9`](https://github.com/stepannassyr/blis/commit/6ffc1d35fcafe74788b2aeaf00d5792d2618e0b9) (Kernel source commit: `7039e98a746f10290208670b837662b4ff85193e`)
- **Location:** `kernels/x60/3/`
- **Design & Scheduling:** Generated using `asmgen-gemm` and optimized with `x60_multiloopexplore.py`. Specifically structured for the dual-issue in-order pipeline of the SpacemiT X60 core, accounting for shared issue pipelines between vector memory loads (`vle`) and fused multiply-accumulate (`vfmacc`).

### Upstream FLAME BLIS Kernels (x86-64 & AArch64)
All x86-64 (AVX2, AVX-512) and AArch64 (NEON) microkernels originate from the official upstream repository:
- **Repository:** [`https://github.com/flame/blis`](https://github.com/flame/blis)
- **Tracked Commit:** [`a8037cfc4330030c726efaec1ffeb1f93d468aa3`](https://github.com/flame/blis/commit/a8037cfc4330030c726efaec1ffeb1f93d468aa3) (Release baseline: BLIS 2.1)
- **Project:** The FLAME Project, Department of Computer Science, The University of Texas at Austin.
- **Copyright Holders:** The University of Texas at Austin, Advanced Micro Devices, Inc., and The University of Tokyo.
