# GEMMBench — Cross-Microarchitecture CRR Performance & Hardware Execution Summary

| SIMD | Microarchitecture | CPU Model | Champion Kernel | Tile ($M_R \times N_R$) | Comp | Pipe Saturation (%) | FLOP/cycle | $f_{\text{eff}}$ (GHz) | $f_{\text{boost}}$ (GHz) | Achieved (GFLOP/s) | Peak Theo (GFLOP/s) | % Peak Theo |
|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| AVX512 | AMD Zen 5 Strix Point | Ryzen AI 9 HX 370 | `mippv2_firestorm_mr4_nr4_fmaddi_crr` | 4x32 | gcc | **100.0%** | 16.00 / 16 | 5.03 | 5.16 | 80.5 | 82.5 | 97.6% |
| AVX512 | AMD Zen 4 | Ryzen 9 7945HX | `mippv2_firestorm_mr4_nr4_fmaddi_crr` | 4x32 | gcc | **99.6%** | 15.94 / 16 | 5.27 | 5.46 | 84.1 | 87.4 | 96.2% |
| AVX2 | Intel Meteor Lake | Core Ultra 9 185H | `mippv2_meteorlake_mr4_nr3_crr` | 4x12 | clang | **96.7%** | 15.47 / 16 | 4.68 | 4.78 | 72.5 | 76.5 | 94.8% |
| AVX2 | Intel Skylake | i5-6200U | `mippv2_meteorlake_mr4_nr3_crr` | 4x12 | clang | **96.2%** | 15.39 / 16 | 2.28 | 2.30 | 35.1 | 36.8 | 95.5% |
| NEON | Apple M1 | Firestorm | `mippv2_x60_mr6_nr4_fmaddi_crr` | 6x8 | gcc | **99.6%** | 15.93 / 16 | 3.03 | 3.04 | 48.3 | 48.6 | 99.4% |
| NEON | Raspberry Pi 5 | Cortex-A76 | `mippv2_a76_mr6_nr4_fmaddi_crr` | 6x8 | gcc | **98.9%** | 7.91 / 8 | 2.39 | 2.40 | 18.9 | 19.2 | 98.6% |
| RVV | SpacemiT X100 | SpacemiT X100 | `mippv2_x100_register_blocked_apack4_crr` | 3x16 | clang | **96.5%** | 7.72 / 8 | 2.40 | 2.40 | 18.5 | 19.2 | 96.4% |
| RVV | SpacemiT A100 | SpacemiT A100 | `mippv2_a100_mr7_nr2_lmul2_pipe_crr` | 7x64 | gcc | **83.8%** | 6.71 / 8 | 1.99 | 2.00 | 13.4 | 16.0 | 83.5% |
| RVV | SpacemiT X60 | SpacemiT X60 | `mippv2_a100_mr7_nr4_pipe_crr` | 7x16 | clang | **58.1%** | 4.65 / 8 | 1.60 | 1.60 | 7.4 | 12.8 | 58.0% |

> **Notes on Metrics:**
> - **Pipe Saturation**: Execution unit saturation $\text{FLOP/cycle} / \text{Peak\_FLOP/cycle}$. Measures pure code generation and FMA port saturation independently of frequency scaling.
> - $f_{\text{eff}}$: Effective core frequency measured in-process during microkernel execution ($\text{Cycles} / \text{Time}$ using Linux `perf_event_open` hardware cycle counters).
> - $f_{\text{boost}}$: Rated single-core burst frequency (from processor specifications / sysfs).
> - **Achieved**: Measured sustained performance $R_{\text{achieved}} = 2MNK / t_{\text{min}}$.
> - **Peak Theo**: Rated theoretical ceiling $R_{\text{peak}} = f_{\text{boost}} \times \text{Peak\_FLOP/cycle}$.
> - **% Peak Theo**: Theoretical ceiling efficiency $R_{\text{achieved}} / R_{\text{peak}}$. Lower than Pipe Saturation when thermal or power scaling causes $f_{\text{eff}} < f_{\text{boost}}$ (e.g. Zen 4 operating at 5.25 GHz sustained vs 5.46 GHz burst).
