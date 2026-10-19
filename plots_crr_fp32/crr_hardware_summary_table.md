# GEMMBench — Cross-Microarchitecture CRR Performance & Hardware Execution Summary

| SIMD | Microarchitecture | CPU Model | Champion Kernel | Tile ($M_R \times N_R$) | Comp | Pipe Saturation (%) | FLOP/cycle | $f_{\text{eff}}$ (GHz) | $f_{\text{boost}}$ (GHz) | Achieved (GFLOP/s) | Peak Theo (GFLOP/s) | % Peak Theo |
|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| AVX512 | AMD Zen 5 Strix Point | Ryzen AI 9 HX 370 | `mippv2_firestorm_mr4_nr4_fmaddi_crr` | 4x64 | gcc | **100.0%** | 31.99 / 32 | 5.12 | 5.16 | 163.7 | 165.1 | 99.2% |
| AVX512 | AMD Zen 4 | Ryzen 9 7945HX | `mippv2_firestorm_mr4_nr4_fmaddi_crr` | 4x64 | gcc | **99.6%** | 31.87 / 32 | 5.42 | 5.46 | 172.7 | 174.8 | 98.8% |
| AVX2 | Intel Meteor Lake | Core Ultra 9 185H | `mippv2_meteorlake_mr4_nr3_crr` | 4x24 | clang | **96.5%** | 30.89 / 32 | 4.74 | 4.78 | 146.5 | 153.0 | 95.8% |
| NEON | Apple M1 | Firestorm | `mippv2_x60_mr6_nr4_fmaddi_crr` | 6x16 | gcc | **99.6%** | 31.87 / 32 | 3.03 | 3.04 | 96.6 | 97.2 | 99.4% |
| NEON | Raspberry Pi 5 | Cortex-A76 | `mippv2_a76_mr6_nr4_fmaddi_crr` | 6x16 | gcc | **99.3%** | 15.89 / 16 | 2.39 | 2.40 | 38.0 | 38.4 | 98.9% |
| RVV | SpacemiT X100 | SpacemiT X100 | `mippv2_x100_register_blocked_apack4_crr` | 3x32 | clang | **96.6%** | 15.45 / 16 | 2.40 | 2.40 | 37.0 | 38.4 | 96.4% |
| RVV | SpacemiT A100 | SpacemiT A100 | `mippv2_a100_mr7_nr2_lmul2_pipe_crr` | 7x128 | clang | **76.4%** | 24.44 / 32 | 1.99 | 2.00 | 48.7 | 64.0 | 76.1% |
| RVV | SpacemiT X60 | SpacemiT X60 | `mippv2_a100_mr7_nr4_pipe_crr` | 7x32 | clang | **57.6%** | 9.21 / 16 | 1.60 | 1.60 | 14.7 | 25.6 | 57.5% |

> **Notes on Metrics:**
> - **Pipe Saturation**: Execution unit saturation $\text{FLOP/cycle} / \text{Peak\_FLOP/cycle}$. Measures pure code generation and FMA port saturation independently of frequency scaling.
> - $f_{\text{eff}}$: Effective core frequency measured in-process during microkernel execution ($\text{Cycles} / \text{Time}$ using Linux `perf_event_open` hardware cycle counters).
> - $f_{\text{boost}}$: Rated single-core burst frequency (from processor specifications / sysfs).
> - **Achieved**: Measured sustained performance $R_{\text{achieved}} = 2MNK / t_{\text{median}}$.
> - **Peak Theo**: Rated theoretical ceiling $R_{\text{peak}} = f_{\text{boost}} \times \text{Peak\_FLOP/cycle}$.
> - **% Peak Theo**: Theoretical ceiling efficiency $R_{\text{achieved}} / R_{\text{peak}}$. Lower than Pipe Saturation when thermal or power scaling causes $f_{\text{eff}} < f_{\text{boost}}$ (e.g. Zen 4 operating at 5.25 GHz sustained vs 5.46 GHz burst).
