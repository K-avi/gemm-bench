# GEMMBench — Cross-Microarchitecture CRR Performance & Hardware Execution Summary

| SIMD | Microarchitecture | CPU Model | Champion Kernel | Tile ($M_R \times N_R$) | Comp | $f_{\text{boost}}$ (GHz) | $f_{\text{eff}}$ (GHz) | Peak Theo (GFLOP/s) | Achieved (GFLOP/s) | % Peak Theo | FLOP/cycle | Pipe Saturation (%) |
|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| AVX512 | AMD Zen 5 Strix Point | Ryzen AI 9 HX 370 | `mippv2_firestorm_mr4_nr4_fmaddi_crr` | 4x32 | gcc | 5.16 | 5.03 | 82.5 | 80.5 | **97.6%** | 16.00 / 16 | **100.0%** |
| AVX512 | AMD Zen 4 | Ryzen 9 7945HX | `mippv2_firestorm_mr4_nr4_fmaddi_crr` | 4x32 | gcc | 5.46 | 5.25 | 87.4 | 83.8 | **95.9%** | 15.95 / 16 | **99.7%** |
| AVX2 | Intel Skylake | i5-6200U | `mippv2_meteorlake_mr4_nr3_crr` | 4x12 | clang | 2.30 | 2.28 | 36.8 | 35.1 | **95.5%** | 15.39 / 16 | **96.2%** |
| AVX2 | Intel Meteor Lake | Core Ultra 9 185H | `mippv2_meteorlake_mr4_nr3_crr` | 4x12 | clang | 4.78 | 4.69 | 76.5 | 72.9 | **95.3%** | 15.53 / 16 | **97.1%** |
| NEON | Apple M1 | Firestorm | `mippv2_x60_mr6_nr4_fmaddi_crr` | 6x8 | gcc | 3.04 | 3.03 | 48.6 | 48.3 | **99.4%** | 15.93 / 16 | **99.6%** |
| NEON | Raspberry Pi 5 | Cortex-A76 | `mippv2_a76_mr6_nr4_fmaddi_crr` | 6x8 | clang | 2.40 | 2.39 | 19.2 | 19.0 | **98.8%** | 7.93 / 8 | **99.1%** |
| RVV | SpacemiT X100 | SpacemiT X100 | `mippv2_x100_register_blocked_apack4_crr` | 3x16 | clang | 2.40 | 2.40 | 19.2 | 18.5 | **96.4%** | 7.73 / 8 | **96.6%** |
| RVV | SpacemiT A100 | SpacemiT A100 | `mippv2_a100_mr7_nr2_lmul2_pipe_crr` | 7x64 | gcc | 2.00 | 1.99 | 16.0 | 13.3 | **83.4%** | 6.70 / 8 | **83.8%** |
| RVV | SpacemiT X60 | SpacemiT X60 | `mippv2_a100_mr7_nr4_pipe_crr` | 7x16 | clang | 1.60 | 1.60 | 12.8 | 7.4 | **58.0%** | 4.65 / 8 | **58.1%** |

> **Notes on Microarchitectural Metrics:**
> - $f_{\text{boost}}$: Rated single-core burst frequency (from processor specifications / sysfs).
> - $f_{\text{eff}}$: Effective core frequency measured in-process during microkernel execution ($\text{Cycles} / \text{Time}$ using Linux `perf_event_open` hardware cycle counters).
> - **Peak Theo**: Rated theoretical ceiling $R_{\text{peak}} = f_{\text{boost}} \times \text{Peak\_FLOP/cycle}$.
> - **Achieved**: Measured sustained performance $R_{\text{achieved}} = 2MNK / t_{\text{min}}$.
> - **% Peak Theo**: Absolute theoretical efficiency $R_{\text{achieved}} / R_{\text{peak}}$.
> - **Pipe Saturation**: Execution unit saturation $\text{FLOP/cycle} / \text{Peak\_FLOP/cycle}$. For example, on AMD Zen 4, 15.95 / 16.0 FLOP/cycle corresponds to 99.7% pipeline saturation, while thermal/power limits under sustained AVX-512 drop $f_{\text{eff}}$ from 5.46 GHz to 5.25 GHz, explaining the 95.9% of rated burst peak.
