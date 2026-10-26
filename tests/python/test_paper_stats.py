#!/usr/bin/env python3
"""
Unit tests for tools/paper_stats.py.
Run with: python3 -m unittest discover -s tests/python
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Add repo root and tools/ to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

from paper_stats import (
    bootstrap_ratio_of_medians,
    flop_per_cycle_samples,
    geomean,
    get_l1d_bytes,
    load_raw,
    parse_cache_size_bytes,
    parse_filename_meta,
    working_set_bytes,
)


class TestPaperStats(unittest.TestCase):

    def test_working_set_bytes(self):
        # M=6, N=32, K=256, elem_bytes=4
        # Expected: K * (M + N) * elem_bytes = 256 * (6 + 32) * 4 = 256 * 38 * 4 = 38912
        ws = working_set_bytes(6, 32, 256, 4)
        self.assertEqual(ws, 256 * 38 * 4)
        self.assertEqual(ws, 38912)

    def test_geomean(self):
        self.assertAlmostEqual(geomean([2.0, 8.0]), 4.0)
        self.assertAlmostEqual(geomean(np.array([1.0, 2.0, 4.0])), 2.0)
        self.assertTrue(np.isnan(geomean([])))
        self.assertTrue(np.isnan(geomean([-1.0, 2.0])))

    def test_bootstrap_deterministic(self):
        # Same seed must yield identical point estimate and confidence interval
        a = {1: np.array([12.0, 12.2, 11.9]), 2: np.array([12.1, 12.0, 12.3])}
        b = {1: np.array([10.0, 10.1, 9.9]), 2: np.array([10.0, 10.2, 10.1])}
        pe1, low1, high1 = bootstrap_ratio_of_medians(a, b, n_boot=2000, seed=42)
        pe2, low2, high2 = bootstrap_ratio_of_medians(a, b, n_boot=2000, seed=42)
        self.assertEqual(pe1, pe2)
        self.assertEqual(low1, low2)
        self.assertEqual(high1, high2)

    def test_bootstrap_identical_samples(self):
        # Two identical distributions must give point estimate 1.0 and CI containing 1.0
        a = {1: np.array([10.0, 10.1, 9.9, 10.0]), 2: np.array([10.1, 9.9, 10.0, 10.2])}
        b = {1: np.array([10.0, 10.1, 9.9, 10.0]), 2: np.array([10.1, 9.9, 10.0, 10.2])}
        pe, low, high = bootstrap_ratio_of_medians(a, b, n_boot=2000, seed=123)
        self.assertAlmostEqual(pe, 1.0, places=5)
        self.assertTrue(low <= 1.0 <= high, f"Expected 1.0 in [{low}, {high}]")

    def test_bootstrap_shifted_samples(self):
        # Shifted by 10% with low noise -> CI must strictly exclude 1.0 (low > 1.0)
        rng = np.random.default_rng(999)
        a = {
            1: rng.normal(11.0, 0.05, size=15),
            2: rng.normal(11.0, 0.05, size=15),
            3: rng.normal(11.0, 0.05, size=15),
        }
        b = {
            1: rng.normal(10.0, 0.05, size=15),
            2: rng.normal(10.0, 0.05, size=15),
            3: rng.normal(10.0, 0.05, size=15),
        }
        pe, low, high = bootstrap_ratio_of_medians(a, b, n_boot=2000, seed=456)
        self.assertGreater(low, 1.0)
        self.assertAlmostEqual(pe, 1.10, delta=0.03)

    def test_ci_coverage_simulation(self):
        # Simulated coverage test: true median ratio is 1.05. Over 100 datasets, at least 90 CIs cover true ratio.
        true_ratio = 1.05
        covered = 0
        n_sim = 100
        rng = np.random.default_rng(777)

        for s in range(n_sim):
            a = {
                1: rng.normal(10.5, 0.3, size=10),
                2: rng.normal(10.5, 0.3, size=10),
            }
            b = {
                1: rng.normal(10.0, 0.3, size=10),
                2: rng.normal(10.0, 0.3, size=10),
            }
            _, low, high = bootstrap_ratio_of_medians(a, b, n_boot=500, seed=1000 + s, ci=0.95)
            if low <= true_ratio <= high:
                covered += 1

        self.assertGreaterEqual(
            covered,
            90,
            f"Expected at least 90/100 coverage for 95% CI, got {covered}",
        )

    def test_parse_filename_meta(self):
        meta = parse_filename_meta("gemm_results_zen4_avx512_crr_nofp_gcc.csv")
        self.assertEqual(meta["Platform"], "zen4_avx512")
        self.assertEqual(meta["Layout"], "crr")
        self.assertEqual(meta["Variant"], "nofp")
        self.assertEqual(meta["Precision"], "fp64")
        self.assertEqual(meta["Compiler"], "gcc")

        meta32 = parse_filename_meta("gemm_results_meteorlake_avx2_crr_coldc_fp32_clang.raw.csv")
        self.assertEqual(meta32["Platform"], "meteorlake_avx2")
        self.assertEqual(meta32["Layout"], "crr")
        self.assertEqual(meta32["Variant"], "coldc")
        self.assertEqual(meta32["Precision"], "fp32")
        self.assertEqual(meta32["Compiler"], "clang")

        meta_def = parse_filename_meta("gemm_results_x100_rvv_crr_gcc.csv")
        self.assertEqual(meta_def["Variant"], "default")

    def test_load_raw_and_flop_per_cycle(self):
        # Create a temporary raw CSV and test load_raw + flop_per_cycle_samples
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "gemm_results_zen4_avx512_crr_gcc.raw.csv"
            raw_content = (
                "Kernel,M,N,K,Alpha,Beta,Fastpath,CCold,Precision,RunId,Rep,Time_s,Cycles,Instructions\n"
                "blis_native,6,8,64,1,0,1,0,fp64,1,0,1.8e-07,400.0,1400.0\n"
                "blis_native,6,8,64,1,0,1,0,fp64,1,1,1.8e-07,410.0,1400.0\n"
                "blis_native,6,8,64,1,0,1,0,fp64,2,0,1.8e-07,405.0,1400.0\n"
                "blis_native,6,8,64,1,0,1,0,fp64,2,1,1.8e-07,395.0,1400.0\n"
            )
            csv_path.write_text(raw_content, encoding="utf-8")

            df = load_raw(csv_path)
            self.assertEqual(len(df), 4)
            self.assertEqual(df["Platform"].iloc[0], "zen4_avx512")
            self.assertEqual(df["Compiler"].iloc[0], "gcc")

            samples = flop_per_cycle_samples(df, "blis_native", 64)
            self.assertEqual(set(samples.keys()), {1, 2})
            # Ops = 2 * 6 * 8 * 64 = 6144
            # Run 1 Rep 0 Cycles = 400 -> FC = 6144 / 400 = 15.36
            self.assertAlmostEqual(samples[1][0], 6144.0 / 400.0)

    def test_parse_cache_and_l1d_bytes(self):
        self.assertEqual(parse_cache_size_bytes("32K"), 32 * 1024)
        self.assertEqual(parse_cache_size_bytes("256K"), 256 * 1024)
        self.assertEqual(parse_cache_size_bytes("3072K"), 3072 * 1024)
        self.assertEqual(parse_cache_size_bytes("32M"), 32 * 1024 * 1024)

        uarch_spec = {"l1d_kib": 48}
        self.assertEqual(get_l1d_bytes(None, uarch_spec), 48 * 1024)

    def test_analyze_dataset_synthetic(self):
        from compare_mippv2_blis import analyze_dataset

        # Synthetic dataset with 3 kernels at K=128 and K=256:
        # - blis_native
        # - mippv2_firestorm_mr4_nr4_fmaddi_crr (official pre-registered champion for zen4_avx512)
        # - mippv2_faster_explo_kernel (faster than champion, e.g. 15.5 FLOP/cyc vs 14.0 FLOP/cyc)
        data = [
            # BLIS native: 10.0 FLOP/cyc at K=128 and 256
            {"Platform": "zen4_avx512", "Compiler": "gcc", "Precision": "fp64", "Variant": "default", "Layout": "crr",
             "Kernel": "blis_native", "M": 4, "N": 16, "K": 128, "FLOP_per_cycle_median": 10.0, "GFLOPS_median": 40.0, "Eff_GHz": 4.0},
            {"Platform": "zen4_avx512", "Compiler": "gcc", "Precision": "fp64", "Variant": "default", "Layout": "crr",
             "Kernel": "blis_native", "M": 4, "N": 16, "K": 256, "FLOP_per_cycle_median": 10.0, "GFLOPS_median": 40.0, "Eff_GHz": 4.0},

            # Official champion: 14.0 FLOP/cyc at K=128 and 256 (Ratio = 1.40)
            {"Platform": "zen4_avx512", "Compiler": "gcc", "Precision": "fp64", "Variant": "default", "Layout": "crr",
             "Kernel": "mippv2_firestorm_mr4_nr4_fmaddi_crr", "M": 4, "N": 16, "K": 128, "FLOP_per_cycle_median": 14.0, "GFLOPS_median": 56.0, "Eff_GHz": 4.0},
            {"Platform": "zen4_avx512", "Compiler": "gcc", "Precision": "fp64", "Variant": "default", "Layout": "crr",
             "Kernel": "mippv2_firestorm_mr4_nr4_fmaddi_crr", "M": 4, "N": 16, "K": 256, "FLOP_per_cycle_median": 14.0, "GFLOPS_median": 56.0, "Eff_GHz": 4.0},

            # Faster non-champion: 15.5 FLOP/cyc
            {"Platform": "zen4_avx512", "Compiler": "gcc", "Precision": "fp64", "Variant": "default", "Layout": "crr",
             "Kernel": "mippv2_faster_explo_kernel", "M": 4, "N": 16, "K": 128, "FLOP_per_cycle_median": 15.5, "GFLOPS_median": 62.0, "Eff_GHz": 4.0},
            {"Platform": "zen4_avx512", "Compiler": "gcc", "Precision": "fp64", "Variant": "default", "Layout": "crr",
             "Kernel": "mippv2_faster_explo_kernel", "M": 4, "N": 16, "K": 256, "FLOP_per_cycle_median": 15.5, "GFLOPS_median": 62.0, "Eff_GHz": 4.0},
        ]
        df = pd.DataFrame(data)

        # Pre-registered mode: MUST pick mippv2_firestorm_mr4_nr4_fmaddi_crr (not the faster kernel)
        summary_df, detailed_df, details = analyze_dataset(df, legacy_argmax=False, k_window=(128, 256))
        self.assertEqual(len(summary_df), 1)
        self.assertEqual(summary_df["MIPP_Champion"].iloc[0], "mippv2_firestorm_mr4_nr4_fmaddi_crr")
        self.assertAlmostEqual(summary_df["Ratio_geomean"].iloc[0], 1.40)
        self.assertEqual(len(detailed_df), 2)
        # Check matched-K ratios
        self.assertAlmostEqual(detailed_df.loc[detailed_df["K"] == 128, "Ratio"].iloc[0], 1.40)
        self.assertAlmostEqual(detailed_df.loc[detailed_df["K"] == 256, "Ratio"].iloc[0], 1.40)

        # Legacy argmax mode: selects mippv2_faster_explo_kernel
        summary_leg, _, _ = analyze_dataset(df, legacy_argmax=True, k_window=(128, 256))
        self.assertEqual(summary_leg["MIPP_Champion"].iloc[0], "mippv2_faster_explo_kernel")
        self.assertAlmostEqual(summary_leg["Ratio_geomean"].iloc[0], 1.55)

        # Precision filter test: add fp32 rows and verify precision_filter isolates fp64 vs fp32
        df_fp32 = df.copy()
        df_fp32["Precision"] = "fp32"
        df_mixed = pd.concat([df, df_fp32], ignore_index=True)
        sum_all, _, _ = analyze_dataset(df_mixed, legacy_argmax=False, k_window=(128, 256))
        self.assertEqual(len(sum_all), 2)
        sum_fp64, _, _ = analyze_dataset(df_mixed, legacy_argmax=False, k_window=(128, 256), precision_filter="fp64")
        self.assertEqual(len(sum_fp64), 1)
        self.assertEqual(sum_fp64["Precision"].iloc[0], "fp64")
        sum_fp32, _, _ = analyze_dataset(df_mixed, legacy_argmax=False, k_window=(128, 256), precision_filter="fp32")
        self.assertEqual(len(sum_fp32), 1)
        self.assertEqual(sum_fp32["Precision"].iloc[0], "fp32")


if __name__ == "__main__":
    unittest.main()
