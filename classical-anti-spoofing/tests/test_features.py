import unittest

import numpy as np
import pandas as pd

from src.models.features import columns_for_variant, is_log_energy, select_features, without_log_energy


class FeatureSelectionTests(unittest.TestCase):
    def test_validation_signal_is_not_visible(self):
        rng = np.random.default_rng(0)
        train = pd.DataFrame({"noise": rng.normal(size=80), "useful": np.repeat([0.0, 1.0], 40)})
        y = np.repeat([0, 1], 40)
        selected = select_features(
            train,
            y,
            ["noise", "useful"],
            max_features=1,
            max_nan_fraction=0.5,
            min_finite=10,
            max_pairwise_correlation=0.98,
        )
        self.assertEqual(selected, ["useful"])

    def test_log_energy_coefficients_are_removed_before_selection(self):
        columns = [
            "mfcc_static_00_mean__mean",
            "lfcc_delta_00_std__p25",
            "cqcc_delta2_00_max__max",
            "energy_rms__median",
            "mfcc_static_01_mean__mean",
            "f0_voiced_fraction__mean",
            "spectral_centroid_hz_mean__mean",
        ]
        kept = without_log_energy(columns)
        self.assertEqual(
            kept,
            ["mfcc_static_01_mean__mean", "f0_voiced_fraction__mean", "spectral_centroid_hz_mean__mean"],
        )
        self.assertTrue(is_log_energy("mfcc_static_00_mean__mean"))
        self.assertFalse(is_log_energy("mfcc_static_10_mean__mean"))

    def test_shape_and_voicing_are_disjoint(self):
        columns = [
            "mfcc_static_00_mean__mean",
            "mfcc_static_01_mean__mean",
            "spectral_centroid_hz_mean__mean",
            "zero_crossing_rate_mean__mean",
            "f0_voiced_fraction__mean",
            "acf_harmonicity_mean__min",
            "jitter_local__mean",
            "energy_rms__median",
        ]
        shape, _ = columns_for_variant(columns, "shape")
        voicing, _ = columns_for_variant(columns, "voicing")
        self.assertEqual(
            shape,
            ["mfcc_static_01_mean__mean", "spectral_centroid_hz_mean__mean", "zero_crossing_rate_mean__mean"],
        )
        self.assertEqual(
            voicing,
            ["f0_voiced_fraction__mean", "acf_harmonicity_mean__min", "jitter_local__mean"],
        )
        self.assertEqual(set(shape) & set(voicing), set())

    def test_mfcc_and_spectral_split_shape(self):
        columns = [
            "mfcc_static_00_mean__mean",
            "mfcc_delta_06_max__mean",
            "lfcc_static_03_mean__mean",
            "spectral_centroid_hz_mean__mean",
            "zero_crossing_rate_mean__mean",
            "f0_mean__mean",
        ]
        mfcc, _ = columns_for_variant(columns, "mfcc")
        spectral, _ = columns_for_variant(columns, "spectral")
        self.assertEqual(mfcc, ["mfcc_delta_06_max__mean"])
        self.assertEqual(spectral, ["spectral_centroid_hz_mean__mean", "zero_crossing_rate_mean__mean"])


if __name__ == "__main__":
    unittest.main()
