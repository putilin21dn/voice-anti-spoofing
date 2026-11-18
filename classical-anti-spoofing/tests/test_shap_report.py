import unittest

import numpy as np

from src.models.shap_report import coverage_count, describe_feature


class ShapReportTests(unittest.TestCase):
    def test_feature_groups(self):
        energy = describe_feature("mfcc_static_00_mean__mean")
        self.assertEqual(energy["family"], "energy")
        self.assertEqual(energy["cep_bucket"], "c0")
        self.assertEqual(energy["call_agg"], "mean")

        shape = describe_feature("lfcc_delta_12_std__p75")
        self.assertEqual(shape["family"], "lfcc")
        self.assertEqual(shape["cep_kind"], "delta")
        self.assertEqual(shape["cep_index"], "12")
        self.assertEqual(shape["cep_bucket"], "c10-19")

        self.assertEqual(describe_feature("f0_voiced_fraction__min")["family"], "voicing")
        self.assertEqual(describe_feature("spectral_centroid_hz_mean__max")["family"], "spectral")
        self.assertEqual(describe_feature("cqcc_delta2_03_mean__median")["cep_bucket"], "c1-4")

    def test_coverage_count(self):
        self.assertEqual(coverage_count(np.array([5.0, 3.0, 2.0]), 0.5), 1)
        self.assertEqual(coverage_count(np.array([5.0, 3.0, 2.0]), 0.8), 2)
        self.assertEqual(coverage_count(np.array([5.0, 3.0, 2.0]), 0.95), 3)
        self.assertEqual(coverage_count(np.zeros(4), 0.95), 0)
