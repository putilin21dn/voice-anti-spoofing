from __future__ import annotations

import unittest

from src.research.model_assisted_annotation.rules import (
    borderline,
    disagreement_codes,
    prediction,
    public_reasons,
    sample_control,
    technical_codes,
)


class QueueRuleTest(unittest.TestCase):
    def test_missing_score_is_not_human(self) -> None:
        self.assertEqual(prediction("spectra0", "", "ok"), "")
        self.assertEqual(prediction("lightgbm_new", "0.9", "insufficient_speech"), "")
        self.assertEqual(prediction("svm_new", "nan", "ok"), "")

    def test_threshold_direction(self) -> None:
        self.assertEqual(prediction("spectra0", "0.22710928916931153", "ok"), "human")
        self.assertEqual(prediction("spectra0", "0.2", "ok"), "synthetic")
        self.assertEqual(prediction("svm_new", "0.0", "ok"), "synthetic")
        self.assertEqual(prediction("svm_new", "-0.01", "ok"), "human")
        self.assertEqual(prediction("lightgbm_new", "0.3460832258410117", "ok"), "synthetic")

    def test_borderline_uses_own_width(self) -> None:
        self.assertTrue(borderline("svm_new", "0.1", "ok"))
        self.assertFalse(borderline("svm_new", "1.0", "ok"))
        self.assertFalse(borderline("spectra0", "", "no_speech"))

    def test_disagreement_needs_both_sides(self) -> None:
        codes = disagreement_codes(
            {"spectra0": "synthetic", "lightgbm_new": ""},
            "no",
            "",
        )
        self.assertIn("spectra_campaign_amd", codes)
        self.assertNotIn("spectra_lightgbm", codes)
        self.assertNotIn("lightgbm_campaign_amd", codes)

    def test_public_text_hides_the_predicted_class(self) -> None:
        text = public_reasons(["spectra_lightgbm", "technical_short_speech", "control_sample"])
        self.assertNotIn("synthetic", text)
        self.assertNotIn("human", text)
        self.assertIn("расхождение моделей", text)
        self.assertIn("короткая речь", text)
        self.assertIn("контрольная выборка", text)

    def test_technical_short_speech_uses_one_spectra_window(self) -> None:
        codes = technical_codes(
            {
                "spectra_status": "ok",
                "classical_status": "ok",
                "spectra_n_segments": "4",
                "spectra_speech_duration_s": "3.0",
                "spectra_n_windows": "1",
                "spectra_full_windows": "0",
                "duration_seconds": "40",
                "client_channel": "0",
            }
        )
        self.assertIn("technical_short_speech", codes)
        self.assertIn("technical_all_padded", codes)

    def test_control_sample_is_stable_and_weighted(self) -> None:
        strata = {
            ("2026-11-11", "PACK1", "human", "lt30"): [f"a{i}" for i in range(3)],
            ("2026-11-11", "PACK1", "human", "ge60"): [f"b{i}" for i in range(20)],
        }
        first = sample_control(strata, target=10, seed=20261111)
        second = sample_control(strata, target=10, seed=20261111)
        self.assertEqual(first, second)
        sampled_b = [key for key in first if key.startswith("b")]
        self.assertEqual(len(sampled_b), 7)
        self.assertGreater(first[sampled_b[0]]["sampling_weight"], 1)
        self.assertEqual(first["a0"]["sampling_weight"], 1)


if __name__ == "__main__":
    unittest.main()
