from __future__ import annotations

import unittest

from src.evaluation.dataset import load_evaluation_config, prepare_dataset
from src.evaluation.metrics import binary_metrics, classical_class, disagreement_flags, spectra0_class


class EvaluationRulesTest(unittest.TestCase):
    def test_spectra_threshold_stays_at_boundary(self) -> None:
        self.assertEqual(spectra0_class(1.02), "bonafide")
        self.assertEqual(spectra0_class(1.019999), "spoof")

    def test_classical_threshold_is_not_renamed(self) -> None:
        threshold = 0.49265447022723596
        self.assertEqual(classical_class(threshold, threshold), "machine")
        self.assertEqual(classical_class(threshold - 0.001, threshold), "human")

    def test_binary_metrics_stay_empty_without_synthetic(self) -> None:
        calls = [
            {
                "category": "human_live",
                "in_binary_metrics": True,
                "spectra0_status": "ok",
                "spectra0_class": "spoof",
            }
        ]
        result = binary_metrics(calls, "spectra0")
        self.assertNotIn("recall_synthetic", result)
        self.assertIn("синтетики", result["reason"])

    def test_aasist_has_no_invented_call_threshold(self) -> None:
        result = binary_metrics(
            [
                {
                    "category": "human_live",
                    "in_binary_metrics": True,
                    "aasist3_status": "ok",
                    "aasist3_class": "",
                },
                {
                    "category": "synthetic_tts",
                    "in_binary_metrics": True,
                    "aasist3_status": "ok",
                    "aasist3_class": "",
                },
            ],
            "aasist3",
        )
        self.assertIn("не зафиксирован", result["reason"])
        self.assertNotIn("f1", result)

    def test_confirmed_batch_keeps_robots_out_of_category_metrics(self) -> None:
        calls, errors = prepare_dataset(load_evaluation_config())
        self.assertEqual(errors, [])
        confirmed = [call for call in calls if call.cohort == "confirmed_001"]
        legacy = [call for call in calls if call.cohort == "legacy_test"]
        self.assertEqual(len(confirmed), 24)
        self.assertEqual(len(legacy), 43)
        robots = [call for call in confirmed if call.listener_label == "robot"]
        humans = [call for call in confirmed if call.primary_label == "human_live"]
        self.assertEqual(len(robots), 12)
        self.assertEqual(len(humans), 12)
        self.assertTrue(all(call.category == "unclassified_robot" for call in robots))
        self.assertTrue(all(not call.in_category_metrics for call in robots))
        self.assertTrue(all(call.in_category_metrics and call.in_binary_metrics for call in humans))
        self.assertTrue(all(call.primary_label == "" and not call.in_category_metrics for call in legacy))
        self.assertEqual({call.legacy_label for call in legacy}, {"human", "machine"})
        self.assertEqual(sum(call.legacy_label == "human" for call in legacy), 18)
        self.assertEqual(sum(call.legacy_label == "machine" for call in legacy), 25)
        self.assertTrue(all(call.wav_path.is_file() and call.exclusion == "" for call in legacy))
        self.assertEqual(sum(call.exclusion == "locked_baseline" for call in calls), 0)
        self.assertEqual(sum(call.exclusion == "batch_001" for call in calls), 0)

    def test_disagreement_does_not_treat_machine_as_synthetic_truth(self) -> None:
        flags = disagreement_flags(
            {
                "spectra0_class": "bonafide",
                "logistic_class": "machine",
                "svm_class": "human",
                "lightgbm_class": "human",
            }
        )
        self.assertIn("spectra0_bonafide_logistic_machine", flags)
        self.assertNotIn("logistic_is_correct", flags)


if __name__ == "__main__":
    unittest.main()
