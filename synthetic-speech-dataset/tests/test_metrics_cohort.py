from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.collect.metrics_cohort import EXPECTED_CALLS, HEADER, cohort_path, load_cohort, pending, wav_path


ROOT = Path(__file__).resolve().parents[1]


class MetricsCohortTest(unittest.TestCase):
    def test_committed_list_is_the_metric_cohort(self) -> None:
        rows = load_cohort(cohort_path(ROOT))
        self.assertEqual(len(rows), EXPECTED_CALLS)
        self.assertEqual(len({row["call_id"] for row in rows}), EXPECTED_CALLS)
        self.assertEqual(sum(row["dataset"] == "11–12" for row in rows), 457)
        self.assertEqual(sum(row["dataset"] == "16" for row in rows), 361)
        self.assertEqual(sum(row["primary_label"] == "synthetic_tts" for row in rows), 192)
        self.assertEqual(sum(row["primary_label"] == "human_live" for row in rows), 626)
        self.assertEqual(HEADER, tuple(cohort_path(ROOT).read_text(encoding="utf-8").splitlines()[0].split("\t")))

    def test_wav_path_stays_inside_audio(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            destination = wav_path(root, "data/audio/day_20261116/003a5cad-83af-4bb7-96e5-de8f36878e4b.wav")
            self.assertEqual(destination.parent.name, "day_20261116")
            with self.assertRaises(ValueError):
                wav_path(root, "data/audio/../secret.wav")

    def test_pending_skips_a_real_wav(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = "data/audio/day_20261116/003a5cad-83af-4bb7-96e5-de8f36878e4b.wav"
            path = wav_path(root, source)
            path.parent.mkdir(parents=True)
            path.write_bytes(b"RIFFxxxxWAVE" + b"\0" * 40)
            row = {
                "call_id": "003a5cad-83af-4bb7-96e5-de8f36878e4b",
                "source_file": source,
                "client_channel": "0",
                "dataset": "16",
                "primary_label": "human_live",
            }
            self.assertEqual(pending(root, [row]), [])
            path.write_bytes(b"tiny")
            self.assertEqual(pending(root, [row]), [row])


if __name__ == "__main__":
    unittest.main()
