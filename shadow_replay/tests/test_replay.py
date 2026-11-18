"""Часы записанного звонка и таблица приоритета unified-priority-v1."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from shadow_replay.acoustic import model_decision
from shadow_replay.constants import (
    BORDER_WIDTHS,
    LIGHTGBM_NEW_THRESHOLD,
    SPECTRA0_THRESHOLD,
    SVM_NEW_THRESHOLD,
)
from shadow_replay.__main__ import main
from shadow_replay.replay import Replay, play_tape

WINDOW = 64600


def _final(elapsed: float, **predicates) -> dict:
    event = {"kind": "final", "elapsed_s": elapsed, "is_final": True}
    event.update(predicates)
    return event


def _acoustic(origin: str, elapsed: float = 5.0) -> list[dict]:
    if origin == "synthetic":
        margin, svm, lightgbm = 0.1, 1.0, 0.5
    elif origin == "human":
        margin, svm, lightgbm = 1.0, -1.0, 0.0
    else:
        raise AssertionError(origin)
    return [
        {"kind": "audio", "elapsed_s": elapsed, "speech_samples": WINDOW, "spectra_margin": margin},
        {
            "kind": "classical",
            "elapsed_s": elapsed + 0.01,
            "svm_score": svm,
            "lightgbm_score": lightgbm,
        },
    ]


def _play(*events: dict) -> list[dict]:
    return play_tape({"shadow_session_id": "session-1", "call_id": None, "events": list(events)})


class ThresholdTests(unittest.TestCase):
    def test_boundaries_match_the_spec(self) -> None:
        self.assertEqual(model_decision("spectra0", SPECTRA0_THRESHOLD, "ok"), "human")
        self.assertEqual(model_decision("spectra0", SPECTRA0_THRESHOLD - 0.0000001, "ok"), "synthetic")
        self.assertEqual(model_decision("svm", SVM_NEW_THRESHOLD, "ok"), "synthetic")
        self.assertEqual(model_decision("svm", -0.01, "ok"), "human")
        self.assertEqual(model_decision("lightgbm", LIGHTGBM_NEW_THRESHOLD, "ok"), "synthetic")
        self.assertEqual(model_decision("lightgbm", LIGHTGBM_NEW_THRESHOLD - 0.001, "ok"), "human")
        self.assertIsNone(model_decision("spectra0", None, "error"))

    def test_numbers_match_rules_py(self) -> None:
        path = (
            Path(__file__).resolve().parents[2]
            / "synthetic-speech-dataset"
            / "src"
            / "research"
            / "model_assisted_annotation"
            / "rules.py"
        )
        spec = importlib.util.spec_from_file_location("frozen_rules", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        self.assertEqual(module.THRESHOLDS["spectra0"], SPECTRA0_THRESHOLD)
        self.assertEqual(module.THRESHOLDS["svm_new"], SVM_NEW_THRESHOLD)
        self.assertEqual(module.THRESHOLDS["lightgbm_new"], LIGHTGBM_NEW_THRESHOLD)
        self.assertEqual(module.WIDTHS["spectra0"], BORDER_WIDTHS["spectra0"])
        self.assertEqual(module.WIDTHS["svm_new"], BORDER_WIDTHS["svm"])
        self.assertEqual(module.WIDTHS["lightgbm_new"], BORDER_WIDTHS["lightgbm"])


class PriorityTableTests(unittest.TestCase):
    def test_operator_voicemail_is_high_for_every_acoustic_state(self) -> None:
        for origin, voice in (("synthetic", "synthetic"), ("human", "human")):
            rows = _play(_final(1.0, operator_voicemail_indexes=[0]), *_acoustic(origin))
            self.assertEqual(rows[0]["priority"], "high")
            self.assertEqual(rows[0]["fusion"]["voice_origin"], "unknown")
            last = rows[-1]
            self.assertEqual(last["priority"], "high")
            self.assertEqual(last["primary_reason"], "operator_voicemail")
            self.assertEqual(last["fusion"]["voice_origin"], voice)
            self.assertEqual(last["fusion"]["behavior"], "automated")
            self.assertEqual(last["fusion"]["scenario_types"], ["voicemail_ivr"])
            self.assertTrue(last["text_signals"][0]["hangup_eligible"])

        quiet = _play(_final(1.0, operator_voicemail_indexes=[0]), {"kind": "call_end", "elapsed_s": 3})
        self.assertEqual(quiet[-1]["priority"], "high")
        self.assertEqual(quiet[-1]["fusion"]["voice_origin"], "unknown")
        self.assertEqual(quiet[-1]["acoustic"]["ensemble_decision"], "no_decision")

    def test_ai_assistant_needs_synthetic_voice_for_high(self) -> None:
        synthetic = _play(_final(1.0, ai_assistant_indexes=[2]), *_acoustic("synthetic"))
        human = _play(_final(1.0, ai_assistant_indexes=[2]), *_acoustic("human"))
        unknown = _play(_final(1.0, ai_assistant_indexes=[2]))
        self.assertEqual(synthetic[-1]["priority"], "high")
        self.assertEqual(synthetic[-1]["primary_reason"], "acoustic_synthetic_and_ai_assistant")
        self.assertEqual(human[-1]["priority"], "medium")
        self.assertEqual(human[-1]["primary_reason"], "strong_text_without_acoustic_agreement")
        self.assertEqual(unknown[-1]["priority"], "medium")
        self.assertEqual(unknown[-1]["primary_reason"], "strong_text_without_acoustic_agreement")

    def test_synthetic_without_strong_text_stays_medium(self) -> None:
        rows = _play(*_acoustic("synthetic"))
        self.assertEqual(rows[-1]["priority"], "medium")
        self.assertEqual(rows[-1]["primary_reason"], "acoustic_synthetic_without_strong_text")

    def test_weak_text_and_synthetic_keep_both_medium_reasons(self) -> None:
        rows = _play(_final(1.0, machine_phrase_indexes=[4]), *_acoustic("synthetic"))
        last = rows[-1]
        self.assertEqual(last["priority"], "medium")
        self.assertEqual(last["primary_reason"], "acoustic_synthetic_without_strong_text")
        self.assertIn("weak_text", last["reason_codes"])
        self.assertEqual(last["fusion"]["behavior"], "automated")

    def test_weak_text_with_human_or_no_decision_is_medium(self) -> None:
        human = _play(_final(1.0, subscriber_mention=True), *_acoustic("human"))
        unknown = _play(_final(1.0, subscriber_mention=True))
        self.assertEqual(human[-1]["primary_reason"], "weak_text")
        self.assertEqual(unknown[-1]["primary_reason"], "weak_text")
        self.assertEqual(human[-1]["fusion"]["behavior"], "suspicious")

    def test_suspicion_is_medium_for_human_and_for_no_decision(self) -> None:
        human = _play(_final(1.0, explicit_assistant_claim=True), *_acoustic("human"))
        unknown = _play(_final(1.0, secretary_role_claim=True))
        self.assertEqual(human[-1]["primary_reason"], "human_check_suspicion")
        self.assertEqual(unknown[-1]["primary_reason"], "human_check_suspicion")
        self.assertEqual(human[-1]["fusion"]["scenario_types"], ["ai_assistant"])

    def test_human_without_signals_is_low_and_silence_is_undetermined(self) -> None:
        human = _play(*_acoustic("human"))
        silence = _play({"kind": "call_end", "elapsed_s": 1})
        self.assertEqual(human[-1]["priority"], "low")
        self.assertEqual(human[-1]["primary_reason"], "acoustic_human_no_suspicion")
        self.assertEqual(silence[-1]["priority"], "undetermined")
        self.assertEqual(silence[-1]["primary_reason"], "insufficient_information")
        self.assertEqual(silence[-1]["acoustic"]["no_decision_reason"], "no_client_audio")

    def test_self_identification_stays_high_when_llm_says_human(self) -> None:
        rows = _play(
            _final(
                1.0,
                automated_self_identification=True,
                human_check={"mode": "shadow", "verdict": "human_claim", "state": "monitoring"},
            )
        )
        self.assertEqual(rows[-1]["priority"], "high")
        self.assertEqual(rows[-1]["primary_reason"], "automated_self_identification")
        self.assertEqual(rows[-1]["conflicts"], ["C3"])

    def test_interim_voicemail_does_not_raise_high(self) -> None:
        rows = _play({"kind": "final", "elapsed_s": 1.0, "is_final": False, "operator_voicemail_indexes": [0]})
        self.assertNotEqual(rows[-1]["priority"], "high")
        self.assertIn("not_final", rows[-1]["text_signals"][0]["gates"])
        self.assertFalse(rows[-1]["text_signals"][0]["hangup_eligible"])

    def test_blocked_voicemail_is_high_and_does_not_hang_up(self) -> None:
        rows = _play(
            _final(
                1.0,
                operator_voicemail_indexes=[0],
                human_check={"mode": "enforce", "blocks_business": True, "state": "assessing"},
            )
        )
        last = rows[-1]
        self.assertEqual(last["priority"], "high")
        self.assertEqual(last["conflicts"], ["C7"])
        self.assertFalse(last["text_signals"][0]["hangup_eligible"])
        self.assertIn("blocked_by_human_check", last["text_signals"][0]["gates"])
        played = Replay(shadow_session_id="session-1")
        played.play([
            _final(
                1.0,
                operator_voicemail_indexes=[0],
                human_check={"mode": "enforce", "blocks_business": True},
            )
        ])
        self.assertEqual(played.advice()["action"], "none")
        self.assertFalse(played.advice()["termination_candidate"])
        self.assertFalse(played.advice()["additional_check"])


class ClockTests(unittest.TestCase):
    def test_later_window_does_not_rewrite_the_earlier_decision(self) -> None:
        replay = Replay(shadow_session_id="session-1")
        first_window = replay.push(
            {"kind": "audio", "elapsed_s": 4.0375, "speech_samples": WINDOW, "spectra_margin": 0.10}
        )
        opened = replay.push(
            {"kind": "classical", "elapsed_s": 4.05, "svm_score": 1.0, "lightgbm_score": 0.0}
        )
        flipped = replay.push(
            {"kind": "audio", "elapsed_s": 8.075, "speech_samples": WINDOW, "spectra_margin": 0.90}
        )
        self.assertIsNotNone(first_window)
        assert opened is not None and flipped is not None
        self.assertEqual(opened["acoustic"]["spectra0_score"], 0.10)
        self.assertEqual(opened["acoustic"]["ensemble_decision"], "synthetic")
        self.assertEqual(flipped["acoustic"]["spectra0_score"], 0.50)
        self.assertEqual(flipped["acoustic"]["ensemble_decision"], "human")
        self.assertEqual(opened["acoustic"]["ensemble_decision"], "synthetic")
        self.assertEqual(flipped["fusion"]["decision_changes"], 1)
        self.assertEqual(flipped["fusion"]["no_decision_gaps"], 0)
        self.assertEqual(replay.advice()["action"], "none")

    def test_one_vote_stays_no_decision(self) -> None:
        rows = _play({"kind": "audio", "elapsed_s": 4.0375, "speech_samples": WINDOW, "spectra_margin": 0.01})
        self.assertEqual(rows[-1]["acoustic"]["ensemble_decision"], "no_decision")
        self.assertEqual(rows[-1]["acoustic"]["valid_vote_count"], 1)
        self.assertEqual(rows[-1]["acoustic"]["no_decision_reason"], "insufficient_models")
        self.assertEqual(rows[-1]["priority"], "undetermined")

    def test_unfinished_window_is_not_a_vote(self) -> None:
        replay = Replay(shadow_session_id="session-1")
        self.assertIsNone(replay.push({"kind": "audio", "elapsed_s": 1.0, "speech_samples": WINDOW - 1}))
        end = replay.push({"kind": "call_end", "elapsed_s": 2.0})
        assert end is not None
        self.assertEqual(end["acoustic"]["no_decision_reason"], "insufficient_audio")
        self.assertTrue(end["acoustic"]["padded_tail"])
        self.assertEqual(end["speech_elapsed_s"], 0.0)

    def test_gap_through_no_decision_is_not_a_direct_flip(self) -> None:
        rows = _play(
            *_acoustic("synthetic", 4.0),
            {"kind": "classical", "elapsed_s": 6.0, "svm_score": None, "svm_status": "error", "lightgbm_score": None, "lightgbm_status": "error"},
            *_acoustic("human", 9.0),
        )
        end = rows[-1]
        self.assertEqual(end["acoustic"]["ensemble_decision"], "human")
        self.assertGreaterEqual(end["fusion"]["no_decision_gaps"], 1)
        self.assertEqual(end["fusion"]["decision_changes"], 0)

    def test_text_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _play(_final(1.0, text="абонент занят"))

    def test_cli_writes_jsonl_without_calling_models(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tape = Path(folder) / "tape.json"
            output = Path(folder) / "events.jsonl"
            tape.write_text(
                json.dumps({"events": [{"kind": "call_end", "elapsed_s": 0.5}]}),
                encoding="utf-8",
            )
            self.assertEqual(main([str(tape), "--output", str(output)]), 0)
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(row["priority"], "undetermined")
            self.assertEqual(row["update_kind"], "call_end")


if __name__ == "__main__":
    unittest.main()
