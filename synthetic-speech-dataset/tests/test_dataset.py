from __future__ import annotations

import json
import struct
import threading
import unittest
import urllib.error
import urllib.request
import wave
from pathlib import Path

from src.annotation.schema import validate_annotation
from src.annotation.storage import AnnotationStore
from src.audio.loader import wav_bytes
from src.audio.preview import preview_bytes
from src.data.manifest import init_batch
from src.data.selector import build_batch, client_channel_index
from src.annotation.review_app import make_server
from src.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def _component(name: str) -> Path:
    import importlib.util

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "repo_paths.py"
        if candidate.is_file() and (parent / "pyproject.toml").is_file():
            spec = importlib.util.spec_from_file_location("voice_repo_paths", candidate)
            module = importlib.util.module_from_spec(spec)
            if spec.loader is None:
                break
            spec.loader.exec_module(module)
            return module.component_dir(here, name)
    raise RuntimeError("Код вне репозитория voice-anti-spoofing не используется")


def stereo(path: Path, count: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(struct.pack("<hh", index, 1000 + index) for index in range(count))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(payload)


def call_id(tail: str) -> str:
    return f"00000000-0000-4000-8000-{tail:0>12}"


class AudioTests(unittest.TestCase):
    def test_channel_keeps_its_samples(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            wav = Path(temporary) / "call.wav"
            stereo(wav, 4)
            right = wav_bytes(wav, 1)
            with wave.open(__import__("io").BytesIO(right), "rb") as handle:
                channels = handle.getnchannels()
                frames = handle.readframes(handle.getnframes())
            self.assertEqual(channels, 1)
            values = struct.unpack("<" + "h" * (len(frames) // 2), frames)
            self.assertEqual(values, (1000, 1001, 1002, 1003))

    def test_source_file_cannot_escape(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "audio"
            audio.mkdir()
            cfg = {"_root": str(root), "paths": {"collector_root": str(root), "audio_root": str(audio)}}
            with self.assertRaises(PermissionError):
                preview_bytes(cfg, "../outside.wav", "mix")


class LabelTests(unittest.TestCase):
    def test_old_campaign_label_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_annotation({"primary_label": "machine", "label_source": "manual_listening", "label_confidence": "high", "review_status": "reviewed"}, duration_s=10)

    def test_unknown_stays_without_confidence(self) -> None:
        clean = validate_annotation({"primary_label": "unknown", "label_source": "unknown_source", "review_status": "unreviewed"}, duration_s=10)
        self.assertEqual(clean["primary_label"], "unknown")
        self.assertEqual(clean["label_source"], "unknown_source")

    def test_recorded_human_is_not_synthetic(self) -> None:
        clean = validate_annotation(
            {
                "primary_label": "human_recorded",
                "secondary_label": "voicemail",
                "label_source": "manual_listening",
                "label_confidence": "high",
                "review_status": "reviewed",
            },
            duration_s=10,
        )
        self.assertEqual(clean["primary_label"], "human_recorded")

    def test_mixed_required_when_segment_origins_differ(self) -> None:
        payload = {
            "primary_label": "human_live",
            "label_source": "manual_listening",
            "label_confidence": "high",
            "review_status": "reviewed",
            "segments": [
                {"start_time": 0, "end_time": 1, "segment_label": "synthetic_tts", "label_source": "manual_listening", "label_confidence": "medium"},
                {"start_time": 1, "end_time": 2, "segment_label": "human_live", "label_source": "manual_listening", "label_confidence": "medium"},
            ],
        }
        with self.assertRaises(ValueError):
            validate_annotation(payload, duration_s=5)

    def test_ai_secretary_cannot_be_marked_human(self) -> None:
        with self.assertRaises(ValueError):
            validate_annotation(
                {
                    "primary_label": "human_live",
                    "secondary_label": "ai_secretary",
                    "label_source": "manual_listening",
                    "label_confidence": "high",
                    "review_status": "reviewed",
                },
                duration_s=5,
            )


class BatchTests(unittest.TestCase):
    def test_selection_drops_phone_fields_and_keeps_unknown(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector = root / "collector"
            audio = collector / "data" / "audio" / "b2x" / "2026-11"
            human_id = call_id("aaa")
            excel_id = call_id("bbb")
            machine_id = call_id("ccc")
            live_id = call_id("ddd")
            stereo(audio / "human" / f"{human_id}.wav")
            stereo(audio / "machine" / f"{excel_id}.wav")
            stereo(audio / "machine" / f"{machine_id}.wav")
            stereo(audio / "machine" / f"{live_id}.wav")
            manifests = collector / "data" / "manifests"
            manifests.mkdir(parents=True)
            human_csv = manifests / "human.csv"
            machine_csv = manifests / "machine.csv"
            human_csv.write_text(
                "call_id,outcome,classification_result,campaign_name,goal_status,external_id,timestamp\n"
                f"{human_id},answering_machine,answering_machine|manual_human,B2X,undetermined,excel-room-should-not-leak,2026-11-03T00:00:00+00:00\n",
                encoding="utf-8",
            )
            machine_csv.write_text(
                "call_id,outcome,classification_result,campaign_name,goal_status,external_id,timestamp\n"
                f"{excel_id},answering_machine,автоответчик,B2X,undetermined,excel-room-1,2026-11-04T00:00:00+00:00\n"
                f"{machine_id},answering_machine,автоответчик,B2X,undetermined,room-2,2026-11-05T00:00:00+00:00\n"
                f"{live_id},b2x_lead,перезвон|manual_review,B2X,goal_achieved,room-3,2026-11-06T00:00:00+00:00\n",
                encoding="utf-8",
            )
            splits = root / "splits"
            splits.mkdir()
            (splits / "test.csv").write_text(f"call_id,label\n{machine_id},machine\n", encoding="utf-8")
            for name in ("train", "validation", "spectra_threshold"):
                (splits / f"{name}.csv").write_text("call_id,label\n", encoding="utf-8")
            cfg = {
                "_root": str(root),
                "batch_id": "batch_001",
                "paths": {
                    "collector_root": str(collector),
                    "human_manifest": str(human_csv),
                    "machine_manifest": str(machine_csv),
                    "splits_dir": str(splits),
                    "audio_root": str(audio),
                    "manifest_out": str(root / "manifest.csv"),
                    "calls_out": str(root / "calls.csv"),
                    "segments_out": str(root / "segments.csv"),
                    "quality_report": str(root / "quality.md"),
                    "logs_dir": str(root / "logs"),
                },
                "selection": {
                    "exclude_locked_splits": ["test", "spectra_threshold"],
                    "quotas": {"human_pool": 12, "machine_answering": 12, "machine_live_outcome": 8},
                    "always_include_conflicts": True,
                    "live_outcomes": ["b2x_lead"],
                    "machine_outcome": "answering_machine",
                },
            }
            rows, stats = build_batch(cfg)
            dumped = json.dumps(rows)
            self.assertNotIn("excel-room", dumped)
            self.assertNotIn("external_id", dumped)
            by_id = {row["call_id"]: row for row in rows}
            self.assertNotIn(machine_id, by_id)
            self.assertEqual(by_id[human_id]["selection_reason"], "label_conflict")
            self.assertEqual(by_id[human_id]["client_channel"], "1")
            self.assertEqual(by_id[excel_id]["client_channel"], "1")
            self.assertEqual(by_id[live_id]["selection_reason"], "machine_live_outcome")
            self.assertEqual(stats["pool_excluded_locked"]["machine_answering"], 1)
            init_batch(cfg)
            store = AnnotationStore(Path(cfg["paths"]["manifest_out"]), Path(cfg["paths"]["calls_out"]), Path(cfg["paths"]["segments_out"]))
            saved = store.save(
                live_id,
                {
                    "primary_label": "unknown",
                    "label_source": "unknown_source",
                    "review_status": "needs_second_review",
                    "annotation_notes": "сомнительно",
                    "annotator_id": "toolcheck",
                },
            )
            self.assertEqual(saved["primary_label"], "unknown")
            self.assertEqual(saved["prior_sample"], "machine")
            self.assertEqual(saved["outcome"], "b2x_lead")
            template = [row for row in store.calls() if row["call_id"] == live_id and row["annotator_id"] == ""]
            self.assertEqual(template[0]["primary_label"], "unknown")

    def test_client_channel_rule(self) -> None:
        self.assertEqual(client_channel_index("excel-room", 2), 1)
        self.assertEqual(client_channel_index("room", 2), 0)
        self.assertEqual(client_channel_index("excel-room", 1), 0)


class HttpTests(unittest.TestCase):
    def test_local_page_and_channels(self) -> None:
        page = (ROOT / "src" / "annotation" / "static" / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("http://", page)
        self.assertNotIn("https://", page)
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector = root / "collector"
            audio = collector / "data" / "audio" / "b2x" / "2026-11" / "human"
            identifier = call_id("eee")
            stereo(audio / f"{identifier}.wav", 4)
            cfg = {
                "_root": str(root),
                "host": "127.0.0.1",
                "port": 0,
                "batch_id": "batch_001",
                "paths": {
                    "collector_root": str(collector),
                    "audio_root": str(audio.parent),
                    "manifest_out": str(root / "manifest.csv"),
                    "calls_out": str(root / "calls.csv"),
                    "segments_out": str(root / "segments.csv"),
                },
            }
            from src.annotation.schema import FACT_FIELDS, CALL_FIELDS, SEGMENT_FIELDS
            from src.annotation.storage import blank_annotation, write_csv

            fact = {
                "call_id": identifier,
                "source_file": f"data/audio/b2x/2026-11/human/{identifier}.wav",
                "data_source": "telephone_call",
                "campaign_id": "",
                "campaign_name": "B2X",
                "call_status": "",
                "goal_status": "undetermined",
                "outcome": "answering_machine",
                "codec": "pcm_s16le",
                "codec_source": "wav_header",
                "sample_rate": "8000",
                "duration_seconds": "0.001",
                "channel_count": "2",
                "annotated_channel": "client",
                "client_channel": "0",
                "prior_sample": "human",
                "prior_classification": "отказ_лпр",
                "locked_split": "train",
                "selection_reason": "human_pool",
                "training_role": "not_for_training",
                "batch_id": "batch_001",
            }
            # 4 frames at 8000 Hz is 0.0005 s, but the header duration used by the server is the manifest value.
            fact["duration_seconds"] = "1.000"
            write_csv(Path(cfg["paths"]["manifest_out"]), FACT_FIELDS, [fact])
            write_csv(Path(cfg["paths"]["calls_out"]), CALL_FIELDS, [blank_annotation(fact)])
            write_csv(Path(cfg["paths"]["segments_out"]), SEGMENT_FIELDS, [])
            server = make_server(cfg, "127.0.0.1", 0)
            port = server.server_address[1]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as response:
                    html = response.read().decode("utf-8")
                self.assertIn("human_recorded", html)
                request = urllib.request.Request(f"http://127.0.0.1:{port}/api/audio/{identifier}?channel=1", headers={"Range": "bytes=0-80"})
                with urllib.request.urlopen(request) as response:
                    self.assertEqual(response.status, 206)
                    body = response.read()
                self.assertTrue(body.startswith(b"RIFF") or len(body) > 0)
                left = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/audio/{identifier}?channel=0").read()
                with wave.open(__import__("io").BytesIO(left), "rb") as handle:
                    samples = struct.unpack("<hhhh", handle.readframes(4))
                self.assertEqual(samples, (0, 1, 2, 3))
                bad = json.dumps({"primary_label": "machine", "label_source": "manual_listening", "label_confidence": "high", "review_status": "reviewed"}).encode()
                try:
                    urllib.request.urlopen(urllib.request.Request(
                        f"http://127.0.0.1:{port}/api/calls/{identifier}",
                        data=bad,
                        headers={"Content-Type": "application/json"},
                    ))
                    self.fail("старая метка machine не должна сохраняться")
                except urllib.error.HTTPError as exc:
                    self.assertEqual(exc.code, 400)
            finally:
                server.shutdown()
                server.server_close()


class RealSelectionTests(unittest.TestCase):
    def test_real_batch_is_deterministic_and_unlocked(self) -> None:
        manifest = _component("voice-dataset-collector") / "data" / "manifests" / "b2x_2026-11_human.csv"
        if not manifest.exists():
            self.skipTest("локальная выгрузка B2X не найдена")
        cfg = load_config(ROOT / "config" / "config.yaml")
        first, _stats = build_batch(cfg)
        second, _stats = build_batch(cfg)
        self.assertEqual([row["call_id"] for row in first], [row["call_id"] for row in second])
        self.assertGreaterEqual(len(first), 20)
        self.assertEqual(len(first), len({row["call_id"] for row in first}))
        for row in first:
            self.assertNotIn("external_id", row)
            self.assertNotIn("s3_key", row)
            self.assertNotIn("phone_number", row)
            self.assertEqual(row["data_source"], "telephone_call")
            self.assertIn(row["client_channel"], {"0", "1"})
            self.assertTrue((_component("voice-dataset-collector") / row["source_file"]).is_file())
            if row["selection_reason"] != "label_conflict":
                self.assertNotIn(row["locked_split"], {"test", "spectra_threshold"})


if __name__ == "__main__":
    unittest.main()
