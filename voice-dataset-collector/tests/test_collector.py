from __future__ import annotations

import os
import sys
import tempfile
import unittest
import wave
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import Settings
from src.collector.audio_downloader import download_audio, probe_wav
from src.collector.call_selector import local_audio_name, merge_rows, read_manifest, row_from_db, write_manifest
from src.db.queries import build_calls_query, recording_key
from src.storage.s3_client import ReadOnlyS3, S3StorageError


def _settings(**overrides) -> Settings:
    values = dict(
        project_root=ROOT,
        db_host="db.internal",
        db_port=5432,
        db_name="genai",
        db_user="reader",
        db_password="super-secret-value",
        db_schema="public",
        s3_endpoint_url="",
        s3_host="minio",
        s3_port=9000,
        s3_secure=False,
        s3_region="eu-central-1",
        s3_bucket="genai",
        s3_access_key_id="access",
        s3_secret_access_key="secret-value",
        s3_session_token="",
    )
    values.update(overrides)
    return Settings(**values)


def _sql_text(composed) -> str:
    chunks: list[str] = []

    def walk(node) -> None:
        seq = getattr(node, "seq", None)
        if seq is not None:
            for item in seq:
                walk(item)
            return
        string = getattr(node, "string", None)
        if string is not None:
            chunks.append(string)
            return
        strings = getattr(node, "strings", None)
        if strings is not None:
            chunks.extend(strings)
            return
        chunks.append(str(node))

    walk(composed)
    return "".join(chunks)


class RecordingKeyTests(unittest.TestCase):
    def test_key_matches_backend_and_worker(self) -> None:
        external_id = "room-123"
        self.assertEqual(recording_key(external_id), "calls/room-123/room-123.wav")

    def test_local_name_uses_call_id_only(self) -> None:
        call_id = "11111111-1111-1111-1111-111111111111"
        self.assertEqual(local_audio_name(call_id), f"{call_id}.wav")
        with self.assertRaises(ValueError):
            local_audio_name("../calls/room.wav")


class SettingsTests(unittest.TestCase):
    def test_minio_endpoint_matches_worker_shape(self) -> None:
        self.assertEqual(_settings().s3_endpoint(), "http://minio:9000")

    def test_aws_host_has_no_custom_endpoint(self) -> None:
        settings = _settings(s3_host="s3.eu-central-1.amazonaws.com", s3_port=443, s3_secure=True)
        self.assertIsNone(settings.s3_endpoint())

    def test_explicit_endpoint_wins(self) -> None:
        settings = _settings(s3_endpoint_url="https://storage.example")
        self.assertEqual(settings.s3_endpoint(), "https://storage.example")

    def test_summary_hides_secrets(self) -> None:
        summary = _settings().public_summary()
        self.assertNotIn("super-secret-value", summary)
        self.assertNotIn("secret-value", summary)
        self.assertIn("bucket=genai", summary.replace("s3_bucket", "bucket"))


class QueryTests(unittest.TestCase):
    def test_query_uses_calls_and_bound_filters(self) -> None:
        query, params = build_calls_query(
            _settings(),
            from_date=datetime(2026, 1, 1),
            to_date=datetime(2026, 2, 1),
            goal_statuses=["goal_achieved"],
            outcomes=[],
            end_reasons=[],
            min_duration_s=1,
            max_duration_s=None,
            require_audio=True,
            require_finalized=True,
            exclude_ids=[],
            limit=3,
            offset=0,
        )
        text = _sql_text(query)
        self.assertIn("FROM", text)
        self.assertIn("with_audio", text)
        self.assertIn("is_finalized", text)
        self.assertNotIn("phone", text.lower())
        self.assertEqual(params["goal_statuses"], ["goal_achieved"])
        self.assertEqual(params["limit"], 3)
        self.assertNotIn("super-secret-value", text)


class ManifestTests(unittest.TestCase):
    def test_merge_does_not_duplicate_or_reset_download(self) -> None:
        existing = [
            {
                "call_id": "11111111-1111-1111-1111-111111111111",
                "download_status": "downloaded",
                "local_path": "data/audio/11111111-1111-1111-1111-111111111111.wav",
                "wav_ok": "true",
                "file_size_bytes": "128",
                "external_id": "room",
            }
        ]
        fresh = row_from_db(
            {
                "call_id": "11111111-1111-1111-1111-111111111111",
                "created_at": datetime(2026, 1, 2, 3, 4, 5),
                "timestamp": datetime(2026, 1, 2, 3, 0, 0),
                "duration_s": 12,
                "duration_ms": 12000,
                "external_id": "room",
                "with_audio": True,
                "is_finalized": True,
                "goal_status": "undetermined",
                "outcome": None,
                "end_reason": "user_hangup",
                "disconnected_by": "client",
                "direction": "OUTBOUND",
            },
            bucket="genai",
        )
        another = dict(fresh)
        another["call_id"] = "22222222-2222-2222-2222-222222222222"
        merged, added = merge_rows(existing, [fresh, another])
        self.assertEqual(added, 1)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["download_status"], "downloaded")
        self.assertEqual(merged[0]["local_path"], existing[0]["local_path"])
        self.assertEqual(merged[0]["s3_key"], "calls/room/room.wav")
        self.assertEqual(merged[0]["duration_ms"], "12000")
        self.assertEqual(merged[1]["download_status"], "pending")

    def test_roundtrip_csv_has_no_secrets(self) -> None:
        row = row_from_db(
            {
                "call_id": "33333333-3333-3333-3333-333333333333",
                "created_at": None,
                "timestamp": None,
                "duration_s": 0,
                "duration_ms": 0,
                "external_id": "room-x",
                "with_audio": False,
                "is_finalized": False,
                "goal_status": None,
                "outcome": None,
                "end_reason": None,
                "disconnected_by": None,
                "direction": "INBOUND",
            },
            bucket="genai",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calls_manifest.csv"
            write_manifest(path, [row])
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("password", text.lower())
            self.assertNotIn("secret", text.lower())
            loaded = read_manifest(path)
        self.assertEqual(loaded[0]["s3_key"], "calls/room-x/room-x.wav")
        self.assertEqual(loaded[0]["audio_available"], "false")


class WavTests(unittest.TestCase):
    def test_probe_reads_header_without_changing_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.wav"
            with wave.open(str(path), "wb") as wav_file:
                wav_file.setnchannels(2)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(b"\x00\x00\x00\x00" * 16000)
            before = path.read_bytes()
            probed = probe_wav(path)
            self.assertEqual(path.read_bytes(), before)
        self.assertEqual(probed["wav_channels"], "2")
        self.assertEqual(probed["wav_sample_rate"], "16000")
        self.assertEqual(probed["wav_duration_ms"], "1000")
        self.assertEqual(probed["wav_ok"], "true")


class S3ErrorTests(unittest.TestCase):
    def test_missing_object_is_classified(self) -> None:
        client = ReadOnlyS3.__new__(ReadOnlyS3)
        client._bucket = "genai"
        error = ClientError({"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "HeadObject")
        with self.assertRaises(S3StorageError) as caught:
            client._raise_client(error, key="calls/room/room.wav")
        self.assertEqual(caught.exception.kind, "missing")

    def test_access_denied_is_auth(self) -> None:
        client = ReadOnlyS3.__new__(ReadOnlyS3)
        client._bucket = "genai"
        error = ClientError({"Error": {"Code": "AccessDenied", "Message": "no"}}, "HeadObject")
        with self.assertRaises(S3StorageError) as caught:
            client._raise_client(error, key="calls/room/room.wav")
        self.assertEqual(caught.exception.kind, "auth_error")
        self.assertNotIn("secret", str(caught.exception).lower())


class DownloadSkipTests(unittest.TestCase):
    def test_complete_file_is_not_downloaded_again(self) -> None:
        call_id = "44444444-4444-4444-4444-444444444444"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "data" / "audio"
            audio.mkdir(parents=True)
            manifests = root / "data" / "manifests"
            manifests.mkdir(parents=True)
            destination = audio / f"{call_id}.wav"
            with wave.open(str(destination), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(8000)
                wav_file.writeframes(b"\x01\x00" * 800)
            row = row_from_db(
                {
                    "call_id": call_id,
                    "created_at": datetime(2026, 3, 1, 0, 0, 0),
                    "timestamp": datetime(2026, 3, 1, 0, 0, 0),
                    "duration_s": 1,
                    "duration_ms": 1000,
                    "external_id": "room-4",
                    "with_audio": True,
                    "is_finalized": True,
                    "goal_status": "undetermined",
                    "outcome": None,
                    "end_reason": None,
                    "disconnected_by": None,
                    "direction": "OUTBOUND",
                },
                bucket="genai",
            )
            row["download_status"] = "pending"
            write_manifest(manifests / "calls_manifest.csv", [row])
            settings = _settings(project_root=root)

            class Storage:
                bucket = "genai"

                def head(self, key: str):
                    self.key = key
                    return type("Info", (), {"bucket": "genai", "key": key, "size": destination.stat().st_size, "content_type": "audio/wav"})()

                def download(self, key: str, path: Path) -> None:
                    raise AssertionError("повторное скачивание не нужно")

            stats = download_audio(
                settings,
                limit=1,
                offset=0,
                goal_statuses=[],
                from_date="",
                to_date="",
                dry_run=False,
                retry_missing=False,
                storage=Storage(),
            )
            self.assertEqual(stats["downloaded"], 0)
            self.assertEqual(stats["already_present"], 1)
            self.assertEqual(stats["wav_ok"], 1)
            saved = read_manifest(settings.manifest_path)
            self.assertEqual(saved[0]["download_status"], "already_present")
            self.assertEqual(saved[0]["s3_key"], "calls/room-4/room-4.wav")


class SampleAssignmentTests(unittest.TestCase):
    def test_machine_wins_over_human_label(self) -> None:
        from src.collector.samples import assign_sample

        self.assertEqual(
            assign_sample(
                outcome=None,
                classifications=["лид", "автоответчик"],
                has_answering_machine_text=False,
            ),
            "machine",
        )

    def test_human_dialogue_label(self) -> None:
        from src.collector.samples import assign_sample

        self.assertEqual(
            assign_sample(outcome="b2x_lead", classifications=["отказ_лпр"], has_answering_machine_text=False),
            "human",
        )

    def test_silence_is_neither_sample(self) -> None:
        from src.collector.samples import assign_sample

        self.assertIsNone(
            assign_sample(outcome=None, classifications=["абонент_молчит"], has_answering_machine_text=False)
        )

    def test_manual_human_overrides_answering_machine(self) -> None:
        from src.collector.samples import apply_manual_override

        self.assertEqual(
            apply_manual_override("machine", "call-1", set(), {"call-1"}),
            "human",
        )
        self.assertEqual(
            apply_manual_override("human", "call-1", {"call-1"}, {"call-1"}),
            "machine",
        )

    def test_november_bounds_use_moscow(self) -> None:
        from src.collector.samples import month_bounds

        start, end = month_bounds("2026-11")
        self.assertEqual(start.isoformat(), "2026-11-01T00:00:00+03:00")
        self.assertEqual(end.isoformat(), "2026-12-01T00:00:00+03:00")


class ClientChannelTests(unittest.TestCase):
    def test_excel_recording_uses_second_channel(self) -> None:
        from src.collector.extract_client import client_channel_index, extract_channel

        self.assertEqual(client_channel_index("room-1", 2), 0)
        self.assertEqual(client_channel_index("excel-room", 2), 1)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "stereo.wav"
            with wave.open(str(source), "wb") as wav_file:
                wav_file.setnchannels(2)
                wav_file.setsampwidth(2)
                wav_file.setframerate(8000)
                frames = bytearray()
                for index in range(4):
                    frames.extend((index).to_bytes(2, "little"))
                    frames.extend((100 + index).to_bytes(2, "little"))
                wav_file.writeframes(bytes(frames))
            destination = Path(directory) / "client.wav"
            extract_channel(source, destination, 0)
            with wave.open(str(destination), "rb") as wav_file:
                self.assertEqual(wav_file.getnchannels(), 1)
                self.assertEqual(wav_file.getframerate(), 8000)
                payload = wav_file.readframes(wav_file.getnframes())
            self.assertEqual(payload, bytes(frames[0:2] + frames[4:6] + frames[8:10] + frames[12:14]))


class EvenSubsetTests(unittest.TestCase):
    def test_takes_one_hundred_spread_across_the_list(self) -> None:
        from src.collector.extract_client import even_subset

        rows = [{"wav_ok": "true", "local_path": __file__, "call_id": str(index)} for index in range(1000)]
        picked = even_subset(rows, 100)
        self.assertEqual(len(picked), 100)
        self.assertEqual(picked[0]["call_id"], "0")
        self.assertEqual(picked[-1]["call_id"], "990")


class LoadSettingsTests(unittest.TestCase):
    def test_schema_rejects_injection(self) -> None:
        from config.settings import load_settings

        with patch.dict(os.environ, {"DB_SCHEMA": "public;drop"}, clear=False):
            with self.assertRaises(ValueError):
                load_settings()


if __name__ == "__main__":
    unittest.main()
