"""Скачивает записи B2X за московский день и оставляет те, что не короче порога.

external_id живёт только в памяти на время запроса к S3. В manifest, лог и исключение
он не попадает: в ключе S3 есть номер телефона.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]


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


COLLECTOR = _component("voice-dataset-collector")


def _load_identity():
    import importlib.util

    path = Path(__file__).resolve().parent / "identity.py"
    spec = importlib.util.spec_from_file_location("day_identity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_identity = _load_identity()
client_channel = _identity.client_channel
room_suffix = _identity.room_suffix

if "src" in sys.modules and not hasattr(sys.modules["src"], "db"):
    os_exec = __import__("os").execv
    launcher = ROOT / "fetch_day.py"
    os_exec(sys.executable, [sys.executable, str(launcher), *sys.argv[1:]])

sys.path.insert(0, str(COLLECTOR))
from config.settings import load_settings  # noqa: E402
from src.db.connection import connect  # noqa: E402
from src.db.queries import recording_key  # noqa: E402
from src.storage.s3_client import ReadOnlyS3, S3StorageError  # noqa: E402


MIN_DURATION_S = 15
DEFAULT_BATCH_ID = "day_20261113"
DEFAULT_DAY = "2026-11-13"
FORBIDDEN = {"phone", "phone_number", "external_id", "s3_key", "s3_bucket", "msisdn"}
FIELDS = [
    "call_id",
    "source_file",
    "data_source",
    "campaign_id",
    "campaign_name",
    "call_status",
    "goal_status",
    "outcome",
    "codec",
    "codec_source",
    "sample_rate",
    "duration_seconds",
    "channel_count",
    "annotated_channel",
    "client_channel",
    "prior_sample",
    "prior_classification",
    "locked_split",
    "selection_reason",
    "training_role",
    "batch_id",
    "timestamp",
    "direction",
    "room_suffix",
]


def probe_wav(path: Path) -> dict[str, str] | None:
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            frames = handle.getnframes()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            compression = handle.getcomptype()
    except (wave.Error, EOFError, OSError):
        return None
    if rate <= 0 or channels <= 0 or frames <= 0 or width != 2 or compression != "NONE":
        return None
    return {
        "sample_rate": str(rate),
        "channel_count": str(channels),
        "duration_seconds": f"{frames / rate:.3f}",
        "codec": "pcm_s16le",
        "codec_source": "wav_header",
    }


def _query(connection, day: datetime) -> list[dict]:
    moscow = ZoneInfo("Europe/Moscow")
    start = datetime(day.year, day.month, day.day, tzinfo=moscow)
    end = start + timedelta(days=1)
    pack1 = f"B2X-{start.strftime('%d-%m-%Y')}-PACK1"
    pack2 = f"B2X-{start.strftime('%d-%m-%Y')}-PACK2"
    print(f"кампании: {pack1}, {pack2}", flush=True)
    sql = """
        SELECT
            c.id::text AS call_id,
            c.timestamp,
            c.duration AS duration_s,
            c.external_id,
            c.goal_status,
            c.outcome,
            c.direction::text AS direction,
            coalesce(
                array_agg(DISTINCT r.classification_result) FILTER (WHERE r.classification_result IS NOT NULL),
                ARRAY[]::text[]
            ) AS classifications,
            coalesce(
                array_agg(DISTINCT camp.name) FILTER (WHERE camp.name IS NOT NULL),
                ARRAY[]::text[]
            ) AS campaign_names
        FROM calls AS c
        JOIN call_campaign_results AS r ON r.call_id = c.id
        JOIN call_campaigns AS camp ON camp.id = r.call_campaign_id
        WHERE camp.analysis_type = 'b2x'
          AND c.with_audio IS TRUE
          AND c.is_finalized IS TRUE
          AND c.direction::text = 'OUTBOUND'
          AND c.timestamp >= %(start)s
          AND c.timestamp < %(end)s
          AND c.duration >= %(min_duration)s
          AND camp.name IN (%(pack1)s, %(pack2)s)
        GROUP BY c.id
        ORDER BY min(c.timestamp), c.id
    """
    rows = connection.execute(
        sql,
        {"start": start, "end": end, "min_duration": MIN_DURATION_S, "pack1": pack1, "pack2": pack2},
    ).fetchall()
    items = []
    for call_id, timestamp, duration_s, external_id, goal_status, outcome, direction, classifications, names in rows:
        items.append(
            {
                "call_id": str(call_id),
                "timestamp": timestamp.isoformat() if timestamp is not None else "",
                "duration_s": int(duration_s or 0),
                "external_id": str(external_id or ""),
                "goal_status": goal_status or "",
                "outcome": outcome or "",
                "direction": direction or "",
                "classifications": [item for item in (classifications or []) if item],
                "campaign_names": [item for item in (names or []) if item],
            }
        )
    return items


def _materialize(storage: ReadOnlyS3, row: dict, audio_dir: Path, confirmed_dir: Path, batch_id: str) -> dict:
    call_id = row["call_id"]
    external_id = row["external_id"]
    destination = audio_dir / f"{call_id}.wav"
    copied = False
    try:
        existing = confirmed_dir / f"{call_id}.wav"
        if existing.is_file() and not destination.is_file():
            shutil.copyfile(existing, destination)
            copied = True
        elif not destination.is_file() or destination.stat().st_size <= 44:
            storage.download(recording_key(external_id), destination)
        channel = client_channel(external_id, 2)
        suffix = room_suffix(external_id)
    except S3StorageError as exc:
        return {"call_id": call_id, "error": exc.kind}
    except OSError:
        return {"call_id": call_id, "error": "write_error"}
    probed = probe_wav(destination)
    if probed is None:
        return {"call_id": call_id, "error": "invalid_wav"}
    if float(probed["duration_seconds"]) < MIN_DURATION_S:
        return {"call_id": call_id, "error": "short_wav"}
    channels = int(probed["channel_count"])
    if int(channel) >= channels:
        channel = "0" if channels == 1 else ""
    if channel == "":
        return {"call_id": call_id, "error": "channel_uncertain"}
    source = f"data/audio/{batch_id}/{call_id}.wav"
    return {
        "call_id": call_id,
        "source_file": source,
        "data_source": "telephone_call",
        "campaign_id": "",
        "campaign_name": "|".join(row["campaign_names"]),
        "call_status": "",
        "goal_status": row["goal_status"],
        "outcome": row["outcome"],
        "codec": probed["codec"],
        "codec_source": probed["codec_source"],
        "sample_rate": probed["sample_rate"],
        "duration_seconds": probed["duration_seconds"],
        "channel_count": probed["channel_count"],
        "annotated_channel": "client",
        "client_channel": channel,
        "prior_sample": "",
        "prior_classification": "|".join(row["classifications"]),
        "locked_split": "",
        "selection_reason": "day_audio",
        "training_role": "awaiting_review",
        "batch_id": batch_id,
        "timestamp": row["timestamp"],
        "direction": row["direction"],
        "room_suffix": suffix,
        "error": "",
        "copied": "yes" if copied else "",
    }


def _write(path: Path, rows: list[dict]) -> None:
    leaked = FORBIDDEN & set(FIELDS)
    if leaked:
        raise RuntimeError("В manifest попали запрещённые колонки")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", default=DEFAULT_DAY, help="Московская дата YYYY-MM-DD")
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    args = parser.parse_args()
    day = datetime.strptime(args.day, "%Y-%m-%d")
    batch_id = args.batch_id
    if batch_id == "day_20261113" and args.day != "2026-11-13":
        raise SystemExit("Партию 13 ноября другой датой не перезаписываем")
    settings = load_settings()
    audio_dir = ROOT / "data" / "audio" / batch_id
    audio_dir.mkdir(parents=True, exist_ok=True)
    confirmed_dir = ROOT / "data" / "audio" / "confirmed_001"
    with connect(settings) as connection:
        rows = _query(connection, day)
    print(f"к скачиванию, длительность базы >= {MIN_DURATION_S} с: {len(rows)}", flush=True)
    storage = ReadOnlyS3(settings)
    kept: list[dict] = []
    failures: dict[str, int] = {}
    copied = 0
    done = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_materialize, storage, row, audio_dir, confirmed_dir, batch_id) for row in rows]
        for future in as_completed(futures):
            done += 1
            result = future.result()
            error = result.get("error") or ""
            if error:
                failures[error] = failures.get(error, 0) + 1
            else:
                if result.get("copied") == "yes":
                    copied += 1
                kept.append(result)
            if done % 25 == 0 or done == len(rows):
                print(f"обработано {done}/{len(rows)}, в партии {len(kept)}", flush=True)
            if kept and len(kept) % 20 == 0:
                _write(ROOT / "data" / "manifests" / f"{batch_id}.csv", kept)
    kept.sort(key=lambda item: (item["timestamp"], item["call_id"]))
    manifest = ROOT / "data" / "manifests" / f"{batch_id}.csv"
    _write(manifest, kept)
    print(f"партия: {len(kept)}", flush=True)
    print(f"уже были локально и скопированы: {copied}", flush=True)
    print(f"отбраковано: {failures or {}}", flush=True)
    print(f"manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
