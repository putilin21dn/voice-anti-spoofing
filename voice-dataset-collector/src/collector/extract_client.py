from __future__ import annotations

import shutil
import wave
from pathlib import Path

import csv

from config.settings import Settings
from src.collector.call_selector import read_manifest
from src.collector.samples import sample_paths


SAMPLE_FIELDS = [
    "call_id",
    "sample",
    "client_channel",
    "source_channels",
    "sample_rate",
    "sample_width",
    "duration_ms",
    "wav_ok",
    "error",
    "local_path",
]


def client_channel_index(external_id: str, channels: int) -> int:
    """Канал абонента. В backend excel-записи каналы переставлены."""

    if channels <= 1:
        return 0
    if external_id.startswith("excel-"):
        return 1
    return 0


def extract_channel(source: Path, destination: Path, channel_index: int) -> dict[str, str]:
    """Копирует один канал в новый WAV. Частота, глубина и байты отсчётов не меняются."""

    with wave.open(str(source), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        payload = wav_file.readframes(frame_count)
    if channel_index >= channels:
        raise wave.Error(f"Нет канала {channel_index}, в файле {channels}")
    if channels == 1:
        mono = payload
    else:
        frame_size = sample_width * channels
        selected = bytearray()
        offset = channel_index * sample_width
        for position in range(0, len(payload) - frame_size + 1, frame_size):
            selected.extend(payload[position + offset : position + offset + sample_width])
        mono = bytes(selected)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with wave.open(str(temporary), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(sample_width)
        out.setframerate(sample_rate)
        out.writeframes(mono)
    temporary.replace(destination)
    duration_ms = (frame_count * 1000) // sample_rate if sample_rate else 0
    return {
        "source_channels": str(channels),
        "sample_rate": str(sample_rate),
        "sample_width": str(sample_width),
        "duration_ms": str(duration_ms),
        "wav_ok": "true",
        "error": "",
    }


def extract_client_samples(settings: Settings, *, analysis_type: str, month: str) -> dict[str, int]:
    stats = {"written": 0, "already_present": 0, "failed": 0, "human": 0, "machine": 0}
    rows: list[dict[str, str]] = []
    root = settings.project_root / "data" / "samples" / analysis_type / month
    for sample in ("human", "machine"):
        manifest, _audio = sample_paths(settings, analysis_type, month, sample)
        for record in read_manifest(manifest):
            call_id = record["call_id"]
            destination = root / sample / f"{call_id}.wav"
            row = {field: "" for field in SAMPLE_FIELDS}
            row["call_id"] = call_id
            row["sample"] = sample
            row["local_path"] = str(destination)
            source = Path(record["local_path"]) if record.get("local_path") else None
            if source is None or not source.is_file():
                row["wav_ok"] = "false"
                row["error"] = "Исходный WAV не найден"
                stats["failed"] += 1
                rows.append(row)
                continue
            if destination.is_file() and destination.stat().st_size > 44:
                try:
                    with wave.open(str(destination), "rb") as existing:
                        frames = existing.getnframes()
                        rate = existing.getframerate()
                        row["sample_rate"] = str(rate)
                        row["sample_width"] = str(existing.getsampwidth())
                        row["duration_ms"] = str((frames * 1000) // rate if rate else 0)
                        row["wav_ok"] = "true"
                    row["client_channel"] = str(client_channel_index(record.get("external_id") or "", 2))
                    stats["already_present"] += 1
                    stats[sample] += 1
                    rows.append(row)
                    continue
                except (wave.Error, EOFError):
                    destination.unlink(missing_ok=True)
            try:
                with wave.open(str(source), "rb") as probe:
                    channel = client_channel_index(record.get("external_id") or "", probe.getnchannels())
                extracted = extract_channel(source, destination, channel)
            except (wave.Error, EOFError, OSError) as exc:
                row["wav_ok"] = "false"
                row["error"] = f"WAV не открывается: {exc.__class__.__name__}"
                stats["failed"] += 1
                rows.append(row)
                continue
            row.update(extracted)
            row["client_channel"] = str(channel)
            stats["written"] += 1
            stats[sample] += 1
            rows.append(row)
    manifest_path = root / "samples.csv"
    write_manifest_rows(manifest_path, rows)
    stats["rows"] = len(rows)
    return stats


def even_subset(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    """Берёт count записей равномерно по уже отсортированному списку."""

    usable = [row for row in rows if row.get("wav_ok") == "true" and Path(row.get("local_path") or "").is_file()]
    if count < 1 or len(usable) <= count:
        return usable
    step = len(usable) / count
    return [usable[int(index * step)] for index in range(count)]


def copy_subset(settings: Settings, *, analysis_type: str, month: str, count: int) -> dict[str, int]:
    source_root = settings.project_root / "data" / "samples" / analysis_type / month
    manifest = source_root / "samples.csv"
    rows = []
    if manifest.exists():
        import csv

        with manifest.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    destination_root = source_root / f"n{count}"
    chosen: list[dict[str, str]] = []
    stats = {"human": 0, "machine": 0}
    for sample in ("human", "machine"):
        picked = even_subset([row for row in rows if row.get("sample") == sample], count)
        for row in picked:
            source = Path(row["local_path"])
            destination = destination_root / sample / f"{row['call_id']}.wav"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied = dict(row)
            copied["local_path"] = str(destination)
            chosen.append(copied)
            stats[sample] += 1
    write_manifest_rows(destination_root / "samples.csv", chosen)
    stats["rows"] = len(chosen)
    return stats


def write_manifest_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SAMPLE_FIELDS})
    temporary.replace(path)
