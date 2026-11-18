from __future__ import annotations

import wave
from pathlib import Path

from config.settings import Settings
from src.collector.call_selector import (
    DONE_STATUSES,
    local_audio_name,
    read_manifest,
    write_manifest,
)
from src.storage.s3_client import ReadOnlyS3, S3StorageError


def probe_wav(path: Path) -> dict[str, str]:
    """Читает заголовок WAV и не меняет байты файла."""

    size = path.stat().st_size
    try:
        with wave.open(str(path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            frames = wav_file.getnframes()
            sample_width = wav_file.getsampwidth()
    except (wave.Error, EOFError) as exc:
        return {
            "file_size_bytes": str(size),
            "wav_channels": "",
            "wav_sample_rate": "",
            "wav_duration_ms": "",
            "wav_sample_width": "",
            "wav_ok": "false",
            "download_error": f"WAV не открывается: {exc.__class__.__name__}",
        }
    duration_ms = (frames * 1000) // sample_rate if sample_rate else 0
    return {
        "file_size_bytes": str(size),
        "wav_channels": str(channels),
        "wav_sample_rate": str(sample_rate),
        "wav_duration_ms": str(duration_ms),
        "wav_sample_width": str(sample_width),
        "wav_ok": "true",
        "download_error": "",
    }


def _matches(row: dict[str, str], *, goal_statuses: list[str], from_date: str, to_date: str) -> bool:
    if goal_statuses and row.get("goal_status", "") not in goal_statuses:
        return False
    created = row.get("created_at", "")
    if from_date and created < from_date:
        return False
    if to_date and created >= to_date:
        return False
    return True


def _file_ready(row: dict[str, str]) -> bool:
    """Локальный файл уже проверен: он на месте, размер совпадает с manifest, WAV открывался."""

    raw_path = row.get("local_path") or ""
    if not raw_path or row.get("wav_ok") == "false":
        return False
    path = Path(raw_path)
    if not path.is_file():
        return False
    size = path.stat().st_size
    if size <= 44:
        return False
    recorded = row.get("file_size_bytes") or ""
    return not recorded or recorded == str(size)


def _needs_transfer(row: dict[str, str], *, retry_missing: bool) -> bool:
    status = row.get("download_status", "")
    if status in DONE_STATUSES and _file_ready(row):
        return False
    if status in {"missing", "invalid_wav"} and not retry_missing:
        return False
    if status == "skipped_no_key":
        return False
    return True


def download_audio(
    settings: Settings,
    *,
    limit: int,
    offset: int,
    goal_statuses: list[str],
    from_date: str,
    to_date: str,
    dry_run: bool,
    retry_missing: bool,
    storage: ReadOnlyS3 | None = None,
    manifest_path: Path | None = None,
    audio_dir: Path | None = None,
) -> dict[str, int]:
    manifest_path = manifest_path or settings.manifest_path
    audio_dir = audio_dir or settings.audio_dir
    rows = read_manifest(manifest_path)
    stats = {
        "manifest_rows": len(rows),
        "selected": 0,
        "s3_exists": 0,
        "downloaded": 0,
        "already_present": 0,
        "missing": 0,
        "connection_errors": 0,
        "download_errors": 0,
        "wav_ok": 0,
        "auth_errors": 0,
    }
    client = storage
    seen_actionable = 0
    attempted = 0
    changed = False

    for row in rows:
        if not _matches(row, goal_statuses=goal_statuses, from_date=from_date, to_date=to_date):
            continue
        if not _needs_transfer(row, retry_missing=retry_missing):
            if row.get("download_status") in DONE_STATUSES:
                stats["already_present"] += 1
            elif row.get("download_status") == "missing":
                stats["missing"] += 1
            continue
        if seen_actionable < offset:
            seen_actionable += 1
            continue
        if attempted >= limit:
            break
        seen_actionable += 1
        attempted += 1
        stats["selected"] += 1

        key = row.get("s3_key", "")
        if not key:
            row["download_status"] = "skipped_no_key"
            row["download_error"] = "У звонка пустой external_id, ключ S3 не строится"
            changed = True
            stats["download_errors"] += 1
            continue

        try:
            name = local_audio_name(row["call_id"])
        except ValueError as exc:
            row["download_status"] = "download_error"
            row["download_error"] = str(exc)
            changed = True
            stats["download_errors"] += 1
            continue

        destination = audio_dir / name
        if client is None:
            client = ReadOnlyS3(settings)

        try:
            info = client.head(key)
        except S3StorageError as exc:
            _mark_storage_error(row, exc, stats)
            changed = True
            if exc.kind in {"auth_error", "network_error"}:
                break
            continue

        stats["s3_exists"] += 1
        row["s3_bucket"] = info.bucket
        row["s3_object_status"] = "exists"
        if _local_is_complete(destination, info.size):
            probe = probe_wav(destination)
            row.update(probe)
            row["local_path"] = str(destination)
            row["download_status"] = "already_present"
            row["audio_available"] = "true"
            stats["already_present"] += 1
            if probe.get("wav_ok") == "true":
                stats["wav_ok"] += 1
            else:
                row["download_status"] = "invalid_wav"
            changed = True
            continue

        if dry_run:
            row["s3_object_status"] = "exists"
            continue

        try:
            client.download(key, destination)
        except S3StorageError as exc:
            _mark_storage_error(row, exc, stats)
            changed = True
            if exc.kind in {"auth_error", "network_error"}:
                break
            continue

        probe = probe_wav(destination)
        row.update(probe)
        row["local_path"] = str(destination)
        row["audio_available"] = "true"
        if probe.get("wav_ok") == "true":
            row["download_status"] = "downloaded"
            stats["downloaded"] += 1
            stats["wav_ok"] += 1
        else:
            row["download_status"] = "invalid_wav"
            stats["download_errors"] += 1
        changed = True

    if changed and not dry_run:
        write_manifest(manifest_path, rows)
    return stats


def _local_is_complete(path: Path, remote_size: int) -> bool:
    if not path.is_file():
        return False
    size = path.stat().st_size
    if size <= 44:
        return False
    if remote_size and size != remote_size:
        return False
    return True


def _mark_storage_error(row: dict[str, str], exc: S3StorageError, stats: dict[str, int]) -> None:
    row["download_error"] = str(exc)
    if exc.kind == "missing":
        row["download_status"] = "missing"
        row["s3_object_status"] = "missing"
        stats["missing"] += 1
        return
    if exc.kind == "auth_error":
        row["download_status"] = "auth_error"
        stats["auth_errors"] += 1
        return
    if exc.kind == "network_error":
        row["download_status"] = "network_error"
        stats["connection_errors"] += 1
        return
    row["download_status"] = "download_error"
    stats["download_errors"] += 1
