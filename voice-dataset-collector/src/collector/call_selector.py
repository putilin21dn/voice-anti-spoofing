from __future__ import annotations

import csv
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from config.settings import Settings
from src.db.connection import connect
from src.db.queries import fetch_calls, recording_key


MANIFEST_FIELDS = [
    "call_id",
    "created_at",
    "timestamp",
    "goal_status",
    "outcome",
    "end_reason",
    "is_finalized",
    "disconnected_by",
    "direction",
    "duration_s",
    "duration_ms",
    "external_id",
    "with_audio",
    "s3_bucket",
    "s3_key",
    "audio_available",
    "download_status",
    "download_error",
    "local_path",
    "file_size_bytes",
    "wav_channels",
    "wav_sample_rate",
    "wav_duration_ms",
    "wav_sample_width",
    "wav_ok",
    "s3_object_status",
    "sample",
    "analysis_type",
    "classification_result",
    "campaign_name",
]

PRESERVE_ON_REEXPORT = [
    "download_status",
    "download_error",
    "local_path",
    "file_size_bytes",
    "wav_channels",
    "wav_sample_rate",
    "wav_duration_ms",
    "wav_sample_width",
    "wav_ok",
    "s3_object_status",
]

CALL_ID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
DONE_STATUSES = {"downloaded", "already_present"}


def local_audio_name(call_id: str) -> str:
    if not CALL_ID_RE.fullmatch(call_id):
        raise ValueError(f"Небезопасный call_id: {call_id!r}")
    return f"{call_id}.wav"


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for raw in reader:
            row = {field: (raw.get(field) or "") for field in MANIFEST_FIELDS}
            if row["call_id"]:
                rows.append(row)
        return rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})
    temporary.replace(path)


def row_from_db(record: dict[str, Any], *, bucket: str) -> dict[str, str]:
    external_id = str(record.get("external_id") or "")
    with_audio = bool(record.get("with_audio"))
    key = recording_key(external_id) if external_id else ""
    duration_s = record.get("duration_s")
    duration_ms = record.get("duration_ms")
    if duration_ms is None and duration_s is not None:
        duration_ms = int(duration_s) * 1000
    row = {field: "" for field in MANIFEST_FIELDS}
    row.update(
        {
            "call_id": _cell(record.get("call_id")),
            "created_at": _cell(record.get("created_at")),
            "timestamp": _cell(record.get("timestamp")),
            "goal_status": _cell(record.get("goal_status")),
            "outcome": _cell(record.get("outcome")),
            "end_reason": _cell(record.get("end_reason")),
            "is_finalized": _cell(record.get("is_finalized")),
            "disconnected_by": _cell(record.get("disconnected_by")),
            "direction": _cell(record.get("direction")),
            "duration_s": _cell(duration_s),
            "duration_ms": _cell(duration_ms),
            "external_id": external_id,
            "with_audio": _cell(with_audio),
            "s3_bucket": bucket,
            "s3_key": key,
            "audio_available": _cell(with_audio),
            "download_status": "pending" if key else "skipped_no_key",
        }
    )
    return row


def merge_rows(existing: list[dict[str, str]], fresh: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    """Добавляет новые call_id и обновляет поля БД, не затирая статус скачивания."""

    by_id = {row["call_id"]: dict(row) for row in existing}
    added = 0
    order = [row["call_id"] for row in existing]
    for row in fresh:
        current = by_id.get(row["call_id"])
        if current is None:
            by_id[row["call_id"]] = row
            order.append(row["call_id"])
            added += 1
            continue
        preserved = {field: current.get(field, "") for field in PRESERVE_ON_REEXPORT}
        current.update(row)
        if preserved.get("download_status"):
            current.update(preserved)
        by_id[row["call_id"]] = current
    return [by_id[call_id] for call_id in order], added


def parse_day(value: str, *, end: bool) -> datetime:
    if len(value) == 10:
        day = datetime.fromisoformat(value)
        if end:
            return day + timedelta(days=1)
        return day
    parsed = datetime.fromisoformat(value)
    if end and parsed.time() == datetime.min.time() and "T" not in value:
        return parsed + timedelta(days=1)
    return parsed


def split_csv_arg(values: list[str] | None) -> list[str]:
    if not values:
        return []
    items: list[str] = []
    for value in values:
        items.extend(part.strip() for part in value.split(",") if part.strip())
    return items


def export_calls(
    settings: Settings,
    *,
    from_date: datetime | None,
    to_date: datetime | None,
    goal_statuses: list[str],
    outcomes: list[str],
    end_reasons: list[str],
    min_duration_s: int | None,
    max_duration_s: int | None,
    require_audio: bool,
    require_finalized: bool,
    skip_exported: bool,
    limit: int,
    offset: int,
    dry_run: bool,
) -> dict[str, int]:
    existing = read_manifest(settings.manifest_path)
    exclude_ids = [row["call_id"] for row in existing] if skip_exported else []
    with connect(settings) as connection:
        records = fetch_calls(
            connection,
            settings,
            from_date=from_date,
            to_date=to_date,
            goal_statuses=goal_statuses,
            outcomes=outcomes,
            end_reasons=end_reasons,
            min_duration_s=min_duration_s,
            max_duration_s=max_duration_s,
            require_audio=require_audio,
            require_finalized=require_finalized,
            exclude_ids=exclude_ids,
            limit=limit,
            offset=offset,
        )
    fresh = [row_from_db(record, bucket=settings.s3_bucket) for record in records]
    merged, added = merge_rows(existing, fresh)
    if not dry_run:
        write_manifest(settings.manifest_path, merged)
    return {
        "found": len(records),
        "selected": len(fresh),
        "added": 0 if dry_run else added,
        "already_in_manifest": len(existing),
        "manifest_rows": len(existing) if dry_run else len(merged),
    }
