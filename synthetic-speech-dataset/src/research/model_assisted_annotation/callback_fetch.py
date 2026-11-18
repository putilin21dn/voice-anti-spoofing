"""Скачивает B2X-16-11-2026-callback длиннее 15 с. Телефон и ключ S3 на диск не пишет."""

from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / "src" / "collect" / "day_audio.py"


def _day_audio():
    spec = importlib.util.spec_from_file_location("day_audio_callback", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    audio = _day_audio()
    settings = audio.load_settings()
    batch_id = "callback_20261116"
    audio_dir = ROOT / "data" / "audio" / batch_id
    audio_dir.mkdir(parents=True, exist_ok=True)
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
          AND camp.name = 'B2X-16-11-2026-callback'
          AND c.with_audio IS TRUE
          AND c.is_finalized IS TRUE
          AND c.direction::text = 'OUTBOUND'
          AND c.duration >= %(min_duration)s
        GROUP BY c.id
        ORDER BY min(c.timestamp), c.id
    """
    with audio.connect(settings) as connection:
        fetched = connection.execute(sql, {"min_duration": audio.MIN_DURATION_S}).fetchall()
    rows = []
    for call_id, timestamp, duration_s, external_id, goal_status, outcome, direction, classifications, names in fetched:
        rows.append(
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
    print(f"к скачиванию: {len(rows)}", flush=True)
    storage = audio.ReadOnlyS3(settings)
    confirmed = ROOT / "data" / "audio" / "confirmed_001"
    kept = []
    failures: dict[str, int] = {}
    for index, row in enumerate(rows, start=1):
        result = audio._materialize(storage, row, audio_dir, confirmed, batch_id)
        error = result.get("error") or ""
        if error:
            failures[error] = failures.get(error, 0) + 1
        else:
            kept.append(result)
        if index == len(rows) or index % 10 == 0:
            print(f"обработано {index}/{len(rows)}, в партии {len(kept)}", flush=True)
    kept.sort(key=lambda item: (item["timestamp"], item["call_id"]))
    manifest = ROOT / "data" / "manifests" / f"{batch_id}.csv"
    audio._write(manifest, kept)
    print(f"партия: {len(kept)}", flush=True)
    print(f"отбраковано: {failures or {}}", flush=True)
    print(f"manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
