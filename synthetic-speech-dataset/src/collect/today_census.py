"""Считает сегодняшние звонки B2X. Телефон, external_id и ключ S3 не печатает."""

from __future__ import annotations

import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

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
sys.path.insert(0, str(COLLECTOR))

from config.settings import load_settings  # noqa: E402
from src.db.connection import connect  # noqa: E402


def main() -> None:
    moscow = ZoneInfo("Europe/Moscow")
    start = datetime(2026, 11, 13, tzinfo=moscow)
    end = datetime(2026, 11, 14, tzinfo=moscow)
    settings = load_settings()
    sql = """
        SELECT
            c.id::text AS call_id,
            c.duration AS duration_s,
            c.direction::text AS direction,
            c.with_audio,
            c.is_finalized,
            coalesce(array_agg(DISTINCT camp.name) FILTER (WHERE camp.name IS NOT NULL), ARRAY[]::text[]) AS campaigns
        FROM calls AS c
        LEFT JOIN call_campaign_results AS r ON r.call_id = c.id
        LEFT JOIN call_campaigns AS camp ON camp.id = r.call_campaign_id
        WHERE c.timestamp >= %(start)s
          AND c.timestamp < %(end)s
        GROUP BY c.id
    """
    with connect(settings) as connection:
        rows = connection.execute(sql, {"start": start, "end": end}).fetchall()
    print(f"звонков: {len(rows)}")
    by_campaign: Counter[str] = Counter()
    durations: list[int] = []
    for _call_id, duration_s, direction, with_audio, is_finalized, campaigns in rows:
        names = list(campaigns or []) or ["(без кампании)"]
        for name in names:
            by_campaign[name] += 1
        if with_audio and is_finalized and direction == "OUTBOUND":
            durations.append(int(duration_s or 0))
    print("кампании:")
    for name, count in by_campaign.most_common():
        print(f"  {count} {name}")
    print(f"outbound с аудио и finalized: {len(durations)}")
    if not durations:
        return
    buckets = Counter()
    for value in durations:
        if value < 5:
            buckets["<5"] += 1
        elif value < 10:
            buckets["5-9"] += 1
        elif value < 15:
            buckets["10-14"] += 1
        elif value < 20:
            buckets["15-19"] += 1
        elif value < 30:
            buckets["20-29"] += 1
        elif value < 60:
            buckets["30-59"] += 1
        else:
            buckets[">=60"] += 1
    print("длительность секунд, outbound audio finalized:")
    for key in ("<5", "5-9", "10-14", "15-19", "20-29", "30-59", ">=60"):
        print(f"  {key}: {buckets[key]}")
    print(f"min {min(durations)} max {max(durations)} n {len(durations)}")


if __name__ == "__main__":
    main()
