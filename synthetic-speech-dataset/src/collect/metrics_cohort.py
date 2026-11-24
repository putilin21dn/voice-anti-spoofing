"""Список 818 звонков, на которых считались метрики голоса A.

В файле нет телефона, external_id и ключа S3. Путь wav относительный.
"""

from __future__ import annotations

import re
from pathlib import Path


COHORT_NAME = "metrics_20261111_16.txt"
HEADER = ("call_id", "source_file", "client_channel", "dataset", "primary_label")
CALL_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
SOURCE_RE = re.compile(r"^data/audio/day_[0-9]{8}/[0-9a-f-]{36}\.wav$")
EXPECTED_CALLS = 818


def cohort_path(root: Path) -> Path:
    return root / "data" / "cohorts" / COHORT_NAME


def load_cohort(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Список когорты пуст")
    header = tuple(lines[0].split("\t"))
    if header != HEADER:
        raise ValueError("В списке когорты неожиданные колонки")
    rows = []
    seen: set[str] = set()
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) != len(HEADER):
            raise ValueError("Строка когорты разошлась с заголовком")
        row = dict(zip(HEADER, parts))
        call_id = row["call_id"]
        if not CALL_ID_RE.fullmatch(call_id):
            raise ValueError("В когорте неверный call_id")
        if call_id in seen:
            raise ValueError("В когорте повторяется call_id")
        if not SOURCE_RE.fullmatch(row["source_file"]) or not row["source_file"].endswith(f"/{call_id}.wav"):
            raise ValueError("Путь wav выходит из каталога дня")
        if row["client_channel"] not in {"0", "1"}:
            raise ValueError("Канал абонента не 0 и не 1")
        if row["dataset"] not in {"11–12", "16"}:
            raise ValueError("Неизвестная дата когорты")
        seen.add(call_id)
        rows.append(row)
    if len(rows) != EXPECTED_CALLS:
        raise ValueError(f"В когорте {len(rows)} звонков, ожидалось {EXPECTED_CALLS}")
    return rows


def wav_path(root: Path, source_file: str) -> Path:
    relative = Path(source_file)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Путь wav выходит из каталога дня")
    destination = (root / relative).resolve()
    audio_root = (root / "data" / "audio").resolve()
    if audio_root not in destination.parents:
        raise ValueError("Путь wav выходит из каталога дня")
    return destination


def pending(root: Path, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    missing = []
    for row in rows:
        path = wav_path(root, row["source_file"])
        if path.is_file() and path.stat().st_size > 44:
            continue
        missing.append(row)
    return missing
