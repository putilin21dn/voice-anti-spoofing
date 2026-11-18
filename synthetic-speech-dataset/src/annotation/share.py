"""Делит партию на две непересекающиеся части и собирает архив для второго разметчика.

В архиве нет телефона, external_id, ключа S3 и оценок моделей. Метки второго
человека вливаются рядом с местными и не затирают их.
"""

from __future__ import annotations

import csv
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.annotation.schema import (
    CALL_FIELDS,
    FORBIDDEN_COLUMNS,
    SEGMENT_FIELDS,
    valid_annotator,
    valid_call_id,
    validate_annotation,
)
from src.annotation.storage import blank_annotation, read_csv, write_csv
from src.utils.config import resolve_path


SHARES = ("local", "guest")
PACK_FIELDS = [
    "call_id",
    "duration_seconds",
    "channel_count",
    "client_channel",
    "campaign_name",
]
PORTABLE = Path(__file__).resolve().parent / "portable"


def assignment_path(cfg: dict) -> Path:
    calls = resolve_path(cfg, cfg["paths"]["calls_out"])
    name = calls.name
    if not name.endswith("_calls.csv"):
        raise ValueError("Файл разметки этой партии не похож на дневной")
    return calls.with_name(name[: -len("_calls.csv")] + "_assignment.csv")


def load_assignment(path: Path) -> dict[str, str]:
    assigned: dict[str, str] = {}
    for row in read_csv(path):
        call_id = row.get("call_id", "")
        share = row.get("share", "")
        if valid_call_id(call_id) and share in SHARES:
            assigned[call_id] = share
    return assigned


def ensure_assignment(cfg: dict) -> dict[str, str]:
    """Закрепляет уже выданные звонки и раскладывает только новые."""

    manifest = _manifest(cfg)
    path = assignment_path(cfg)
    assigned = load_assignment(path)
    counts = Counter(assigned.values())
    pending = [row for row in manifest if row["call_id"] not in assigned]
    pending.sort(key=lambda row: (row.get("campaign_name", ""), _duration(row), row["call_id"]))
    for row in pending:
        share = min(SHARES, key=lambda name: (counts[name], name))
        assigned[row["call_id"]] = share
        counts[share] += 1
    rows = [{"call_id": call_id, "share": assigned[call_id]} for call_id in sorted(assigned)]
    write_csv(path, ["call_id", "share"], rows)
    return assigned


LISTENED = {"reviewed", "needs_second_review", "disputed"}
GUEST_FIELDS = [
    "call_id",
    "sort_index",
    "date",
    "campaign_name",
    "review_priority",
    "queues",
    "public_reasons",
    "duration_seconds",
    "channel_count",
    "client_channel",
    "speech_duration_s",
    "n_segments",
    "n_windows",
    "padding_ratio",
    "spectra_status",
    "classical_status",
]


def build_priority_assignment(
    queue_rows: list[dict[str, str]],
    labeled: set[str],
    manifest_ids: list[str],
    existing: dict[str, str] | None = None,
) -> dict[str, str]:
    """Уже прослушанное остаётся у local. Остаток каждой страты делится пополам.

    Страта — приоритет, день и кампания. Внутри страты более ранний звонок
    достаётся local. Уже записанное закрепление не переносится.
    """

    assigned = {call_id: share for call_id, share in (existing or {}).items() if share in SHARES}
    for call_id in labeled:
        if assigned.get(call_id) != "guest":
            assigned[call_id] = "local"
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in queue_rows:
        call_id = row.get("call_id", "")
        if not valid_call_id(call_id) or call_id in assigned:
            continue
        key = (row.get("review_priority", ""), row.get("date", ""), row.get("campaign", ""))
        groups.setdefault(key, []).append(row)
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda row: (int(row.get("sort_index") or 0), row["call_id"]))
        for index, row in enumerate(ordered):
            assigned[row["call_id"]] = "local" if index % 2 == 0 else "guest"
    for call_id in manifest_ids:
        if valid_call_id(call_id):
            assigned.setdefault(call_id, "local")
    return assigned


def _queue_rows(cfg: dict) -> list[dict[str, str]]:
    raw = (cfg.get("paths") or {}).get("queue_csv") or ""
    if not raw:
        raise ValueError("В конфиге нет очереди прослушивания")
    return read_csv(resolve_path(cfg, raw))


def _listened(cfg: dict) -> set[str]:
    path = resolve_path(cfg, cfg["paths"]["calls_out"])
    return {
        row.get("call_id", "")
        for row in read_csv(path)
        if row.get("review_status") in LISTENED and valid_call_id(row.get("call_id", ""))
    }


def assign_priority(cfg: dict) -> dict[str, str]:
    manifest = _manifest(cfg)
    path = assignment_path(cfg)
    existing = load_assignment(path) if path.is_file() else {}
    assigned = build_priority_assignment(
        _queue_rows(cfg),
        _listened(cfg),
        [row["call_id"] for row in manifest],
        existing,
    )
    rows = [{"call_id": call_id, "share": assigned[call_id]} for call_id in sorted(assigned)]
    write_csv(path, ["call_id", "share"], rows)
    return assigned


def export_share(cfg: dict, share: str = "guest") -> Path:
    if cfg.get("blind_priority"):
        return _export_priority(cfg, share)
    if share not in SHARES:
        raise ValueError("Часть: local или guest")
    assigned = ensure_assignment(cfg)
    manifest = {row["call_id"]: row for row in _manifest(cfg)}
    chosen = [manifest[call_id] for call_id in sorted(assigned) if assigned[call_id] == share and call_id in manifest]
    if not chosen:
        raise ValueError("В этой части нет звонков")
    root = Path(cfg["_root"])
    destination = root / "shares" / f"{cfg['batch_id']}_{share}.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".zip.part")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(PORTABLE / "annotate.py", "annotate.py")
        archive.write(PORTABLE / "index.html", "index.html")
        archive.writestr("README.txt", _readme(cfg["batch_id"], share, len(chosen)))
        archive.writestr("manifest.csv", _csv_text(PACK_FIELDS, [_pack_row(row) for row in chosen]))
        archive.writestr("annotations.csv", _csv_text(_portable_call_fields(), []))
        archive.writestr("segments.csv", _csv_text(SEGMENT_FIELDS, []))
        for index, row in enumerate(chosen, start=1):
            source = resolve_path(cfg, row["source_file"])
            if not source.is_file():
                raise FileNotFoundError(f"Нет WAV для {row['call_id']}")
            archive.write(source, f"audio/{row['call_id']}.wav")
            if index % 25 == 0 or index == len(chosen):
                print(f"в архив {index}/{len(chosen)}", flush=True)
    temporary.replace(destination)
    counts = Counter(row["campaign_name"] for row in chosen)
    print(f"архив: {destination}", flush=True)
    print(f"звонков: {len(chosen)}", flush=True)
    for name, count in sorted(counts.items()):
        print(f"  {count} {name}", flush=True)
    return destination


def _export_priority(cfg: dict, share: str) -> Path:
    if share not in SHARES:
        raise ValueError("Часть: local или guest")
    assigned = assign_priority(cfg)
    manifest = {row["call_id"]: row for row in _manifest(cfg)}
    chosen_ids = [call_id for call_id, owner in sorted(assigned.items()) if owner == share and call_id in manifest]
    queue = {row["call_id"]: row for row in _queue_rows(cfg)}
    chosen = [queue[call_id] for call_id in chosen_ids if call_id in queue]
    if not chosen:
        raise ValueError("В этой части нет звонков")
    packed = [_guest_row(queue_row, manifest[queue_row["call_id"]]) for queue_row in chosen]
    root = Path(cfg["_root"])
    destination = root / "shares" / f"{cfg['batch_id']}_{share}.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".zip.part")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(PORTABLE / "annotate.py", "annotate.py")
        archive.write(PORTABLE / "priority.html", "priority.html")
        archive.writestr("README.txt", _priority_readme(cfg["batch_id"], share, len(chosen)))
        archive.writestr("manifest.csv", _csv_text(GUEST_FIELDS, packed))
        archive.writestr("annotations.csv", _csv_text(_portable_call_fields(), []))
        archive.writestr("segments.csv", _csv_text(SEGMENT_FIELDS, []))
        for index, row in enumerate(chosen, start=1):
            source = resolve_path(cfg, manifest[row["call_id"]]["source_file"])
            if not source.is_file():
                raise FileNotFoundError(f"Нет WAV для {row['call_id']}")
            archive.write(source, f"audio/{row['call_id']}.wav")
            if index % 25 == 0 or index == len(chosen):
                print(f"в архив {index}/{len(chosen)}", flush=True)
    temporary.replace(destination)
    counts = Counter(row["review_priority"] for row in chosen)
    print(f"архив: {destination}", flush=True)
    print(f"звонков: {len(chosen)}", flush=True)
    for name in ("A", "C", "B", "D"):
        if counts[name]:
            print(f"  {name}: {counts[name]}", flush=True)
    return destination


def _guest_row(queue_row: dict[str, str], manifest_row: dict[str, str]) -> dict[str, str]:
    packed = {
        "call_id": queue_row["call_id"],
        "sort_index": queue_row.get("sort_index", ""),
        "date": queue_row.get("date", ""),
        "campaign_name": queue_row.get("campaign", ""),
        "review_priority": queue_row.get("review_priority", ""),
        "queues": queue_row.get("queues", ""),
        "public_reasons": queue_row.get("public_reasons", ""),
        "duration_seconds": queue_row.get("duration_seconds", ""),
        "channel_count": manifest_row.get("channel_count", ""),
        "client_channel": queue_row.get("client_channel", ""),
        "speech_duration_s": queue_row.get("spectra_speech_duration_s", ""),
        "n_segments": queue_row.get("spectra_n_segments", ""),
        "n_windows": queue_row.get("spectra_n_windows", ""),
        "padding_ratio": queue_row.get("padding_ratio", ""),
        "spectra_status": queue_row.get("spectra_status", ""),
        "classical_status": queue_row.get("classical_status", ""),
    }
    if set(packed).intersection(FORBIDDEN_COLUMNS):
        raise ValueError("В строку архива попало запрещённое поле")
    return packed


def _priority_readme(batch_id: str, share: str, count: int) -> str:
    return (
        f"Часть {share} партии {batch_id}. Звонков: {count}.\n"
        "Это только ваши звонки. Телефонов и оценок моделей в архиве нет.\n"
        "Порядок в списке: A, затем C, затем B, затем D.\n"
        "\n"
        "1. Распакуйте архив.\n"
        "2. В этой папке выполните: py -3 annotate.py\n"
        "3. Откройте http://127.0.0.1:8771\n"
        "4. Впишите своё имя латиницей в поле «Разметчик» и больше его не меняйте.\n"
        "5. Слушайте канал абонента. Голос робота на другом канале меткой не является.\n"
        "6. Когда закончите, пришлите обратно только annotations.csv и segments.csv.\n"
        "Аудио отправлять обратно не нужно. Страница слушает только этот компьютер.\n"
    )


def import_share(cfg: dict, annotations: Path, segments: Path | None, share: str = "guest") -> dict[str, int]:
    if share not in SHARES:
        raise ValueError("Часть: local или guest")
    assigned = assign_priority(cfg) if cfg.get("blind_priority") else ensure_assignment(cfg)
    allowed = {call_id for call_id, owner in assigned.items() if owner == share}
    manifest = {row["call_id"]: row for row in _manifest(cfg)}
    incoming = read_csv(annotations)
    if incoming:
        foreign = FORBIDDEN_COLUMNS.intersection(incoming[0])
        if foreign:
            raise ValueError("В файле меток есть запрещённые столбцы")
    segment_rows = read_csv(segments) if segments and segments.is_file() else []
    if segment_rows:
        foreign = FORBIDDEN_COLUMNS.intersection(segment_rows[0])
        if foreign:
            raise ValueError("В файле интервалов есть запрещённые столбцы")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in segment_rows:
        grouped.setdefault((row.get("call_id", ""), row.get("annotator_id", "")), []).append(row)

    accepted: list[tuple[dict[str, str], list[dict[str, str]]]] = []
    rejected: list[str] = []
    for row in incoming:
        call_id = row.get("call_id", "")
        annotator = row.get("annotator_id", "")
        if call_id not in allowed or call_id not in manifest:
            rejected.append(call_id or "(пустой)")
            continue
        if not annotator or not valid_annotator(annotator):
            rejected.append(call_id)
            continue
        payload = dict(row)
        payload["segments"] = grouped.get((call_id, annotator), [])
        try:
            clean = validate_annotation(payload, duration_s=float(manifest[call_id].get("duration_seconds") or 0))
        except ValueError:
            rejected.append(call_id)
            continue
        if clean["primary_label"] == "unknown" and clean["review_status"] == "unreviewed" and not clean["segments"]:
            continue
        accepted.append(({"call_id": call_id, **clean}, clean["segments"]))

    calls_path = resolve_path(cfg, cfg["paths"]["calls_out"])
    segments_path = resolve_path(cfg, cfg["paths"]["segments_out"])
    calls = read_csv(calls_path)
    stored_segments = read_csv(segments_path)
    replaced = 0
    for clean, raw_segments in accepted:
        call_id = clean["call_id"]
        annotator = clean["annotator_id"]
        before = len(calls)
        calls = [row for row in calls if not (row.get("call_id") == call_id and row.get("annotator_id", "") == annotator)]
        replaced += int(len(calls) != before)
        fact = blank_annotation(manifest[call_id])
        for key, value in clean.items():
            if key != "segments":
                fact[key] = value
        fact["updated_at"] = datetime.now(timezone.utc).isoformat()
        calls.append(fact)
        stored_segments = [
            row
            for row in stored_segments
            if not (row.get("call_id") == call_id and row.get("annotator_id", "") == annotator)
        ]
        for segment in raw_segments:
            stored_segments.append(
                {
                    "call_id": call_id,
                    "segment_id": segment.get("segment_id", ""),
                    "start_time": segment.get("start_time", ""),
                    "end_time": segment.get("end_time", ""),
                    "speaker_id": segment.get("speaker_id", ""),
                    "segment_label": segment.get("segment_label", ""),
                    "label_confidence": segment.get("label_confidence", ""),
                    "label_source": segment.get("label_source", ""),
                    "notes": segment.get("notes", ""),
                    "annotator_id": annotator,
                }
            )
    calls.sort(key=lambda row: (row.get("call_id", ""), row.get("annotator_id", "")))
    stored_segments.sort(key=lambda row: (row.get("call_id", ""), row.get("annotator_id", ""), row.get("start_time", "")))
    write_csv(calls_path, CALL_FIELDS, calls)
    write_csv(segments_path, SEGMENT_FIELDS, stored_segments)
    return {"accepted": len(accepted), "replaced": replaced, "rejected": len(rejected)}


def _manifest(cfg: dict) -> list[dict[str, str]]:
    path = resolve_path(cfg, cfg["paths"]["manifest_out"])
    rows = []
    for row in read_csv(path):
        if valid_call_id(row.get("call_id", "")) and row.get("source_file"):
            rows.append(row)
    return rows


def _duration(row: dict[str, str]) -> float:
    try:
        return float(row.get("duration_seconds") or 0)
    except ValueError:
        return 0.0


def _pack_row(row: dict[str, str]) -> dict[str, str]:
    packed = {field: row.get(field, "") for field in PACK_FIELDS}
    if set(packed).intersection(FORBIDDEN_COLUMNS):
        raise ValueError("В строку архива попало запрещённое поле")
    return packed


def _portable_call_fields() -> list[str]:
    return [
        "call_id",
        "annotator_id",
        "primary_label",
        "secondary_label",
        "label_confidence",
        "label_source",
        "tts_generator",
        "voice_conversion_system",
        "speaker_id",
        "playback_method",
        "annotation_notes",
        "review_status",
        "updated_at",
    ]


def _csv_text(fields: list[str], rows: list[dict[str, str]]) -> str:
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return buffer.getvalue()


def _readme(batch_id: str, share: str, count: int) -> str:
    return (
        f"Часть {share} партии {batch_id}. Звонков: {count}.\n"
        "Это только ваши звонки. Телефонов и оценок моделей в архиве нет.\n"
        "\n"
        "1. Распакуйте архив.\n"
        "2. В этой папке выполните: py -3 annotate.py\n"
        "3. Откройте http://127.0.0.1:8768\n"
        "4. Впишите своё имя латиницей в поле «Разметчик» и больше его не меняйте.\n"
        "5. Слушайте канал абонента. Голос нашего робота меткой не является.\n"
        "6. Когда закончите, пришлите обратно только annotations.csv и segments.csv.\n"
        "Аудио отправлять обратно не нужно. Страница слушает только этот компьютер.\n"
    )
