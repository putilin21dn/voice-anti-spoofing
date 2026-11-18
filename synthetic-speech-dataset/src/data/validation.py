from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.annotation.schema import (
    FORBIDDEN_COLUMNS,
    PRIMARY_LABELS,
    valid_call_id,
)
from src.annotation.storage import AnnotationStore
from src.audio.loader import probe_wav
from src.audio.preview import resolve_source
from src.utils.config import resolve_path


def _headers(path: Path) -> list[str]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = handle.readline().strip()
    return [item for item in header.split(",") if item]


def validate_dataset(cfg: dict) -> dict:
    manifest_path = resolve_path(cfg, cfg["paths"]["manifest_out"])
    calls_path = resolve_path(cfg, cfg["paths"]["calls_out"])
    segments_path = resolve_path(cfg, cfg["paths"]["segments_out"])
    store = AnnotationStore(manifest_path, calls_path, segments_path)
    errors: list[str] = []
    findings: list[str] = []
    manifest = store.manifest()
    calls = store.calls()
    segments = store.segments()

    for path in (manifest_path, calls_path, segments_path):
        forbidden = sorted(set(_headers(path)) & FORBIDDEN_COLUMNS)
        if forbidden:
            errors.append(f"{path.name}: запрещённые колонки {forbidden}")

    ids = [row["call_id"] for row in manifest]
    if len(ids) != len(set(ids)):
        errors.append("В manifest есть повторяющийся call_id")
    if not ids:
        errors.append("Manifest пуст")

    by_call = {row["call_id"]: row for row in manifest}
    template = {(row["call_id"], row.get("annotator_id", "")): row for row in calls}
    for call_id, fact in by_call.items():
        if not valid_call_id(call_id):
            errors.append(f"Некорректный call_id: {call_id}")
            continue
        try:
            path = resolve_source(cfg, fact.get("source_file", ""))
        except (PermissionError, FileNotFoundError):
            errors.append(f"{call_id}: исходный WAV недоступен")
            continue
        probed = probe_wav(path)
        if probed is None:
            errors.append(f"{call_id}: WAV не открывается")
            continue
        for field in ("sample_rate", "channel_count", "duration_seconds", "codec"):
            if (fact.get(field) or "") != probed[field]:
                errors.append(f"{call_id}: {field} в manifest не совпадает с WAV")
        channels = int(probed["channel_count"])
        client = fact.get("client_channel") or ""
        if client not in {str(index) for index in range(channels)}:
            errors.append(f"{call_id}: client_channel вне диапазона")
        if fact.get("annotated_channel") != "client":
            errors.append(f"{call_id}: размечается не канал абонента")
        if fact.get("data_source") != "telephone_call":
            errors.append(f"{call_id}: data_source не telephone_call")
        if fact.get("prior_sample") not in {"human", "machine"}:
            errors.append(f"{call_id}: нет прежней метки кампании")
        if not fact.get("campaign_id"):
            pass
        row = template.get((call_id, ""))
        if row is None:
            errors.append(f"{call_id}: нет строчки разметки")
            continue
        primary = row.get("primary_label") or ""
        if primary not in PRIMARY_LABELS:
            errors.append(f"{call_id}: основная метка вне схемы")
        if primary in {"human", "machine"} or primary == row.get("prior_sample"):
            errors.append(f"{call_id}: прежняя метка кампании попала в новую схему")

    if all(not (row.get("campaign_id") or "") for row in manifest):
        findings.append("campaign_id пуст у всех звонков: в выгрузке есть только campaign_name.")
    if all(not (row.get("call_status") or "") for row in manifest):
        findings.append("call_status пуст у всех звонков: в таблице calls нет колонки status.")
    if all(not (row.get("speaker_id") or "") for row in calls):
        findings.append("speaker_id неизвестен.")
    if all(not (row.get("tts_generator") or "") for row in calls):
        findings.append("tts_generator абонента неизвестен. TTS нашего робота сюда не подставляется.")
    if all(not (row.get("voice_conversion_system") or "") for row in calls):
        findings.append("voice_conversion_system неизвестна.")
    if all(not (row.get("playback_method") or "") for row in calls):
        findings.append("playback_method неизвестен.")

    template_rows = [row for row in calls if not row.get("annotator_id")]
    labels = Counter((row.get("primary_label") or "unknown") for row in template_rows)
    confirmed = sum(count for label, count in labels.items() if label not in {"unknown", ""})
    if confirmed == 0:
        findings.append("Подтверждённых меток происхождения пока нет: шаблон партии оставлен как unknown.")
    for label in ("synthetic_tts", "synthetic_voice_conversion", "synthetic_playback", "human_recorded", "mixed"):
        if labels.get(label, 0) == 0:
            findings.append(f"Категория {label} в этой партии не подтверждена.")
    if confirmed and all((row.get("label_source") or "") == "manual_listening" for row in template_rows if (row.get("primary_label") or "") != "unknown"):
        findings.append("Подтверждённые метки получены прослушиванием. Имени генератора в них нет.")
    open_reviews = [
        row["call_id"]
        for row in template_rows
        if (row.get("primary_label") or "unknown") != "unknown" and (row.get("review_status") or "") != "reviewed"
    ]
    if open_reviews:
        findings.append(
            "Метка сохранена, но статус проверки не reviewed: " + ", ".join(open_reviews) + "."
        )

    segment_ids = [row.get("segment_id") for row in segments if row.get("segment_id")]
    if len(segment_ids) != len(set(segment_ids)):
        errors.append("Повторяющийся segment_id")
    for segment in segments:
        fact = by_call.get(segment.get("call_id", ""))
        if fact is None:
            errors.append("Сегмент ссылается на звонок вне manifest")
            continue
        try:
            start = float(segment.get("start_time") or 0)
            end = float(segment.get("end_time") or 0)
            duration = float(fact.get("duration_seconds") or 0)
        except ValueError:
            errors.append(f"{segment.get('call_id')}: время сегмента не число")
            continue
        if end <= start or end > duration + 0.05:
            errors.append(f"{segment.get('call_id')}: сегмент вне записи")

    return {
        "errors": errors,
        "findings": findings,
        "rows": len(manifest),
        "labels": dict(labels),
        "reasons": dict(Counter(row.get("selection_reason") or "" for row in manifest)),
        "training_roles": dict(Counter(row.get("training_role") or "" for row in manifest)),
        "locked_splits": dict(Counter(row.get("locked_split") or "none" for row in manifest)),
        "client_channels": dict(Counter(row.get("client_channel") or "" for row in manifest)),
        "sample_rates": dict(Counter(row.get("sample_rate") or "" for row in manifest)),
        "channel_counts": dict(Counter(row.get("channel_count") or "" for row in manifest)),
        "codecs": dict(Counter(row.get("codec") or "" for row in manifest)),
        "segments": len(segments),
        "manifest_ids": ids,
        "secondary": dict(Counter((row.get("secondary_label") or "none") for row in template_rows)),
        "confidence": dict(Counter((row.get("label_confidence") or "none") for row in template_rows)),
        "review_status": dict(Counter((row.get("review_status") or "") for row in template_rows)),
        "label_sources": dict(Counter((row.get("label_source") or "") for row in template_rows)),
        "by_reason": {
            reason: dict(Counter((row.get("primary_label") or "unknown") for row in group))
            for reason, group in _grouped(template_rows, "selection_reason").items()
        },
        "soft_labels": [
            {
                "call_id": row["call_id"],
                "primary_label": row.get("primary_label") or "",
                "secondary_label": row.get("secondary_label") or "",
                "label_confidence": row.get("label_confidence") or "",
                "review_status": row.get("review_status") or "",
            }
            for row in template_rows
            if (row.get("label_confidence") or "") in {"low", "medium"}
            or (
                (row.get("primary_label") or "unknown") != "unknown"
                and (row.get("review_status") or "") != "reviewed"
            )
        ],
    }


def _grouped(rows: list[dict], field: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row.get(field) or "", []).append(row)
    return groups
