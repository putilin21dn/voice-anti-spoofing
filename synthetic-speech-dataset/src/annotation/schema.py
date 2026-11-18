from __future__ import annotations

import math
import re
from uuid import UUID


CALL_ID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
ANNOTATOR_RE = re.compile(r"^[\w.\-]{0,32}$")

PRIMARY_LABELS = (
    "human_live",
    "human_recorded",
    "synthetic_tts",
    "synthetic_voice_conversion",
    "synthetic_playback",
    "mixed",
    "unknown",
)
HUMAN_LABELS = frozenset({"human_live", "human_recorded"})
SYNTHETIC_LABELS = frozenset(
    {"synthetic_tts", "synthetic_voice_conversion", "synthetic_playback"}
)
SECONDARY_LABELS = frozenset({"", "ai_secretary", "voicemail", "anti_spam", "greeting_only"})
CONFIDENCE_VALUES = frozenset({"", "high", "medium", "low"})
LABEL_SOURCES = frozenset(
    {"manual_listening", "verified_metadata", "known_generator", "unknown_source"}
)
REVIEW_STATUSES = frozenset(
    {"unreviewed", "in_progress", "reviewed", "needs_second_review", "disputed"}
)
FORBIDDEN_COLUMNS = frozenset(
    {"phone", "phone_number", "external_id", "s3_key", "s3_bucket", "msisdn"}
)

FACT_FIELDS = [
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
]

LABEL_FIELDS = [
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
    "annotator_id",
    "updated_at",
]

CALL_FIELDS = [
    "call_id",
    "source_file",
    "data_source",
    "primary_label",
    "secondary_label",
    "label_confidence",
    "label_source",
    "campaign_id",
    "campaign_name",
    "call_status",
    "goal_status",
    "outcome",
    "tts_generator",
    "voice_conversion_system",
    "speaker_id",
    "playback_method",
    "codec",
    "codec_source",
    "sample_rate",
    "duration_seconds",
    "channel_count",
    "annotated_channel",
    "client_channel",
    "annotation_notes",
    "review_status",
    "annotator_id",
    "prior_sample",
    "prior_classification",
    "locked_split",
    "selection_reason",
    "training_role",
    "batch_id",
    "updated_at",
]

SEGMENT_FIELDS = [
    "call_id",
    "segment_id",
    "start_time",
    "end_time",
    "speaker_id",
    "segment_label",
    "label_confidence",
    "label_source",
    "notes",
    "annotator_id",
]


def valid_call_id(value: str) -> bool:
    return bool(CALL_ID_RE.fullmatch(value or ""))


def valid_annotator(value: str) -> bool:
    return bool(ANNOTATOR_RE.fullmatch(value or ""))


def _text(value: object, limit: int) -> str:
    text = "" if value is None else str(value).strip()
    if "\n" in text or "\r" in text:
        raise ValueError("Поле содержит перевод строки")
    if len(text) > limit:
        raise ValueError(f"Поле длиннее {limit} символов")
    return text


def _time(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Время сегмента должно быть числом в секундах") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError("Время сегмента должно быть неотрицательным")
    return number


def check_label_pair(primary: str, secondary: str) -> str | None:
    if secondary == "ai_secretary" and primary in HUMAN_LABELS:
        return "AI-секретарь с этой меткой — синтетический голос. Для человека уберите secondary ai_secretary."
    if secondary == "voicemail" and primary == "human_live":
        return "Заранее записанная почта — human_recorded, а не human_live."
    return None


def validate_annotation(payload: dict, *, duration_s: float) -> dict:
    """Проверяет разметку одного звонка. Факты звонка сюда не входят."""

    primary = _text(payload.get("primary_label"), 64)
    secondary = _text(payload.get("secondary_label"), 64)
    confidence = _text(payload.get("label_confidence"), 16)
    source = _text(payload.get("label_source"), 64)
    status = _text(payload.get("review_status"), 32)
    annotator = _text(payload.get("annotator_id"), 32)
    notes = "" if payload.get("annotation_notes") is None else str(payload.get("annotation_notes")).strip()
    if len(notes) > 4000:
        raise ValueError("Заметка длиннее 4000 символов")

    if primary not in PRIMARY_LABELS:
        raise ValueError("Неизвестная основная метка. Старые метки human и machine сюда не подходят.")
    if secondary not in SECONDARY_LABELS:
        raise ValueError("Неизвестный дополнительный признак")
    if confidence not in CONFIDENCE_VALUES:
        raise ValueError("Уверенность: high, medium, low или пусто")
    if source not in LABEL_SOURCES:
        raise ValueError("Неизвестный источник метки")
    if status not in REVIEW_STATUSES:
        raise ValueError("Неизвестный статус проверки")
    if not valid_annotator(annotator):
        raise ValueError("Код разметчика: до 32 букв, цифр, точки, _ или -")

    pair_error = check_label_pair(primary, secondary)
    if pair_error:
        raise ValueError(pair_error)

    if primary == "unknown":
        if source != "unknown_source":
            raise ValueError("Для unknown источник — unknown_source")
        if confidence:
            raise ValueError("Для unknown уверенность не ставится")
    else:
        if source == "unknown_source":
            raise ValueError("Подтверждённая метка требует источник: прослушивание, метаданные или известный генератор")
        if confidence not in {"high", "medium", "low"}:
            raise ValueError("Для подтверждённой метки укажите уверенность")

    tts = _text(payload.get("tts_generator"), 200)
    conversion = _text(payload.get("voice_conversion_system"), 200)
    speaker = _text(payload.get("speaker_id"), 200)
    playback = _text(payload.get("playback_method"), 200)
    if source == "known_generator" and not any((tts, conversion, speaker, playback)):
        raise ValueError("known_generator требует известный генератор, систему, говорящего или способ воспроизведения")
    if source == "known_generator":
        if primary == "synthetic_tts" and not tts:
            raise ValueError("known_generator для TTS требует имя генератора")
        if primary == "synthetic_voice_conversion" and not conversion:
            raise ValueError("known_generator для voice conversion требует имя системы")
        if primary == "synthetic_playback" and not playback:
            raise ValueError("known_generator для воспроизведения требует способ playback")
        if primary in HUMAN_LABELS and not speaker:
            raise ValueError("known_generator для человеческой речи требует speaker_id")

    raw_segments = payload.get("segments") or []
    if not isinstance(raw_segments, list):
        raise ValueError("Сегменты должны быть списком")
    if len(raw_segments) > 50:
        raise ValueError("Не больше 50 сегментов на звонок")

    segments: list[dict[str, str]] = []
    families: set[str] = set()
    for index, raw in enumerate(raw_segments, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Сегмент {index} повреждён")
        start = _time(raw.get("start_time"))
        end = _time(raw.get("end_time"))
        if end <= start:
            raise ValueError(f"Сегмент {index}: конец должен быть позже начала")
        if end > duration_s + 0.05:
            raise ValueError(f"Сегмент {index} выходит за длительность записи")
        label = _text(raw.get("segment_label"), 64)
        if label not in PRIMARY_LABELS:
            raise ValueError(f"Сегмент {index}: неизвестная метка")
        seg_confidence = _text(raw.get("label_confidence"), 16)
        seg_source = _text(raw.get("label_source"), 64)
        if seg_confidence not in CONFIDENCE_VALUES or seg_source not in LABEL_SOURCES:
            raise ValueError(f"Сегмент {index}: уверенность или источник не подходят")
        if label == "unknown" and seg_source != "unknown_source":
            raise ValueError(f"Сегмент {index}: для unknown источник unknown_source")
        if label != "unknown" and seg_source == "unknown_source":
            raise ValueError(f"Сегмент {index}: подтверждённая метка требует источник")
        seg_secondary_error = check_label_pair(label, secondary)
        if seg_secondary_error and label in HUMAN_LABELS:
            raise ValueError(f"Сегмент {index}: {seg_secondary_error}")
        speaker_id = _text(raw.get("speaker_id"), 200)
        seg_notes = "" if raw.get("notes") is None else str(raw.get("notes")).strip()
        if len(seg_notes) > 1000:
            raise ValueError(f"Сегмент {index}: заметка длиннее 1000 символов")
        segment_id = _text(raw.get("segment_id"), 64)
        if segment_id:
            try:
                UUID(segment_id)
            except ValueError as exc:
                raise ValueError(f"Сегмент {index}: segment_id должен быть UUID") from exc
        if label in HUMAN_LABELS:
            families.add("human")
        elif label in SYNTHETIC_LABELS:
            families.add("synthetic")
        segments.append(
            {
                "segment_id": segment_id,
                "start_time": f"{start:.3f}",
                "end_time": f"{end:.3f}",
                "speaker_id": speaker_id,
                "segment_label": label,
                "label_confidence": seg_confidence,
                "label_source": seg_source,
                "notes": seg_notes,
            }
        )

    if families == {"human", "synthetic"} and primary not in {"mixed", "unknown"}:
        raise ValueError("В сегментах есть и человек, и синтетика. Звонку нужна метка mixed или unknown.")

    return {
        "primary_label": primary,
        "secondary_label": secondary,
        "label_confidence": confidence,
        "label_source": source,
        "tts_generator": tts,
        "voice_conversion_system": conversion,
        "speaker_id": speaker,
        "playback_method": playback,
        "annotation_notes": notes,
        "review_status": status,
        "annotator_id": annotator,
        "segments": segments,
    }
