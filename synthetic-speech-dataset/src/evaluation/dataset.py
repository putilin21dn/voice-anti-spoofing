from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.audio.loader import probe_wav
from src.utils.config import load_config, project_root, resolve_path

ORIGIN_LABELS = (
    "human_live",
    "human_recorded",
    "synthetic_tts",
    "synthetic_voice_conversion",
    "synthetic_playback",
    "mixed",
    "unknown",
)
HUMAN_ORIGINS = frozenset({"human_live", "human_recorded"})
SYNTHETIC_ORIGINS = frozenset(
    {"synthetic_tts", "synthetic_voice_conversion", "synthetic_playback"}
)
AGGREGATIONS = frozenset({"mean", "median", "std", "min", "max", "p25", "p75"})


@dataclass
class PreparedCall:
    call_id: str
    source_file: str
    wav_path: Path
    listener_label: str
    primary_label: str
    category: str
    client_channel: str
    sample_rate: str
    duration_seconds: str
    channel_count: str
    campaign_name: str
    outcome: str
    prior_classification: str
    timestamp: str
    training_role: str
    cohort: str
    legacy_label: str
    in_category_metrics: bool
    in_binary_metrics: bool
    exclusion: str
    exclusion_reason: str
    row: dict = field(repr=False)


def load_evaluation_config(path: str = "config/evaluation.yaml") -> dict:
    return load_config(path)


def _ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["call_id"] for row in csv.DictReader(handle) if row.get("call_id")}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [{key: value or "" for key, value in row.items() if key} for row in csv.DictReader(handle)]


def prepare_dataset(cfg: dict) -> tuple[list[PreparedCall], list[str]]:
    manifest_path = resolve_path(cfg, cfg["paths"]["confirmed_manifest"])
    audio_root = resolve_path(cfg, cfg["paths"]["audio_root"])
    rows = _read_rows(manifest_path)
    errors: list[str] = []
    seen: set[str] = set()
    locked = _ids(resolve_path(cfg, cfg["paths"]["test_split"])) | _ids(
        resolve_path(cfg, cfg["paths"]["spectra_threshold_split"])
    )
    batch_ids = _ids(resolve_path(cfg, cfg["paths"]["batch_001_manifest"]))
    prepared: list[PreparedCall] = []
    for row in rows:
        call_id = row.get("call_id", "")
        if not call_id:
            errors.append("В manifest есть строка без call_id")
            continue
        if call_id in seen:
            errors.append(f"Повторяющийся call_id: {call_id}")
        seen.add(call_id)
        prepared.append(_prepare_one(row, audio_root, locked, batch_ids))
    legacy, legacy_errors = _legacy_calls(cfg)
    errors.extend(legacy_errors)
    for call in legacy:
        if call.call_id in seen:
            errors.append(f"Повторяющийся call_id: {call.call_id}")
        seen.add(call.call_id)
        prepared.append(call)
    return prepared, errors


def _prepare_one(row: dict[str, str], audio_root: Path, locked: set[str], batch_ids: set[str]) -> PreparedCall:
    call_id = row["call_id"]
    source = row.get("source_file", "")
    wav_path = (audio_root / source).resolve() if source else Path()
    listener = row.get("listener_label", "")
    primary = row.get("primary_label", "")
    role = row.get("training_role", "")
    exclusion = ""
    reason = ""
    category = primary or "unknown"
    if call_id in batch_ids or role == "not_for_training":
        exclusion, reason = "batch_001", "Партия batch_001 не входит в основную оценку"
        category = "excluded"
    elif role == "locked_baseline" or call_id in locked:
        exclusion, reason = "locked_baseline", "Старый test20, n100 или locked_baseline"
        category = "excluded"
    elif listener == "robot" and not primary:
        exclusion, reason = "unclassified_robot", "Робот без подтипа. В категориальные таблицы не входит"
        category = "unclassified_robot"
    elif primary == "unknown" or not primary:
        exclusion, reason = "unknown", "Происхождение не подтверждено"
        category = "unknown"
    elif primary not in ORIGIN_LABELS:
        exclusion, reason = "unknown_label", f"Метка {primary} вне схемы происхождения"
        category = "unknown"

    probe_error = _probe_error(row, wav_path)
    if probe_error and not exclusion:
        exclusion, reason = probe_error
        category = "excluded"
    elif probe_error and exclusion == "unclassified_robot":
        # Робот без файла тоже не оценивается, причина файла важнее.
        exclusion, reason = probe_error
        category = "excluded"

    in_category = exclusion == "" and category in ORIGIN_LABELS and category != "unknown"
    in_binary = in_category and category in HUMAN_ORIGINS | SYNTHETIC_ORIGINS
    return PreparedCall(
        call_id=call_id,
        source_file=source,
        wav_path=wav_path,
        listener_label=listener,
        primary_label=primary,
        category=category,
        client_channel=row.get("client_channel", ""),
        sample_rate=row.get("sample_rate", ""),
        duration_seconds=row.get("duration_seconds", ""),
        channel_count=row.get("channel_count", ""),
        campaign_name=row.get("campaign_name", ""),
        outcome=row.get("outcome", ""),
        prior_classification=row.get("prior_classification", ""),
        timestamp=row.get("timestamp", ""),
        training_role=role,
        cohort="confirmed_001",
        legacy_label="",
        in_category_metrics=in_category,
        in_binary_metrics=in_binary,
        exclusion=exclusion,
        exclusion_reason=reason,
        row=row,
    )


def _legacy_calls(cfg: dict) -> tuple[list[PreparedCall], list[str]]:
    """Прежний test20. Метка human/machine остаётся меткой старой задачи."""

    path = resolve_path(cfg, cfg["paths"]["legacy_test_manifest"])
    if not path.is_file():
        return [], [f"Нет manifest прежнего test: {path}"]
    rows = [row for row in _read_rows(path) if row.get("locked_split") == "test"]
    if not rows:
        return [], ["В manifest нет звонков прежнего test"]
    return [_prepare_legacy(row) for row in rows], []


def _prepare_legacy(row: dict[str, str]) -> PreparedCall:
    call_id = row["call_id"]
    wav_path = Path(row.get("resolved_path", ""))
    probed = probe_wav(wav_path) if wav_path.is_file() else None
    label = row.get("label", "")
    exclusion = ""
    reason = ""
    category = "legacy_test"
    if probed is None or not probed.get("codec"):
        exclusion = "missing_wav" if not wav_path.is_file() else "broken_wav"
        reason = "WAV прежнего test не найден" if exclusion == "missing_wav" else "WAV прежнего test не открывается"
        category = "excluded"
        sample_rate = row.get("sample_rate", "")
        channels = ""
        duration = ""
    else:
        sample_rate = probed["sample_rate"]
        channels = probed["channel_count"]
        duration = probed["duration_seconds"]
        client = row.get("client_channel", "")
        if client not in {str(index) for index in range(int(channels))}:
            exclusion, reason, category = "channel_uncertain", "Канал абонента не совпадает с файлом test", "excluded"
    if not exclusion and label not in {"human", "machine"}:
        exclusion, reason, category = "unknown_label", f"Старая метка {label} не human и не machine", "excluded"
    try:
        source = str(wav_path.resolve().relative_to(project_root().parent))
    except ValueError:
        source = str(wav_path)
    return PreparedCall(
        call_id=call_id,
        source_file=source.replace("\\", "/"),
        wav_path=wav_path,
        listener_label="",
        primary_label="",
        category=category,
        client_channel=row.get("client_channel", ""),
        sample_rate=sample_rate,
        duration_seconds=duration,
        channel_count=channels,
        campaign_name=row.get("campaign_name", ""),
        outcome=row.get("outcome", ""),
        prior_classification=row.get("classification_result", ""),
        timestamp=row.get("timestamp", ""),
        training_role="legacy_test",
        cohort="legacy_test",
        legacy_label=label if label in {"human", "machine"} else "",
        in_category_metrics=False,
        in_binary_metrics=False,
        exclusion=exclusion,
        exclusion_reason=reason,
        row=row,
    )


def _probe_error(row: dict[str, str], wav_path: Path) -> tuple[str, str] | None:
    if not wav_path.is_file():
        return "missing_wav", "WAV не найден"
    probed = probe_wav(wav_path)
    if probed is None or not probed.get("codec"):
        return "broken_wav", "WAV не открывается или это не pcm_s16le"
    channels = int(probed["channel_count"])
    client = row.get("client_channel", "")
    if client not in {str(index) for index in range(channels)}:
        return "channel_uncertain", "Канал абонента не указан или выходит за число каналов"
    if row.get("sample_rate") and row["sample_rate"] != probed["sample_rate"]:
        return "broken_wav", "sample rate в manifest не совпадает с WAV"
    if row.get("channel_count") and row["channel_count"] != probed["channel_count"]:
        return "broken_wav", "Число каналов в manifest не совпадает с WAV"
    return None


def batch_status(cfg: dict) -> dict:
    path = resolve_path(cfg, cfg["paths"]["confirmed_status"])
    if not path.exists():
        return {"status": "open", "path": str(path)}
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    loaded["path"] = str(path)
    loaded["status"] = str(loaded.get("status") or "open")
    return loaded


def readiness(cfg: dict, calls: list[PreparedCall], prepare_errors: list[str]) -> list[str]:
    errors = list(prepare_errors)
    status = batch_status(cfg)
    if status.get("status") != "closed":
        errors.append("Пачка не закрыта. Полный evaluation не запускается.")
    if float(cfg["spectra0_threshold"]) != 1.02:
        errors.append("Порог Spectra-0 в конфиге не равен 1.02")
    for key in ("logistic", "svm", "lightgbm", "spectra0_dir", "aasist3_dir"):
        path = resolve_path(cfg, cfg["paths"][key])
        if key.endswith("_dir"):
            if not (path / "model.py").is_file() or not (path / "model.safetensors").is_file():
                errors.append(f"Нет артефакта {key}: {path}")
        elif not path.is_file():
            errors.append(f"Нет артефакта {key}: {path}")
    python = resolve_path(cfg, cfg["paths"]["classical_python"])
    if not python.is_file():
        errors.append(f"Нет Python классического эксперимента: {python}")
    critical = [call for call in calls if call.exclusion in {"missing_wav", "broken_wav", "channel_uncertain"}]
    if critical:
        errors.append(
            "Критическая проблема с файлом или каналом: "
            + ", ".join(f"{call.call_id} ({call.exclusion})" for call in critical)
        )
    return errors


def feature_name_errors(bundle_features: list[str], stems: set[str]) -> list[str]:
    errors = []
    for name in bundle_features:
        if "__" not in name:
            errors.append(f"Признак без агрегации: {name}")
            continue
        stem, aggregation = name.rsplit("__", 1)
        if aggregation not in AGGREGATIONS:
            errors.append(f"Неизвестная агрегация: {name}")
        elif stem not in stems:
            errors.append(f"В схеме признаков нет {stem}")
    return errors
