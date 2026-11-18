from __future__ import annotations

import numpy as np

from src.features.cqcc import compute_cqcc, cqcc_available, cqcc_names
from src.features.lfcc import compute_lfcc, lfcc_names
from src.features.mfcc import compute_mfcc, mfcc_names
from src.features.pitch import compute_pitch, pitch_names
from src.features.schema import FLAG_COLUMNS
from src.features.spectral import compute_spectral, spectral_names
from src.features.voice_quality import compute_voice_quality, voice_quality_names


def feature_names(cfg: dict) -> list[str]:
    names: list[str] = []
    if cfg["features"]["lfcc"]["enabled"]:
        names.extend(lfcc_names(cfg))
    if cfg["features"]["mfcc"]["enabled"]:
        names.extend(mfcc_names(cfg))
    if cfg["features"]["cqcc"]["enabled"]:
        names.extend(cqcc_names(cfg))
    if cfg["features"]["pitch"]["enabled"]:
        names.extend(pitch_names(cfg))
    if cfg["features"]["voice_quality"]["enabled"]:
        names.extend(voice_quality_names(cfg))
    if cfg["features"]["spectral"]["enabled"]:
        names.extend(spectral_names(cfg))
    return names


def empty_features(cfg: dict) -> dict[str, float]:
    return {name: float("nan") for name in feature_names(cfg)}


def _sanitize(names: list[str], values: dict[str, float]) -> tuple[dict[str, float], int]:
    cleaned: dict[str, float] = {}
    inf_count = 0
    for name in names:
        raw = values.get(name, float("nan"))
        try:
            number = float(raw)
        except (TypeError, ValueError):
            number = float("nan")
        if np.isinf(number):
            inf_count += 1
            number = float("nan")
        elif not np.isfinite(number):
            number = float("nan")
        cleaned[name] = number
    return cleaned, inf_count


def _run_group(name: str, names: list[str], function, audio, sample_rate, cfg) -> tuple[dict[str, float], int]:
    try:
        values = function(audio, sample_rate, cfg)
        return _sanitize(names, values)
    except Exception:
        return {item: float("nan") for item in names}, -1


def extract_segment_features(audio: np.ndarray, sample_rate: int, cfg: dict) -> tuple[dict[str, float], dict[str, int | str]]:
    flags: dict[str, int | str] = {name: 0 for name in FLAG_COLUMNS}
    flags["voice_quality_reliability"] = ""
    values = empty_features(cfg)
    partial = False

    if cfg["features"]["lfcc"]["enabled"]:
        group_names = lfcc_names(cfg)
        group, status = _run_group("lfcc", group_names, compute_lfcc, audio, sample_rate, cfg)
        values.update(group)
        if status < 0:
            flags["flag_lfcc_error"] = 1
            partial = True
        elif _deltas_missing(group_names, group):
            flags["flag_delta_unavailable"] = 1

    if cfg["features"]["mfcc"]["enabled"]:
        group_names = mfcc_names(cfg)
        group, status = _run_group("mfcc", group_names, compute_mfcc, audio, sample_rate, cfg)
        values.update(group)
        if status < 0:
            flags["flag_mfcc_error"] = 1
            partial = True

    if cfg["features"]["cqcc"]["enabled"]:
        ready, _reason = cqcc_available()
        group_names = cqcc_names(cfg)
        if not ready:
            flags["flag_cqcc_disabled"] = 1
            partial = True
        else:
            group, status = _run_group("cqcc", group_names, compute_cqcc, audio, sample_rate, cfg)
            values.update(group)
            if status < 0:
                flags["flag_cqcc_error"] = 1
                partial = True
    else:
        flags["flag_cqcc_disabled"] = 1

    if cfg["features"]["pitch"]["enabled"]:
        group_names = pitch_names(cfg)
        group, status = _run_group("pitch", group_names, compute_pitch, audio, sample_rate, cfg)
        values.update(group)
        if status < 0:
            flags["flag_pitch_error"] = 1
            partial = True
        elif group.get("f0_voiced_fraction", 0.0) == 0.0:
            flags["flag_unvoiced"] = 1

    if cfg["features"]["voice_quality"]["enabled"]:
        try:
            group, voice_flags = compute_voice_quality(audio, sample_rate, cfg)
            group, inf_count = _sanitize(voice_quality_names(cfg), group)
            values.update(group)
            flags.update(voice_flags)
            if inf_count:
                flags["flag_nonfinite"] = 1
        except Exception:
            flags["flag_voice_quality_unavailable"] = 1
            partial = True

    if cfg["features"]["spectral"]["enabled"]:
        group_names = spectral_names(cfg)
        group, status = _run_group("spectral", group_names, compute_spectral, audio, sample_rate, cfg)
        values.update(group)
        if status < 0:
            flags["flag_spectral_error"] = 1
            partial = True

    flags["flag_partial_features"] = int(partial)
    return values, flags


def _deltas_missing(names: list[str], values: dict[str, float]) -> bool:
    delta = [name for name in names if "_delta_" in name]
    static = [name for name in names if "_static_" in name]
    if not delta:
        return False
    static_ok = any(np.isfinite(values.get(name, np.nan)) for name in static)
    delta_missing = all(not np.isfinite(values.get(name, np.nan)) for name in delta)
    return static_ok and delta_missing
