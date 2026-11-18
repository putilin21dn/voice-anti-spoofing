from __future__ import annotations

import re

import numpy as np
import pandas as pd


_LOG_ENERGY = re.compile(r"^(?:energy_rms__|(?:lfcc|mfcc|cqcc)_(?:static|delta|delta2)_00_)")
_CEPS = re.compile(r"^(?:lfcc|mfcc|cqcc)_(?:static|delta|delta2)_(\d+)_")
_MFCC = re.compile(r"^mfcc_(?:static|delta|delta2)_(\d+)_")
_SPECTRAL = re.compile(r"^(?:spectral_|zero_crossing_rate)")
_VOICING = re.compile(r"^(?:f0_|jitter_local|shimmer_local|hnr_praat_db|acf_harmonicity_mean)")
VARIANTS = ("full", "no_energy", "shape", "voicing", "mfcc", "spectral")


def is_log_energy(name: str) -> bool:
    """Нулевой кепстр и RMS: уровень сегмента и его изменение, не форма спектра."""

    return _LOG_ENERGY.match(name) is not None


def is_shape(name: str) -> bool:
    """Кепстр начиная с коэффициента 1 и спектральная форма. Уровень и F0 сюда не входят."""

    match = _CEPS.match(name)
    if match is not None:
        return match.group(1) != "00"
    return _SPECTRAL.match(name) is not None


def is_voicing(name: str) -> bool:
    """F0, доля озвученных кадров, гармоничность, jitter, shimmer и HNR."""

    return _VOICING.match(name) is not None


def is_mfcc_shape(name: str) -> bool:
    """MFCC начиная с коэффициента 1. Нулевой коэффициент — энергия."""

    match = _MFCC.match(name)
    return match is not None and match.group(1) != "00"


def is_spectral_measure(name: str) -> bool:
    """Центроид, полоса, rolloff, flatness и zero-crossing rate."""

    return _SPECTRAL.match(name) is not None


def without_log_energy(columns: list[str]) -> list[str]:
    return [name for name in columns if not is_log_energy(name)]


def columns_for_variant(columns: list[str], variant: str) -> tuple[list[str], list[str]]:
    if variant not in VARIANTS:
        raise ValueError(f"Неизвестный вариант модели: {variant}")
    if variant == "full":
        kept = list(columns)
    elif variant == "no_energy":
        kept = without_log_energy(columns)
    elif variant == "shape":
        kept = [name for name in columns if is_shape(name)]
    elif variant == "voicing":
        kept = [name for name in columns if is_voicing(name)]
    elif variant == "mfcc":
        kept = [name for name in columns if is_mfcc_shape(name)]
    else:
        kept = [name for name in columns if is_spectral_measure(name)]
    kept_set = set(kept)
    excluded = [name for name in columns if name not in kept_set]
    return kept, excluded


def select_features(
    frame: pd.DataFrame,
    y: pd.Series | np.ndarray,
    columns: list[str],
    *,
    max_features: int,
    max_nan_fraction: float,
    min_finite: int,
    max_pairwise_correlation: float,
) -> list[str]:
    """Оставляет признаки по статистикам train. Validation и test сюда не передаются."""

    matrix = frame[columns]
    target = pd.Series(np.asarray(y), index=frame.index, dtype=float)
    finite_count = matrix.notna().sum()
    nan_fraction = 1.0 - finite_count / max(len(matrix), 1)
    variance = matrix.var(skipna=True)
    usable = [
        name
        for name in columns
        if finite_count[name] >= min_finite and nan_fraction[name] <= max_nan_fraction and variance[name] > 1e-8
    ]
    if not usable:
        raise RuntimeError("После отбора на train не осталось признаков")
    association = matrix[usable].corrwith(target).abs().dropna().sort_values(ascending=False)
    candidates = list(association.head(max(max_features * 4, max_features)).index)
    filled = matrix[candidates].fillna(matrix[candidates].median())
    correlation = filled.corr().abs()
    selected: list[str] = []
    for name in candidates:
        if any(float(correlation.loc[name, previous]) > max_pairwise_correlation for previous in selected):
            continue
        selected.append(name)
        if len(selected) >= max_features:
            break
    return selected
