"""Shared constants, group splits, and effect sizes for the follow-up."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.research.acoustic_representations.config import (
    CQCC_CEPS,
    LFCC_CEPS,
    PREEMPHASIS,
    SAMPLE_RATE,
    SPECTRA_MODEL_WINDOW_S,
    SPECTRA_THRESHOLD,
)
from src.research.acoustic_representations.signal import write_csv

ROOT = Path(__file__).resolve().parents[3]
PREVIOUS = ROOT / "reports" / "day_20261113_acoustic_representations"
OUTPUT = ROOT / "reports" / "day_20261113_cepstral_padding_followup"
WINDOWS = ROOT / "evaluation" / "day_20261113" / "20261113T175850" / "window_results" / "spectra0.csv"
SCORES = ROOT / "data" / "manifests" / "day_20261113_scores.csv"

TARGET_SAMPLES = int(round(SPECTRA_MODEL_WINDOW_S * SAMPLE_RATE))
RMS_TARGET = 0.1
N_BOOT = 2000
BOOT_SEED = 42
MIN_N_CI = 8
FN_CALLS = (
    "a037652f-505c-4f81-a49d-a2ed36434761",
    "7eb5d6cd-8612-484b-a770-eb3113faf791",
    "60f772ee-765d-48b3-9099-2f9a50cc0e25",
    "cf7c2727-4fd9-4032-aefd-71871f58c582",
    "8b34a750-b5df-4a6d-be75-7fbc207a9546",
)

FAMILIES = (("lfcc", LFCC_CEPS), ("cqcc", CQCC_CEPS), ("spafe_cqcc", CQCC_CEPS))
BETWEEN_ENDPOINTS = (
    "lfcc_between_segment_std_mean",
    "cqcc_between_segment_std_mean",
    "spafe_cqcc_between_segment_std_mean",
    "cqcc_c01_mean",
)
FP_ENDPOINTS = (
    "speech_duration_s",
    "n_speech_segments",
    "speech_ratio",
    "spectra_padding_ratio",
    "lfcc_between_segment_std_mean",
    "cqcc_between_segment_std_mean",
    "spafe_cqcc_between_segment_std_mean",
    "cqcc_c01_mean",
    "raw_speech_rms",
    "explanatory_f0_iqr_hz",
)


def coeff_columns(prefix: str, count: int, stat: str = "mean") -> list[str]:
    return [f"{prefix}_c{index:02d}_{stat}" for index in range(count)]


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def as_float(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def cell(value) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return "1" if bool(value) else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    number = as_float(value)
    if not np.isfinite(number):
        return ""
    return f"{number:.8g}"


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    rows = []
    for record in frame.to_dict(orient="records"):
        rows.append({key: cell(value) for key, value in record.items()})
    write_csv(path, rows, list(frame.columns))


def finite(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    return values[np.isfinite(values)]


def cliffs_delta(reference: np.ndarray, other: np.ndarray) -> float:
    """Positive when `other` tends to exceed `reference`. Ties contribute nothing."""

    if reference.size == 0 or other.size == 0:
        return np.nan
    ref = np.sort(np.asarray(reference, dtype=np.float64))
    other = np.asarray(other, dtype=np.float64)
    greater = np.searchsorted(ref, other, side="left")
    smaller = ref.size - np.searchsorted(ref, other, side="right")
    return float((greater.sum() - smaller.sum()) / (ref.size * other.size))


def cliffs_interval(reference: np.ndarray, other: np.ndarray) -> tuple[float, float, str]:
    if reference.size < MIN_N_CI or other.size < MIN_N_CI:
        return np.nan, np.nan, "insufficient"
    rng = np.random.default_rng(BOOT_SEED)
    stats = np.empty(N_BOOT, dtype=np.float64)
    n_ref = reference.size
    n_other = other.size
    for index in range(N_BOOT):
        stats[index] = cliffs_delta(
            reference[rng.integers(0, n_ref, n_ref)],
            other[rng.integers(0, n_other, n_other)],
        )
    low, high = np.percentile(stats, [2.5, 97.5])
    return float(low), float(high), "ok"


def spearman(left: pd.Series, right: pd.Series) -> tuple[float, int]:
    frame = pd.DataFrame(
        {
            "left": pd.to_numeric(left, errors="coerce"),
            "right": pd.to_numeric(right, errors="coerce"),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 5 or frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
        return np.nan, int(len(frame))
    return float(frame["left"].corr(frame["right"], method="spearman")), int(len(frame))


def rank_tertile(frame: pd.DataFrame, column: str) -> pd.Series:
    """Three groups of nearly equal size. Ties break by call_id, not by file order."""

    labels = pd.Series(pd.NA, index=frame.index, dtype="object")
    values = pd.to_numeric(frame[column], errors="coerce")
    usable = frame.loc[np.isfinite(values)].copy()
    usable["_sort_value"] = values.loc[usable.index]
    order = usable.sort_values(["_sort_value", "call_id"], kind="mergesort")
    count = len(order)
    if count < 3:
        return labels
    cuts = [int(round(index * count / 3)) for index in range(4)]
    names = ("low", "mid", "high")
    assigned = np.empty(count, dtype=object)
    for index, name in enumerate(names):
        assigned[cuts[index] : cuts[index + 1]] = name
    labels.loc[order.index] = assigned
    return labels


def segment_count_bin(count: int) -> str:
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 4:
        return "2-4"
    return "5+"


def quartile_row(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"median": np.nan, "q25": np.nan, "q75": np.nan}
    q25, median, q75 = np.percentile(values, [25, 50, 75])
    return {"median": float(median), "q25": float(q25), "q75": float(q75)}
