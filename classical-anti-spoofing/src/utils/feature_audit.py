from __future__ import annotations

import numpy as np
import pandas as pd


def audit_feature_table(segments: pd.DataFrame) -> dict:
    if segments.empty:
        return {"segments": 0}
    usable = segments[pd.to_numeric(segments["usable_for_features"], errors="coerce").fillna(0).astype(int) == 1]
    report: dict = {
        "segments": int(len(segments)),
        "usable_segments": int(len(usable)),
        "calls": int(segments["call_id"].nunique()),
        "by_label": {},
    }
    for label, group in segments.groupby("label"):
        usable_group = group[pd.to_numeric(group["usable_for_features"], errors="coerce").fillna(0).astype(int) == 1]
        report["by_label"][str(label)] = {
            "calls": int(group["call_id"].nunique()),
            "segments": int(len(group)),
            "usable_segments": int(len(usable_group)),
            "speech_duration_s_median": _median(group.groupby("call_id")["speech_duration_s"].first()),
            "recording_duration_s_median": _median(group.groupby("call_id")["recording_duration_s"].first()),
            "speech_ratio_median": _median(group.groupby("call_id")["speech_ratio"].first()),
            "snr_db_median": _median(group.groupby("call_id")["snr_db"].first()),
        }
    numeric = usable.apply(pd.to_numeric, errors="coerce")
    groups = ("lfcc_", "mfcc_", "cqcc_", "f0_", "jitter_", "shimmer_", "hnr_", "acf_", "spectral_", "zero_")
    coverage = {}
    for prefix in groups:
        columns = [column for column in numeric.columns if column.startswith(prefix)]
        if not columns:
            continue
        values = numeric[columns]
        total = int(values.shape[0] * values.shape[1])
        finite = int(np.isfinite(values.to_numpy(dtype=float)).sum()) if total else 0
        coverage[prefix] = {"columns": len(columns), "finite_fraction": (finite / total) if total else None}
    report["coverage"] = coverage
    report["sample_rates"] = sorted({int(float(value)) for value in segments["sample_rate_hz"].unique()})
    report["resample_actions"] = segments["resample_action"].value_counts().to_dict()
    report["physical"] = _physical(usable)
    report["flags"] = {
        column: float(pd.to_numeric(segments[column], errors="coerce").fillna(0).mean())
        for column in segments.columns
        if column.startswith("flag_")
    }
    return report


def _median(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.median(finite))


def _range(frame: pd.DataFrame, column: str) -> dict | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"finite": 0}
    return {
        "finite": int(finite.size),
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "max": float(np.max(finite)),
    }


def _physical(usable: pd.DataFrame) -> dict:
    checks = {
        "f0_median": _range(usable, "f0_median"),
        "spectral_centroid_hz_mean": _range(usable, "spectral_centroid_hz_mean"),
        "spectral_rolloff_hz_mean": _range(usable, "spectral_rolloff_hz_mean"),
        "zero_crossing_rate_mean": _range(usable, "zero_crossing_rate_mean"),
        "spectral_flatness_mean": _range(usable, "spectral_flatness_mean"),
        "acf_harmonicity_mean": _range(usable, "acf_harmonicity_mean"),
        "hnr_praat_db": _range(usable, "hnr_praat_db"),
    }
    problems = []
    centroid = checks["spectral_centroid_hz_mean"]
    rolloff = checks["spectral_rolloff_hz_mean"]
    zcr = checks["zero_crossing_rate_mean"]
    flatness = checks["spectral_flatness_mean"]
    f0 = checks["f0_median"]
    if centroid and centroid.get("max", 0) > 4000:
        problems.append("спектральный центроид выше 4000 Гц")
    if rolloff and rolloff.get("max", 0) > 4000:
        problems.append("spectral rolloff выше 4000 Гц")
    if zcr and (zcr.get("min", 0) < 0 or zcr.get("max", 0) > 1):
        problems.append("zero crossing rate вне [0, 1]")
    if flatness and (flatness.get("min", 0) < 0 or flatness.get("max", 0) > 1.01):
        problems.append("spectral flatness вне [0, 1]")
    if f0 and (f0.get("min", 60) < 59 or f0.get("max", 400) > 401):
        problems.append("F0 вне заданного диапазона 60–400 Гц")
    checks["problems"] = problems
    return checks
