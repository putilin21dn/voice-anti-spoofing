"""Group comparisons for the acoustic study. No classifier is fit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.acoustic_representations.config import (
    CANDIDATE_MAX_ABS_SPEARMAN,
    CANDIDATE_MIN_CLIFF,
    CANDIDATE_MIN_MID_CLIFF,
    FEATURE_NOTES,
    FORBIDDEN_COLUMNS,
)

KEY_FEATURES = list(FEATURE_NOTES)


def _finite(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    return values[np.isfinite(values)]


def cliffs_delta(reference: np.ndarray, other: np.ndarray) -> float:
    """Positive when `other` tends to be larger than `reference`."""

    if reference.size == 0 or other.size == 0:
        return np.nan
    difference = other[:, None] - reference[None, :]
    return float((np.sum(difference > 0) - np.sum(difference < 0)) / difference.size)


def spearman(left: pd.Series, right: pd.Series) -> float:
    frame = pd.DataFrame({"left": pd.to_numeric(left, errors="coerce"), "right": pd.to_numeric(right, errors="coerce")})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 5 or frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
        return np.nan
    return float(frame["left"].corr(frame["right"], method="spearman"))


def _median(series: pd.Series) -> float:
    values = _finite(series)
    if values.size == 0:
        return np.nan
    return float(np.median(values))


def duration_tertiles(binary: pd.DataFrame) -> pd.Series:
    ranks = binary["duration_s"].rank(method="first")
    return pd.qcut(ranks, 3, labels=["short", "mid", "long"])


def feature_rows(frame: pd.DataFrame) -> list[dict]:
    binary = frame[frame["binary_group"].isin(["human_voice", "synthetic"])].copy()
    human = binary[binary["binary_group"] == "human_voice"]
    synth = binary[binary["binary_group"] == "synthetic"]
    binary["duration_tertile"] = duration_tertiles(binary)
    rows = []
    for name in KEY_FEATURES:
        if name not in frame.columns:
            continue
        human_values = _finite(human[name])
        synth_values = _finite(synth[name])
        delta = cliffs_delta(human_values, synth_values)
        pack_delta = {}
        pack_median = {}
        for pack in ("PACK1", "PACK2"):
            pack_human = _finite(human.loc[human["campaign"] == pack, name])
            pack_synth = _finite(synth.loc[synth["campaign"] == pack, name])
            pack_delta[pack] = cliffs_delta(pack_human, pack_synth)
            pack_median[f"{pack}_human"] = float(np.median(pack_human)) if pack_human.size else np.nan
            pack_median[f"{pack}_synth"] = float(np.median(pack_synth)) if pack_synth.size else np.nan
        tertile_delta = {}
        for label, part in binary.groupby("duration_tertile", observed=True):
            tertile_delta[str(label)] = cliffs_delta(
                _finite(part.loc[part["binary_group"] == "human_voice", name]),
                _finite(part.loc[part["binary_group"] == "synthetic", name]),
            )
        rho_duration = spearman(binary[name], binary["duration_s"])
        rho_speech = spearman(binary[name], binary["speech_ratio"])
        rho_rms = spearman(binary[name], binary["raw_speech_rms"])
        rho_padding = spearman(binary[name], binary["spectra_padding_ratio"])
        signs = [np.sign(delta), np.sign(pack_delta["PACK1"]), np.sign(pack_delta["PACK2"])]
        signs.extend(np.sign(value) for value in tertile_delta.values())
        finite_signs = [sign for sign in signs if np.isfinite(sign) and sign != 0]
        same_sign = len(set(finite_signs)) == 1 and len(finite_signs) == len(signs)
        weak_confound = all(
            np.isfinite(value) and abs(value) < CANDIDATE_MAX_ABS_SPEARMAN
            for value in (rho_duration, rho_speech, rho_rms)
        )
        mid = tertile_delta.get("mid", np.nan)
        candidate = bool(
            same_sign
            and np.isfinite(delta)
            and abs(delta) >= CANDIDATE_MIN_CLIFF
            and np.isfinite(mid)
            and abs(mid) >= CANDIDATE_MIN_MID_CLIFF
            and weak_confound
        )
        reasons = []
        if not same_sign:
            reasons.append("знак эффекта не совпал между кампаниями или третями длительности")
        if not (np.isfinite(delta) and abs(delta) >= CANDIDATE_MIN_CLIFF):
            reasons.append("общий эффект меньше порога")
        if not (np.isfinite(mid) and abs(mid) >= CANDIDATE_MIN_MID_CLIFF):
            reasons.append("в средней трети длительности эффект слабый")
        if not weak_confound:
            reasons.append("есть связь с длительностью, долей речи или громкостью")
        rows.append(
            {
                "feature": name,
                "explanation": FEATURE_NOTES[name],
                "n_human": int(human_values.size),
                "n_synth": int(synth_values.size),
                "median_human": _median(human[name]),
                "median_synth": _median(synth[name]),
                "cliffs_delta_synth_minus_human": delta,
                "median_pack1_human": pack_median["PACK1_human"],
                "median_pack1_synth": pack_median["PACK1_synth"],
                "cliffs_pack1": pack_delta["PACK1"],
                "median_pack2_human": pack_median["PACK2_human"],
                "median_pack2_synth": pack_median["PACK2_synth"],
                "cliffs_pack2": pack_delta["PACK2"],
                "cliffs_short": tertile_delta.get("short", np.nan),
                "cliffs_mid": mid,
                "cliffs_long": tertile_delta.get("long", np.nan),
                "spearman_duration": rho_duration,
                "spearman_speech_ratio": rho_speech,
                "spearman_rms": rho_rms,
                "spearman_padding": rho_padding,
                "candidate_for_next_experiment": int(candidate),
                "note": "кандидат на следующий независимый эксперимент" if candidate else "; ".join(reasons),
            }
        )
    return rows


def _percentile(value: float, reference: np.ndarray) -> float:
    if not np.isfinite(value) or reference.size == 0:
        return np.nan
    return float(np.mean(reference <= value) * 100.0)


def error_table(frame: pd.DataFrame, role: str) -> pd.DataFrame:
    binary = frame[frame["binary_group"].isin(["human_voice", "synthetic"])]
    human = binary[binary["binary_group"] == "human_voice"]
    synth = binary[binary["binary_group"] == "synthetic"]
    chosen = frame[frame["spectra_error_role"] == role].copy()
    for name in KEY_FEATURES:
        if name not in frame.columns:
            continue
        human_values = _finite(human[name])
        synth_values = _finite(synth[name])
        chosen[f"{name}__pct_in_human"] = [
            _percentile(value, human_values) for value in pd.to_numeric(chosen[name], errors="coerce")
        ]
        chosen[f"{name}__pct_in_synthetic"] = [
            _percentile(value, synth_values) for value in pd.to_numeric(chosen[name], errors="coerce")
        ]
    return chosen


def fp_pattern(frame: pd.DataFrame, feature_table: list[dict]) -> list[dict]:
    binary = frame[frame["binary_group"].isin(["human_voice", "synthetic"])]
    rows = []
    iqr = {}
    for item in feature_table:
        name = item["feature"]
        values = _finite(binary[name])
        if values.size < 4:
            iqr[name] = np.nan
        else:
            q25, q75 = np.percentile(values, [25, 75])
            iqr[name] = float(q75 - q25)
    fp = frame[frame["spectra_error_role"] == "FP"]
    tn = frame[frame["spectra_error_role"] == "TN"]
    tp = frame[frame["spectra_error_role"] == "TP"]
    for item in feature_table:
        name = item["feature"]
        med_fp = _median(fp[name])
        med_tn = _median(tn[name])
        med_tp = _median(tp[name])
        scale = iqr[name]
        if not np.isfinite(scale) or scale == 0:
            closer = ""
        else:
            closer = "synthetic" if abs(med_fp - med_tp) < abs(med_fp - med_tn) else "human_voice"
        rows.append(
            {
                "feature": name,
                "median_fp": med_fp,
                "median_tn": med_tn,
                "median_tp": med_tp,
                "fp_closer_to": closer,
                "candidate_for_next_experiment": item["candidate_for_next_experiment"],
            }
        )
    return rows


def normalization_summary(comparison: pd.DataFrame) -> dict:
    def change(left: str, right: str) -> float:
        if left not in comparison or right not in comparison:
            return np.nan
        delta = pd.to_numeric(comparison[right], errors="coerce") - pd.to_numeric(comparison[left], errors="coerce")
        values = delta.to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return np.nan
        return float(np.median(np.abs(values)))

    return {
        "median_abs_change_log_energy_peak_minus_raw": change("raw_log_energy_mean", "peak_log_energy_mean"),
        "median_abs_change_centroid_peak_minus_raw": change("raw_centroid_hz_mean", "peak_centroid_hz_mean"),
        "median_abs_change_flatness_peak_minus_raw": change("raw_flatness_mean", "peak_flatness_mean"),
        "median_abs_change_tilt_peak_minus_raw": change("raw_tilt", "peak_tilt"),
        "median_abs_change_log_energy_preemph_minus_raw": change("raw_log_energy_mean", "preemph_log_energy_mean"),
        "median_abs_change_centroid_preemph_minus_raw": change("raw_centroid_hz_mean", "preemph_centroid_hz_mean"),
        "median_abs_change_flatness_preemph_minus_raw": change("raw_flatness_mean", "preemph_flatness_mean"),
        "median_abs_change_tilt_preemph_minus_raw": change("raw_tilt", "preemph_tilt"),
        "median_mean_log_band_range": _median(comparison["mean_log_band_range"]),
        "median_log_of_mean_band_range": _median(comparison["log_of_mean_band_range"]),
        "median_logmel_band_range": _median(comparison["logmel_band_range"]),
        "median_mel_log_of_mean_band_range": _median(comparison["mel_log_of_mean_band_range"]),
        "median_preemph_fraction_above_1": _median(comparison["preemph_fraction_above_1"]),
        "calls_with_preemph_peak_above_1": int((pd.to_numeric(comparison["preemph_peak_max"], errors="coerce") > 1).sum()),
    }


def assert_no_forbidden(frame: pd.DataFrame, name: str) -> None:
    found = set(frame.columns) & set(FORBIDDEN_COLUMNS)
    if found:
        raise RuntimeError(f"{name} содержит запрещённые колонки: {sorted(found)}")


def write_analysis(output_dir: Path, calls: pd.DataFrame, comparison: pd.DataFrame) -> dict:
    assert_no_forbidden(calls, "call_level_features")
    features = feature_rows(calls)
    fp = error_table(calls, "FP")
    fn = error_table(calls, "FN")
    pattern = fp_pattern(calls, features)
    summary = {
        "n_calls": int(len(calls)),
        "n_feature_ok": int((calls["status"] == "ok").sum()),
        "n_no_speech": int((calls["status"] == "no_speech").sum()),
        "roles": calls["spectra_error_role"].value_counts(dropna=False).to_dict(),
        "binary_by_campaign": (
            calls[calls["binary_group"].isin(["human_voice", "synthetic"])]
            .groupby(["campaign", "binary_group"], observed=True)
            .size()
            .astype(int)
            .to_dict()
        ),
        "candidates": [row["feature"] for row in features if row["candidate_for_next_experiment"]],
        "fp_pattern": pattern,
        "normalization": normalization_summary(comparison),
        "focus_call_ids": [
            "8b34a750-b5df-4a6d-be75-7fbc207a9546",
            "fc0ba21c-245c-4388-b0ce-5da9ff149fae",
            "8dbd9d14-84ef-456d-922d-64a95f861196",
            "ed9640a9-b85e-43d4-8220-76726f51697f",
        ],
    }
    summary["binary_by_campaign"] = {str(key): int(value) for key, value in summary["binary_by_campaign"].items()}
    _write_frame(output_dir / "confounding_analysis.csv", pd.DataFrame(features))
    _write_frame(output_dir / "error_analysis_fp.csv", fp)
    _write_frame(output_dir / "error_analysis_fn.csv", fn)
    (output_dir / "summary_stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return summary


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        if isinstance(value, np.floating) and not np.isfinite(value):
            return None
        return value.item()
    raise TypeError(type(value))


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    forbidden = set(frame.columns) & set(FORBIDDEN_COLUMNS)
    if forbidden:
        raise RuntimeError(f"Запрещённые колонки в {path.name}: {sorted(forbidden)}")
    frame.to_csv(path, index=False, encoding="utf-8-sig")
