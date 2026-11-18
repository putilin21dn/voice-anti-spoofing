"""Call and segment tables built from the frozen scores and the previous cepstra."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from src.research.acoustic_representations.config import CQCC_CEPS, LFCC_CEPS
from src.research.acoustic_representations.signal import read_csv
from src.research.cepstral_padding_followup.common import (
    BETWEEN_ENDPOINTS,
    FAMILIES,
    FN_CALLS,
    FP_ENDPOINTS,
    OUTPUT,
    PREVIOUS,
    SCORES,
    SPECTRA_MODEL_WINDOW_S,
    WINDOWS,
    cliffs_delta,
    cliffs_interval,
    coeff_columns,
    finite,
    quartile_row,
    rank_tertile,
    read_table,
    segment_count_bin,
    spearman,
    write_frame,
)


def _windows_by_segment() -> dict[tuple[str, int], list[dict]]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in read_csv(WINDOWS):
        grouped[(row["call_id"], int(row["segment_index"]))].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: int(item["window_index"]))
    return grouped


def _recompute_between(segments: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    max_abs = {prefix: 0.0 for prefix, _ in FAMILIES}
    compared = {prefix: 0 for prefix, _ in FAMILIES}
    for call_id, part in segments.groupby("call_id", sort=False):
        durations = pd.to_numeric(part["duration_s"], errors="coerce").to_numpy(dtype=np.float64)
        row: dict = {
            "call_id": call_id,
            "n_speech_segments": int(len(part)),
            "segment_duration_min_s": float(np.nanmin(durations)) if durations.size else np.nan,
            "segment_duration_median_s": float(np.nanmedian(durations)) if durations.size else np.nan,
            "segment_duration_max_s": float(np.nanmax(durations)) if durations.size else np.nan,
            "segment_durations_s": ";".join(f"{value:.3f}" for value in durations if np.isfinite(value)),
            "n_shorter_than_cqt_window": int(pd.to_numeric(part["shorter_than_cqt_window"], errors="coerce").fillna(0).sum()),
        }
        for prefix, count in FAMILIES:
            columns = coeff_columns(prefix, count, "mean")
            matrix = part[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
            valid = np.isfinite(matrix).all(axis=1)
            used = matrix[valid]
            row[f"n_valid_{prefix}"] = int(valid.sum())
            row[f"n_excluded_{prefix}"] = int((~valid).sum())
            if used.shape[0] < 2:
                std = np.full(count, np.nan)
            else:
                std = np.std(used, axis=0, ddof=1)
            for index, value in enumerate(std):
                row[f"{prefix}_c{index:02d}_between_segment_std"] = float(value)
            finite_std = std[np.isfinite(std)]
            row[f"{prefix}_between_segment_std_mean"] = float(np.mean(finite_std)) if finite_std.size else np.nan
        rows.append(row)
    recomputed = pd.DataFrame(rows)
    stored = read_table(PREVIOUS / "call_level_features.csv")
    merged = recomputed.merge(stored, on="call_id", suffixes=("", "_stored"))
    for prefix, count in FAMILIES:
        for index in range(count):
            name = f"{prefix}_c{index:02d}_between_segment_std"
            left = pd.to_numeric(merged[name], errors="coerce")
            right = pd.to_numeric(merged[f"{name}_stored"], errors="coerce")
            both = np.isfinite(left) & np.isfinite(right)
            if both.any():
                diff = np.max(np.abs(left[both] - right[both]))
                max_abs[prefix] = max(max_abs[prefix], float(diff))
                compared[prefix] += int(both.sum())
            left_missing = np.isfinite(left) ^ np.isfinite(right)
            if left_missing.any():
                max_abs[prefix] = max(max_abs[prefix], 1.0)
    pooled = {}
    for prefix, _count in FAMILIES:
        name = f"{prefix}_between_segment_std_mean"
        left = pd.to_numeric(merged[name], errors="coerce")
        right = pd.to_numeric(merged[f"{name}_stored"], errors="coerce")
        both = np.isfinite(left) & np.isfinite(right)
        pooled[prefix] = float(np.max(np.abs(left[both] - right[both]))) if both.any() else np.nan
    checksum = {"max_abs_coefficient": max_abs, "n_compared": compared, "max_abs_mean": pooled}
    return recomputed, checksum


def _effect_row(
    comparison: str,
    endpoint: str,
    family: str,
    stratum: str,
    reference: np.ndarray,
    other: np.ndarray,
    note: str,
    n_reference: int | None = None,
    n_other: int | None = None,
) -> dict:
    low, high, status = cliffs_interval(reference, other)
    ref_q = quartile_row(reference)
    other_q = quartile_row(other)
    if family == "campaign" and stratum == "PACK2":
        status = "pack2_small" if status == "ok" else status
    return {
        "comparison": comparison,
        "endpoint": endpoint,
        "stratum_family": family,
        "stratum": stratum,
        "n_reference": int(reference.size if n_reference is None else n_reference),
        "n_other": int(other.size if n_other is None else n_other),
        "n_reference_finite": int(reference.size),
        "n_other_finite": int(other.size),
        "reference_median": ref_q["median"],
        "reference_q25": ref_q["q25"],
        "reference_q75": ref_q["q75"],
        "other_median": other_q["median"],
        "other_q25": other_q["q25"],
        "other_q75": other_q["q75"],
        "cliffs_delta": cliffs_delta(reference, other),
        "ci_low": low,
        "ci_high": high,
        "ci_status": status,
        "spearman": np.nan,
        "spearman_n": np.nan,
        "note": note,
    }


def _group_effects(frame: pd.DataFrame, endpoints: tuple[str, ...], comparison: str, reference_role: str, other_role: str, note: str) -> list[dict]:
    rows = []
    families = {
        "overall": [("all", frame)],
        "rms": [(label, part) for label, part in frame.groupby("rms_tertile", dropna=True)],
        "call_duration": [(label, part) for label, part in frame.groupby("call_duration_tertile", dropna=True)],
        "speech_duration": [(label, part) for label, part in frame.groupby("speech_duration_tertile", dropna=True)],
        "speech_ratio": [(label, part) for label, part in frame.groupby("speech_ratio_tertile", dropna=True)],
        "campaign": [(label, part) for label, part in frame.groupby("campaign", dropna=True)],
        "n_segments": [(label, part) for label, part in frame.groupby("segment_count_bin", dropna=True) if label in {"1", "2-4", "5+"}],
    }
    for family, parts in families.items():
        for label, part in parts:
            if family == "n_segments" and label == "1":
                allowed = [name for name in endpoints if "between_segment" not in name]
            else:
                allowed = list(endpoints)
            reference_frame = part[part["spectra_error_role"] == reference_role] if comparison == "fp_vs_tn" else part[part["binary_group"] == "human_voice"]
            other_frame = part[part["spectra_error_role"] == other_role] if comparison == "fp_vs_tn" else part[part["binary_group"] == "synthetic"]
            if comparison == "fp_vs_tn":
                reference_frame = part[part["spectra_error_role"] == "TN"]
                other_frame = part[part["spectra_error_role"] == "FP"]
            for endpoint in allowed:
                if endpoint not in part.columns:
                    continue
                rows.append(
                    _effect_row(
                        comparison,
                        endpoint,
                        family,
                        str(label),
                        finite(reference_frame[endpoint]),
                        finite(other_frame[endpoint]),
                        note,
                        n_reference=int(len(reference_frame)),
                        n_other=int(len(other_frame)),
                    )
                )
    return rows


def _association_rows(frame: pd.DataFrame) -> list[dict]:
    rows = []
    covariates = (
        "n_speech_segments",
        "speech_duration_s",
        "duration_s",
        "speech_ratio",
        "raw_speech_rms",
        "spectra_padding_ratio",
    )
    groups = {
        "binary": frame,
        "human_voice": frame[frame["binary_group"] == "human_voice"],
        "synthetic": frame[frame["binary_group"] == "synthetic"],
    }
    for endpoint in BETWEEN_ENDPOINTS:
        for covariate in covariates:
            for group_name, part in groups.items():
                estimate, count = spearman(part[endpoint], part[covariate])
                rows.append(
                    {
                        "comparison": "spearman",
                        "endpoint": endpoint,
                        "stratum_family": group_name,
                        "stratum": covariate,
                        "n_reference": count,
                        "n_other": count,
                        "n_reference_finite": count,
                        "n_other_finite": count,
                        "reference_median": np.nan,
                        "reference_q25": np.nan,
                        "reference_q75": np.nan,
                        "other_median": np.nan,
                        "other_q25": np.nan,
                        "other_q75": np.nan,
                        "cliffs_delta": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "ci_status": "",
                        "spearman": estimate,
                        "spearman_n": count,
                        "note": "Связь внутри уже отобранного дня. Это не влияние ковариаты.",
                    }
                )
    margin_groups = {
        "binary": frame,
        "human_voice": frame[frame["binary_group"] == "human_voice"],
        "synthetic": frame[frame["binary_group"] == "synthetic"],
        "TN": frame[frame["spectra_error_role"] == "TN"],
        "FP": frame[frame["spectra_error_role"] == "FP"],
        "TP": frame[frame["spectra_error_role"] == "TP"],
        "FN": frame[frame["spectra_error_role"] == "FN"],
    }
    for covariate in ("spectra_padding_ratio", "speech_duration_s", "spectra_n_windows", "n_speech_segments"):
        for group_name, part in margin_groups.items():
            estimate, count = spearman(part["spectra0_margin_frozen"], part[covariate])
            rows.append(
                {
                    "comparison": "spearman_margin",
                    "endpoint": "spectra0_margin_frozen",
                    "stratum_family": group_name,
                    "stratum": covariate,
                    "n_reference": count,
                    "n_other": count,
                    "n_reference_finite": count,
                    "n_other_finite": count,
                    "reference_median": np.nan,
                    "reference_q25": np.nan,
                    "reference_q75": np.nan,
                    "other_median": np.nan,
                    "other_q25": np.nan,
                    "other_q75": np.nan,
                    "cliffs_delta": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "ci_status": "fragile" if count < 8 else "",
                    "spearman": estimate,
                    "spearman_n": count,
                    "note": "Сохранённый margin. Повтор хвоста не отделён от короткого хвоста.",
                }
            )
    return rows


def _window_fields(windows: dict[tuple[str, int], list[dict]]) -> pd.DataFrame:
    rows = []
    for (call_id, segment_index), items in windows.items():
        valid = np.array([float(item["valid_speech_audio_sec"]) for item in items], dtype=np.float64)
        padded = np.array([item["padded_to_model_length"] == "True" for item in items])
        margins = np.array([float(item["margin"]) for item in items], dtype=np.float64)
        repeated = np.maximum(np.where(padded, SPECTRA_MODEL_WINDOW_S - valid, 0.0), 0.0)
        last = items[-1]
        denominator = float(valid.sum() + repeated.sum())
        rows.append(
            {
                "call_id": call_id,
                "segment_index": segment_index,
                "n_windows": len(items),
                "n_full_windows": int((~padded).sum()),
                "n_padded_windows": int(padded.sum()),
                "tail_valid_s": float(last["valid_speech_audio_sec"]),
                "tail_repeated_s": float(max(SPECTRA_MODEL_WINDOW_S - float(last["valid_speech_audio_sec"]), 0.0))
                if last["padded_to_model_length"] == "True"
                else 0.0,
                "segment_real_fraction": float(valid.sum() / denominator) if denominator else np.nan,
                "segment_repeated_fraction": float(repeated.sum() / denominator) if denominator else np.nan,
                "mean_window_margin": float(np.mean(margins)),
                "mean_full_window_margin": float(np.mean(margins[~padded])) if (~padded).any() else np.nan,
                "mean_padded_window_margin": float(np.mean(margins[padded])) if padded.any() else np.nan,
                "window_margins": ";".join(f"{value:.6f}" for value in margins),
                "window_valid_s": ";".join(f"{value:.3f}" for value in valid),
                "window_padded": ";".join("1" if flag else "0" for flag in padded),
                "tail_start_time": float(last["start_time"]),
                "tail_end_time": float(last["end_time"]),
            }
        )
    return pd.DataFrame(rows)


def _percentile(value: float, reference: np.ndarray) -> float:
    reference = reference[np.isfinite(reference)]
    if not np.isfinite(value) or reference.size == 0:
        return np.nan
    return float(np.mean(reference <= value))


def _shape_distance(row: pd.Series, median: np.ndarray, columns: list[str]) -> float:
    values = row[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not np.isfinite(median).all():
        return np.nan
    return float(np.sqrt(np.mean((values - median) ** 2)))


def _false_negative_table(segments: pd.DataFrame, calls: pd.DataFrame) -> pd.DataFrame:
    binary_segments = segments[segments["binary_group"].isin(["human_voice", "synthetic"])].copy()
    cqcc_columns = coeff_columns("cqcc", CQCC_CEPS, "mean")[1:]
    lfcc_columns = coeff_columns("lfcc", LFCC_CEPS, "mean")[1:]
    human = binary_segments[binary_segments["binary_group"] == "human_voice"]
    human_c1 = finite(human["cqcc_c01_mean"])
    human_cqcc_median = human[cqcc_columns].apply(pd.to_numeric, errors="coerce").median().to_numpy(dtype=np.float64)
    human_lfcc_median = human[lfcc_columns].apply(pd.to_numeric, errors="coerce").median().to_numpy(dtype=np.float64)
    human_cqcc_distance = human.apply(lambda row: _shape_distance(row, human_cqcc_median, cqcc_columns), axis=1).to_numpy(dtype=np.float64)
    human_lfcc_distance = human.apply(lambda row: _shape_distance(row, human_lfcc_median, lfcc_columns), axis=1).to_numpy(dtype=np.float64)
    rows = []
    for call_id in FN_CALLS:
        part = segments[segments["call_id"] == call_id].sort_values("segment_index")
        call = calls[calls["call_id"] == call_id].iloc[0]
        others = binary_segments[(binary_segments["binary_group"] == "synthetic") & (binary_segments["call_id"] != call_id)]
        synth_c1 = finite(others["cqcc_c01_mean"])
        synth_cqcc_median = others[cqcc_columns].apply(pd.to_numeric, errors="coerce").median().to_numpy(dtype=np.float64)
        synth_lfcc_median = others[lfcc_columns].apply(pd.to_numeric, errors="coerce").median().to_numpy(dtype=np.float64)
        synth_cqcc_distance = others.apply(lambda row: _shape_distance(row, synth_cqcc_median, cqcc_columns), axis=1).to_numpy(dtype=np.float64)
        synth_lfcc_distance = others.apply(lambda row: _shape_distance(row, synth_lfcc_median, lfcc_columns), axis=1).to_numpy(dtype=np.float64)
        call_c1 = pd.to_numeric(part["cqcc_c01_mean"], errors="coerce")
        for record in part.to_dict(orient="records"):
            c1 = float(record["cqcc_c01_mean"]) if np.isfinite(record["cqcc_c01_mean"]) else np.nan
            human_distance = _shape_distance(pd.Series(record), human_cqcc_median, cqcc_columns)
            synth_distance = _shape_distance(pd.Series(record), synth_cqcc_median, cqcc_columns)
            lfcc_human = _shape_distance(pd.Series(record), human_lfcc_median, lfcc_columns)
            lfcc_synth = _shape_distance(pd.Series(record), synth_lfcc_median, lfcc_columns)
            if np.isfinite(c1):
                closer = "human_voice" if abs(c1 - np.median(human_c1)) <= abs(c1 - np.median(synth_c1)) else "synthetic"
            else:
                closer = ""
            rows.append(
                {
                    "call_id": call_id,
                    "primary_label": call["primary_label"],
                    "campaign": call["campaign"],
                    "spectra_error_role": "FN",
                    "spectra0_margin_frozen": call["spectra0_margin_frozen"],
                    "n_speech_segments": call["n_speech_segments"],
                    "lfcc_between_segment_std_mean": call["lfcc_between_segment_std_mean"],
                    "cqcc_between_segment_std_mean": call["cqcc_between_segment_std_mean"],
                    "spafe_cqcc_between_segment_std_mean": call["spafe_cqcc_between_segment_std_mean"],
                    "cqcc_c01_call_mean": call["cqcc_c01_mean"],
                    "segment_index": record["segment_index"],
                    "duration_s": record["duration_s"],
                    "raw_rms": record["raw_rms"],
                    "f0_median_hz": record["f0_median_hz"],
                    "f0_iqr_hz": record["f0_iqr_hz"],
                    "f0_voiced_fraction": record["f0_voiced_fraction"],
                    "lfcc_c01_mean": record["lfcc_c01_mean"],
                    "cqcc_c01_mean": c1,
                    "spafe_cqcc_c01_mean": record["spafe_cqcc_c01_mean"],
                    "cqcc_c01_deviation_from_call_median": c1 - float(np.nanmedian(call_c1)),
                    "n_windows": record["n_windows"],
                    "n_full_windows": record["n_full_windows"],
                    "tail_valid_s": record["tail_valid_s"],
                    "tail_repeated_s": record["tail_repeated_s"],
                    "segment_repeated_fraction": record["segment_repeated_fraction"],
                    "mean_window_margin": record["mean_window_margin"],
                    "mean_full_window_margin": record["mean_full_window_margin"],
                    "mean_padded_window_margin": record["mean_padded_window_margin"],
                    "window_margins": record["window_margins"],
                    "window_valid_s": record["window_valid_s"],
                    "window_padded": record["window_padded"],
                    "cqcc_c1_percentile_in_human_segments": _percentile(c1, human_c1),
                    "cqcc_c1_percentile_in_other_synthetic_segments": _percentile(c1, synth_c1),
                    "cqcc_c1_closer_segment_median": closer,
                    "cqcc_shape_rms_to_human_median": human_distance,
                    "cqcc_shape_rms_to_other_synthetic_median": synth_distance,
                    "cqcc_shape_distance_percentile_in_human": _percentile(human_distance, human_cqcc_distance),
                    "cqcc_shape_distance_percentile_in_other_synthetic": _percentile(synth_distance, synth_cqcc_distance),
                    "lfcc_shape_rms_to_human_median": lfcc_human,
                    "lfcc_shape_rms_to_other_synthetic_median": lfcc_synth,
                    "lfcc_shape_distance_percentile_in_human": _percentile(lfcc_human, human_lfcc_distance),
                    "lfcc_shape_distance_percentile_in_other_synthetic": _percentile(lfcc_synth, synth_lfcc_distance),
                }
            )
    return pd.DataFrame(rows)


def _cross_counts(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for duration_bin, duration_part in frame.groupby("speech_duration_tertile", dropna=True):
        for padding_bin, part in duration_part.groupby("padding_tertile", dropna=True):
            rows.append(
                {
                    "row_type": "crosstab",
                    "call_id": "",
                    "speech_duration_tertile": duration_bin,
                    "padding_tertile": padding_bin,
                    "n_fp": int((part["spectra_error_role"] == "FP").sum()),
                    "n_tn": int((part["spectra_error_role"] == "TN").sum()),
                    "n_tp": int((part["spectra_error_role"] == "TP").sum()),
                    "n_fn": int((part["spectra_error_role"] == "FN").sum()),
                    "note": "Границы терцилей зафиксированы по всем 251 бинарным звонкам до сравнения FP и TN.",
                }
            )
    return rows


def build_tables() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stored_calls = read_table(PREVIOUS / "call_level_features.csv")
    stored_segments = read_table(PREVIOUS / "segment_level_features.csv")
    scores = {row["call_id"]: row for row in read_csv(SCORES)}
    windows = _windows_by_segment()
    recomputed, checksum = _recompute_between(stored_segments)
    if max(checksum["max_abs_coefficient"].values()) > 0.05:
        raise RuntimeError(f"Межсегментное СКО не сходится с прошлым отчётом: {checksum}")

    window_frame = _window_fields(windows)
    segments = stored_segments.merge(window_frame, on=["call_id", "segment_index"], how="left", validate="one_to_one")
    segments = segments.merge(
        stored_calls[["call_id", "primary_label", "binary_group", "campaign", "spectra_error_role"]],
        on="call_id",
        how="left",
        validate="many_to_one",
    )
    segment_keep = [
        "call_id",
        "segment_index",
        "primary_label",
        "binary_group",
        "campaign",
        "spectra_error_role",
        "start_time",
        "end_time",
        "duration_s",
        "n_samples",
        "status",
        "shorter_than_cqt_window",
        "raw_rms",
        "raw_peak",
        "f0_median_hz",
        "f0_iqr_hz",
        "f0_voiced_fraction",
        *coeff_columns("lfcc", LFCC_CEPS, "mean"),
        *coeff_columns("cqcc", CQCC_CEPS, "mean"),
        *coeff_columns("spafe_cqcc", CQCC_CEPS, "mean"),
        "n_windows",
        "n_full_windows",
        "n_padded_windows",
        "tail_valid_s",
        "tail_repeated_s",
        "tail_start_time",
        "tail_end_time",
        "segment_real_fraction",
        "segment_repeated_fraction",
        "mean_window_margin",
        "mean_full_window_margin",
        "mean_padded_window_margin",
        "window_margins",
        "window_valid_s",
        "window_padded",
    ]
    segment_out = segments[segment_keep].sort_values(["call_id", "segment_index"])
    write_frame(OUTPUT / "segment_level_cepstral_features.csv", segment_out)

    calls = stored_calls.merge(recomputed, on="call_id", how="left", suffixes=("_previous", ""))
    if "n_speech_segments_previous" in calls.columns:
        calls["n_speech_segments"] = calls["n_speech_segments"].fillna(calls["n_speech_segments_previous"])
    def _margin(call_id: str) -> float:
        raw = scores[call_id]["spectra0_score"]
        return float(raw) if raw else np.nan

    calls["spectra0_margin_frozen"] = calls["call_id"].map(_margin)
    call_window_rows = []
    by_call: dict[str, list[dict]] = defaultdict(list)
    for (call_id, _segment), items in windows.items():
        by_call[call_id].extend(items)
    max_margin_gap = 0.0
    for call_id, items in by_call.items():
        margins = np.array([float(item["margin"]) for item in items], dtype=np.float64)
        padded = np.array([item["padded_to_model_length"] == "True" for item in items])
        stored_margin = scores[call_id]["spectra0_score"]
        if stored_margin:
            max_margin_gap = max(max_margin_gap, abs(float(np.mean(margins)) - float(stored_margin)))
        call_window_rows.append(
            {
                "call_id": call_id,
                "mean_full_window_margin": float(np.mean(margins[~padded])) if (~padded).any() else np.nan,
                "mean_padded_window_margin": float(np.mean(margins[padded])) if padded.any() else np.nan,
                "n_full_windows_check": int((~padded).sum()),
                "n_padded_windows_check": int(padded.sum()),
            }
        )
    calls = calls.merge(pd.DataFrame(call_window_rows), on="call_id", how="left")
    binary_mask = calls["binary_group"].isin(["human_voice", "synthetic"])
    for column, name in (
        ("raw_speech_rms", "rms_tertile"),
        ("duration_s", "call_duration_tertile"),
        ("speech_duration_s", "speech_duration_tertile"),
        ("speech_ratio", "speech_ratio_tertile"),
        ("spectra_padding_ratio", "padding_tertile"),
    ):
        calls[name] = pd.NA
        calls.loc[binary_mask, name] = rank_tertile(calls.loc[binary_mask].copy(), column)
    calls["segment_count_bin"] = calls["n_speech_segments"].fillna(0).astype(int).map(segment_count_bin)
    calls["within_segment_lfcc_delta_mad_mean"] = calls["lfcc_temporal_mad_mean"]
    calls["within_segment_cqcc_delta_mad_mean"] = calls["cqcc_temporal_mad_mean"]
    calls["within_segment_spafe_cqcc_delta_mad_mean"] = calls["spafe_cqcc_temporal_mad_mean"]

    summary_columns = [
        "call_id",
        "status",
        "primary_label",
        "binary_group",
        "campaign",
        "spectra_error_role",
        "spectra0_margin_frozen",
        "duration_s",
        "speech_duration_s",
        "speech_ratio",
        "raw_speech_rms",
        "explanatory_f0_median_hz",
        "explanatory_f0_iqr_hz",
        "explanatory_f0_voiced_fraction",
        "n_speech_segments",
        "segment_count_bin",
        "segment_duration_min_s",
        "segment_duration_median_s",
        "segment_duration_max_s",
        "segment_durations_s",
        "n_shorter_than_cqt_window",
        "n_valid_lfcc",
        "n_excluded_lfcc",
        "n_valid_cqcc",
        "n_excluded_cqcc",
        "n_valid_spafe_cqcc",
        "n_excluded_spafe_cqcc",
        "lfcc_between_segment_std_mean",
        "cqcc_between_segment_std_mean",
        "spafe_cqcc_between_segment_std_mean",
        "cqcc_c01_mean",
        "within_segment_lfcc_delta_mad_mean",
        "within_segment_cqcc_delta_mad_mean",
        "within_segment_spafe_cqcc_delta_mad_mean",
        "spectra_n_windows",
        "spectra_n_padded_windows",
        "spectra_padding_ratio",
        "spectra_valid_audio_s",
        "spectra_repeated_audio_s",
        "mean_full_window_margin",
        "mean_padded_window_margin",
        "rms_tertile",
        "call_duration_tertile",
        "speech_duration_tertile",
        "speech_ratio_tertile",
        "padding_tertile",
    ]
    for prefix, count in FAMILIES:
        summary_columns.extend(f"{prefix}_c{index:02d}_between_segment_std" for index in range(count))
    # Prefer the recomputed names. merge suffixes may have renamed stored copies.
    available = []
    for name in summary_columns:
        if name in calls.columns:
            available.append(name)
        elif f"{name}_previous" in calls.columns:
            calls[name] = calls[f"{name}_previous"]
            available.append(name)
        else:
            raise RuntimeError(f"Нет колонки {name}")
    call_out = calls[available].sort_values("call_id")
    write_frame(OUTPUT / "call_level_cepstral_summary.csv", call_out)

    binary = calls[binary_mask].copy()
    effect_rows = _group_effects(
        binary,
        BETWEEN_ENDPOINTS,
        "synthetic_vs_human",
        "human_voice",
        "synthetic",
        "reference = human_voice, other = synthetic. Положительная delta: у синтетики значение обычно больше. Тот же 251 звонок, не новый день.",
    )
    effect_rows.extend(
        _group_effects(
            binary,
            FP_ENDPOINTS,
            "fp_vs_tn",
            "TN",
            "FP",
            "reference = TN, other = FP. Положительная delta: у FP значение обычно больше, чем у TN.",
        )
    )
    effect_rows.extend(_association_rows(binary))
    effects = pd.DataFrame(effect_rows)
    write_frame(OUTPUT / "matched_group_analysis.csv", effects)

    fn = _false_negative_table(segment_out, call_out)
    write_frame(OUTPUT / "false_negative_segment_analysis.csv", fn)

    human_errors = call_out[call_out["spectra_error_role"].isin(["FP", "TN"])].copy()
    fp_columns = [
        "row_type",
        "call_id",
        "primary_label",
        "campaign",
        "spectra_error_role",
        "spectra0_margin_frozen",
        "speech_duration_s",
        "duration_s",
        "n_speech_segments",
        "segment_count_bin",
        "speech_ratio",
        "spectra_padding_ratio",
        "spectra_n_windows",
        "spectra_n_padded_windows",
        "mean_full_window_margin",
        "mean_padded_window_margin",
        "lfcc_between_segment_std_mean",
        "cqcc_between_segment_std_mean",
        "spafe_cqcc_between_segment_std_mean",
        "cqcc_c01_mean",
        "raw_speech_rms",
        "explanatory_f0_iqr_hz",
        "explanatory_f0_voiced_fraction",
        "speech_duration_tertile",
        "padding_tertile",
        "n_fp",
        "n_tn",
        "n_tp",
        "n_fn",
        "note",
    ]
    fp_rows = human_errors.copy()
    fp_rows["row_type"] = "call"
    fp_rows["n_fp"] = np.nan
    fp_rows["n_tn"] = np.nan
    fp_rows["n_tp"] = np.nan
    fp_rows["n_fn"] = np.nan
    fp_rows["note"] = "Строка звонка. Сравнение группы лежит в matched_group_analysis.csv, comparison=fp_vs_tn."
    cross = pd.DataFrame(_cross_counts(binary))
    for column in fp_columns:
        if column not in fp_rows.columns:
            fp_rows[column] = np.nan
        if column not in cross.columns:
            cross[column] = np.nan
    fp_out = pd.concat([fp_rows[fp_columns], cross[fp_columns]], ignore_index=True)
    write_frame(OUTPUT / "false_positive_padding_analysis.csv", fp_out)

    edges = {}
    for column, label in (
        ("raw_speech_rms", "rms"),
        ("duration_s", "call_duration"),
        ("speech_duration_s", "speech_duration"),
        ("speech_ratio", "speech_ratio"),
        ("spectra_padding_ratio", "padding_ratio"),
    ):
        tertile_column = {
            "rms": "rms_tertile",
            "call_duration": "call_duration_tertile",
            "speech_duration": "speech_duration_tertile",
            "speech_ratio": "speech_ratio_tertile",
            "padding_ratio": "padding_tertile",
        }[label]
        edges[label] = {}
        for name, part in binary.groupby(tertile_column, dropna=True):
            values = finite(part[column])
            edges[label][str(name)] = {
                "n": int(values.size),
                "min": float(values.min()) if values.size else np.nan,
                "max": float(values.max()) if values.size else np.nan,
            }
    return {
        "checksum": checksum,
        "max_margin_gap": max_margin_gap,
        "n_calls": int(len(call_out)),
        "n_segments": int(len(segment_out)),
        "edges": edges,
        "n_between_human": int(np.isfinite(pd.to_numeric(binary.loc[binary["binary_group"] == "human_voice", "lfcc_between_segment_std_mean"], errors="coerce")).sum()),
        "n_between_synth": int(np.isfinite(pd.to_numeric(binary.loc[binary["binary_group"] == "synthetic", "lfcc_between_segment_std_mean"], errors="coerce")).sum()),
    }
