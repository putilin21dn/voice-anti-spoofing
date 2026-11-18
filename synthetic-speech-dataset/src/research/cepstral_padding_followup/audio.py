"""Offline cepstral checks: repeated tail, zero fill, and loudness gain.

These waveforms are not Spectra-0 inputs. Logits are not recomputed.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from src.research.acoustic_representations.config import CQCC_CEPS, LFCC_CEPS, PEAK_NORM_TARGET, PREEMPHASIS
from src.research.acoustic_representations.features import cqcc_librosa, cqcc_spafe, lfcc_matrix, stft_power
from src.research.acoustic_representations.run import load_records
from src.research.acoustic_representations.signal import load_subscriber, peak_normalize, preemphasis
from src.research.cepstral_padding_followup.common import (
    BETWEEN_ENDPOINTS,
    FAMILIES,
    OUTPUT,
    RMS_TARGET,
    SAMPLE_RATE,
    TARGET_SAMPLES,
    cliffs_delta,
    cliffs_interval,
    coeff_columns,
    finite,
    quartile_row,
    read_table,
    write_frame,
)

ORDER_NOTE = "tile_or_gain_on_raw_then_preemphasis_0.97_then_cepstrum"
ALIGNMENT_NOTE = "times_rounded_to_0.001s_slice_round_time_times_8000_not_spectra_16khz_buffer"


def _empty_measure(status: str) -> dict:
    row = {"status": status, "n_samples": 0, "peak_before_preemphasis": np.nan, "fraction_above_1_before_preemphasis": np.nan}
    for prefix, count in FAMILIES:
        row[f"n_{prefix}_frames"] = 0
        for index in range(count):
            row[f"{prefix}_c{index:02d}_mean"] = np.nan
            row[f"{prefix}_c{index:02d}_delta_mad"] = np.nan
    return row


def _store_matrix(row: dict, prefix: str, matrix: np.ndarray, count: int) -> None:
    if matrix.size == 0:
        return
    means = matrix.mean(axis=0)
    if matrix.shape[0] >= 2:
        delta = np.mean(np.abs(np.diff(matrix, axis=0)), axis=0)
    else:
        delta = np.full(count, np.nan)
    row[f"n_{prefix}_frames"] = int(matrix.shape[0])
    for index in range(count):
        row[f"{prefix}_c{index:02d}_mean"] = float(means[index])
        row[f"{prefix}_c{index:02d}_delta_mad"] = float(delta[index])


def measure(raw: np.ndarray) -> dict:
    if raw.size < 160:
        return _empty_measure("shorter_than_one_frame")
    row = _empty_measure("ok")
    row["n_samples"] = int(raw.size)
    row["peak_before_preemphasis"] = float(np.max(np.abs(raw)))
    row["fraction_above_1_before_preemphasis"] = float(np.mean(np.abs(raw) >= 1.0))
    emphasized = preemphasis(raw)
    try:
        _store_matrix(row, "lfcc", lfcc_matrix(stft_power(emphasized)), LFCC_CEPS)
    except Exception:
        row["status"] = "lfcc_error"
    try:
        _store_matrix(row, "cqcc", cqcc_librosa(emphasized), CQCC_CEPS)
    except Exception:
        row["status"] = "cqcc_error"
    try:
        _store_matrix(row, "spafe_cqcc", cqcc_spafe(emphasized), CQCC_CEPS)
    except Exception:
        row["status"] = "spafe_error"
    return row


def _tile(chunk: np.ndarray, target: int) -> np.ndarray:
    if chunk.size >= target:
        return chunk[:target].copy()
    repeats = int(np.ceil(target / chunk.size))
    return np.tile(chunk, repeats)[:target]


def _zero_fill(chunk: np.ndarray, target: int) -> np.ndarray:
    output = np.zeros(target, dtype=np.float64)
    used = min(chunk.size, target)
    output[:used] = chunk[:used]
    return output


def _rms_scale(raw: np.ndarray) -> tuple[np.ndarray, float]:
    if raw.size == 0:
        return raw.copy(), np.nan
    rms = float(np.sqrt(np.mean(raw * raw)))
    if rms < 1e-8:
        return raw.copy(), np.nan
    gain = RMS_TARGET / rms
    return raw * gain, gain


def _base_row(segment: pd.Series, scope: str, condition: str, waveform: np.ndarray, n_real: int, copied: bool) -> dict:
    real_fraction = n_real / waveform.size if waveform.size else np.nan
    return {
        "call_id": segment["call_id"],
        "segment_index": int(segment["segment_index"]),
        "binary_group": segment["binary_group"],
        "campaign": segment["campaign"],
        "spectra_error_role": segment["spectra_error_role"],
        "scope": scope,
        "condition": condition,
        "copied_from_tail": int(copied),
        "sample_rate_hz": SAMPLE_RATE,
        "preemphasis": PREEMPHASIS,
        "target_samples": TARGET_SAMPLES,
        "target_duration_s": TARGET_SAMPLES / SAMPLE_RATE,
        "n_waveform_samples": int(waveform.size),
        "n_real_samples": int(n_real),
        "real_audio_fraction": real_fraction,
        "repeated_or_zero_fraction": 1.0 - real_fraction if np.isfinite(real_fraction) else np.nan,
        "construct_order": ORDER_NOTE,
        "sample_alignment": ALIGNMENT_NOTE,
        "tail_valid_s": segment["tail_valid_s"],
        "segment_duration_s": segment["duration_s"],
    }


def _attach(base: dict, measured: dict) -> dict:
    base.update(measured)
    return base


def _waveforms(audio: np.ndarray, segment: pd.Series) -> tuple[dict[tuple[str, str], tuple[np.ndarray, int, bool]], str]:
    n_audio = audio.size
    start = min(n_audio, max(0, int(round(float(segment["start_time"]) * SAMPLE_RATE))))
    end = min(n_audio, max(0, int(round(float(segment["end_time"]) * SAMPLE_RATE))))
    tail_start = min(n_audio, max(0, int(round(float(segment["tail_start_time"]) * SAMPLE_RATE))))
    tail_end = min(n_audio, max(0, int(round(float(segment["tail_end_time"]) * SAMPLE_RATE))))
    note = "rounded_1ms"
    if tail_start < start or tail_end > end:
        note = "tail_bounds_outside_segment_span"
    elif tail_end != end:
        note = "segment_extends_past_recorded_tail"
    segment_raw = audio[start:end]
    tail = audio[tail_start:tail_end]
    prefix = audio[start:tail_start] if tail_start >= start else np.empty(0, dtype=np.float64)
    same_span = start == tail_start and end == tail_end
    built: dict[tuple[str, str], tuple[np.ndarray, int, bool]] = {}
    if tail.size == 0 or segment_raw.size == 0:
        return built, "empty_slice"
    n_real_tail = int(min(tail.size, TARGET_SAMPLES))
    built[("tail", "real")] = (tail, int(tail.size), False)
    built[("tail", "repeat")] = (_tile(tail, TARGET_SAMPLES), n_real_tail, False)
    built[("tail", "zero")] = (_zero_fill(tail, TARGET_SAMPLES), n_real_tail, False)
    built[("segment", "real")] = (segment_raw, int(segment_raw.size), False)
    n_real_extended = int(prefix.size + n_real_tail)
    if same_span:
        built[("segment", "repeat")] = (*built[("tail", "repeat")][:2], True)
        built[("segment", "zero")] = (*built[("tail", "zero")][:2], True)
    else:
        built[("segment", "repeat")] = (np.concatenate([prefix, _tile(tail, TARGET_SAMPLES)]), n_real_extended, False)
        built[("segment", "zero")] = (np.concatenate([prefix, _zero_fill(tail, TARGET_SAMPLES)]), n_real_extended, False)
    return built, note


def _padding_columns() -> list[str]:
    columns = [
        "call_id",
        "segment_index",
        "binary_group",
        "campaign",
        "spectra_error_role",
        "scope",
        "condition",
        "copied_from_tail",
        "status",
        "sample_rate_hz",
        "preemphasis",
        "target_samples",
        "target_duration_s",
        "n_waveform_samples",
        "n_real_samples",
        "real_audio_fraction",
        "repeated_or_zero_fraction",
        "construct_order",
        "sample_alignment",
        "bounds_note",
        "tail_valid_s",
        "segment_duration_s",
        "n_samples",
        "peak_before_preemphasis",
        "fraction_above_1_before_preemphasis",
    ]
    for prefix, count in FAMILIES:
        columns.append(f"n_{prefix}_frames")
        columns.extend(coeff_columns(prefix, count, "mean"))
    return columns


def _aggregate_call(call_id: str, meta: dict, condition: str, measures: list[dict], gains: list[float], above: list[float]) -> dict:
    row = {
        "call_id": call_id,
        "binary_group": meta["binary_group"],
        "campaign": meta["campaign"],
        "spectra_error_role": meta["spectra_error_role"],
        "condition": condition,
        "rms_target": RMS_TARGET if condition == "rms" else np.nan,
        "peak_target": PEAK_NORM_TARGET if condition == "peak" else np.nan,
        "preemphasis": PREEMPHASIS,
        "construct_order": "gain_on_raw_segment_then_preemphasis_0.97_then_cepstrum",
        "n_segments_audio": len(measures),
        "gain_median": float(np.median(gains)) if gains else np.nan,
        "fraction_above_1_before_preemphasis_median": float(np.median(above)) if above else np.nan,
    }
    for prefix, count in FAMILIES:
        used = []
        pooled_weight = []
        within_values = []
        within_weights = []
        for item in measures:
            means = np.array([item[f"{prefix}_c{index:02d}_mean"] for index in range(count)], dtype=np.float64)
            if not np.isfinite(means).all():
                continue
            frames = int(item[f"n_{prefix}_frames"])
            used.append(means)
            pooled_weight.append(frames)
            delta = np.array([item[f"{prefix}_c{index:02d}_delta_mad"] for index in range(count)], dtype=np.float64)
            weight = max(frames - 1, 0)
            if weight and np.isfinite(delta).all():
                within_values.append(delta)
                within_weights.append(weight)
        row[f"n_segments_used_{prefix}"] = len(used)
        if len(used) >= 2:
            between = np.std(np.vstack(used), axis=0, ddof=1)
        else:
            between = np.full(count, np.nan)
        for index, value in enumerate(between):
            row[f"{prefix}_c{index:02d}_between_segment_std"] = float(value)
        finite_between = between[np.isfinite(between)]
        row[f"{prefix}_between_segment_std_mean"] = float(np.mean(finite_between)) if finite_between.size else np.nan
        if pooled_weight and sum(pooled_weight) > 0:
            stacked = np.vstack(used)
            weights = np.array(pooled_weight, dtype=np.float64)
            pooled = np.average(stacked, axis=0, weights=weights)
        else:
            pooled = np.full(count, np.nan)
        row[f"{prefix}_c01_pooled_mean"] = float(pooled[1]) if pooled.size > 1 else np.nan
        row[f"{prefix}_c00_pooled_mean"] = float(pooled[0]) if pooled.size else np.nan
        if within_values:
            weight = np.array(within_weights, dtype=np.float64)
            within = np.average(np.vstack(within_values), axis=0, weights=weight)
            row[f"within_segment_{prefix}_delta_mad_mean"] = float(np.mean(within))
        else:
            row[f"within_segment_{prefix}_delta_mad_mean"] = np.nan
    row["cqcc_c01_mean"] = row["cqcc_c01_pooled_mean"]
    return row


def _norm_columns(sample: dict) -> list[str]:
    preferred = [
        "call_id",
        "binary_group",
        "campaign",
        "spectra_error_role",
        "condition",
        "rms_target",
        "peak_target",
        "preemphasis",
        "construct_order",
        "n_segments_audio",
        "gain_median",
        "fraction_above_1_before_preemphasis_median",
        "n_segments_used_lfcc",
        "n_segments_used_cqcc",
        "n_segments_used_spafe_cqcc",
        "lfcc_between_segment_std_mean",
        "cqcc_between_segment_std_mean",
        "spafe_cqcc_between_segment_std_mean",
        "cqcc_c01_mean",
        "cqcc_c01_pooled_mean",
        "cqcc_c00_pooled_mean",
        "lfcc_c01_pooled_mean",
        "lfcc_c00_pooled_mean",
        "within_segment_lfcc_delta_mad_mean",
        "within_segment_cqcc_delta_mad_mean",
        "within_segment_spafe_cqcc_delta_mad_mean",
    ]
    for prefix, count in FAMILIES:
        preferred.extend(f"{prefix}_c{index:02d}_between_segment_std" for index in range(count))
        preferred.append(f"{prefix}_c00_pooled_mean")
        preferred.append(f"{prefix}_c01_pooled_mean")
    ordered = []
    for name in preferred:
        if name in sample and name not in ordered:
            ordered.append(name)
    for name in sample:
        if name not in ordered:
            ordered.append(name)
    return ordered


def append_normalization_effects() -> None:
    norm = read_table(OUTPUT / "normalization_sensitivity.csv")
    matched_path = OUTPUT / "matched_group_analysis.csv"
    matched = read_table(matched_path)
    matched = matched[matched["stratum_family"] != "normalization"]
    endpoints = list(BETWEEN_ENDPOINTS) + [
        "lfcc_c00_between_segment_std",
        "cqcc_c00_between_segment_std",
        "within_segment_lfcc_delta_mad_mean",
        "within_segment_cqcc_delta_mad_mean",
    ]
    rows = []
    for condition, part in norm.groupby("condition"):
        human = part[part["binary_group"] == "human_voice"]
        synth = part[part["binary_group"] == "synthetic"]
        for endpoint in endpoints:
            if endpoint not in part.columns:
                continue
            reference = finite(human[endpoint])
            other = finite(synth[endpoint])
            low, high, status = cliffs_interval(reference, other)
            ref_q = quartile_row(reference)
            other_q = quartile_row(other)
            rows.append(
                {
                    "comparison": "synthetic_vs_human",
                    "endpoint": endpoint,
                    "stratum_family": "normalization",
                    "stratum": condition,
                    "n_reference": int(len(human)),
                    "n_other": int(len(synth)),
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
                    "note": "Та же выборка после отдельной нормировки амплитуды. c00 — контроль громкости. within_segment не является межсегментным СКО.",
                }
            )
    write_frame(matched_path, pd.concat([matched, pd.DataFrame(rows)], ignore_index=True))


def run_audio(resume: bool = True) -> dict:
    warnings.filterwarnings("ignore", message="n_fft=.*too large")
    segments = read_table(OUTPUT / "segment_level_cepstral_features.csv")
    records, _intervals, _padding = load_records()
    by_id = {row["call_id"]: row for row in records}
    padding_path = OUTPUT / "padding_sensitivity.csv"
    norm_path = OUTPUT / "normalization_sensitivity.csv"
    padding_rows: list[dict] = []
    norm_rows: list[dict] = []
    done_padding: set[str] = set()
    done_norm: set[str] = set()
    if resume and padding_path.exists():
        previous = read_table(padding_path)
        expected = segments.groupby("call_id").size() * 6
        counts = previous.groupby("call_id").size()
        done_padding = {call_id for call_id, count in counts.items() if int(count) == int(expected.get(call_id, -1))}
        padding_rows = previous[previous["call_id"].isin(done_padding)].to_dict(orient="records")
    if resume and norm_path.exists():
        previous_norm = read_table(norm_path)
        condition_count = previous_norm.groupby("call_id")["condition"].nunique()
        done_norm = {call_id for call_id, count in condition_count.items() if int(count) >= 3}
        norm_rows = previous_norm[previous_norm["call_id"].isin(done_norm)].to_dict(orient="records")
    checksum = []
    started = time.perf_counter()
    call_ids = sorted(segments["call_id"].unique())
    for index, call_id in enumerate(call_ids, start=1):
        need_padding = call_id not in done_padding
        need_norm = call_id not in done_norm
        if not need_padding and not need_norm:
            continue
        record = by_id[call_id]
        audio = load_subscriber(record["source_file"])
        call_segments = segments[segments["call_id"] == call_id]
        measures = {"none": [], "peak": [], "rms": []}
        gains = {"none": [], "peak": [], "rms": []}
        above = {"none": [], "peak": [], "rms": []}
        new_padding = []
        for _, segment_row in call_segments.iterrows():
            built, bounds_note = _waveforms(audio, segment_row)
            cache: dict[tuple[str, str], dict] = {}
            if need_padding:
                if not built:
                    for scope in ("tail", "segment"):
                        for condition in ("real", "repeat", "zero"):
                            empty = _base_row(segment_row, scope, condition, np.zeros(1), 0, False)
                            empty["n_waveform_samples"] = 0
                            empty["real_audio_fraction"] = np.nan
                            empty["repeated_or_zero_fraction"] = np.nan
                            empty["bounds_note"] = bounds_note
                            new_padding.append(_attach(empty, _empty_measure("empty_slice")))
                for (scope, condition), (waveform, n_real, copied) in built.items():
                    if copied:
                        measured = cache[("tail", condition)]
                    else:
                        measured = measure(waveform)
                        cache[(scope, condition)] = measured
                    row = _base_row(segment_row, scope, condition, waveform, n_real, copied)
                    row["bounds_note"] = bounds_note
                    new_padding.append(_attach(row, measured))
            if need_norm:
                segment_raw = built.get(("segment", "real"))
                if segment_raw is None:
                    continue
                raw = segment_raw[0]
                variants = {"none": (raw, np.nan)}
                peaked, peak_gain = peak_normalize(raw)
                variants["peak"] = (peaked, peak_gain)
                scaled, rms_gain = _rms_scale(raw)
                variants["rms"] = (scaled, rms_gain)
                for condition, (waveform, gain) in variants.items():
                    measured = measure(waveform)
                    measures[condition].append(measured)
                    if np.isfinite(gain):
                        gains[condition].append(gain)
                    if np.isfinite(measured["fraction_above_1_before_preemphasis"]):
                        above[condition].append(measured["fraction_above_1_before_preemphasis"])
                    if condition == "none" and np.isfinite(measured["cqcc_c01_mean"]) and np.isfinite(segment_row["cqcc_c01_mean"]):
                        checksum.append(abs(float(measured["cqcc_c01_mean"]) - float(segment_row["cqcc_c01_mean"])))
        if need_padding:
            padding_rows.extend(new_padding)
        if need_norm:
            meta = call_segments.iloc[0]
            for condition in ("none", "peak", "rms"):
                norm_rows.append(_aggregate_call(call_id, meta, condition, measures[condition], gains[condition], above[condition]))
        if index % 25 == 0 or index == len(call_ids):
            write_frame(padding_path, pd.DataFrame(padding_rows, columns=_padding_columns()))
            if norm_rows:
                write_frame(norm_path, pd.DataFrame(norm_rows)[_norm_columns(norm_rows[0])])
            elapsed = time.perf_counter() - started
            print(f"audio {index}/{len(call_ids)} elapsed {elapsed:.0f}s", flush=True)
    if padding_rows:
        write_frame(padding_path, pd.DataFrame(padding_rows, columns=_padding_columns()))
    if norm_rows:
        write_frame(norm_path, pd.DataFrame(norm_rows)[_norm_columns(norm_rows[0])])
    append_normalization_effects()
    max_c1 = float(np.max(checksum)) if checksum else np.nan
    print(f"none-condition cqcc c1 max abs diff vs stored segment means: {max_c1}", flush=True)
    return {"cqcc_c1_max_abs_diff_none_vs_stored": max_c1, "n_checksum_segments": len(checksum)}
