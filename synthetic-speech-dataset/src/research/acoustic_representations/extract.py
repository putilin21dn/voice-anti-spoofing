"""Turn one subscriber channel into call-level and segment-level feature rows."""

from __future__ import annotations

import numpy as np

from src.research.acoustic_representations.config import (
    CQT_LONG_WINDOW_SAMPLES,
    CQCC_CEPS,
    HOP_LENGTH,
    LFCC_CEPS,
    N_MELS,
    N_STFT_BANDS,
    SAMPLE_RATE,
    WIN_LENGTH,
)
from src.research.acoustic_representations.features import (
    band_log_means,
    cqcc_librosa,
    cqcc_spafe,
    f0_track,
    lfcc_matrix,
    log_of_band_means,
    log_safe,
    matrix_flux,
    mel_power,
    mel_tilt,
    spectral_shape,
    stft_power,
)
from src.research.acoustic_representations.signal import peak_normalize, preemphasis, signal_quality

COEFF_STATS = ("mean", "std", "median", "q25", "q75", "iqr", "delta_mad", "between_segment_std")


def _nan_row(keys: list[str]) -> dict[str, float]:
    return {key: np.nan for key in keys}


def coeff_columns(prefix: str, count: int) -> list[str]:
    return [f"{prefix}_c{index:02d}_{stat}" for index in range(count) for stat in COEFF_STATS]


def aggregate_columns(prefix: str) -> list[str]:
    return [f"{prefix}_temporal_mad_mean", f"{prefix}_between_segment_std_mean"]


def call_feature_columns() -> list[str]:
    columns = [
        "logmel_mean",
        "logmel_std",
        "logmel_median",
        "logmel_q25",
        "logmel_q75",
        "logmel_iqr",
        "logmel_temporal_std_mean",
        "logmel_level_temporal_std",
        "logmel_low_high_tilt",
        "mel_flux_l1shape_mean",
        "stft_centroid_hz_mean",
        "stft_centroid_hz_std",
        "stft_rolloff85_hz_mean",
        "stft_flatness_mean",
        "stft_flux_l1shape_mean",
        "logstft_mean",
        "logstft_std",
        "logstft_median",
        "logstft_q25",
        "logstft_q75",
        "logstft_iqr",
        "logstft_temporal_std_mean",
        "stft_floor_fraction",
        "n_stft_frames",
        "stft_discarded_samples",
        "n_segments_shorter_than_cqt_window",
        "cqcc_error_segments",
        "spafe_error_segments",
        "f0_error_segments",
    ]
    columns.extend(f"logmel_b{index:02d}_mean" for index in range(N_MELS))
    columns.extend(f"logmel_b{index:02d}_std" for index in range(N_MELS))
    columns.extend(f"logstft_band{index:02d}_mean" for index in range(N_STFT_BANDS))
    columns.extend(f"logstft_band{index:02d}_std" for index in range(N_STFT_BANDS))
    for prefix, count in (("lfcc", LFCC_CEPS), ("cqcc", CQCC_CEPS), ("spafe_cqcc", CQCC_CEPS)):
        columns.extend(coeff_columns(prefix, count))
        columns.extend(aggregate_columns(prefix))
    return columns


def comparison_columns() -> list[str]:
    bases = (
        "log_energy_mean",
        "centroid_hz_mean",
        "flatness_mean",
        "tilt",
        "flux_l1shape_mean",
    )
    columns = ["call_id"]
    for variant in ("raw", "peak", "preemph"):
        columns.extend(f"{variant}_{name}" for name in bases)
    columns.extend(
        [
            "peak_gain_median",
            "peak_gain_min",
            "peak_gain_max",
            "preemph_peak_max",
            "preemph_fraction_above_1",
            "mean_log_band_range",
            "log_of_mean_band_range",
            "logmel_band_range",
            "mel_log_of_mean_band_range",
        ]
    )
    return columns


def segment_columns() -> list[str]:
    columns = [
        "call_id",
        "segment_index",
        "start_time",
        "end_time",
        "duration_s",
        "n_samples",
        "n_stft_frames",
        "stft_discarded_samples",
        "status",
        "shorter_than_cqt_window",
        "raw_rms",
        "raw_peak",
        "clip_fraction",
        "preemph_peak",
        "preemph_fraction_above_1",
        "logmel_mean",
        "logmel_low_high_tilt",
        "stft_centroid_hz_mean",
        "stft_flatness_mean",
        "stft_rolloff85_hz_mean",
        "stft_flux_l1shape_mean",
        "f0_median_hz",
        "f0_iqr_hz",
        "f0_voiced_fraction",
    ]
    for prefix, count in (("lfcc", LFCC_CEPS), ("cqcc", CQCC_CEPS), ("spafe_cqcc", CQCC_CEPS)):
        columns.extend(f"{prefix}_c{index:02d}_mean" for index in range(count))
    return columns


def _global(matrix: np.ndarray) -> dict[str, float]:
    if matrix.size == 0:
        return {key: np.nan for key in ("mean", "std", "median", "q25", "q75", "iqr")}
    values = matrix.reshape(-1)
    q25, median, q75 = np.percentile(values, [25, 50, 75])
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else np.nan,
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
    }


def _weighted_delta(parts: list[np.ndarray]) -> np.ndarray:
    width = parts[0].shape[1]
    total = np.zeros(width, dtype=np.float64)
    weight = 0
    for matrix in parts:
        if matrix.shape[0] < 2:
            continue
        step = matrix.shape[0] - 1
        total += np.mean(np.abs(np.diff(matrix, axis=0)), axis=0) * step
        weight += step
    if weight == 0:
        return np.full(width, np.nan)
    return total / weight


def _between(means: list[np.ndarray], width: int) -> np.ndarray:
    if len(means) < 2:
        return np.full(width, np.nan)
    stacked = np.vstack(means)
    return np.std(stacked, axis=0, ddof=1)


def _fill_coeffs(target: dict, prefix: str, parts: list[np.ndarray]) -> None:
    count = parts[0].shape[1] if parts else {"lfcc": LFCC_CEPS, "cqcc": CQCC_CEPS, "spafe_cqcc": CQCC_CEPS}[prefix]
    if not parts:
        for index in range(count):
            for stat in COEFF_STATS:
                target[f"{prefix}_c{index:02d}_{stat}"] = np.nan
        target[f"{prefix}_temporal_mad_mean"] = np.nan
        target[f"{prefix}_between_segment_std_mean"] = np.nan
        return
    pooled = np.vstack(parts)
    means = [matrix.mean(axis=0) for matrix in parts]
    delta = _weighted_delta(parts)
    between = _between(means, count)
    for index in range(count):
        column = pooled[:, index]
        q25, median, q75 = np.percentile(column, [25, 50, 75])
        target[f"{prefix}_c{index:02d}_mean"] = float(np.mean(column))
        target[f"{prefix}_c{index:02d}_std"] = float(np.std(column, ddof=1)) if column.size > 1 else np.nan
        target[f"{prefix}_c{index:02d}_median"] = float(median)
        target[f"{prefix}_c{index:02d}_q25"] = float(q25)
        target[f"{prefix}_c{index:02d}_q75"] = float(q75)
        target[f"{prefix}_c{index:02d}_iqr"] = float(q75 - q25)
        target[f"{prefix}_c{index:02d}_delta_mad"] = float(delta[index])
        target[f"{prefix}_c{index:02d}_between_segment_std"] = float(between[index])
    finite_delta = delta[np.isfinite(delta)]
    finite_between = between[np.isfinite(between)]
    target[f"{prefix}_temporal_mad_mean"] = float(np.mean(finite_delta)) if finite_delta.size else np.nan
    target[f"{prefix}_between_segment_std_mean"] = float(np.mean(finite_between)) if finite_between.size else np.nan


def _band_std(series: np.ndarray) -> np.ndarray:
    if series.shape[0] < 2:
        return np.full(series.shape[1], np.nan)
    return np.std(series, axis=0, ddof=1)


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    pairs = [(value, weight) for value, weight in zip(values, weights) if np.isfinite(value) and weight > 0]
    if not pairs:
        return np.nan
    total = sum(weight for _, weight in pairs)
    return float(sum(value * weight for value, weight in pairs) / total)


def _waveform_summary(audio: np.ndarray) -> dict[str, float | np.ndarray]:
    power = stft_power(audio)
    mel = mel_power(power)
    logged_mel = log_safe(mel) if mel.size else mel
    shape = spectral_shape(power)
    flux_weight = max(power.shape[0] - 1, 0)
    return {
        "power": power,
        "logmel": logged_mel,
        "mel": mel,
        "lfcc": lfcc_matrix(power),
        "shape": shape,
        "tilt": mel_tilt(logged_mel),
        "mel_flux": matrix_flux(mel) if mel.size else np.nan,
        "flux_weight": flux_weight,
        "n_frames": int(power.shape[0]),
        "band_log": band_log_means(power),
        "band_log_of_mean": log_of_band_means(power),
    }


def _empty_segment(call_id: str, index: int, start: float, end: float, status: str) -> dict:
    row = {column: "" for column in segment_columns()}
    row.update(
        {
            "call_id": call_id,
            "segment_index": index,
            "start_time": round(start, 3),
            "end_time": round(end, 3),
            "duration_s": round(end - start, 3),
            "n_samples": 0,
            "n_stft_frames": 0,
            "stft_discarded_samples": 0,
            "status": status,
            "shorter_than_cqt_window": 0,
        }
    )
    return row


def _coeff_means(matrix: np.ndarray, prefix: str, count: int) -> dict[str, float]:
    values = {}
    for index in range(count):
        key = f"{prefix}_c{index:02d}_mean"
        values[key] = float(matrix[:, index].mean()) if matrix.size else np.nan
    return values


def extract_call(
    call_id: str, audio: np.ndarray, intervals: list[tuple[int, float, float]]
) -> tuple[dict, list[dict], dict, dict]:
    """Primary features use per-segment pre-emphasis. Raw and peak-norm stay in the comparison row."""

    feature_row = _nan_row(call_feature_columns())
    comparison = {column: np.nan for column in comparison_columns() if column != "call_id"}
    comparison["call_id"] = call_id
    empty_quality = {
        "raw_speech_rms": np.nan,
        "raw_speech_peak": np.nan,
        "raw_clip_fraction": np.nan,
        "explanatory_f0_median_hz": np.nan,
        "explanatory_f0_iqr_hz": np.nan,
        "explanatory_f0_voiced_fraction": np.nan,
    }
    segments: list[dict] = []
    if not intervals:
        return feature_row, segments, comparison, empty_quality

    logmel_parts: list[np.ndarray] = []
    band_parts: list[np.ndarray] = []
    lfcc_parts: list[np.ndarray] = []
    cqcc_parts: list[np.ndarray] = []
    spafe_parts: list[np.ndarray] = []
    centroids: list[np.ndarray] = []
    rolloffs: list[np.ndarray] = []
    flatness: list[np.ndarray] = []
    flux_values: list[float] = []
    flux_weights: list[float] = []
    mel_flux_values: list[float] = []
    mel_flux_weights: list[float] = []
    discarded = 0
    short_cqt = 0
    cqcc_errors = 0
    spafe_errors = 0
    f0_errors = 0
    f0_values: list[np.ndarray] = []
    voiced_values: list[np.ndarray] = []
    floor_frames = 0
    floor_total = 0
    sum_squares = 0.0
    n_speech = 0
    peak = 0.0
    n_clip = 0
    preemph_peak = 0.0
    preemph_above = 0
    preemph_count = 0
    gains: list[float] = []

    variant_store = {
        name: {"log_energy": [], "centroid": [], "flatness": [], "tilt": [], "flux": [], "weight": [], "flux_weight": []}
        for name in ("raw", "peak", "preemph")
    }
    sum_band_log = np.zeros(N_STFT_BANDS)
    sum_band_linear = np.zeros(N_STFT_BANDS)
    sum_mel_log = np.zeros(N_MELS)
    sum_mel_linear = np.zeros(N_MELS)
    band_frames = 0

    for index, start, end in intervals:
        raw = audio[max(0, int(round(start * SAMPLE_RATE))) : min(audio.size, int(round(end * SAMPLE_RATE)))]
        if raw.size < WIN_LENGTH:
            segments.append(_empty_segment(call_id, index, start, end, "too_short"))
            continue
        quality = signal_quality(raw)
        sum_squares += float(np.dot(raw, raw))
        n_speech += int(raw.size)
        peak = max(peak, quality["peak"])
        n_clip += int(round(quality["clip_fraction"] * raw.size))
        emphasized = preemphasis(raw)
        emphasized_peak = float(np.max(np.abs(emphasized)))
        emphasized_above = float(np.mean(np.abs(emphasized) > 1.0))
        preemph_peak = max(preemph_peak, emphasized_peak)
        preemph_above += emphasized_above * emphasized.size
        preemph_count += int(emphasized.size)
        normalized, gain = peak_normalize(raw)
        if np.isfinite(gain):
            gains.append(gain)

        primary = _waveform_summary(emphasized)
        raw_summary = _waveform_summary(raw)
        peak_summary = _waveform_summary(normalized)
        for name, summary in (("raw", raw_summary), ("peak", peak_summary), ("preemph", primary)):
            store = variant_store[name]
            store["log_energy"].append(summary["shape"]["log_energy_mean"])
            store["centroid"].append(summary["shape"]["centroid_hz_mean"])
            store["flatness"].append(summary["shape"]["flatness_mean"])
            store["tilt"].append(summary["tilt"])
            store["flux"].append(summary["shape"]["flux_l1shape_mean"])
            store["weight"].append(summary["n_frames"])
            store["flux_weight"].append(summary["flux_weight"])

        n_frames = primary["n_frames"]
        used = (n_frames - 1) * HOP_LENGTH + WIN_LENGTH if n_frames else 0
        segment_discarded = int(raw.size - used)
        discarded += segment_discarded
        is_short_cqt = int(raw.size < CQT_LONG_WINDOW_SAMPLES)
        short_cqt += is_short_cqt

        try:
            cqcc = cqcc_librosa(emphasized)
        except Exception:
            cqcc = np.empty((0, CQCC_CEPS))
            cqcc_errors += 1
        try:
            spafe = cqcc_spafe(emphasized)
        except Exception:
            spafe = np.empty((0, CQCC_CEPS))
            spafe_errors += 1
        try:
            f0, voiced = f0_track(raw)
        except Exception:
            f0, voiced = np.empty(0), np.empty(0, dtype=bool)
            f0_errors += 1

        if primary["logmel"].size:
            logmel_parts.append(primary["logmel"])
        if n_frames:
            band_series = _band_series(primary["power"])
            band_parts.append(band_series)
            linear_bands = _band_linear(primary["power"])
            sum_band_log += primary["band_log"] * n_frames
            sum_band_linear += linear_bands * n_frames
            sum_mel_log += primary["logmel"].mean(axis=0) * n_frames
            sum_mel_linear += np.maximum(primary["mel"].mean(axis=0), 1e-10) * n_frames
            band_frames += n_frames
        if primary["lfcc"].size:
            lfcc_parts.append(primary["lfcc"])
        if cqcc.size:
            cqcc_parts.append(cqcc)
        if spafe.size:
            spafe_parts.append(spafe)
        if f0.size:
            f0_values.append(f0)
            voiced_values.append(voiced)
        shape = primary["shape"]
        floor_frames += int(round(shape["floor_fraction"] * n_frames)) if n_frames and np.isfinite(shape["floor_fraction"]) else 0
        floor_total += n_frames
        flux_values.append(shape["flux_l1shape_mean"])
        flux_weights.append(primary["flux_weight"])
        mel_flux_values.append(primary["mel_flux"])
        mel_flux_weights.append(max(primary["logmel"].shape[0] - 1, 0))

        voiced_f0 = f0[voiced] if voiced.size else np.empty(0)
        voiced_f0 = voiced_f0[np.isfinite(voiced_f0)]
        segment = {
            "call_id": call_id,
            "segment_index": index,
            "start_time": round(start, 3),
            "end_time": round(end, 3),
            "duration_s": round(end - start, 3),
            "n_samples": int(raw.size),
            "n_stft_frames": n_frames,
            "stft_discarded_samples": segment_discarded,
            "status": "ok",
            "shorter_than_cqt_window": is_short_cqt,
            "raw_rms": quality["rms"],
            "raw_peak": quality["peak"],
            "clip_fraction": quality["clip_fraction"],
            "preemph_peak": emphasized_peak,
            "preemph_fraction_above_1": emphasized_above,
            "logmel_mean": float(np.mean(primary["logmel"])) if primary["logmel"].size else np.nan,
            "logmel_low_high_tilt": primary["tilt"],
            "stft_centroid_hz_mean": shape["centroid_hz_mean"],
            "stft_flatness_mean": shape["flatness_mean"],
            "stft_rolloff85_hz_mean": shape["rolloff85_hz_mean"],
            "stft_flux_l1shape_mean": shape["flux_l1shape_mean"],
            "f0_median_hz": float(np.median(voiced_f0)) if voiced_f0.size else np.nan,
            "f0_iqr_hz": float(np.subtract(*np.percentile(voiced_f0, [75, 25]))) if voiced_f0.size else np.nan,
            "f0_voiced_fraction": float(np.mean(voiced)) if voiced.size else np.nan,
        }
        segment.update(_coeff_means(primary["lfcc"], "lfcc", LFCC_CEPS))
        segment.update(_coeff_means(cqcc, "cqcc", CQCC_CEPS))
        segment.update(_coeff_means(spafe, "spafe_cqcc", CQCC_CEPS))
        segments.append(segment)
        centroids.append(_per_frame_centroid(primary["power"]))
        rolloffs.append(_per_frame_rolloff(primary["power"]))
        flatness.append(_per_frame_flatness(primary["power"]))

    if logmel_parts:
        logmel = np.vstack(logmel_parts)
        stats = _global(logmel)
        feature_row["logmel_mean"] = stats["mean"]
        feature_row["logmel_std"] = stats["std"]
        feature_row["logmel_median"] = stats["median"]
        feature_row["logmel_q25"] = stats["q25"]
        feature_row["logmel_q75"] = stats["q75"]
        feature_row["logmel_iqr"] = stats["iqr"]
        band_std = _band_std(logmel)
        feature_row["logmel_temporal_std_mean"] = float(np.nanmean(band_std))
        level = logmel.mean(axis=1)
        feature_row["logmel_level_temporal_std"] = float(np.std(level, ddof=1)) if level.size > 1 else np.nan
        feature_row["logmel_low_high_tilt"] = float(np.mean(logmel[:, :8].mean(axis=1) - logmel[:, -8:].mean(axis=1)))
        for index in range(N_MELS):
            feature_row[f"logmel_b{index:02d}_mean"] = float(np.mean(logmel[:, index]))
            feature_row[f"logmel_b{index:02d}_std"] = float(band_std[index])
    feature_row["mel_flux_l1shape_mean"] = _weighted_mean(mel_flux_values, [float(item) for item in mel_flux_weights])

    if band_parts:
        bands = np.vstack(band_parts)
        band_stats = _global(bands)
        feature_row["logstft_mean"] = band_stats["mean"]
        feature_row["logstft_std"] = band_stats["std"]
        feature_row["logstft_median"] = band_stats["median"]
        feature_row["logstft_q25"] = band_stats["q25"]
        feature_row["logstft_q75"] = band_stats["q75"]
        feature_row["logstft_iqr"] = band_stats["iqr"]
        series_std = _band_std(bands)
        feature_row["logstft_temporal_std_mean"] = float(np.nanmean(series_std))
        for index in range(N_STFT_BANDS):
            feature_row[f"logstft_band{index:02d}_mean"] = float(np.mean(bands[:, index]))
            feature_row[f"logstft_band{index:02d}_std"] = float(series_std[index])

    centroid = _concat(centroids)
    rolloff = _concat(rolloffs)
    flat = _concat(flatness)
    feature_row["stft_centroid_hz_mean"] = _nanmean(centroid)
    feature_row["stft_centroid_hz_std"] = _nanstd(centroid)
    feature_row["stft_rolloff85_hz_mean"] = _nanmean(rolloff)
    feature_row["stft_flatness_mean"] = _nanmean(flat)
    feature_row["stft_flux_l1shape_mean"] = _weighted_mean(flux_values, [float(item) for item in flux_weights])
    feature_row["stft_floor_fraction"] = float(floor_frames / floor_total) if floor_total else np.nan
    feature_row["n_stft_frames"] = int(sum(part.shape[0] for part in logmel_parts))
    feature_row["stft_discarded_samples"] = discarded
    feature_row["n_segments_shorter_than_cqt_window"] = short_cqt
    feature_row["cqcc_error_segments"] = cqcc_errors
    feature_row["spafe_error_segments"] = spafe_errors
    feature_row["f0_error_segments"] = f0_errors
    _fill_coeffs(feature_row, "lfcc", lfcc_parts)
    _fill_coeffs(feature_row, "cqcc", cqcc_parts)
    _fill_coeffs(feature_row, "spafe_cqcc", spafe_parts)

    for name, store in variant_store.items():
        weights = [float(item) for item in store["weight"]]
        flux_w = [float(item) for item in store["flux_weight"]]
        comparison[f"{name}_log_energy_mean"] = _weighted_mean(store["log_energy"], weights)
        comparison[f"{name}_centroid_hz_mean"] = _weighted_mean(store["centroid"], weights)
        comparison[f"{name}_flatness_mean"] = _weighted_mean(store["flatness"], weights)
        comparison[f"{name}_tilt"] = _weighted_mean(store["tilt"], weights)
        comparison[f"{name}_flux_l1shape_mean"] = _weighted_mean(store["flux"], flux_w)
    if gains:
        comparison["peak_gain_median"] = float(np.median(gains))
        comparison["peak_gain_min"] = float(np.min(gains))
        comparison["peak_gain_max"] = float(np.max(gains))
    comparison["preemph_peak_max"] = preemph_peak if preemph_count else np.nan
    comparison["preemph_fraction_above_1"] = preemph_above / preemph_count if preemph_count else np.nan
    if band_frames:
        mean_log = sum_band_log / band_frames
        log_mean = np.log(np.maximum(sum_band_linear / band_frames, 1e-10))
        mel_mean_log = sum_mel_log / band_frames
        mel_log_mean = np.log(np.maximum(sum_mel_linear / band_frames, 1e-10))
        comparison["mean_log_band_range"] = float(np.nanmax(mean_log) - np.nanmin(mean_log))
        comparison["log_of_mean_band_range"] = float(np.nanmax(log_mean) - np.nanmin(log_mean))
        comparison["logmel_band_range"] = float(np.nanmax(mel_mean_log) - np.nanmin(mel_mean_log))
        comparison["mel_log_of_mean_band_range"] = float(np.nanmax(mel_log_mean) - np.nanmin(mel_log_mean))

    if f0_values:
        f0 = np.concatenate(f0_values)
        voiced = np.concatenate(voiced_values)
        voiced_f0 = f0[voiced]
        voiced_f0 = voiced_f0[np.isfinite(voiced_f0)]
        f0_median = float(np.median(voiced_f0)) if voiced_f0.size else np.nan
        f0_iqr = float(np.subtract(*np.percentile(voiced_f0, [75, 25]))) if voiced_f0.size else np.nan
        voiced_fraction = float(np.mean(voiced))
    else:
        f0_median = f0_iqr = voiced_fraction = np.nan
    quality = {
        "raw_speech_rms": float(np.sqrt(sum_squares / n_speech)) if n_speech else np.nan,
        "raw_speech_peak": float(peak) if n_speech else np.nan,
        "raw_clip_fraction": float(n_clip / n_speech) if n_speech else np.nan,
        "explanatory_f0_median_hz": f0_median,
        "explanatory_f0_iqr_hz": f0_iqr,
        "explanatory_f0_voiced_fraction": voiced_fraction,
    }
    return feature_row, segments, comparison, quality


def _band_series(power: np.ndarray) -> np.ndarray:
    from src.research.acoustic_representations.features import frequencies, log_safe as take_log

    logged = take_log(power)
    freqs = frequencies()
    edges = np.linspace(0.0, 4000.0, N_STFT_BANDS + 1)
    series = np.empty((power.shape[0], N_STFT_BANDS), dtype=np.float64)
    for index in range(N_STFT_BANDS):
        if index == N_STFT_BANDS - 1:
            mask = (freqs >= edges[index]) & (freqs <= edges[index + 1])
        else:
            mask = (freqs >= edges[index]) & (freqs < edges[index + 1])
        series[:, index] = np.mean(logged[:, mask], axis=1)
    return series


def _band_linear(power: np.ndarray) -> np.ndarray:
    from src.research.acoustic_representations.features import frequencies

    freqs = frequencies()
    edges = np.linspace(0.0, 4000.0, N_STFT_BANDS + 1)
    means = np.empty(N_STFT_BANDS, dtype=np.float64)
    for index in range(N_STFT_BANDS):
        if index == N_STFT_BANDS - 1:
            mask = (freqs >= edges[index]) & (freqs <= edges[index + 1])
        else:
            mask = (freqs >= edges[index]) & (freqs < edges[index + 1])
        means[index] = float(np.mean(power[:, mask]))
    return means


def _per_frame_centroid(power: np.ndarray) -> np.ndarray:
    from src.research.acoustic_representations.features import frequencies

    if power.size == 0:
        return np.empty(0)
    freqs = frequencies()
    total = np.maximum(power.sum(axis=1), 1e-10)
    return (power * freqs[None, :]).sum(axis=1) / total


def _per_frame_rolloff(power: np.ndarray) -> np.ndarray:
    from src.research.acoustic_representations.config import ROLLOFF
    from src.research.acoustic_representations.features import frequencies

    if power.size == 0:
        return np.empty(0)
    freqs = frequencies()
    total = power.sum(axis=1)
    cumulative = np.cumsum(power, axis=1)
    crossed = cumulative >= (ROLLOFF * np.maximum(total, 1e-10))[:, None]
    index = np.argmax(crossed, axis=1)
    values = freqs[index].astype(np.float64)
    values[~crossed.any(axis=1)] = np.nan
    return values


def _per_frame_flatness(power: np.ndarray) -> np.ndarray:
    from src.research.acoustic_representations.config import LOG_FLOOR

    if power.size == 0:
        return np.empty(0)
    geometric = np.exp(np.mean(np.log(np.maximum(power, LOG_FLOOR)), axis=1))
    arithmetic = np.maximum(np.mean(power, axis=1), LOG_FLOOR)
    return geometric / arithmetic


def _concat(parts: list[np.ndarray]) -> np.ndarray:
    usable = [part for part in parts if part.size]
    if not usable:
        return np.empty(0)
    return np.concatenate(usable)


def _nanmean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)] if values.size else values
    if finite.size == 0:
        return np.nan
    return float(np.mean(finite))


def _nanstd(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)] if values.size else values
    if finite.size < 2:
        return np.nan
    return float(np.std(finite, ddof=1))

