from __future__ import annotations

import numpy as np


VOICE_NAMES = (
    "jitter_local",
    "shimmer_local",
    "hnr_praat_db",
    "acf_harmonicity_mean",
)


def voice_quality_names(cfg: dict) -> list[str]:
    return list(VOICE_NAMES)


def acf_harmonicity(audio: np.ndarray, sample_rate: int, fmin: float, fmax: float, win: int, hop: int) -> float:
    """Пик нормированной автокорреляции в диапазоне лагов F0. Это не Praat HNR."""

    if audio.size < win:
        return float("nan")
    lag_min = max(1, int(sample_rate / fmax))
    lag_max = min(win - 1, int(sample_rate / fmin))
    if lag_max <= lag_min:
        return float("nan")
    peaks: list[float] = []
    for start in range(0, audio.size - win + 1, hop):
        frame = audio[start : start + win]
        energy = float(np.dot(frame, frame))
        if energy <= 1e-8:
            continue
        spectrum = np.fft.rfft(frame, n=win * 2)
        corr = np.fft.irfft(spectrum * np.conj(spectrum))[:win]
        peak = float(np.max(corr[lag_min : lag_max + 1]) / corr[0])
        if np.isfinite(peak):
            peaks.append(peak)
    if not peaks:
        return float("nan")
    return float(np.mean(peaks))


def compute_voice_quality(audio: np.ndarray, sample_rate: int, cfg: dict) -> tuple[dict[str, float], dict[str, int | str]]:
    group = cfg["features"]["voice_quality"]
    values = {name: float("nan") for name in VOICE_NAMES}
    flags: dict[str, int | str] = {
        "flag_voice_quality_8khz_limited": 1 if sample_rate <= 8000 else 0,
        "flag_voice_quality_unavailable": 0,
        "voice_quality_reliability": "telephone_8khz_limited" if sample_rate <= 8000 else "wideband",
    }
    win, hop, _n_fft = _frame(sample_rate, cfg)
    values["acf_harmonicity_mean"] = acf_harmonicity(
        audio, sample_rate, float(group["fmin"]), float(group["fmax"]), win, hop
    )
    min_samples = int(sample_rate * float(group["min_duration_ms"]) / 1000.0)
    if audio.size < min_samples:
        flags["flag_voice_quality_unavailable"] = 1
        flags["voice_quality_reliability"] = "insufficient_duration"
        return values, flags
    try:
        import parselmouth
        from parselmouth.praat import call
    except Exception:
        flags["flag_voice_quality_unavailable"] = 1
        flags["voice_quality_reliability"] = "parselmouth_missing"
        return values, flags
    try:
        sound = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sample_rate)
        point_process = call(sound, "To PointProcess (periodic, cc)", float(group["fmin"]), float(group["fmax"]))
        values["jitter_local"] = float(call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
        values["shimmer_local"] = float(
            call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        )
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, float(group["fmin"]), 0.1, 1.0)
        values["hnr_praat_db"] = float(call(harmonicity, "Get mean", 0, 0))
    except Exception:
        flags["flag_voice_quality_unavailable"] = 1
        flags["voice_quality_reliability"] = "praat_undefined"
    for name in ("jitter_local", "shimmer_local", "hnr_praat_db"):
        if not np.isfinite(values[name]):
            values[name] = float("nan")
    return values, flags


def _frame(sample_rate: int, cfg: dict) -> tuple[int, int, int]:
    from src.audio.preprocessing import frame_params

    return frame_params(sample_rate, cfg["features"]["frame_ms"], cfg["features"]["hop_ms"])
