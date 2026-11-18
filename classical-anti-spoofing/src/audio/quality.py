from __future__ import annotations

import numpy as np


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def mask_from_segments(n_samples: int, sample_rate: int, segments: list[tuple[float, float]]) -> np.ndarray:
    mask = np.zeros(n_samples, dtype=bool)
    for start, end in segments:
        left = max(0, int(round(start * sample_rate)))
        right = min(n_samples, int(round(end * sample_rate)))
        if right > left:
            mask[left:right] = True
    return mask


def recording_quality(
    audio: np.ndarray,
    sample_rate: int,
    segments: list[tuple[float, float]],
    *,
    min_noise_ms: float,
) -> dict[str, float | str]:
    duration = len(audio) / float(sample_rate) if sample_rate else 0.0
    speech = float(sum(end - start for start, end in segments))
    speech_ratio = speech / duration if duration else 0.0
    mask = mask_from_segments(len(audio), sample_rate, segments)
    speech_rms = rms(audio[mask]) if mask.any() else float("nan")
    noise = audio[~mask]
    noise_ms = 1000.0 * noise.size / sample_rate if sample_rate else 0.0
    noise_rms = rms(noise) if noise.size else float("nan")
    if noise_ms < min_noise_ms or not np.isfinite(noise_rms) or noise_rms <= 1e-8 or not np.isfinite(speech_rms):
        snr_db = float("nan")
        method = "unavailable"
    else:
        snr_db = float(20.0 * np.log10(speech_rms / noise_rms))
        method = "vad_rms_ratio"
    return {
        "recording_duration_s": duration,
        "speech_duration_s": speech,
        "speech_ratio": speech_ratio,
        "silence_ratio": 1.0 - speech_ratio if duration else 0.0,
        "speech_rms": speech_rms,
        "noise_rms": noise_rms,
        "snr_db": snr_db,
        "snr_method": method,
    }
