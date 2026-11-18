from __future__ import annotations

import numpy as np


PITCH_NAMES = (
    "f0_mean",
    "f0_median",
    "f0_std",
    "f0_range",
    "f0_iqr",
    "f0_voiced_fraction",
)


def pitch_names(cfg: dict) -> list[str]:
    return list(PITCH_NAMES)


def compute_pitch(audio: np.ndarray, sample_rate: int, cfg: dict) -> dict[str, float]:
    import librosa

    group = cfg["features"]["pitch"]
    empty = {name: float("nan") for name in PITCH_NAMES}
    frame_length = int(group["frame_length"])
    hop = int(round(sample_rate * cfg["features"]["hop_ms"] / 1000.0))
    if audio.size < frame_length:
        empty["f0_voiced_fraction"] = 0.0
        return empty
    f0, voiced_flag, _probability = librosa.pyin(
        np.ascontiguousarray(audio, dtype=np.float32),
        fmin=float(group["fmin"]),
        fmax=float(group["fmax"]),
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop,
        center=False,
        fill_na=np.nan,
    )
    voiced = np.asarray(f0, dtype=np.float64)
    voiced = voiced[np.isfinite(voiced) & (voiced > 0)]
    fraction = float(np.mean(voiced_flag)) if voiced_flag is not None and len(voiced_flag) else 0.0
    if voiced.size == 0:
        empty["f0_voiced_fraction"] = fraction
        return empty
    q25, q75 = np.percentile(voiced, [25, 75])
    return {
        "f0_mean": float(np.mean(voiced)),
        "f0_median": float(np.median(voiced)),
        "f0_std": float(np.std(voiced)) if voiced.size > 1 else 0.0,
        "f0_range": float(np.max(voiced) - np.min(voiced)),
        "f0_iqr": float(q75 - q25),
        "f0_voiced_fraction": fraction,
    }
