from __future__ import annotations

import numpy as np

from src.audio.preprocessing import frame_params, resolve_fmax
from src.features.stats import scalar_names, summarize_vector


MEASURES = (
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "spectral_rolloff_hz",
    "spectral_flatness",
    "zero_crossing_rate",
)


def spectral_names(cfg: dict) -> list[str]:
    names: list[str] = []
    for measure in MEASURES:
        names.extend(scalar_names(measure))
    return names


def compute_spectral(audio: np.ndarray, sample_rate: int, cfg: dict) -> dict[str, float]:
    import librosa

    names = spectral_names(cfg)
    win, hop, n_fft = frame_params(sample_rate, cfg["features"]["frame_ms"], cfg["features"]["hop_ms"])
    if audio.size < win:
        return {name: float("nan") for name in names}
    signal = np.ascontiguousarray(audio, dtype=np.float32)
    fmax = resolve_fmax(sample_rate, cfg["features"]["spectral"].get("fmax", sample_rate / 2.0))
    kwargs = dict(sr=sample_rate, n_fft=n_fft, hop_length=hop, win_length=win, center=False)
    centroid = librosa.feature.spectral_centroid(y=signal, **kwargs)
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, **kwargs)
    rolloff = librosa.feature.spectral_rolloff(
        y=signal, roll_percent=float(cfg["features"]["spectral"]["rolloff"]), **kwargs
    )
    flatness = librosa.feature.spectral_flatness(y=signal, n_fft=n_fft, hop_length=hop, win_length=win, center=False)
    zcr = librosa.feature.zero_crossing_rate(signal, frame_length=win, hop_length=hop, center=False)
    values: dict[str, float] = {}
    values.update(summarize_vector("spectral_centroid_hz", centroid))
    values.update(summarize_vector("spectral_bandwidth_hz", bandwidth))
    values.update(summarize_vector("spectral_rolloff_hz", rolloff))
    values.update(summarize_vector("spectral_flatness", flatness))
    values.update(summarize_vector("zero_crossing_rate", zcr))
    for key, value in list(values.items()):
        if "hz" in key and np.isfinite(value) and value > fmax + 1.0:
            raise ValueError(f"{key}={value} вышел за полосу {fmax} Гц")
    return {name: values.get(name, float("nan")) for name in names}
