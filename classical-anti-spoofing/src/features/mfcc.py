from __future__ import annotations

import numpy as np

from src.audio.preprocessing import frame_params, preemphasis, resolve_fmax
from src.features.lfcc import _pack
from src.features.stats import coefficient_names


def _kinds(include_deltas: bool) -> tuple[str, ...]:
    if include_deltas:
        return ("static", "delta", "delta2")
    return ("static",)


def mfcc_names(cfg: dict) -> list[str]:
    group = cfg["features"]["mfcc"]
    return coefficient_names("mfcc", int(group["n_mfcc"]), _kinds(bool(group["include_deltas"])))


def compute_mfcc(audio: np.ndarray, sample_rate: int, cfg: dict) -> dict[str, float]:
    import librosa

    group = cfg["features"]["mfcc"]
    names = mfcc_names(cfg)
    win, hop, n_fft = frame_params(sample_rate, cfg["features"]["frame_ms"], cfg["features"]["hop_ms"])
    if audio.size < win:
        return {name: float("nan") for name in names}
    fmax = resolve_fmax(sample_rate, float(group["fmax"]))
    signal = preemphasis(audio, float(group["preemphasis"]))
    cepstra = librosa.feature.mfcc(
        y=np.ascontiguousarray(signal, dtype=np.float32),
        sr=sample_rate,
        n_mfcc=int(group["n_mfcc"]),
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        n_mels=int(group["n_mels"]),
        fmin=float(group["fmin"]),
        fmax=fmax,
        center=False,
    ).T
    return _pack("mfcc", cepstra.astype(np.float64), names, bool(group["include_deltas"]), int(group["delta_width"]))
