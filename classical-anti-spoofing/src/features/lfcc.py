from __future__ import annotations

import numpy as np

from src.audio.preprocessing import frame_signal, preemphasis, resolve_fmax
from src.features.stats import coefficient_names, delta_matrix, summarize_matrix


def linear_filterbank(n_fft: int, sample_rate: int, n_filters: int, fmin: float, fmax: float) -> np.ndarray:
    fmax = resolve_fmax(sample_rate, fmax)
    if n_filters < 1:
        raise ValueError("Нужен хотя бы один линейный фильтр")
    if fmax <= fmin:
        raise ValueError("fmax должен быть выше fmin")
    n_freqs = n_fft // 2 + 1
    freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs)
    edges = np.linspace(fmin, fmax, n_filters + 2)
    weights = np.zeros((n_filters, n_freqs), dtype=np.float64)
    for index in range(n_filters):
        left, center, right = edges[index : index + 3]
        rising = (freqs - left) / max(center - left, 1e-12)
        falling = (right - freqs) / max(right - center, 1e-12)
        weights[index] = np.maximum(0.0, np.minimum(rising, falling))
    return weights


def _kinds(include_deltas: bool) -> tuple[str, ...]:
    if include_deltas:
        return ("static", "delta", "delta2")
    return ("static",)


def lfcc_names(cfg: dict) -> list[str]:
    group = cfg["features"]["lfcc"]
    return coefficient_names("lfcc", int(group["n_ceps"]), _kinds(bool(group["include_deltas"])))


def compute_lfcc(audio: np.ndarray, sample_rate: int, cfg: dict) -> dict[str, float]:
    from scipy.fft import dct

    group = cfg["features"]["lfcc"]
    names = lfcc_names(cfg)
    win, hop, n_fft = _window(sample_rate, cfg)
    signal = preemphasis(audio, float(group["preemphasis"]))
    frames = frame_signal(signal, win, hop)
    empty = {name: float("nan") for name in names}
    if frames.shape[0] == 0:
        return empty
    fmax = resolve_fmax(sample_rate, float(group["fmax"]))
    weights = linear_filterbank(n_fft, sample_rate, int(group["n_filters"]), float(group["fmin"]), fmax)
    power = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2
    bands = np.maximum(power @ weights.T, 1e-10)
    cepstra = dct(np.log(bands), type=2, norm="ortho", axis=1)[:, : int(group["n_ceps"])]
    return _pack("lfcc", cepstra, names, bool(group["include_deltas"]), int(group["delta_width"]))


def _window(sample_rate: int, cfg: dict) -> tuple[int, int, int]:
    from src.audio.preprocessing import frame_params

    return frame_params(sample_rate, cfg["features"]["frame_ms"], cfg["features"]["hop_ms"])


def _pack(prefix: str, cepstra: np.ndarray, names: list[str], include_deltas: bool, width: int) -> dict[str, float]:
    values = summarize_matrix(prefix, "static", cepstra, cepstra.shape[1])
    if include_deltas:
        first = delta_matrix(cepstra, width)
        second = None if first is None else delta_matrix(first, width)
        values.update(summarize_matrix(prefix, "delta", first, cepstra.shape[1]))
        values.update(summarize_matrix(prefix, "delta2", second, cepstra.shape[1]))
    return {name: values.get(name, float("nan")) for name in names}
