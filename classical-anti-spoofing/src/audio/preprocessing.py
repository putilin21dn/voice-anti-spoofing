from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly


def resolve_fmax(sample_rate: int, fmax: float | None) -> float:
    """Верхняя частота не выше Найквиста: новые частоты не создаются."""

    nyquist = sample_rate / 2.0
    if fmax is None:
        return nyquist
    if float(fmax) > nyquist + 1e-6:
        raise ValueError(f"fmax {fmax} Гц выше Найквиста {nyquist} Гц при {sample_rate} Гц")
    if float(fmax) <= 0:
        raise ValueError("fmax должен быть положительным")
    return float(fmax)


def frame_params(sample_rate: int, frame_ms: float, hop_ms: float) -> tuple[int, int, int]:
    win = int(round(sample_rate * frame_ms / 1000.0))
    hop = int(round(sample_rate * hop_ms / 1000.0))
    if win < 8 or hop < 1:
        raise ValueError("Слишком короткое окно или шаг для этой частоты дискретизации")
    n_fft = 1
    while n_fft < win:
        n_fft *= 2
    return win, hop, n_fft


def preemphasis(audio: np.ndarray, coefficient: float) -> np.ndarray:
    if coefficient == 0 or audio.size == 0:
        return audio
    out = np.empty_like(audio, dtype=np.float64)
    out[0] = audio[0]
    out[1:] = audio[1:] - coefficient * audio[:-1]
    return out


def frame_signal(audio: np.ndarray, win_length: int, hop_length: int) -> np.ndarray:
    """Режет сигнал на кадры без дополнения повтором. Короткий хвост отбрасывается."""

    if audio.size < win_length:
        return np.empty((0, win_length), dtype=np.float64)
    count = 1 + (audio.size - win_length) // hop_length
    offsets = hop_length * np.arange(count, dtype=np.int64)
    index = offsets[:, None] + np.arange(win_length, dtype=np.int64)[None, :]
    window = np.hamming(win_length).astype(np.float64)
    return audio[index] * window


def prepare_analysis_audio(
    audio: np.ndarray,
    sample_rate: int,
    *,
    target_sample_rate: int,
    allow_upsample: bool,
) -> tuple[np.ndarray, int, str]:
    """Возвращает сигнал для признаков. Повышение частоты по умолчанию запрещено."""

    signal = np.ascontiguousarray(audio, dtype=np.float64)
    if signal.size == 0:
        raise ValueError("Пустой аудиосигнал")
    if not np.isfinite(signal).all():
        raise ValueError("В сигнале есть NaN или Inf")
    if sample_rate == target_sample_rate:
        return signal, sample_rate, "native"
    if sample_rate < target_sample_rate:
        if not allow_upsample:
            return signal, sample_rate, "upsample_refused_native_kept"
        raise ValueError("Повышение частоты дискретизации в этой версии не выполняется")
    divisor = gcd(target_sample_rate, sample_rate)
    resampled = resample_poly(signal, target_sample_rate // divisor, sample_rate // divisor)
    return np.ascontiguousarray(resampled, dtype=np.float64), target_sample_rate, "downsampled"
