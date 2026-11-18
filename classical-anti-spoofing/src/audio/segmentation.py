from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly


def merge_voice_frames(
    flags: list[tuple[float, bool]],
    frame_s: float,
    merge_gap_s: float,
) -> list[tuple[float, float]]:
    """Склеивает соседние речевые кадры, если пауза между ними не длиннее порога.

    Правило совпадает с detect_speech в Spectra-0: пауза длиннее порога
    закрывает сегмент. Короткие сегменты здесь не отбрасываются.
    """

    raw: list[tuple[float, float]] = []
    active_start: float | None = None
    last_voice_end: float | None = None
    for start, is_voice in flags:
        end = start + frame_s
        if is_voice:
            if active_start is None:
                active_start = start
            last_voice_end = end
        elif active_start is not None and last_voice_end is not None and (start - last_voice_end) > merge_gap_s:
            raw.append((active_start, last_voice_end))
            active_start = None
            last_voice_end = None
    if active_start is not None and last_voice_end is not None:
        raw.append((active_start, last_voice_end))
    return raw


def _pcm16(audio: np.ndarray, sample_rate: int, vad_rate: int = 8000) -> np.ndarray:
    signal = audio.astype(np.float64)
    if sample_rate != vad_rate:
        divisor = gcd(vad_rate, sample_rate)
        signal = resample_poly(signal, vad_rate // divisor, sample_rate // divisor)
    signal = np.clip(signal, -1.0, 1.0)
    return (signal * 32767.0).astype("<i2")


def detect_speech(
    audio: np.ndarray,
    sample_rate: int,
    *,
    vad_mode: int = 2,
    frame_ms: int = 20,
    merge_gap_ms: int = 300,
) -> list[tuple[float, float]]:
    """WebRTC VAD. Время сегментов — в секундах исходного сигнала."""

    import webrtcvad

    if frame_ms not in (10, 20, 30):
        raise ValueError("WebRTC VAD принимает кадр 10, 20 или 30 мс")
    vad_rate = 8000
    pcm = _pcm16(audio, sample_rate, vad_rate)
    frame_n = vad_rate * frame_ms // 1000
    vad = webrtcvad.Vad(int(vad_mode))
    flags: list[tuple[float, bool]] = []
    for pos in range(0, len(pcm) - frame_n + 1, frame_n):
        frame = pcm[pos : pos + frame_n].tobytes()
        flags.append((pos / vad_rate, bool(vad.is_speech(frame, vad_rate))))
    duration = len(audio) / float(sample_rate) if sample_rate else 0.0
    merged = merge_voice_frames(flags, frame_ms / 1000.0, merge_gap_ms / 1000.0)
    segments: list[tuple[float, float]] = []
    for start, end in merged:
        start = max(0.0, start)
        end = min(duration, end)
        if end > start:
            segments.append((start, end))
    return segments
