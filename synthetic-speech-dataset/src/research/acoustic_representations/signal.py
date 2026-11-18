"""Load the subscriber channel and the speech intervals already used by Spectra-0."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from src.research.acoustic_representations.config import (
    CLIENT_CHANNEL,
    CLIP_LEVEL,
    MIN_NONSPEECH_S,
    PEAK_NORM_TARGET,
    PREEMPHASIS,
    SAMPLE_RATE,
    SPECTRA_MODEL_WINDOW_S,
    SPECTRA_THRESHOLD,
    SYNTH_LABELS,
    HUMAN_LABELS,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def reconstruct_intervals(windows: list[dict[str, str]]) -> dict[str, list[tuple[int, float, float]]]:
    """Speech intervals Spectra-0 actually scored, in seconds on the original timeline."""

    grouped: dict[str, dict[int, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in windows:
        grouped[row["call_id"]][int(row["segment_index"])].append(row)
    intervals: dict[str, list[tuple[int, float, float]]] = {}
    for call_id, segments in grouped.items():
        spans = []
        for index in sorted(segments):
            rows = segments[index]
            start = min(float(row["start_time"]) for row in rows)
            end = max(float(row["end_time"]) for row in rows)
            spans.append((index, start, end))
        intervals[call_id] = spans
    return intervals


def padding_stats(windows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in windows:
        grouped[row["call_id"]].append(row)
    stats: dict[str, dict[str, float]] = {}
    for call_id, rows in grouped.items():
        valid = np.array([float(row["valid_speech_audio_sec"]) for row in rows], dtype=np.float64)
        padded = np.array([row["padded_to_model_length"] == "True" for row in rows])
        repeated = np.where(padded, SPECTRA_MODEL_WINDOW_S - valid, 0.0)
        repeated = np.maximum(repeated, 0.0)
        denominator = float(valid.sum() + repeated.sum())
        tails = valid[padded]
        stats[call_id] = {
            "spectra_n_windows": float(len(rows)),
            "spectra_n_padded_windows": float(padded.sum()),
            "spectra_valid_audio_s": float(valid.sum()),
            "spectra_repeated_audio_s": float(repeated.sum()),
            "spectra_padding_ratio": float(repeated.sum() / denominator) if denominator else np.nan,
            "spectra_min_tail_s": float(tails.min()) if tails.size else np.nan,
            "spectra_n_tails_below_0_1s": float(np.sum(tails < 0.1)) if tails.size else 0.0,
            "spectra_n_tails_below_0_5s": float(np.sum(tails < 0.5)) if tails.size else 0.0,
        }
    return stats


def binary_role(primary: str, margin: str) -> tuple[str, str]:
    if margin == "":
        predicted = ""
    else:
        predicted = "spoof" if float(margin) < SPECTRA_THRESHOLD else "bonafide"
    if primary in HUMAN_LABELS:
        side = "human_voice"
    elif primary in SYNTH_LABELS:
        side = "synthetic"
    else:
        return "excluded", predicted
    if predicted == "":
        return "excluded", predicted
    if side == "synthetic" and predicted == "spoof":
        return "TP", predicted
    if side == "synthetic" and predicted == "bonafide":
        return "FN", predicted
    if side == "human_voice" and predicted == "spoof":
        return "FP", predicted
    return "TN", predicted


def campaign_code(name: str) -> str:
    if name.endswith("PACK1"):
        return "PACK1"
    if name.endswith("PACK2"):
        return "PACK2"
    raise ValueError(f"Неизвестная кампания: {name}")


def load_subscriber(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if int(sample_rate) != SAMPLE_RATE:
        raise ValueError(f"{path.name}: частота {sample_rate}, ожидалась {SAMPLE_RATE}")
    if audio.shape[1] < CLIENT_CHANNEL + 1:
        raise ValueError(f"{path.name}: нет канала абонента")
    return np.ascontiguousarray(audio[:, CLIENT_CHANNEL], dtype=np.float64)


def slice_segment(audio: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    start = max(0, int(round(start_s * SAMPLE_RATE)))
    end = min(audio.size, int(round(end_s * SAMPLE_RATE)))
    if end <= start:
        return np.empty(0, dtype=np.float64)
    return audio[start:end]


def preemphasis(audio: np.ndarray, coefficient: float = PREEMPHASIS) -> np.ndarray:
    if audio.size == 0:
        return audio
    out = np.empty_like(audio)
    out[0] = audio[0]
    out[1:] = audio[1:] - coefficient * audio[:-1]
    return out


def peak_normalize(audio: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale the peak to 0.99. A clipped waveform stays flat; only its gain changes."""

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0.0 or not np.isfinite(peak):
        return audio.copy(), np.nan
    gain = PEAK_NORM_TARGET / peak
    return audio * gain, gain


def signal_quality(audio: np.ndarray) -> dict[str, float]:
    if audio.size == 0:
        return {"rms": np.nan, "peak": np.nan, "clip_fraction": np.nan}
    return {
        "rms": float(np.sqrt(np.mean(audio * audio))),
        "peak": float(np.max(np.abs(audio))),
        "clip_fraction": float(np.mean(np.abs(audio) >= CLIP_LEVEL)),
    }


def nonspeech_rms(audio: np.ndarray, intervals: list[tuple[int, float, float]]) -> float:
    mask = np.ones(audio.size, dtype=bool)
    for _, start_s, end_s in intervals:
        start = max(0, int(round(start_s * SAMPLE_RATE)))
        end = min(audio.size, int(round(end_s * SAMPLE_RATE)))
        mask[start:end] = False
    if int(mask.sum()) < int(MIN_NONSPEECH_S * SAMPLE_RATE):
        return np.nan
    chunk = audio[mask]
    return float(np.sqrt(np.mean(chunk * chunk)))
