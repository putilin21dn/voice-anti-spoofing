"""Spectrogram and summary figures. Scales come from the two typical calls, then stay fixed."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.research.acoustic_representations.features import (
    cqcc_librosa,
    cqcc_spafe,
    lfcc_matrix,
    log_safe,
    mel_power,
    stft_power,
)
from src.research.acoustic_representations.signal import load_subscriber, preemphasis, slice_segment

plt.rcParams["font.family"] = "DejaVu Sans"

KINDS = ("logmel", "logstft", "lfcc", "cqcc", "spafe_cqcc")
CEPS = {"lfcc", "cqcc", "spafe_cqcc"}
MAX_FRAMES = 400


def _collect(audio: np.ndarray, intervals: list[tuple[int, float, float]]) -> list[dict[str, np.ndarray]]:
    pieces = []
    for index, start, end in intervals:
        raw = slice_segment(audio, start, end)
        if raw.size < 160:
            continue
        emphasized = preemphasis(raw)
        power = stft_power(emphasized)
        pieces.append(
            {
                "index": index,
                "start": start,
                "logmel": log_safe(mel_power(power)),
                "logstft": log_safe(power),
                "lfcc": lfcc_matrix(power),
                "cqcc": cqcc_librosa(emphasized),
                "spafe_cqcc": cqcc_spafe(emphasized),
            }
        )
    return pieces


def _stitch(pieces: list[dict[str, np.ndarray]], kind: str, limit: int) -> np.ndarray | None:
    chunks = []
    remaining = limit
    for piece in pieces:
        matrix = piece[kind]
        if matrix.size == 0 or remaining <= 0:
            continue
        chunks.append(matrix[:remaining])
        remaining -= matrix.shape[0]
    if not chunks:
        return None
    return np.vstack(chunks)


def _middle(pieces: list[dict[str, np.ndarray]], kind: str) -> np.ndarray | None:
    if not pieces:
        return None
    durations = []
    for piece in pieces:
        durations.append(piece["logmel"].shape[0] if piece["logmel"].size else 0)
    total = sum(durations)
    if total == 0:
        return None
    half = total / 2
    running = 0
    chosen = pieces[0]
    for piece, length in zip(pieces, durations):
        running += length
        chosen = piece
        if running >= half:
            break
    matrix = chosen[kind]
    if matrix.size == 0:
        return None
    return matrix[:MAX_FRAMES]


def _scale(matrices: list[np.ndarray], kind: str) -> tuple[float, float]:
    parts = []
    for matrix in matrices:
        if matrix is None or matrix.size == 0:
            continue
        values = matrix[:, 1:] if kind in CEPS and matrix.shape[1] > 1 else matrix
        parts.append(values.reshape(-1))
    if not parts:
        return 0.0, 1.0
    values = np.concatenate(parts)
    values = values[np.isfinite(values)]
    low, high = np.percentile(values, [1, 99])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return float(low if np.isfinite(low) else 0.0), float(low + 1 if np.isfinite(low) else 1.0)
    return float(low), float(high)


def _outside_fraction(matrix: np.ndarray, kind: str, vmin: float, vmax: float) -> float:
    values = matrix[:, 1:] if kind in CEPS and matrix.shape[1] > 1 else matrix
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.mean((finite < vmin) | (finite > vmax)))


def _draw(path: Path, matrix: np.ndarray, kind: str, vmin: float, vmax: float, title: str) -> None:
    fig, axis = plt.subplots(figsize=(8.2, 3.6))
    show = matrix[:, 1:] if kind in CEPS and matrix.shape[1] > 1 else matrix
    image = axis.imshow(show.T, origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax, interpolation="nearest")
    axis.set_xlabel("Кадр, шаг 10 мс" if kind != "spafe_cqcc" else "Кадр spafe, шаг 10 мс")
    axis.set_ylabel("Коэффициент, начиная с c1" if kind in CEPS else "Полоса")
    axis.set_title(title, fontsize=8)
    fig.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_protocol(root: Path, output_dir: Path, protocol: pd.DataFrame, intervals: dict, audio_paths: dict[str, Path]) -> None:
    cache: dict[str, list[dict[str, np.ndarray]]] = {}
    for call_id in protocol["call_id"]:
        cache[call_id] = _collect(load_subscriber(audio_paths[call_id]), intervals.get(call_id, []))
    scale_ids = set(protocol.loc[protocol["reasons"].str.contains("scale_source"), "call_id"])
    scales = {}
    for kind in KINDS:
        matrices = [_stitch(cache[call_id], kind, MAX_FRAMES) for call_id in scale_ids]
        scales[kind] = _scale(matrices, kind)
    (output_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    (output_dir / "visualizations" / "color_scales.json").write_text(
        json.dumps({kind: {"vmin": scales[kind][0], "vmax": scales[kind][1], "source": "1st-99th percentile of typical human and typical synthetic early slices, cepstra without c0"} for kind in KINDS}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    picture_dir = output_dir / "visualizations" / "protocol"
    for row in protocol.itertuples(index=False):
        pieces = cache[row.call_id]
        views = {"early": {kind: _stitch(pieces, kind, MAX_FRAMES) for kind in KINDS}}
        if float(row.duration_s) > 60:
            views["middle"] = {kind: _middle(pieces, kind) for kind in KINDS}
        for view, matrices in views.items():
            for kind, matrix in matrices.items():
                if matrix is None or matrix.size == 0:
                    continue
                vmin, vmax = scales[kind]
                outside = _outside_fraction(matrix, kind, vmin, vmax)
                title = f"{row.call_id} | {row.primary_label} | {row.spectra_error_role} | {view}"
                if outside > 0.05:
                    title += f" | {outside:.0%} значений вне шкалы"
                _draw(picture_dir / f"{row.call_id}_{kind}_{view}.png", matrix, kind, vmin, vmax, title)


def plot_summaries(output_dir: Path, calls: pd.DataFrame) -> None:
    summary_dir = output_dir / "visualizations" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    binary = calls[calls["binary_group"].isin(["human_voice", "synthetic"])].copy()
    _band_profile(summary_dir / "logmel_band_profile.png", binary, "logmel_b", 26, "Мел-полоса")
    _band_profile(summary_dir / "logstft_band_profile.png", binary, "logstft_band", 16, "Полоса STFT, 250 Гц")
    _coeff_profile(summary_dir / "lfcc_profile.png", binary, "lfcc", 20)
    _coeff_profile(summary_dir / "cqcc_profile.png", binary, "cqcc", 13)
    _coeff_profile(summary_dir / "spafe_cqcc_profile.png", binary, "spafe_cqcc", 13)
    _scatter(summary_dir / "duration_vs_centroid.png", binary, "duration_s", "stft_centroid_hz_mean", "Длительность, с", "Центроид STFT, Гц")
    _scatter(summary_dir / "speech_ratio_vs_flatness.png", binary, "speech_ratio", "stft_flatness_mean", "Доля речи", "Плоскостность STFT")


def _series(frame: pd.DataFrame, prefix: str, count: int, stat: str) -> np.ndarray:
    columns = [f"{prefix}{index:02d}_{stat}" if prefix.endswith("_b") or prefix.endswith("band") else f"{prefix}_c{index:02d}_{stat}" for index in range(count)]
    if prefix in {"logmel_b", "logstft_band"}:
        columns = [f"{prefix}{index:02d}_{stat}" for index in range(count)]
    return frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)


def _band_profile(path: Path, binary: pd.DataFrame, prefix: str, count: int, xlabel: str) -> None:
    fig, axis = plt.subplots(figsize=(8.2, 4.2))
    x = np.arange(count)
    for group, color in (("human_voice", "#1f4e79"), ("synthetic", "#b85c38")):
        part = binary[binary["binary_group"] == group]
        values = _series(part, prefix, count, "mean")
        median = np.nanmedian(values, axis=0)
        q25 = np.nanpercentile(values, 25, axis=0)
        q75 = np.nanpercentile(values, 75, axis=0)
        axis.plot(x, median, color=color, label=f"{group}, медиана")
        axis.fill_between(x, q25, q75, color=color, alpha=0.18)
    fp = binary[binary["spectra_error_role"] == "FP"]
    if len(fp):
        axis.plot(x, np.nanmedian(_series(fp, prefix, count, "mean"), axis=0), color="#2f6f4e", linestyle="--", label="FP, медиана")
    fn = binary[binary["spectra_error_role"] == "FN"]
    for row_index, values in enumerate(_series(fn, prefix, count, "mean")):
        axis.plot(x, values, color="#6b3fa0", alpha=0.45, linewidth=1, label="FN" if row_index == 0 else None)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Средний натуральный логарифм энергии")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _coeff_profile(path: Path, binary: pd.DataFrame, prefix: str, count: int) -> None:
    fig, axis = plt.subplots(figsize=(8.2, 4.2))
    x = np.arange(count)
    for group, color in (("human_voice", "#1f4e79"), ("synthetic", "#b85c38")):
        values = _series(binary[binary["binary_group"] == group], prefix, count, "mean")
        axis.plot(x, np.nanmedian(values, axis=0), color=color, label=f"{group}, медиана")
    fp = binary[binary["spectra_error_role"] == "FP"]
    if len(fp):
        axis.plot(x, np.nanmedian(_series(fp, prefix, count, "mean"), axis=0), color="#2f6f4e", linestyle="--", label="FP, медиана")
    fn = binary[binary["spectra_error_role"] == "FN"]
    for row_index, values in enumerate(_series(fn, prefix, count, "mean")):
        axis.plot(x, values, color="#6b3fa0", alpha=0.45, linewidth=1, label="FN" if row_index == 0 else None)
    axis.set_xlabel("Коэффициент")
    axis.set_ylabel("Среднее значение")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _scatter(path: Path, binary: pd.DataFrame, x_name: str, y_name: str, xlabel: str, ylabel: str) -> None:
    fig, axis = plt.subplots(figsize=(8.2, 4.4))
    for group, color in (("human_voice", "#1f4e79"), ("synthetic", "#b85c38")):
        part = binary[binary["binary_group"] == group]
        axis.scatter(part[x_name], part[y_name], s=16, alpha=0.55, c=color, label=group)
    fp = binary[binary["spectra_error_role"] == "FP"]
    fn = binary[binary["spectra_error_role"] == "FN"]
    axis.scatter(fp[x_name], fp[y_name], s=36, facecolors="none", edgecolors="#2f6f4e", linewidths=1.2, label="FP")
    axis.scatter(fn[x_name], fn[y_name], s=42, marker="x", c="#6b3fa0", label="FN")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
