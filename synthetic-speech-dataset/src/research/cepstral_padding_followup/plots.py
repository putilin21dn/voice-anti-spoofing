"""Figures for the cepstral and padding follow-up. Saved scores are not recomputed."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.research.acoustic_representations.signal import read_csv
from src.research.cepstral_padding_followup.common import (
    FN_CALLS,
    OUTPUT,
    SPECTRA_THRESHOLD,
    WINDOWS,
    read_table,
)

plt.rcParams["font.family"] = "DejaVu Sans"

COLORS = {
    "human_voice": "#4C78A8",
    "synthetic": "#F58518",
    "TN": "#4C78A8",
    "FP": "#E45756",
    "TP": "#F58518",
    "FN": "#54A24B",
}


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    copy = frame.copy()
    for column in columns:
        if column in copy.columns:
            copy[column] = pd.to_numeric(copy[column], errors="coerce")
    return copy


def _save(figure: plt.Figure, name: str) -> None:
    path = OUTPUT / "visualizations" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _scatter_by_role(calls: pd.DataFrame, x_name: str, x_label: str, file_name: str) -> None:
    binary = calls[calls["binary_group"].isin(["human_voice", "synthetic"])]
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for role, marker in (("TN", "o"), ("TP", "o"), ("FP", "X"), ("FN", "D")):
        part = binary[binary["spectra_error_role"] == role]
        axis.scatter(
            part[x_name],
            part["spectra0_margin_frozen"],
            s=28 if role in {"FP", "FN"} else 16,
            alpha=0.8,
            marker=marker,
            c=COLORS[role],
            label=role,
        )
    axis.axhline(SPECTRA_THRESHOLD, color="black", lw=0.8, ls="--")
    axis.set_xlabel(x_label)
    axis.set_ylabel("Сохранённый margin Spectra-0")
    axis.set_title("Ниже 1.02 модель относит звонок к spoof")
    axis.legend(frameon=False)
    _save(figure, file_name)


def _padding_deltas(padding: pd.DataFrame) -> pd.DataFrame:
    keys = ["call_id", "segment_index", "scope", "binary_group", "spectra_error_role"]
    repeat = padding[padding["condition"] == "repeat"]
    zero = padding[padding["condition"] == "zero"]
    merged = repeat.merge(zero, on=keys, suffixes=("_repeat", "_zero"))
    for name in ("cqcc_c01_mean", "lfcc_c01_mean", "spafe_cqcc_c01_mean", "real_audio_fraction"):
        merged[f"{name}_delta"] = merged[f"{name}_repeat"] - merged[f"{name}_zero"]
    return merged


def _window_margins(calls: pd.DataFrame) -> pd.DataFrame:
    labels = calls.set_index("call_id")["binary_group"].to_dict()
    rows = []
    for row in read_csv(WINDOWS):
        rows.append(
            {
                "binary_group": labels.get(row["call_id"], ""),
                "padded": row["padded_to_model_length"] == "True",
                "margin": float(row["margin"]),
                "call_id": row["call_id"],
            }
        )
    return pd.DataFrame(rows)


def make_plots() -> None:
    calls = _numeric(
        read_table(OUTPUT / "call_level_cepstral_summary.csv"),
        [
            "spectra0_margin_frozen",
            "spectra_padding_ratio",
            "speech_duration_s",
            "spectra_n_windows",
            "n_speech_segments",
            "mean_full_window_margin",
            "mean_padded_window_margin",
            "lfcc_between_segment_std_mean",
            "cqcc_between_segment_std_mean",
            "cqcc_c01_mean",
        ],
    )
    _scatter_by_role(calls, "spectra_padding_ratio", "Доля повторённых отсчётов в окнах Spectra-0", "margin_vs_padding_ratio.png")
    _scatter_by_role(calls, "speech_duration_s", "Длительность речи, с", "margin_vs_speech_duration.png")
    _scatter_by_role(calls, "spectra_n_windows", "Число окон Spectra-0", "margin_vs_n_windows.png")
    _scatter_by_role(calls, "n_speech_segments", "Число речевых сегментов", "margin_vs_n_segments.png")

    windows = _window_margins(calls)
    binary_windows = windows[windows["binary_group"].isin(["human_voice", "synthetic"])]
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    positions = []
    data = []
    labels = []
    index = 1
    for group in ("human_voice", "synthetic"):
        for padded, name in ((False, "полное окно"), (True, "хвост с повтором")):
            part = binary_windows[(binary_windows["binary_group"] == group) & (binary_windows["padded"] == padded)]
            data.append(part["margin"].to_numpy())
            positions.append(index)
            labels.append(f"{'human' if group == 'human_voice' else 'synthetic'}\n{name}")
            index += 1
        index += 0.6
    axis.boxplot(data, positions=positions, widths=0.6, showfliers=False)
    axis.axhline(SPECTRA_THRESHOLD, color="black", lw=0.8, ls="--")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylabel("Сохранённый margin окна")
    axis.set_title("Полное окно и повторённый хвост — разные куски сегмента")
    _save(figure, "window_margin_full_vs_padded.png")

    fn = _numeric(
        read_table(OUTPUT / "false_negative_segment_analysis.csv"),
        ["segment_index", "cqcc_c01_mean", "lfcc_c01_mean", "raw_rms", "f0_median_hz", "f0_voiced_fraction", "mean_window_margin"],
    )
    segments = _numeric(read_table(OUTPUT / "segment_level_cepstral_features.csv"), ["cqcc_c01_mean"])
    human_median = float(segments.loc[segments["binary_group"] == "human_voice", "cqcc_c01_mean"].median())
    synth_median = float(segments.loc[segments["binary_group"] == "synthetic", "cqcc_c01_mean"].median())
    figure, axes = plt.subplots(5, 1, figsize=(8, 11), sharex=False)
    for axis, call_id in zip(axes, FN_CALLS):
        part = fn[fn["call_id"] == call_id]
        axis.plot(part["segment_index"], part["cqcc_c01_mean"], marker="o", ms=3, lw=1, color="#54A24B")
        axis.axhline(human_median, color=COLORS["human_voice"], lw=0.8, label="медиана сегментов human")
        axis.axhline(synth_median, color=COLORS["synthetic"], lw=0.8, label="медиана сегментов synthetic")
        axis.set_ylabel("CQCC c1")
        axis.set_title(call_id)
    axes[0].legend(frameon=False, fontsize=8)
    axes[-1].set_xlabel("Индекс речевого сегмента")
    _save(figure, "false_negative_cqcc_c1.png")

    call_id = "8b34a750-b5df-4a6d-be75-7fbc207a9546"
    part = fn[fn["call_id"] == call_id]
    figure, axes = plt.subplots(5, 1, figsize=(8, 10), sharex=True)
    series = (
        ("cqcc_c01_mean", "CQCC librosa c1"),
        ("lfcc_c01_mean", "LFCC c1"),
        ("raw_rms", "RMS сегмента"),
        ("f0_median_hz", "Медиана F0, Гц"),
        ("f0_voiced_fraction", "Доля voiced"),
    )
    for axis, (column, label) in zip(axes, series):
        axis.plot(part["segment_index"], part[column], marker="o", ms=3, lw=1, color="#54A24B")
        axis.set_ylabel(label)
    axes[-1].set_xlabel("Индекс сегмента")
    axes[0].set_title(call_id)
    _save(figure, "false_negative_8b34a750_segments.png")

    a037 = "a037652f-505c-4f81-a49d-a2ed36434761"
    part = fn[fn["call_id"] == a037]
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.bar(part["segment_index"].astype(int), part["mean_window_margin"], color="#54A24B")
    axis.axhline(SPECTRA_THRESHOLD, color="black", lw=0.8, ls="--")
    axis.set_xlabel("Сегмент")
    axis.set_ylabel("Сохранённый margin окна")
    axis.set_title(a037)
    _save(figure, "false_negative_a037652f_windows.png")

    padding_path = OUTPUT / "padding_sensitivity.csv"
    if not padding_path.exists():
        return
    padding = _numeric(
        read_table(padding_path),
        ["cqcc_c01_mean", "lfcc_c01_mean", "spafe_cqcc_c01_mean", "real_audio_fraction", "segment_index"],
    )
    deltas = _padding_deltas(padding)
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), sharey=True)
    for axis, scope, title in zip(axes, ("tail", "segment"), ("Только хвост", "Сегмент с дописанным хвостом")):
        part = deltas[deltas["scope"] == scope]
        for group in ("human_voice", "synthetic"):
            group_part = part[part["binary_group"] == group]
            axis.scatter(
                group_part["real_audio_fraction_repeat"],
                group_part["cqcc_c01_mean_delta"],
                s=10,
                alpha=0.45,
                c=COLORS[group],
                label=group,
            )
        axis.axhline(0, color="black", lw=0.6)
        axis.set_title(title)
        axis.set_xlabel("Доля реальных отсчётов")
        axis.set_ylabel("CQCC c1: повтор минус нули")
    axes[0].legend(frameon=False)
    _save(figure, "padding_cqcc_c1_repeat_vs_zero.png")

    norm_path = OUTPUT / "normalization_sensitivity.csv"
    if not norm_path.exists():
        return
    norm = _numeric(
        read_table(norm_path),
        ["lfcc_between_segment_std_mean", "cqcc_between_segment_std_mean", "spafe_cqcc_between_segment_std_mean", "cqcc_c01_mean"],
    )
    endpoints = (
        ("lfcc_between_segment_std_mean", "Межсегментное СКО LFCC"),
        ("cqcc_between_segment_std_mean", "Межсегментное СКО CQCC"),
        ("spafe_cqcc_between_segment_std_mean", "Межсегментное СКО spafe"),
        ("cqcc_c01_mean", "CQCC c1"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 7.2))
    conditions = ["none", "peak", "rms"]
    for axis, (endpoint, title) in zip(axes.ravel(), endpoints):
        positions = []
        data = []
        labels = []
        index = 1
        for condition in conditions:
            for group, short in (("human_voice", "H"), ("synthetic", "S")):
                values = norm.loc[(norm["condition"] == condition) & (norm["binary_group"] == group), endpoint].to_numpy(dtype=np.float64)
                values = values[np.isfinite(values)]
                data.append(values)
                positions.append(index)
                labels.append(f"{condition}\n{short}")
                index += 1
            index += 0.5
        axis.boxplot(data, positions=positions, widths=0.55, showfliers=False)
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, fontsize=7)
        axis.set_title(title)
    _save(figure, "normalization_between_segment_std.png")
