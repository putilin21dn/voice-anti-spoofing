"""Сравнение семи TTS, пропущенных всеми моделями происхождения, с пойманными TTS PACK1.

Обучения нет. Пороги не двигаются. Аудио читается только чтобы снять RMS.
"""

from __future__ import annotations

import csv
import wave
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "reports" / "model_assisted_annotation_20261111_20261112"
ORIGIN = ["spectra0", "aasist3", "logistic_new", "svm_new", "lightgbm_new"]
SPECTRA_THR = 0.22710928916931153
AASIST_THR = 0.11397590637207031
KEY_FEATURES = [
    "speech_duration_s",
    "n_segments",
    "n_usable_segments",
    "speech_ratio",
    "snr_db",
    "f0_mean__mean",
    "f0_std__mean",
    "f0_range__mean",
    "f0_voiced_fraction__mean",
    "lfcc_static_00_mean__mean",
    "lfcc_static_01_mean__mean",
    "cqcc_static_00_mean__mean",
    "cqcc_static_01_mean__mean",
    "cqcc_delta_12_max__max",
    "lfcc_delta2_00_std__median",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def groups() -> tuple[list[str], list[str], list[str]]:
    pred = {row["call_id"]: row for row in read_csv(REPORT / "per_call_predictions.csv")}
    labels = {row["call_id"]: row for row in read_csv(ROOT / "data" / "annotations" / "priority_20261111_20261112_calls.csv")}
    missed, caught, humans = [], [], []
    for call_id, label in labels.items():
        row = pred[call_id]
        if not str(row["campaign"]).endswith("PACK1"):
            continue
        votes = [row.get(f"{name}_prediction") for name in ORIGIN]
        if label["primary_label"] == "synthetic_tts" and all(vote == "human" for vote in votes):
            missed.append(call_id)
        elif label["primary_label"] == "synthetic_tts" and all(vote == "synthetic" for vote in votes):
            caught.append(call_id)
        elif label["primary_label"] == "human_live" and all(vote == "human" for vote in votes):
            humans.append(call_id)
    return missed, caught, humans


def windows(call_ids: set[str]) -> pd.DataFrame:
    frames = []
    for day, stamp in (("day_20261111", "20261116T195124"), ("day_20261112", "20261116T195754")):
        folder = ROOT / "evaluation" / day / stamp / "window_results"
        for model, score_name in (("spectra0", "margin"), ("aasist3", "bonafide_logit")):
            frame = pd.read_csv(folder / f"{model}.csv")
            frame = frame[frame["call_id"].isin(call_ids)].copy()
            frame["model"] = model
            frame["decision_score"] = pd.to_numeric(frame[score_name], errors="coerce")
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def audio_levels(call_ids: list[str]) -> pd.DataFrame:
    manifest = {}
    for name in ("day_20261111.csv", "day_20261112.csv"):
        for row in read_csv(ROOT / "data" / "manifests" / name):
            manifest[row["call_id"]] = row
    rows = []
    for call_id in call_ids:
        source = ROOT / manifest[call_id]["source_file"]
        channel = int(manifest[call_id].get("client_channel") or 0)
        with wave.open(str(source), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
        channel_audio = frames.reshape(-1, channels)[:, channel].astype(np.float64) / 32768.0
        rms = float(np.sqrt(np.mean(np.square(channel_audio)))) if channel_audio.size else np.nan
        frame = int(0.02 * rate)
        usable = channel_audio[: channel_audio.size // frame * frame].reshape(-1, frame)
        frame_rms = np.sqrt(np.mean(np.square(usable), axis=1))
        db = 20.0 * np.log10(np.maximum(frame_rms, 1e-8))
        active = db[db > -45.0]
        dynamic = float(np.percentile(active, 95) - np.percentile(active, 5)) if active.size >= 8 else np.nan
        rows.append(
            {
                "call_id": call_id,
                "rms_dbfs": 20.0 * np.log10(max(rms, 1e-8)),
                "dynamic_range_db": dynamic,
            }
        )
    return pd.DataFrame(rows)


def median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if len(values) else float("nan")


def summarize(frame: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    return {column: median(frame[column]) for column in columns if column in frame.columns}


def main() -> None:
    missed, caught, humans = groups()
    print(f"пропущены всеми: {len(missed)}")
    print(f"PACK1 TTS, все пять synthetic: {len(caught)}")
    print(f"PACK1 human, все пять human: {len(humans)}")
    pred = pd.DataFrame(read_csv(REPORT / "per_call_predictions.csv")).set_index("call_id")
    features = pd.read_csv(REPORT / "new_classical_features.csv", usecols=lambda name: name == "call_id" or name in KEY_FEATURES)
    features = features.set_index("call_id")
    levels = audio_levels(missed + caught + humans).set_index("call_id")
    window_frame = windows(set(missed + caught))

    print("\n--- семь звонков ---")
    for call_id in missed:
        row = pred.loc[call_id]
        feat = features.loc[call_id] if call_id in features.index else None
        level = levels.loc[call_id]
        print(
            call_id[:8],
            row["date"],
            "речь",
            row["spectra_speech_duration_s"],
            "сегм",
            row["spectra_n_segments"],
            "окон",
            row["spectra_n_windows"],
            "pad",
            row["padding_ratio"],
            "rms",
            f"{level['rms_dbfs']:.1f}",
            "dyn",
            f"{level['dynamic_range_db']:.1f}",
            "f0",
            None if feat is None else round(float(feat["f0_mean__mean"]), 1),
            "voiced",
            None if feat is None else round(float(feat["f0_voiced_fraction__mean"]), 2),
            "amd",
            row["amd_campaign"],
        )
        own = window_frame[window_frame["call_id"] == call_id]
        for model, threshold, below in (("spectra0", SPECTRA_THR, True), ("aasist3", AASIST_THR, True)):
            part = own[own["model"] == model]
            scores = part["decision_score"]
            synthetic = (scores < threshold) if below else (scores >= threshold)
            print(
                " ",
                model,
                "окон",
                len(part),
                "synthetic",
                int(synthetic.sum()),
                "min",
                f"{scores.min():.3f}",
                "median",
                f"{scores.median():.3f}",
                "max",
                f"{scores.max():.3f}",
                "padded",
                int(part["padded_to_model_length"].astype(str).str.lower().eq("true").sum()),
            )

    print("\n--- медианы групп ---")
    acoustic = pred.loc[missed + caught + humans, ["spectra_speech_duration_s", "spectra_n_segments", "spectra_n_windows", "padding_ratio"]].apply(pd.to_numeric, errors="coerce")
    acoustic = acoustic.join(levels)
    for column in ["spectra_speech_duration_s", "spectra_n_segments", "spectra_n_windows", "padding_ratio", "rms_dbfs", "dynamic_range_db"]:
        print(
            column,
            "miss",
            round(median(acoustic.loc[missed, column]), 3),
            "caught",
            round(median(acoustic.loc[caught, column]), 3),
            "human",
            round(median(acoustic.loc[humans, column]), 3),
        )
    print("\n--- признаки ---")
    for column in KEY_FEATURES:
        if column not in features.columns:
            continue
        print(
            column,
            "miss",
            round(median(features.loc[features.index.intersection(missed), column]), 3),
            "caught",
            round(median(features.loc[features.index.intersection(caught), column]), 3),
            "human",
            round(median(features.loc[features.index.intersection(humans), column]), 3),
        )

    model_features = pd.read_csv(ROOT / "reports" / "origin_retrain_20261113" / "features.csv")["feature"].tolist()
    present = [name for name in model_features if name in pd.read_csv(REPORT / "new_classical_features.csv", nrows=0).columns]
    wide = pd.read_csv(REPORT / "new_classical_features.csv", usecols=["call_id", *present]).set_index("call_id")
    caught_med = wide.loc[wide.index.intersection(caught)].median(numeric_only=True)
    human_med = wide.loc[wide.index.intersection(humans)].median(numeric_only=True)
    spread = (caught_med - human_med).abs()
    usable = spread[spread > 1e-6].index
    print("\nпризнаков модели в кадре", len(present), "разделяющих", len(usable))
    for call_id in missed:
        row = wide.loc[call_id, usable]
        closer_human = ((row - human_med[usable]).abs() < (row - caught_med[usable]).abs()).sum()
        print(call_id[:8], "ближе к human по", int(closer_human), "из", len(usable))


if __name__ == "__main__":
    main()
