"""Recall, FPR и precision по длине речи, сегментам, padding и окнам.

Границы терцилей считаются на бинарных звонках 11–12 ноября и теми же
числами применяются к 16 ноября. Сегменты режутся заданными корзинами 1–2 / 3–5 / 6+.
Пороги не двигаются.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from src.research.model_assisted_annotation.build_report import padding_of
from src.research.model_assisted_annotation.common_misses import ORIGIN, REPORT, ROOT, read_csv

HUMAN = {"human_live", "human_recorded"}
SYNTHETIC = {"synthetic_tts", "synthetic_voice_conversion", "synthetic_playback"}
NAMES = {
    "spectra0": "Spectra-0",
    "aasist3": "AASIST3",
    "logistic_new": "Logistic new",
    "svm_new": "SVM new",
    "lightgbm_new": "LightGBM new",
}


def _binary_1112() -> pd.DataFrame:
    pred = pd.DataFrame(read_csv(REPORT / "per_call_predictions.csv"))
    labels = pd.DataFrame(read_csv(ROOT / "data" / "annotations" / "priority_20261111_20261112_calls.csv"))
    labels = labels.drop_duplicates("call_id")
    frame = labels.merge(pred, on="call_id", how="inner")
    frame = frame[frame["primary_label"].isin(HUMAN | SYNTHETIC)].copy()
    frame["synthetic"] = frame["primary_label"].isin(SYNTHETIC)
    return frame


def _binary_16() -> pd.DataFrame:
    long = pd.read_csv(ROOT / "reports" / "day_20261116_model_error_overlap" / "per_call_predictions.csv", encoding="utf-8-sig")
    truth = long.drop_duplicates("call_id")[["call_id", "ground_truth", "primary_label"]]
    wide = long.pivot(index="call_id", columns="model", values="predicted_label")
    wide = wide.rename(columns={name: f"{name}_prediction" for name in wide.columns}).reset_index()
    spectra = pd.read_csv(
        ROOT / "evaluation" / "day_20261116" / "20261116T152222" / "call_results" / "spectra0.csv",
        usecols=["call_id", "status", "n_windows", "speech_segments", "speech_duration_s"],
    )
    windows: dict[str, list[dict]] = defaultdict(list)
    for row in read_csv(ROOT / "evaluation" / "day_20261116" / "20261116T152222" / "window_results" / "spectra0.csv"):
        windows[row["call_id"]].append(row)
    spectra["padding_ratio"] = spectra["call_id"].map(lambda call_id: padding_of(windows.get(call_id, []))[0])
    frame = truth.merge(wide, on="call_id", how="left").merge(spectra, on="call_id", how="left")
    frame = frame.rename(
        columns={
            "speech_duration_s": "spectra_speech_duration_s",
            "speech_segments": "spectra_n_segments",
            "n_windows": "spectra_n_windows",
        }
    )
    frame["synthetic"] = frame["ground_truth"].eq("synthetic")
    return frame


def _edges(values: pd.Series) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    low, high = np.quantile(clean.to_numpy(), [1 / 3, 2 / 3])
    return float(low), float(high)


def _three(values: pd.Series, low: float, high: float, names: tuple[str, str, str]) -> pd.Series:
    number = pd.to_numeric(values, errors="coerce")
    out = pd.Series(pd.NA, index=values.index, dtype="object")
    out[number <= low] = names[0]
    out[(number > low) & (number <= high)] = names[1]
    out[number > high] = names[2]
    return out


def _segments(values: pd.Series) -> pd.Series:
    number = pd.to_numeric(values, errors="coerce")
    out = pd.Series(pd.NA, index=values.index, dtype="object")
    out[number <= 2] = "1–2"
    out[(number >= 3) & (number <= 5)] = "3–5"
    out[number >= 6] = "6+"
    return out


def _metrics(frame: pd.DataFrame, model: str) -> str:
    column = f"{model}_prediction"
    part = frame[frame[column].isin(["human", "synthetic"])]
    y = part["synthetic"].astype(bool)
    pred = part[column].eq("synthetic")
    tp = int((y & pred).sum())
    fn = int((y & ~pred).sum())
    fp = int((~y & pred).sum())
    n_syn = int(y.sum())
    n_hum = int((~y).sum())
    recall = tp / n_syn if n_syn else float("nan")
    fpr = fp / n_hum if n_hum else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    return (
        f"n {len(part)} syn {n_syn} hum {n_hum} "
        f"recall {recall:.3f} fpr {fpr:.3f} prec {precision:.3f} "
        f"tp {tp} fp {fp} fn {fn}"
    )


def _print_slice(title: str, frame: pd.DataFrame, column: str, order: list[str]) -> None:
    print(f"\n## {title}")
    for name in order:
        part = frame[frame[column] == name]
        print(f"\n[{name}] calls {len(part)}")
        for model in ORIGIN:
            print(f"  {NAMES[model]:16} {_metrics(part, model)}")


def main() -> None:
    early = _binary_1112()
    late = _binary_16()
    speech_edges = _edges(early["spectra_speech_duration_s"])
    pad_edges = _edges(early["padding_ratio"])
    window_edges = _edges(early["spectra_n_windows"])
    print("speech edges", speech_edges)
    print("padding edges", pad_edges)
    print("window edges", window_edges)
    print("early segments", pd.to_numeric(early["spectra_n_segments"], errors="coerce").describe().to_string())
    print("early windows value counts\n", pd.to_numeric(early["spectra_n_windows"], errors="coerce").value_counts().sort_index().head(20).to_string())
    for frame, day in ((early, "11-12"), (late, "16")):
        frame = frame.copy()
        frame["speech_bin"] = _three(frame["spectra_speech_duration_s"], *speech_edges, ("короткая", "средняя", "длинная"))
        frame["segment_bin"] = _segments(frame["spectra_n_segments"])
        frame["pad_bin"] = _three(frame["padding_ratio"], *pad_edges, ("низкий", "средний", "высокий"))
        frame["window_bin"] = _three(frame["spectra_n_windows"], *window_edges, ("мало", "средне", "много"))
        print(f"\n===== {day} n {len(frame)} syn {int(frame['synthetic'].sum())} =====")
        _print_slice("речь", frame, "speech_bin", ["короткая", "средняя", "длинная"])
        _print_slice("сегменты", frame, "segment_bin", ["1–2", "3–5", "6+"])
        _print_slice("padding", frame, "pad_bin", ["низкий", "средний", "высокий"])
        _print_slice("окна", frame, "window_bin", ["мало", "средне", "много"])


if __name__ == "__main__":
    main()
