from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.evaluation import binary_metrics
from src.utils.config import resolve_path


def spectra_call_scores(windows: pd.DataFrame) -> pd.DataFrame:
    frame = windows[windows["status"] == "ok"].copy()
    frame["call_id"] = frame["file"].astype(str).str.removesuffix(".wav")
    frame["gap"] = frame["bonafide_logit"].astype(float) - frame["spoof_logit"].astype(float)
    grouped = frame.groupby("call_id", as_index=False).agg(
        spectra0_score=("gap", "mean"),
        spectra0_windows=("gap", "size"),
    )
    return grouped


def evaluate_spectra(scores: pd.DataFrame, labels: pd.DataFrame, threshold: float) -> dict:
    merged = scores.merge(labels[["call_id", "label", "locked_split"]], on="call_id", how="left")
    known = merged[merged["label"].isin(["human", "machine"])].copy()
    y_true = (known["label"] == "machine").astype(int).to_numpy()
    y_pred = (known["spectra0_score"] < threshold).astype(int).to_numpy()
    y_score = (-known["spectra0_score"]).to_numpy()
    metrics = binary_metrics(y_true, y_score, y_pred)
    metrics["threshold"] = threshold
    metrics["decision"] = "machine, если средний зазор bonafide_logit - spoof_logit меньше порога"
    metrics["aggregation"] = "среднее по всем окнам со status=ok, включая короткие добитые хвосты"
    metrics["unlabeled"] = int((~merged["label"].isin(["human", "machine"])).sum())
    metrics["calls"] = int(len(known))
    return metrics


def load_spectra_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def baseline_tables(cfg: dict) -> dict[str, dict]:
    from src.data.dataset_loader import load_dataset

    labels = load_dataset(cfg)[["call_id", "label", "locked_split"]]
    threshold = float(cfg["spectra0"]["threshold"])
    reports = {}
    for name, key in (("n100", "spectra0_n100_csv"), ("test20", "spectra0_test_csv")):
        path = resolve_path(cfg, cfg["paths"][key])
        scores = spectra_call_scores(load_spectra_table(path))
        reports[name] = evaluate_spectra(scores, labels, threshold)
        reports[name]["source"] = str(path)
        if name == "n100":
            reports[name]["caution"] = "На этом наборе выбирался порог Spectra-0. Метрики здесь не являются независимым тестом."
        else:
            reports[name]["caution"] = "Независимый test20. Порог на нём не подбирался."
    return reports
