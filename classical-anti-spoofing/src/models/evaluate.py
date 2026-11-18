from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from src.comparison.spectra0_comparison import load_spectra_table, spectra_call_scores
from src.models.dataset import labels_machine, load_call_frame, modeling_rows
from src.models.evaluation import binary_metrics
from src.models.plots import save_confusion, save_pr, save_roc, save_score_hist
from src.models.training import load_bundle, predict_frame, score_frame
from src.utils.config import resolve_path
from src.utils.reports import next_numbered_dir, write_json


MODEL_NAMES = ("logistic", "svm", "lightgbm")


def evaluate_models(cfg: dict, variant: str = "full") -> dict:
    frame = load_call_frame(cfg)
    bundles = {name: load_bundle(cfg, name, variant) for name in MODEL_NAMES}
    edges = _edges(modeling_rows(frame, "train"))
    spectra = _spectra_scores(cfg)
    metric_rows = []
    slice_rows = []
    curves = []
    score_rows: dict[str, dict] = {}
    report = {"models": {}, "slices": [], "excluded_without_usable_segments": {}}
    for split in ("validation", "test", "spectra_threshold"):
        present = frame[frame["split"] == split]
        part = modeling_rows(frame, split)
        report["excluded_without_usable_segments"][split] = int(len(present) - len(part))
        y_true = labels_machine(part).to_numpy()
        for name, bundle in bundles.items():
            scores = score_frame(bundle, part)
            predicted = predict_frame(bundle, scores)
            metrics = binary_metrics(y_true, scores, predicted)
            metrics["threshold"] = float(bundle["threshold"])
            metrics["calls"] = int(len(part))
            report["models"].setdefault(name, {})[split] = metrics
            metric_rows.append({"model": name, "split": split, **_flat(metrics)})
            for call_id, score, label in zip(part["call_id"], scores, predicted):
                score_rows.setdefault(call_id, {})
                score_rows[call_id][f"{name}_score"] = float(score)
                score_rows[call_id][f"{name}_pred"] = "machine" if int(label) == 1 else "human"
            if split == "test":
                curves.append((name, y_true, scores))
            for column, bins in edges.items():
                slice_rows.extend(_slices(name, split, part, y_true, scores, predicted, column, bins))
    predictions = _assemble_predictions(frame, score_rows, spectra)
    report_dir = next_numbered_dir(resolve_path(cfg, cfg["paths"]["reports_dir"]), "experiment")
    report["variant"] = variant
    report["score_note"] = "Score моделей не калибровался. Пороги взяты из обучения на validation и здесь не менялись."
    report["test_used_for_selection"] = False
    write_json(report_dir / "metrics.json", report)
    pd.DataFrame(metric_rows).to_csv(report_dir / "classification_report.csv", index=False)
    predictions.to_csv(report_dir / "predictions.csv", index=False)
    pd.DataFrame(slice_rows).to_csv(report_dir / "slices.csv", index=False)
    _plots(report_dir, report, curves, predictions)
    _write_importance(cfg, report_dir, variant)
    snapshot = {key: value for key, value in cfg.items() if not str(key).startswith("_")}
    with (report_dir / "experiment_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(snapshot, handle, allow_unicode=True, sort_keys=False)
    text = _summary(report, slice_rows)
    (report_dir / "summary.md").write_text(text, encoding="utf-8")
    return {"report_dir": str(report_dir), "summary": text}


def _assemble_predictions(frame: pd.DataFrame, score_rows: dict[str, dict], spectra: pd.DataFrame) -> pd.DataFrame:
    base = frame[frame["split"].isin(["validation", "test", "spectra_threshold"])][
        ["call_id", "label", "split", "speech_duration_s", "recording_duration_s", "n_usable_segments", "snr_db", "n_segments"]
    ].copy()
    scores = pd.DataFrame.from_dict(score_rows, orient="index").reset_index(names="call_id")
    base = base.merge(scores, on="call_id", how="left")
    return base.merge(spectra, on="call_id", how="left")


def _spectra_scores(cfg: dict) -> pd.DataFrame:
    threshold = float(cfg["spectra0"]["threshold"])
    frames = []
    for key in ("spectra0_n100_csv", "spectra0_test_csv"):
        scores = spectra_call_scores(load_spectra_table(resolve_path(cfg, cfg["paths"][key])))
        frames.append(scores)
    combined = pd.concat(frames, ignore_index=True).drop_duplicates("call_id")
    combined["spectra0_pred"] = np.where(combined["spectra0_score"] < threshold, "machine", "human")
    return combined


def _edges(train: pd.DataFrame) -> dict[str, list[float]]:
    edges = {}
    for column in ("speech_duration_s", "n_usable_segments", "snr_db"):
        values = pd.to_numeric(train[column], errors="coerce").dropna()
        low, high = values.quantile([0.33, 0.66]).tolist()
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            continue
        edges[column] = [float("-inf"), float(low), float(high), float("inf")]
    return edges


def _slices(model, split, frame, y_true, scores, predicted, column, edges) -> list[dict]:
    values = pd.to_numeric(frame[column], errors="coerce")
    labels = ["low", "mid", "high"]
    rows = []
    for index, label in enumerate(labels):
        mask = (values >= edges[index]) & (values < edges[index + 1])
        mask_np = mask.to_numpy()
        if int(mask_np.sum()) < 8 or len(np.unique(y_true[mask_np])) < 2:
            rows.append({"model": model, "split": split, "slice": column, "bin": label, "calls": int(mask_np.sum())})
            continue
        metrics = binary_metrics(y_true[mask_np], scores[mask_np], predicted[mask_np])
        row = {"model": model, "split": split, "slice": column, "bin": label, **_flat(metrics)}
        row["calls"] = int(mask_np.sum())
        rows.append(row)
    return rows


def _plots(report_dir, report, curves, predictions) -> None:
    test = predictions[predictions["split"] == "test"].copy()
    save_roc(curves, report_dir / "roc_curve.png")
    save_pr(curves, report_dir / "precision_recall_curve.png")
    for name in MODEL_NAMES:
        matrix = report["models"][name]["test"]["confusion_matrix"]
        save_confusion(matrix, report_dir / f"confusion_matrix_{name}.png")
        y_true = (test["label"] == "machine").astype(int).to_numpy()
        scores = test[f"{name}_score"].to_numpy(dtype=float)
        finite = np.isfinite(scores)
        save_score_hist(y_true[finite], scores[finite], report_dir / f"scores_{name}.png", f"{name} score")
    save_confusion(report["models"]["lightgbm"]["test"]["confusion_matrix"], report_dir / "confusion_matrix.png")


def _write_importance(cfg: dict, report_dir, variant: str) -> None:
    frames = []
    for name in MODEL_NAMES:
        bundle = load_bundle(cfg, name, variant)
        table = pd.DataFrame(bundle["importance"])
        table.insert(0, "model", name)
        frames.append(table)
    pd.concat(frames, ignore_index=True).to_csv(report_dir / "feature_importance.csv", index=False)


def _flat(metrics: dict) -> dict:
    keys = (
        "human", "machine", "recall_synthetic", "specificity_human", "precision_synthetic",
        "f1_synthetic", "balanced_accuracy", "fpr", "fnr", "roc_auc", "pr_auc", "threshold",
    )
    return {key: metrics.get(key) for key in keys}


def _summary(report: dict, slices: list[dict]) -> str:
    lines = [
        "# Оценка классических моделей",
        "",
        f"Вариант: {report.get('variant', 'full')}.",
        report["score_note"],
        "Порог на test не подбирался.",
        "",
    ]
    for name, splits in report["models"].items():
        lines.append(f"## {name}")
        lines.append("")
        for split, metrics in splits.items():
            lines.append(
                f"- {split}: recall {metrics['recall_synthetic']:.3f}, "
                f"specificity {metrics['specificity_human']:.3f}, "
                f"F1 {metrics['f1_synthetic']:.3f}, "
                f"balanced accuracy {metrics['balanced_accuracy']:.3f}, "
                f"ROC-AUC {metrics['roc_auc']:.3f}, calls {metrics['calls']}"
            )
        lines.append("")
    test_slices = [row for row in slices if row["split"] == "test" and row.get("balanced_accuracy") is not None]
    lines.append("## Test по длительности, числу сегментов и SNR")
    lines.append("")
    lines.append("Границы корзин посчитаны на train.")
    for row in test_slices:
        lines.append(
            f"- {row['model']} {row['slice']} {row['bin']}: calls {row['calls']}, "
            f"balanced accuracy {row.get('balanced_accuracy')}"
        )
    lines.append("")
    return "\n".join(lines)
