from __future__ import annotations

import numpy as np
import pandas as pd

from src.comparison.spectra0_comparison import baseline_tables, load_spectra_table, spectra_call_scores
from src.models.dataset import labels_machine, load_call_frame, modeling_rows
from src.models.evaluation import binary_metrics
from src.models.plots import save_roc
from src.models.training import load_bundle, predict_frame, score_frame
from src.utils.config import resolve_path
from src.utils.reports import next_numbered_dir, write_json


MODEL_NAMES = ("logistic", "svm", "lightgbm")


def compare(cfg: dict, variant: str = "full") -> dict:
    baseline = baseline_tables(cfg)
    try:
        bundles = {name: load_bundle(cfg, name, variant) for name in MODEL_NAMES}
    except FileNotFoundError as exc:
        return {"baseline": baseline, "classical": None, "reason": str(exc)}
    frame = load_call_frame(cfg)
    spectra = _spectra(cfg)
    case_rows = []
    rule_rows = []
    curves = []
    for split in ("test", "spectra_threshold"):
        part = modeling_rows(frame, split).merge(spectra, on="call_id", how="inner")
        y_true = labels_machine(part).to_numpy()
        spectra_pred = (part["spectra0_pred"] == "machine").astype(int).to_numpy()
        spectra_rank = -part["spectra0_score"].to_numpy(dtype=float)
        curves.append((f"spectra0:{split}", y_true, spectra_rank))
        rule_rows.append({"model": "spectra0", "split": split, "rule": "spectra_only", **_flat(binary_metrics(y_true, spectra_rank, spectra_pred))})
        for name, bundle in bundles.items():
            scores = score_frame(bundle, part)
            predicted = predict_frame(bundle, scores)
            curves.append((f"{name}:{split}", y_true, scores))
            rule_rows.append({"model": name, "split": split, "rule": "model_only", **_flat(binary_metrics(y_true, scores, predicted))})
            either = np.maximum(predicted, spectra_pred)
            both = predicted * spectra_pred
            rule_rows.append({"model": name, "split": split, "rule": "either_machine", "note": "заранее задано, не выбиралось по test", **_flat(binary_metrics(y_true, either.astype(float), either))})
            rule_rows.append({"model": name, "split": split, "rule": "both_machine", "note": "заранее задано, не выбиралось по test", **_flat(binary_metrics(y_true, both.astype(float), both))})
            case_rows.extend(_cases(part, y_true, predicted, spectra_pred, scores, name, split))
    overlap = _feature_overlap(frame, spectra, bundles)
    report_dir = next_numbered_dir(resolve_path(cfg, cfg["paths"]["reports_dir"]), "comparison")
    write_json(report_dir / "metrics.json", {"variant": variant, "baseline": baseline, "rules": rule_rows})
    pd.DataFrame(rule_rows).to_csv(report_dir / "classification_report.csv", index=False)
    pd.DataFrame(case_rows).to_csv(report_dir / "predictions.csv", index=False)
    overlap.to_csv(report_dir / "feature_spectra_association.csv", index=False)
    test_curves = [item for item in curves if item[0].endswith(":test")]
    if test_curves:
        save_roc(test_curves, report_dir / "roc_curve.png")
    text = _summary(baseline, rule_rows, case_rows, overlap, variant)
    (report_dir / "summary.md").write_text(text, encoding="utf-8")
    return {"report_dir": str(report_dir), "summary": text}


def _spectra(cfg: dict) -> pd.DataFrame:
    threshold = float(cfg["spectra0"]["threshold"])
    frames = [spectra_call_scores(load_spectra_table(resolve_path(cfg, cfg["paths"][key]))) for key in ("spectra0_n100_csv", "spectra0_test_csv")]
    combined = pd.concat(frames, ignore_index=True).drop_duplicates("call_id")
    combined["spectra0_pred"] = np.where(combined["spectra0_score"] < threshold, "machine", "human")
    return combined


def _cases(part, y_true, predicted, spectra_pred, scores, model, split) -> list[dict]:
    rows = []
    for index, record in enumerate(part.itertuples(index=False)):
        truth = "machine" if int(y_true[index]) == 1 else "human"
        model_label = "machine" if int(predicted[index]) == 1 else "human"
        spectra_label = "machine" if int(spectra_pred[index]) == 1 else "human"
        if model_label == truth and spectra_label == truth:
            relation = "both_correct"
        elif model_label == truth and spectra_label != truth:
            relation = "classical_only"
        elif model_label != truth and spectra_label == truth:
            relation = "spectra_only"
        else:
            relation = "both_wrong"
        rows.append({
            "call_id": record.call_id,
            "split": split,
            "model": model,
            "label": truth,
            "spectra0_pred": spectra_label,
            "model_pred": model_label,
            "relation": relation,
            "spectra0_score": float(record.spectra0_score),
            "model_score": float(scores[index]),
            "speech_duration_s": float(record.speech_duration_s),
            "n_usable_segments": int(record.n_usable_segments),
            "snr_db": float(record.snr_db) if np.isfinite(record.snr_db) else None,
        })
    return rows


def _feature_overlap(frame, spectra, bundles) -> pd.DataFrame:
    part = modeling_rows(frame, "spectra_threshold").merge(spectra, on="call_id", how="inner")
    target = (part["label"] == "machine").astype(float)
    features = []
    for bundle in bundles.values():
        features.extend(bundle["features"])
    unique = list(dict.fromkeys(features))
    rows = []
    for name in unique:
        rows.append({
            "feature": name,
            "corr_machine_n100": float(part[name].corr(target)),
            "corr_spectra_gap_n100": float(part[name].corr(part["spectra0_score"])),
        })
    table = pd.DataFrame(rows)
    table["abs_label"] = table["corr_machine_n100"].abs()
    table["complement_score"] = table["abs_label"] * (1.0 - table["corr_spectra_gap_n100"].abs().clip(0, 1))
    return table.sort_values("complement_score", ascending=False)


def _flat(metrics: dict) -> dict:
    keys = ("calls", "human", "machine", "recall_synthetic", "specificity_human", "precision_synthetic", "f1_synthetic", "balanced_accuracy", "fpr", "fnr", "roc_auc", "pr_auc")
    payload = {key: metrics.get(key) for key in keys}
    payload["calls"] = int(metrics["human"] + metrics["machine"])
    return payload


def _summary(baseline, rules, cases, overlap, variant: str = "full") -> str:
    lines = [
        "# Сравнение со Spectra-0",
        "",
        f"Вариант классических моделей: {variant}.",
        "Порог Spectra-0 остаётся 1.02. Классические пороги взяты с validation.",
        "Правила either/both заранее заданы и не выбирались по test.",
        "Score не является калиброванной вероятностью.",
        "",
        "## Spectra-0 отдельно",
        "",
    ]
    for name, metrics in baseline.items():
        lines.append(f"- {name}: recall {float(metrics['recall_synthetic']):.3f}, specificity {float(metrics['specificity_human']):.3f}, balanced accuracy {float(metrics['balanced_accuracy']):.3f}, ROC-AUC {float(metrics['roc_auc']):.3f}")
    lines.extend(["", "## Модели и заранее заданные объединения на test", ""])
    for row in rules:
        if row["split"] != "test":
            continue
        lines.append(
            f"- {row['model']} {row['rule']}: recall {row['recall_synthetic']:.3f}, "
            f"specificity {row['specificity_human']:.3f}, balanced accuracy {row['balanced_accuracy']:.3f}"
        )
    test_cases = [row for row in cases if row["split"] == "test"]
    lines.extend(["", "## Расхождения на test", ""])
    for model in MODEL_NAMES:
        subset = [row for row in test_cases if row["model"] == model]
        counts = pd.Series([row["relation"] for row in subset]).value_counts().to_dict()
        lines.append(f"- {model}: {counts}")
        rescued = [row["call_id"] for row in subset if row["relation"] == "classical_only"]
        shared = [row["call_id"] for row in subset if row["relation"] == "both_wrong"]
        lines.append(f"  classical only: {', '.join(rescued) if rescued else 'нет'}")
        lines.append(f"  both wrong: {', '.join(shared) if shared else 'нет'}")
    lines.extend(["", "## Признаки, которые на n100 слабее повторяют зазор Spectra-0", ""])
    lines.append("Это описание n100, не независимый test. Корреляция не доказывает причину.")
    for row in overlap.head(12).itertuples(index=False):
        lines.append(f"- {row.feature}: corr machine {row.corr_machine_n100:.3f}, corr gap {row.corr_spectra_gap_n100:.3f}")
    lines.append("")
    return "\n".join(lines)
