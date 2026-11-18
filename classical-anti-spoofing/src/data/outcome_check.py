from __future__ import annotations

import pandas as pd

from src.data.dataset_loader import load_dataset
from src.utils.config import resolve_path
from src.utils.reports import next_numbered_dir, write_json


HUMAN_LIVE = frozenset(
    {
        "refused",
        "b2x_lead",
        "b2x_callback",
        "b2x_not_lpr",
        "b2x_wrong_org",
        "b2x_async",
    }
)
MODELS = ("logistic", "svm", "lightgbm", "spectra0")


def outcome_bucket(label: str, outcome: str) -> str:
    """Согласованность метки sample и поля outcome. В обучение это поле не входит."""

    value = (outcome or "").strip()
    if value in {"silence", "connection_issue"}:
        return "ambiguous"
    if not value:
        return "empty"
    if label == "machine" and value == "answering_machine":
        return "consistent"
    if label == "human" and value in HUMAN_LIVE:
        return "consistent"
    if label == "machine" and value in HUMAN_LIVE:
        return "conflict"
    if label == "human" and value == "answering_machine":
        return "conflict"
    return "other"


def check_outcomes(cfg: dict) -> dict:
    predictions_path = resolve_path(cfg, "reports/experiment_005/predictions.csv")
    if not predictions_path.is_file():
        raise FileNotFoundError("Нет reports/experiment_005/predictions.csv. Сначала выполните evaluate.")
    predictions = pd.read_csv(predictions_path, dtype={"call_id": str})
    meta = load_dataset(cfg)[["call_id", "outcome", "classification_result"]]
    frame = predictions.merge(meta, on="call_id", how="left")
    frame["outcome_bucket"] = [
        outcome_bucket(label, outcome) for label, outcome in zip(frame["label"], frame["outcome"].fillna(""))
    ]
    error_rows = []
    for split, part in frame.groupby("split"):
        for model in MODELS:
            column = f"{model}_pred"
            usable = part[part[column].isin(["human", "machine"])]
            for bucket, group in usable.groupby("outcome_bucket"):
                wrong = group[column] != group["label"]
                error_rows.append(
                    {
                        "split": split,
                        "model": model,
                        "outcome_bucket": bucket,
                        "calls": int(len(group)),
                        "errors": int(wrong.sum()),
                        "error_rate": float(wrong.mean()) if len(group) else float("nan"),
                    }
                )
    score_rows = _score_overlap(frame)
    conflicts = frame[frame["outcome_bucket"] == "conflict"][
        ["call_id", "label", "split", "outcome", "classification_result", "logistic_pred", "svm_pred", "lightgbm_pred", "spectra0_pred"]
    ]
    report_dir = next_numbered_dir(resolve_path(cfg, cfg["paths"]["reports_dir"]), "outcome_check")
    pd.DataFrame(error_rows).to_csv(report_dir / "errors_by_outcome.csv", index=False)
    pd.DataFrame(score_rows).to_csv(report_dir / "score_overlap.csv", index=False)
    conflicts.to_csv(report_dir / "conflicts.csv", index=False)
    payload = {
        "source_predictions": "reports/experiment_005/predictions.csv",
        "thresholds_changed": False,
        "outcome_in_model": False,
        "bucket_counts": {str(name): int(count) for name, count in frame["outcome_bucket"].value_counts().items()},
    }
    write_json(report_dir / "metrics.json", payload)
    text = _summary(frame, error_rows, score_rows)
    (report_dir / "summary.md").write_text(text, encoding="utf-8")
    return {"report_dir": str(report_dir), "summary": text}


def _score_overlap(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for split, part in frame.groupby("split"):
        for model in MODELS:
            column = f"{model}_score"
            if column not in part.columns:
                continue
            values = pd.to_numeric(part[column], errors="coerce")
            human = values[part["label"] == "human"].dropna()
            machine = values[part["label"] == "machine"].dropna()
            if len(human) < 5 or len(machine) < 5:
                continue
            low, high = human.quantile([0.25, 0.75]).tolist()
            rows.append(
                {
                    "split": split,
                    "model": model,
                    "human_median": float(human.median()),
                    "machine_median": float(machine.median()),
                    "machine_inside_human_iqr": float(((machine >= low) & (machine <= high)).mean()),
                }
            )
    return rows


def _summary(frame: pd.DataFrame, error_rows: list[dict], score_rows: list[dict]) -> str:
    lines = [
        "# Метка sample и поле outcome",
        "",
        "Пороги моделей не менялись. Outcome и classification_result в обучение не входят.",
        "consistent: machine с answering_machine или human с живым исходом.",
        "conflict: machine с живым исходом или human с answering_machine.",
        "empty и ambiguous (silence, connection_issue) посчитаны отдельно.",
        "",
        "## Сколько звонков в каждой корзине",
        "",
    ]
    for split, part in frame.groupby("split"):
        counts = part["outcome_bucket"].value_counts().to_dict()
        lines.append(f"- {split}: {counts}")
    lines.extend(["", "## Ошибки замороженных моделей", ""])
    for row in error_rows:
        if row["calls"] < 5:
            continue
        lines.append(
            f"- {row['split']} {row['model']} {row['outcome_bucket']}: "
            f"{row['errors']} из {row['calls']}, доля {row['error_rate']:.3f}"
        )
    lines.extend(["", "## Перекрытие score", "", "Доля machine, чей score попал в межквартильный размах human. Меньше — классы дальше друг от друга.", ""])
    for row in score_rows:
        lines.append(
            f"- {row['split']} {row['model']}: human median {row['human_median']:.3f}, "
            f"machine median {row['machine_median']:.3f}, "
            f"перекрытие {row['machine_inside_human_iqr']:.3f}"
        )
    lines.append("")
    return "\n".join(lines)
