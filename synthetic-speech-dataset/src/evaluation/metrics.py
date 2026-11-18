from __future__ import annotations

from src.evaluation.dataset import HUMAN_ORIGINS, ORIGIN_LABELS, SYNTHETIC_ORIGINS


SPECTRA0_THRESHOLD = 1.02


def spectra0_class(score: float) -> str:
    """spoof, если средний зазор строго меньше 1.02. Равенство — bonafide."""

    return "spoof" if float(score) < SPECTRA0_THRESHOLD else "bonafide"


def classical_class(score: float, threshold: float) -> str:
    """Положительный класс старой задачи — machine, если score не ниже сохранённого порога."""

    return "machine" if float(score) >= float(threshold) else "human"


def category_counts(calls: list[dict], model: str) -> list[dict]:
    rows = []
    for category in ORIGIN_LABELS:
        chosen = [
            call
            for call in calls
            if call.get("category") == category and call.get("in_category_metrics") and _ok(call, model)
        ]
        if category == "unknown":
            chosen = [call for call in calls if call.get("category") == "unknown" and _ok(call, model)]
        row = {"category": category, "calls": len(chosen)}
        if model in {"spectra0", "aasist3"}:
            classes = [_class(call, model) for call in chosen]
            known = [item for item in classes if item in {"bonafide", "spoof"}]
            row.update(
                {
                    "bonafide": sum(item == "bonafide" for item in known),
                    "spoof": sum(item == "spoof" for item in known),
                    "spoof_rate": _rate(sum(item == "spoof" for item in known), len(known)),
                    "class_note": "" if model == "spectra0" else "Порог звонка AASIST3 не зафиксирован",
                }
            )
            if model == "aasist3":
                row["bonafide"] = ""
                row["spoof"] = ""
                row["spoof_rate"] = ""
        else:
            classes = [_class(call, model) for call in chosen]
            known = [item for item in classes if item in {"human", "machine"}]
            row.update(
                {
                    "human": sum(item == "human" for item in known),
                    "machine": sum(item == "machine" for item in known),
                    "machine_rate": _rate(sum(item == "machine" for item in known), len(known)),
                }
            )
        rows.append(row)
    return rows


def binary_metrics(calls: list[dict], model: str) -> dict:
    """Бинарные метрики только когда есть и человек, и подтверждённая синтетика."""

    chosen = [call for call in calls if call.get("in_binary_metrics") and _ok(call, model)]
    humans = [call for call in chosen if call["category"] in HUMAN_ORIGINS]
    synthetic = [call for call in chosen if call["category"] in SYNTHETIC_ORIGINS]
    base = {
        "model": model,
        "human_calls": len(humans),
        "synthetic_calls": len(synthetic),
        "task_note": _task_note(model),
    }
    if model == "aasist3":
        base["reason"] = "Порог звонка AASIST3 не зафиксирован. Бинарные метрики не считаются."
        return base
    if not humans or not synthetic:
        base["reason"] = "В знаменателе нет одновременно подтверждённого человека и подтверждённой синтетики."
        return base
    positive = "spoof" if model == "spectra0" else "machine"
    negative = "bonafide" if model == "spectra0" else "human"
    truth_positive = synthetic
    truth_negative = humans
    tp = sum(_class(call, model) == positive for call in truth_positive)
    fn = sum(_class(call, model) == negative for call in truth_positive)
    tn = sum(_class(call, model) == negative for call in truth_negative)
    fp = sum(_class(call, model) == positive for call in truth_negative)
    recall = _rate(tp, tp + fn)
    specificity = _rate(tn, tn + fp)
    precision = _rate(tp, tp + fp)
    base.update(
        {
            "reason": "",
            "recall_synthetic": recall,
            "specificity_human": specificity,
            "precision_synthetic": precision,
            "f1": _f1(precision, recall),
            "balanced_accuracy": _mean(recall, specificity),
            "fpr": _rate(fp, fp + tn),
            "fnr": _rate(fn, fn + tp),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }
    )
    return base


def legacy_task_counts(calls: list[dict], model: str) -> list[dict]:
    """Счётчики старого test20. human/machine здесь не становятся классом происхождения."""

    rows = []
    for label in ("human", "machine"):
        chosen = [
            call
            for call in calls
            if call.get("category") == "legacy_test" and call.get("legacy_label") == label and _ok(call, model)
        ]
        row = {"model": model, "legacy_label": label, "calls": len(chosen), "task": "live_dialogue_vs_answering_machine"}
        if model == "aasist3":
            scores = [_float(call.get("aasist3_score")) for call in chosen]
            scores = [score for score in scores if score is not None]
            row.update(
                {
                    "predicted_class": "",
                    "mean_bonafide_logit": (sum(scores) / len(scores)) if scores else "",
                    "class_note": "Порог звонка AASIST3 не зафиксирован",
                }
            )
        elif model == "spectra0":
            bonafide = sum(_class(call, model) == "bonafide" for call in chosen)
            spoof = sum(_class(call, model) == "spoof" for call in chosen)
            row.update({"bonafide": bonafide, "spoof": spoof, "spoof_rate": _rate(spoof, bonafide + spoof)})
        else:
            human = sum(_class(call, model) == "human" for call in chosen)
            machine = sum(_class(call, model) == "machine" for call in chosen)
            matched = sum(_class(call, model) == label for call in chosen)
            row.update(
                {
                    "human": human,
                    "machine": machine,
                    "matches_legacy_label": matched,
                    "match_rate": _rate(matched, len(chosen)),
                }
            )
        rows.append(row)
    return rows


def disagreement_flags(call: dict) -> list[str]:
    spectra = _class(call, "spectra0")
    flags = []
    for name in ("logistic", "svm", "lightgbm"):
        classical = _class(call, name)
        if not spectra or not classical:
            continue
        if spectra == "bonafide" and classical == "machine":
            flags.append(f"spectra0_bonafide_{name}_machine")
        if spectra == "spoof" and classical == "human":
            flags.append(f"spectra0_spoof_{name}_human")
        if (spectra == "spoof" and classical == "machine") or (spectra == "bonafide" and classical == "human"):
            flags.append(f"spectra0_agrees_direction_{name}")
        else:
            flags.append(f"spectra0_differs_{name}")
    return flags


def _float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ok(call: dict, model: str) -> bool:
    return (call.get(f"{model}_status") or "") == "ok"


def _class(call: dict, model: str) -> str:
    return call.get(f"{model}_class") or ""


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def _mean(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return (left + right) / 2


def _task_note(model: str) -> str:
    if model in {"logistic", "svm", "lightgbm"}:
        return "Старая задача: живой диалог против автоответчика. Это не метрика детектора синтеза."
    if model == "spectra0":
        return "Класс звонка: средний зазор bonafide_logit - spoof_logit, spoof при зазоре меньше 1.02."
    return "Порог звонка не зафиксирован."
