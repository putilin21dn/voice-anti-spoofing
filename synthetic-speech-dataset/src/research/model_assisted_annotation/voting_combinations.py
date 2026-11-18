"""Сравнение комбинаций голоса 2 из 3 на уже сохранённых решениях.

Нового прогона, обучения и подбора порогов нет. Положительный класс — synthetic.
Правило 2 из 2 для Spectra-0 и AASIST3 считается отдельно и в majority не входит.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.research.model_assisted_annotation.rules import BELOW, THRESHOLDS, prediction
from src.research.model_assisted_annotation.three_experiments import SPEECH_HIGH, SPEECH_LOW

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "reports" / "voting_combinations"
REPORT = ROOT / "reports" / "voting_combinations_analysis.md"

HUMAN = {"human_live", "human_recorded"}
SYNTHETIC = {"synthetic_tts", "synthetic_voice_conversion", "synthetic_playback"}
MODELS = ("spectra0", "aasist3", "svm_new", "lightgbm_new")
MODEL_NAME = {
    "spectra0": "Spectra-0",
    "aasist3": "AASIST3",
    "svm_new": "SVM",
    "lightgbm_new": "LightGBM",
}
SCORE_NAME = {
    "spectra0": "margin",
    "aasist3": "средний bonafide logit",
    "svm_new": "score",
    "lightgbm_new": "score",
}
THRESHOLD_TEXT = {
    "spectra0": "0.22710928916931153",
    "aasist3": "0.11397590637207031",
    "svm_new": "0.0",
    "lightgbm_new": "0.3460832258410117",
    "logistic_new": "0.7354826103642409",
}

# Majority 2 из 3. Пара Spectra-0 + AASIST3 сюда не входит.
COMBOS = {
    "A": ("spectra0", "svm_new", "lightgbm_new"),
    "B": ("spectra0", "aasist3", "lightgbm_new"),
    "C": ("spectra0", "aasist3", "svm_new"),
    "D": ("aasist3", "svm_new", "lightgbm_new"),
}
COMBO_NAME = {
    "A": "Spectra-0 + SVM + LightGBM",
    "B": "Spectra-0 + AASIST3 + LightGBM",
    "C": "Spectra-0 + AASIST3 + SVM",
    "D": "AASIST3 + SVM + LightGBM",
    "pair_2of2": "Spectra-0 + AASIST3, оба голоса",
}
SHORT = {"A": "A", "B": "B", "C": "C", "D": "D", "pair_2of2": "2 из 2"}
DATE_ORDER = ("11–12", "16")
DATE_COLOR = {"11–12": "#215c98", "16": "#d4762c"}
COMBO_COLOR = {"A": "#1f4e79", "B": "#2f7d5d", "C": "#c47b2b", "D": "#6d4c7d", "pair_2of2": "#8d3b3b"}
SPEECH_ORDER = ("короткая", "средняя", "длинная")
PACK_ORDER = ("PACK1", "PACK2")

# Опубликованные числа прежних отчётов. Сверка выборки, не подбор порога.
EXPECTED_TRIO_FN = {
    "11–12": {
        ("spectra0",): 20,
        ("svm_new",): 8,
        ("lightgbm_new",): 6,
        ("spectra0", "svm_new"): 1,
        ("spectra0", "lightgbm_new"): 4,
        ("svm_new", "lightgbm_new"): 13,
        ("lightgbm_new", "spectra0", "svm_new"): 7,
        (): 75,
    },
    "16": {
        ("spectra0",): 8,
        ("svm_new",): 6,
        ("lightgbm_new",): 3,
        ("spectra0", "svm_new"): 0,
        ("spectra0", "lightgbm_new"): 1,
        ("svm_new", "lightgbm_new"): 8,
        ("lightgbm_new", "spectra0", "svm_new"): 6,
        (): 26,
    },
}
EXPECTED_TRIO_FP = {
    "11–12": {
        ("spectra0",): 29,
        ("svm_new",): 8,
        ("lightgbm_new",): 5,
        ("spectra0", "svm_new"): 0,
        ("spectra0", "lightgbm_new"): 1,
        ("svm_new", "lightgbm_new"): 7,
        ("lightgbm_new", "spectra0", "svm_new"): 0,
        (): 273,
    },
    "16": {
        ("spectra0",): 26,
        ("svm_new",): 4,
        ("lightgbm_new",): 2,
        ("spectra0", "svm_new"): 0,
        ("spectra0", "lightgbm_new"): 0,
        ("svm_new", "lightgbm_new"): 3,
        ("lightgbm_new", "spectra0", "svm_new"): 0,
        (): 268,
    },
}
EXPECTED_CELL = {
    ("11–12", "spectra0"): (102, 30),
    ("11–12", "aasist3"): (75, 14),
    ("11–12", "svm_new"): (105, 15),
    ("11–12", "lightgbm_new"): (104, 13),
    ("11–12", "A"): (109, 8),
    ("16", "spectra0"): (43, 26),
    ("16", "aasist3"): (30, 14),
    ("16", "svm_new"): (38, 7),
    ("16", "lightgbm_new"): (40, 5),
    ("16", "A"): (43, 3),
}
EXPECTED_PACK = {
    ("11–12", "PACK1", "spectra0"): (84, 115, 13, 152),
    ("11–12", "PACK1", "svm_new"): (89, 115, 10, 152),
    ("11–12", "PACK1", "lightgbm_new"): (87, 115, 9, 152),
    ("11–12", "PACK1", "A"): (92, 115, 5, 152),
    ("11–12", "PACK2", "spectra0"): (18, 19, 17, 171),
    ("11–12", "PACK2", "svm_new"): (16, 19, 5, 171),
    ("11–12", "PACK2", "lightgbm_new"): (17, 19, 4, 171),
    ("11–12", "PACK2", "A"): (17, 19, 3, 171),
    ("16", "PACK1", "spectra0"): (36, 51, 8, 83),
    ("16", "PACK1", "svm_new"): (32, 51, 4, 83),
    ("16", "PACK1", "lightgbm_new"): (34, 51, 3, 83),
    ("16", "PACK1", "A"): (37, 51, 2, 83),
    ("16", "PACK2", "spectra0"): (7, 7, 18, 220),
    ("16", "PACK2", "svm_new"): (6, 7, 3, 220),
    ("16", "PACK2", "lightgbm_new"): (6, 7, 2, 220),
    ("16", "PACK2", "A"): (6, 7, 1, 220),
}
EXPECTED_SPEECH = {
    ("11–12", "короткая", "spectra0"): (22, 29, 21, 124),
    ("11–12", "короткая", "aasist3"): (16, 29, 9, 124),
    ("11–12", "короткая", "svm_new"): (20, 29, 3, 124),
    ("11–12", "короткая", "lightgbm_new"): (21, 29, 4, 124),
    ("11–12", "короткая", "A"): (24, 29, 3, 124),
    ("11–12", "средняя", "spectra0"): (38, 48, 5, 104),
    ("11–12", "средняя", "aasist3"): (22, 48, 4, 104),
    ("11–12", "средняя", "svm_new"): (33, 48, 6, 104),
    ("11–12", "средняя", "lightgbm_new"): (33, 48, 2, 104),
    ("11–12", "средняя", "A"): (35, 48, 1, 104),
    ("11–12", "длинная", "spectra0"): (42, 57, 4, 95),
    ("11–12", "длинная", "aasist3"): (37, 57, 1, 95),
    ("11–12", "длинная", "svm_new"): (52, 57, 6, 95),
    ("11–12", "длинная", "lightgbm_new"): (50, 57, 7, 95),
    ("11–12", "длинная", "A"): (50, 57, 4, 95),
    ("16", "короткая", "spectra0"): (5, 7, 18, 112),
    ("16", "короткая", "aasist3"): (4, 7, 11, 112),
    ("16", "короткая", "svm_new"): (3, 7, 4, 112),
    ("16", "короткая", "lightgbm_new"): (4, 7, 1, 112),
    ("16", "короткая", "A"): (4, 7, 1, 112),
    ("16", "средняя", "spectra0"): (27, 30, 5, 86),
    ("16", "средняя", "aasist3"): (15, 30, 2, 86),
    ("16", "средняя", "svm_new"): (20, 30, 2, 86),
    ("16", "средняя", "lightgbm_new"): (22, 30, 2, 86),
    ("16", "средняя", "A"): (23, 30, 1, 86),
    ("16", "длинная", "spectra0"): (11, 21, 3, 105),
    ("16", "длинная", "aasist3"): (11, 21, 1, 105),
    ("16", "длинная", "svm_new"): (15, 21, 1, 105),
    ("16", "длинная", "lightgbm_new"): (14, 21, 2, 105),
    ("16", "длинная", "A"): (16, 21, 1, 105),
}
PUBLISHED_DISPUTES = {
    "14708dd0-11e2-4e07-9bfc-fae7808a4249",
    "2d4ccfb6-4da6-4032-843a-0aeb45a52238",
    "ebabebec-7daf-41bc-afd3-1c09a08f63c6",
}


def vote_2of3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Synthetic, если synthetic сказали минимум две из трёх моделей."""

    parts = [np.asarray(item, dtype=bool) for item in (a, b, c)]
    if len({part.shape for part in parts}) != 1:
        raise AssertionError("голоса 2 из 3 разной длины")
    total = parts[0].astype(np.int8) + parts[1].astype(np.int8) + parts[2].astype(np.int8)
    if int(total.max()) > 3:
        raise AssertionError("в голосование попало не бинарное решение")
    return total >= 2


def vote_2of2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Synthetic, только если synthetic сказали обе модели. Это не majority 2 из 3."""

    left = np.asarray(a, dtype=bool)
    right = np.asarray(b, dtype=bool)
    if left.shape != right.shape:
        raise AssertionError("голоса 2 из 2 разной длины")
    return left & right


def speech_bin(values: pd.Series) -> pd.Series:
    """Те же границы, что в срезах 11–12: <=9.68, (9.68, 17.04], >17.04."""

    number = pd.to_numeric(values, errors="coerce")
    out = pd.Series("средняя", index=values.index, dtype="object")
    out[number <= SPEECH_LOW] = "короткая"
    out[number > SPEECH_HIGH] = "длинная"
    out[number.isna()] = pd.NA
    return out


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def pack_of(campaign: object) -> str:
    text = str(campaign or "")
    if text.endswith("PACK1"):
        return "PACK1"
    if text.endswith("PACK2"):
        return "PACK2"
    return ""


def binary_label(label: object) -> bool | None:
    if label in SYNTHETIC:
        return True
    if label in HUMAN:
        return False
    return None


def confusion(y: np.ndarray, pred: np.ndarray) -> dict:
    truth = np.asarray(y, dtype=bool)
    vote = np.asarray(pred, dtype=bool)
    if truth.shape != vote.shape:
        raise AssertionError("метка и голос разной длины")
    if set(np.unique(vote)).difference({False, True}):
        raise AssertionError("голос не бинарный")
    tp = int((truth & vote).sum())
    fp = int((~truth & vote).sum())
    tn = int((~truth & ~vote).sum())
    fn = int((truth & ~vote).sum())
    n_syn = int(truth.sum())
    n_hum = int((~truth).sum())
    n = int(truth.size)
    if tp + fn != n_syn or tn + fp != n_hum or tp + fp + tn + fn != n:
        raise AssertionError("матрица ошибок не сходится с размером выборки")
    recall = tp / n_syn if n_syn else float("nan")
    specificity = tn / n_hum if n_hum else float("nan")
    fpr = fp / n_hum if n_hum else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    balanced = (recall + specificity) / 2 if n_syn and n_hum else float("nan")
    if n_hum and abs((specificity + fpr) - 1.0) > 1e-12:
        raise AssertionError("specificity и FPR не дополняют друг друга")
    return {
        "n": n,
        "n_synthetic": n_syn,
        "n_human": n_hum,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "recall": recall,
        "specificity": specificity,
        "fpr": fpr,
        "precision": precision,
        "balanced_accuracy": balanced,
        "errors": fp + fn,
    }


def apply_stored_rule(model: str, score: pd.Series, status: pd.Series) -> pd.Series:
    return pd.Series(
        [prediction(model, value, state) for value, state in zip(score, status)],
        index=score.index,
        dtype="object",
    )


def load_early(excluded: list[dict]) -> tuple[pd.DataFrame, dict]:
    pred_path = ROOT / "reports" / "model_assisted_annotation_20261111_20261112" / "per_call_predictions.csv"
    label_path = ROOT / "data" / "annotations" / "priority_20261111_20261112_calls.csv"
    pred = read_table(pred_path)
    labels = read_table(label_path)
    if labels["call_id"].duplicated().any():
        raise AssertionError("у 11–12 появились повторы call_id, прежняя логика их не ждала")
    for row in labels.itertuples(index=False):
        if row.primary_label == "unknown":
            excluded.append({"dataset": "11–12", "call_id": row.call_id, "reason": "unknown"})
        elif row.primary_label == "mixed":
            excluded.append({"dataset": "11–12", "call_id": row.call_id, "reason": "mixed"})
    binary = labels[labels["primary_label"].isin(HUMAN | SYNTHETIC)].copy()
    missing = sorted(set(binary["call_id"]) - set(pred["call_id"]))
    for call_id in missing:
        excluded.append({"dataset": "11–12", "call_id": call_id, "reason": "missing_prediction"})
    pred_columns = ["call_id", "campaign", "duration_seconds", "spectra_speech_duration_s"]
    for model in MODELS:
        pred_columns.extend([f"{model}_score", f"{model}_status", f"{model}_prediction"])
    frame = binary.merge(pred[pred_columns], on="call_id", how="inner", suffixes=("", "_scorefile"))
    frame["dataset"] = "11–12"
    frame["y"] = frame["primary_label"].isin(SYNTHETIC)
    frame["pack"] = frame["campaign"].map(pack_of)
    frame["speech_s"] = pd.to_numeric(frame["spectra_speech_duration_s"], errors="coerce")
    duration_column = "duration_seconds_scorefile" if "duration_seconds_scorefile" in frame.columns else "duration_seconds"
    frame["duration_s"] = pd.to_numeric(frame[duration_column], errors="coerce")
    for model in MODELS:
        recomputed = apply_stored_rule(model, frame[f"{model}_score"], frame[f"{model}_status"])
        stored = frame[f"{model}_prediction"].fillna("").astype(str)
        if not recomputed.eq(stored).all():
            mismatch = frame.loc[recomputed.ne(stored), "call_id"].head(5).tolist()
            raise AssertionError(f"11–12, {model}: сохранённое решение не совпало с порогом, примеры {mismatch}")
        frame[f"{model}_recomputed"] = recomputed
        frame[model] = recomputed.map({"synthetic": True, "human": False})
    audit = {
        "predictions": pred_path,
        "labels": label_path,
        "prediction_rows": int(len(pred)),
        "label_rows": int(len(labels)),
        "unknown": int((labels["primary_label"] == "unknown").sum()),
        "mixed": int((labels["primary_label"] == "mixed").sum()),
        "disputes": 0,
        "same_label_duplicates": 0,
    }
    return frame, audit


def load_late(excluded: list[dict]) -> tuple[pd.DataFrame, dict]:
    pred_path = ROOT / "reports" / "day_20261116_model_error_overlap" / "per_call_predictions.csv"
    label_path = ROOT / "data" / "annotations" / "day_20261116_calls.csv"
    manifest_path = ROOT / "data" / "manifests" / "day_20261116.csv"
    speech_path = (
        ROOT / "evaluation" / "day_20261116" / "20261116T152222" / "call_results" / "spectra0.csv"
    )
    long = read_table(pred_path)
    annotations = read_table(label_path)
    manifest = read_table(manifest_path)
    speech = read_table(speech_path)[["call_id", "speech_duration_s", "duration_s", "status"]]
    disputes = []
    kept = []
    same_label_duplicates = 0
    for call_id, group in annotations.groupby("call_id", sort=False):
        names = sorted(set(group["primary_label"]))
        if len(group) > 1 and len(names) == 1:
            same_label_duplicates += 1
        if len(names) > 1:
            disputes.append({"call_id": call_id, "labels": "|".join(names)})
            excluded.append(
                {
                    "dataset": "16",
                    "call_id": call_id,
                    "reason": "label_disagreement:" + "|".join(names),
                }
            )
            continue
        kept.append(group.iloc[0])
    resolved = pd.DataFrame(kept)
    for row in resolved.itertuples(index=False):
        if row.primary_label == "unknown":
            excluded.append({"dataset": "16", "call_id": row.call_id, "reason": "unknown"})
        elif row.primary_label == "mixed":
            excluded.append({"dataset": "16", "call_id": row.call_id, "reason": "mixed"})
    binary = resolved[resolved["primary_label"].isin(HUMAN | SYNTHETIC)].copy()
    binary = binary.merge(
        manifest[["call_id", "campaign_name", "duration_seconds"]].rename(
            columns={"duration_seconds": "manifest_duration_s", "campaign_name": "manifest_campaign"}
        ),
        on="call_id",
        how="inner",
    )
    wide_label = long.pivot(index="call_id", columns="model", values="predicted_label")
    wide_score = long.pivot(index="call_id", columns="model", values="score")
    wide_threshold = long.pivot(index="call_id", columns="model", values="threshold")
    wide_rule = long.pivot(index="call_id", columns="model", values="score_rule")
    truth = long.drop_duplicates("call_id").set_index("call_id")[["ground_truth", "primary_label"]]
    frame = binary.merge(truth, left_on="call_id", right_index=True, how="inner", suffixes=("", "_saved"))
    frame = frame.merge(wide_label[list(MODELS)].add_suffix("_saved_label"), left_on="call_id", right_index=True)
    frame = frame.merge(wide_score[list(MODELS)].add_suffix("_score"), left_on="call_id", right_index=True)
    frame = frame.merge(wide_threshold[list(MODELS)].add_suffix("_saved_threshold"), left_on="call_id", right_index=True)
    frame = frame.merge(wide_rule[list(MODELS)].add_suffix("_saved_rule"), left_on="call_id", right_index=True)
    frame = frame.merge(speech, on="call_id", how="left")
    frame["dataset"] = "16"
    frame["y"] = frame["primary_label"].isin(SYNTHETIC)
    frame["pack"] = frame["manifest_campaign"].map(pack_of)
    frame["speech_s"] = pd.to_numeric(frame["speech_duration_s"], errors="coerce")
    frame["duration_s"] = pd.to_numeric(frame["manifest_duration_s"], errors="coerce")
    saved_truth = frame["ground_truth"].eq("synthetic")
    if not frame["y"].eq(saved_truth).all():
        raise AssertionError("16 ноября: ground_truth не совпал с primary_label")
    for model in MODELS:
        status = pd.Series("ok", index=frame.index)
        recomputed = apply_stored_rule(model, frame[f"{model}_score"], status)
        frame[f"{model}_recomputed"] = recomputed
        frame[model] = recomputed.map({"synthetic": True, "human": False})
        saved = frame[f"{model}_saved_label"]
        if not recomputed.eq(saved).all():
            mismatch = frame.loc[recomputed.ne(saved), "call_id"].head(5).tolist()
            raise AssertionError(f"16 ноября, {model}: сохранённое решение не совпало с порогом, примеры {mismatch}")
    audit = {
        "predictions": pred_path,
        "labels": label_path,
        "manifest": manifest_path,
        "speech": speech_path,
        "prediction_rows": int(len(long)),
        "label_rows": int(len(annotations)),
        "manifest_rows": int(len(manifest)),
        "unknown": int((resolved["primary_label"] == "unknown").sum()),
        "mixed": int((resolved["primary_label"] == "mixed").sum()),
        "disputes": disputes,
        "same_label_duplicates": same_label_duplicates,
    }
    return frame, audit


def attach_votes(frame: pd.DataFrame) -> pd.DataFrame:
    ready = frame.copy()
    missing = [model for model in MODELS if ready[model].isna().any()]
    if missing:
        ids = ready.loc[ready[missing].isna().any(axis=1), "call_id"].tolist()
        raise AssertionError(f"нет бинарного решения {missing}: {ids[:20]}")
    for model in MODELS:
        if set(ready[f"{model}_recomputed"]).difference({"human", "synthetic"}):
            raise AssertionError(f"{model}: решение не только human/synthetic")
    votes = {model: ready[model].to_numpy(dtype=bool) for model in MODELS}
    ready["A"] = vote_2of3(votes["spectra0"], votes["svm_new"], votes["lightgbm_new"])
    ready["B"] = vote_2of3(votes["spectra0"], votes["aasist3"], votes["lightgbm_new"])
    ready["C"] = vote_2of3(votes["spectra0"], votes["aasist3"], votes["svm_new"])
    ready["D"] = vote_2of3(votes["aasist3"], votes["svm_new"], votes["lightgbm_new"])
    ready["pair_2of2"] = vote_2of2(votes["spectra0"], votes["aasist3"])
    ready["speech_bin"] = speech_bin(ready["speech_s"])
    return ready


def iter_slices(frame: pd.DataFrame):
    yield "все", "все", frame
    for pack in PACK_ORDER:
        yield "pack", pack, frame[frame["pack"] == pack]
    for name in SPEECH_ORDER:
        yield "speech", name, frame[frame["speech_bin"] == name]


def collect_metrics(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    rules = {name: "single" for name in MODELS}
    rules.update({name: "majority_2of3" for name in COMBOS})
    rules["pair_2of2"] = "both_2of2"
    for dataset, frame in frames.items():
        for slice_type, slice_value, part in iter_slices(frame):
            y = part["y"].to_numpy(dtype=bool)
            for name in list(MODELS) + list(COMBOS) + ["pair_2of2"]:
                metrics = confusion(y, part[name].to_numpy(dtype=bool))
                metrics.update(
                    {
                        "dataset": dataset,
                        "slice_type": slice_type,
                        "slice_value": slice_value,
                        "rule_id": name,
                        "rule": rules[name],
                        "rule_name": COMBO_NAME[name] if name in COMBO_NAME else MODEL_NAME[name],
                    }
                )
                rows.append(metrics)
    return pd.DataFrame(rows)


def metric_row(metrics: pd.DataFrame, dataset: str, slice_type: str, slice_value: str, rule_id: str) -> pd.Series:
    part = metrics[
        (metrics["dataset"] == dataset)
        & (metrics["slice_type"] == slice_type)
        & (metrics["slice_value"] == slice_value)
        & (metrics["rule_id"] == rule_id)
    ]
    if len(part) != 1:
        raise AssertionError(f"ожидалась одна строка метрик {dataset} {slice_value} {rule_id}, найдено {len(part)}")
    return part.iloc[0]


def region_key(row: pd.Series, models: tuple[str, ...], kind: str) -> tuple[str, ...]:
    if kind == "fn":
        erred = [model for model in models if not bool(row[model])]
    else:
        erred = [model for model in models if bool(row[model])]
    return tuple(sorted(erred))


def region_counts(frame: pd.DataFrame, models: tuple[str, ...], kind: str) -> dict[tuple[str, ...], int]:
    part = frame[frame["y"]] if kind == "fn" else frame[~frame["y"]]
    keys = part.apply(lambda row: region_key(row, models, kind), axis=1)
    counts = {key: int((keys == key).sum()) for key in set(keys)}
    return counts


def pairwise_rows(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    pairs = (
        ("spectra0", "aasist3"),
        ("spectra0", "svm_new"),
        ("spectra0", "lightgbm_new"),
        ("aasist3", "svm_new"),
        ("aasist3", "lightgbm_new"),
        ("svm_new", "lightgbm_new"),
    )
    for dataset, frame in frames.items():
        for kind, mask in (("FN", frame["y"]), ("FP", ~frame["y"])):
            part = frame[mask]
            for left, right in pairs:
                if kind == "FN":
                    left_err = ~part[left]
                    right_err = ~part[right]
                else:
                    left_err = part[left]
                    right_err = part[right]
                both = int((left_err & right_err).sum())
                only_left = int((left_err & ~right_err).sum())
                only_right = int((~left_err & right_err).sum())
                neither = int((~left_err & ~right_err).sum())
                union = both + only_left + only_right
                rows.append(
                    {
                        "dataset": dataset,
                        "error_type": kind,
                        "model_a": MODEL_NAME[left],
                        "model_b": MODEL_NAME[right],
                        "model_a_id": left,
                        "model_b_id": right,
                        "both": both,
                        "only_a": only_left,
                        "only_b": only_right,
                        "neither": neither,
                        "union": union,
                        "jaccard": both / union if union else float("nan"),
                        "class_n": int(len(part)),
                    }
                )
    return pd.DataFrame(rows)


def overlap_matrix(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    if kind == "FN":
        flags = pd.DataFrame({MODEL_NAME[model]: ~frame.loc[frame["y"], model] for model in MODELS})
    else:
        flags = pd.DataFrame({MODEL_NAME[model]: frame.loc[~frame["y"], model] for model in MODELS})
    names = [MODEL_NAME[model] for model in MODELS]
    matrix = pd.DataFrame(index=names, columns=names, dtype=int)
    for left in names:
        for right in names:
            matrix.loc[left, right] = int((flags[left] & flags[right]).sum())
    return matrix


def flip_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset, frame in frames.items():
        y = frame["y"].to_numpy(dtype=bool)
        base = frame["A"].to_numpy(dtype=bool)
        for rule_id in ("B", "C", "D", "pair_2of2"):
            alt = frame[rule_id].to_numpy(dtype=bool)
            changed = base != alt
            fixed_fn = y & changed & alt & ~base
            new_fn = y & changed & ~alt & base
            fixed_fp = ~y & changed & ~alt & base
            new_fp = ~y & changed & alt & ~base
            if int(changed.sum()) != int(fixed_fn.sum() + new_fn.sum() + fixed_fp.sum() + new_fp.sum()):
                raise AssertionError("разложение смены голоса не сошлось")
            base_m = confusion(y, base)
            alt_m = confusion(y, alt)
            if (alt_m["fn"] - base_m["fn"]) != int(new_fn.sum() - fixed_fn.sum()):
                raise AssertionError("ΔFN не совпала со списком смен")
            if (alt_m["fp"] - base_m["fp"]) != int(new_fp.sum() - fixed_fp.sum()):
                raise AssertionError("ΔFP не совпала со списком смен")
            for kind, mask in (
                ("исправленный FN", fixed_fn),
                ("новый FN", new_fn),
                ("исправленный FP", fixed_fp),
                ("новый FP", new_fp),
            ):
                for call_id in frame.loc[mask, "call_id"]:
                    row = frame.loc[frame["call_id"] == call_id].iloc[0]
                    rows.append(
                        {
                            "dataset": dataset,
                            "rule_id": rule_id,
                            "change": kind,
                            "call_id": call_id,
                            "pack": row["pack"],
                            "speech_s": row["speech_s"],
                            "speech_bin": row["speech_bin"],
                            "primary_label": row["primary_label"],
                            "spectra0": "synthetic" if row["spectra0"] else "human",
                            "aasist3": "synthetic" if row["aasist3"] else "human",
                            "svm_new": "synthetic" if row["svm_new"] else "human",
                            "lightgbm_new": "synthetic" if row["lightgbm_new"] else "human",
                        }
                    )
    return pd.DataFrame(rows)


def delta_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATE_ORDER:
        base = metric_row(metrics, dataset, "все", "все", "A")
        for rule_id in ("B", "C", "D", "pair_2of2"):
            alt = metric_row(metrics, dataset, "все", "все", rule_id)
            rows.append(
                {
                    "dataset": dataset,
                    "rule_id": rule_id,
                    "rule": "both_2of2" if rule_id == "pair_2of2" else "majority_2of3",
                    "recall_pp": (alt["recall"] - base["recall"]) * 100,
                    "precision_pp": (alt["precision"] - base["precision"]) * 100,
                    "fpr_pp": (alt["fpr"] - base["fpr"]) * 100,
                    "delta_fp": int(alt["fp"] - base["fp"]),
                    "delta_fn": int(alt["fn"] - base["fn"]),
                    "delta_errors": int(alt["errors"] - base["errors"]),
                    "delta_tp": int(alt["tp"] - base["tp"]),
                }
            )
        for slice_type, values in (("pack", PACK_ORDER), ("speech", SPEECH_ORDER)):
            for slice_value in values:
                base_s = metric_row(metrics, dataset, slice_type, slice_value, "A")
                for rule_id in ("B", "C", "D", "pair_2of2"):
                    alt = metric_row(metrics, dataset, slice_type, slice_value, rule_id)
                    rows.append(
                        {
                            "dataset": dataset,
                            "rule_id": rule_id,
                            "rule": "both_2of2" if rule_id == "pair_2of2" else "majority_2of3",
                            "slice_type": slice_type,
                            "slice_value": slice_value,
                            "recall_pp": (alt["recall"] - base_s["recall"]) * 100,
                            "precision_pp": (alt["precision"] - base_s["precision"]) * 100
                            if np.isfinite(alt["precision"]) and np.isfinite(base_s["precision"])
                            else float("nan"),
                            "fpr_pp": (alt["fpr"] - base_s["fpr"]) * 100,
                            "delta_fp": int(alt["fp"] - base_s["fp"]),
                            "delta_fn": int(alt["fn"] - base_s["fn"]),
                            "delta_errors": int(alt["errors"] - base_s["errors"]),
                            "delta_tp": int(alt["tp"] - base_s["tp"]),
                        }
                    )
    # Полный день тоже получает slice, чтобы таблица была однородной.
    for row in rows:
        row.setdefault("slice_type", "все")
        row.setdefault("slice_value", "все")
    return pd.DataFrame(rows)


def check_rules() -> list[str]:
    passed = []

    def ok(name: str) -> None:
        passed.append(name)

    for model, text in THRESHOLD_TEXT.items():
        if float(text) != THRESHOLDS[model]:
            raise AssertionError(f"текст порога {model} не совпал с rules.THRESHOLDS")
    if THRESHOLDS["spectra0"] != 0.22710928916931153:
        raise AssertionError("порог Spectra-0 изменён")
    if THRESHOLDS["aasist3"] != 0.11397590637207031:
        raise AssertionError("порог AASIST3 изменён")
    if THRESHOLDS["lightgbm_new"] != 0.3460832258410117:
        raise AssertionError("порог LightGBM изменён")
    if THRESHOLDS["logistic_new"] != 0.7354826103642409:
        raise AssertionError("порог logistic regression изменён")
    if THRESHOLDS["svm_new"] != 0.0:
        raise AssertionError("порог SVM изменён")
    if "spectra0" not in BELOW or "aasist3" not in BELOW:
        raise AssertionError("Spectra-0 или AASIST3 больше не сравниваются снизу порога")
    if "svm_new" in BELOW or "lightgbm_new" in BELOW or "logistic_new" in BELOW:
        raise AssertionError("классическая модель сравнивается снизу порога")
    ok("пороги и направление совпадают с rules.THRESHOLDS")

    if SPEECH_LOW != 9.68 or SPEECH_HIGH != 17.04:
        raise AssertionError("границы длительности речи не 9.68 и 17.04")
    probe = pd.Series([9.68, 9.6800001, 17.04, 17.0400001, 0.0, 100.0])
    bins = speech_bin(probe).tolist()
    if bins != ["короткая", "средняя", "средняя", "длинная", "короткая", "длинная"]:
        raise AssertionError(f"границы длительности пересекаются или сдвинуты: {bins}")
    ok("границы речи непересекающиеся: <=9.68, (9.68, 17.04], >17.04")

    demo_a = np.array([True, True, False, False])
    demo_b = np.array([True, False, True, False])
    demo_c = np.array([False, False, False, True])
    voted = vote_2of3(demo_a, demo_b, demo_c)
    if not np.array_equal(voted, np.array([True, False, False, False])):
        raise AssertionError("majority принял меньше двух голосов или отбросил два")
    if not bool(vote_2of3(np.array([True]), np.array([True]), np.array([True]))[0]):
        raise AssertionError("три голоса не дают synthetic")
    pair = vote_2of2(np.array([True, True, False]), np.array([True, False, False]))
    if not np.array_equal(pair, np.array([True, False, False])):
        raise AssertionError("2 из 2 не требует обоих голосов")
    either = np.array([True, True, False]) | np.array([True, False, False])
    if np.array_equal(pair, either):
        raise AssertionError("2 из 2 совпало с правилом хотя бы одного голоса")
    ok("2 из 3 — это сумма бинарных голосов >= 2; 2 из 2 — это пересечение и не смешано с 2 из 3")

    synthetic_below = prediction("spectra0", 0.2, "ok")
    human_above = prediction("spectra0", 0.3, "ok")
    aasist_hit = prediction("aasist3", 0.0, "ok")
    aasist_miss = prediction("aasist3", 0.2, "ok")
    svm_hit = prediction("svm_new", 0.0, "ok")
    svm_miss = prediction("svm_new", -0.01, "ok")
    lgbm_hit = prediction("lightgbm_new", 0.3460832258410117, "ok")
    lgbm_miss = prediction("lightgbm_new", 0.3460832258410116, "ok")
    empty = prediction("spectra0", 0.0, "no_speech")
    if (synthetic_below, human_above, aasist_hit, aasist_miss) != ("synthetic", "human", "synthetic", "human"):
        raise AssertionError("направление Spectra-0 или AASIST3 перевёрнуто")
    if (svm_hit, svm_miss, lgbm_hit, lgbm_miss) != ("synthetic", "human", "synthetic", "human"):
        raise AssertionError("направление SVM или LightGBM перевёрнуто")
    if empty != "":
        raise AssertionError("отсутствие score заменено на human")
    ok("на равном пороге classical — synthetic, Spectra-0 и AASIST3 — human; пустой status не становится human")
    return passed


def check_frames(frames: dict[str, pd.DataFrame], audits: dict[str, dict], metrics: pd.DataFrame) -> list[str]:
    passed = []

    def ok(name: str) -> None:
        passed.append(name)

    early, late = frames["11–12"], frames["16"]
    if set(early["call_id"]).intersection(late["call_id"]):
        raise AssertionError("call_id пересекаются между 11–12 и 16 ноября")
    ok("выборки 11–12 и 16 ноября не пересекаются")
    if len(early) != 457 or int(early["y"].sum()) != 134 or int((~early["y"]).sum()) != 323:
        raise AssertionError("когорта 11–12 не совпала с 457 = 134 synthetic + 323 human")
    if len(late) != 361 or int(late["y"].sum()) != 58 or int((~late["y"]).sum()) != 303:
        raise AssertionError("когорта 16 ноября не совпала с 361 = 58 synthetic + 303 human")
    ok("размеры бинарных когорт совпали с предыдущими отчётами")
    dispute_ids = {item["call_id"] for item in audits["16"]["disputes"]}
    if dispute_ids != PUBLISHED_DISPUTES or audits["16"]["same_label_duplicates"] != 5:
        raise AssertionError("исключения 16 ноября не совпали с тремя спорами и пятью одинаковыми дублями")
    if audits["11–12"]["unknown"] != 23 or audits["11–12"]["mixed"] != 12:
        raise AssertionError("исключения unknown/mixed 11–12 не совпали с 23 и 12")
    if audits["16"]["unknown"] != 15 or audits["16"]["mixed"] != 5:
        raise AssertionError("исключения unknown/mixed 16 ноября не совпали с 15 и 5")
    ok("unknown, mixed и три расхождения меток исключены по прежней логике и не входят в знаменатель")

    for dataset, frame in frames.items():
        if frame["call_id"].duplicated().any():
            raise AssertionError(f"{dataset}: повторяющийся call_id в общей выборке")
        if frame[list(MODELS)].isna().any().any():
            raise AssertionError(f"{dataset}: модели сравниваются не на одном наборе")
        if set(frame["pack"]) != {"PACK1", "PACK2"}:
            raise AssertionError(f"{dataset}: есть звонок без PACK1/PACK2")
        if frame["speech_s"].isna().any() or frame["speech_bin"].isna().any():
            raise AssertionError(f"{dataset}: нет длительности речи")
        groups = [set(frame.loc[frame["speech_bin"] == name, "call_id"]) for name in SPEECH_ORDER]
        if sum(map(len, groups)) != len(set.union(*groups)) or len(set.union(*groups)) != len(frame):
            raise AssertionError(f"{dataset}: звонок попал в несколько интервалов речи или ни в один")
        if int((frame["speech_s"] - frame["duration_s"]).abs().gt(0.05).sum()) == 0:
            raise AssertionError(f"{dataset}: длительность речи совпала с длительностью звонка")
        for model in MODELS:
            score = pd.to_numeric(frame[f"{model}_score"], errors="coerce")
            saved = frame[f"{model}_recomputed"]
            below = score < THRESHOLDS[model]
            above = score > THRESHOLDS[model]
            if not below.any() or not above.any():
                raise AssertionError(f"{dataset} {model}: все score по одну сторону порога, направление не проверить")
            if model in BELOW:
                if not saved[below].eq("synthetic").all() or not saved[above].eq("human").all():
                    raise AssertionError(f"{dataset} {model}: направление score перевёрнуто")
            else:
                if not saved[score >= THRESHOLDS[model]].eq("synthetic").all() or not saved[score < THRESHOLDS[model]].eq("human").all():
                    raise AssertionError(f"{dataset} {model}: направление score перевёрнуто")
        trio = ("spectra0", "svm_new", "lightgbm_new")
        for kind, expected in (("fn", EXPECTED_TRIO_FN), ("fp", EXPECTED_TRIO_FP)):
            counts = region_counts(frame, trio, kind)
            for key, value in expected[dataset].items():
                normalized = tuple(sorted(key))
                if counts.get(normalized, 0) != value:
                    raise AssertionError(f"{dataset} {kind} {normalized}: {counts.get(normalized, 0)} != {value}")
    ok("четыре модели стоят на одном наборе звонков, порог применён к бинарному решению, тройка совпала с прежним разбором")

    for dataset in DATE_ORDER:
        for rule_id, (tp, fp) in ((key[1], value) for key, value in EXPECTED_CELL.items() if key[0] == dataset):
            row = metric_row(metrics, dataset, "все", "все", rule_id)
            if int(row["tp"]) != tp or int(row["fp"]) != fp:
                raise AssertionError(f"{dataset} {rule_id}: TP/FP {int(row['tp'])}/{int(row['fp'])} != {tp}/{fp}")
        for key, (tp, n_syn, fp, n_hum) in EXPECTED_PACK.items():
            if key[0] != dataset:
                continue
            row = metric_row(metrics, dataset, "pack", key[1], key[2])
            if (int(row["tp"]), int(row["n_synthetic"]), int(row["fp"]), int(row["n_human"])) != (tp, n_syn, fp, n_hum):
                raise AssertionError(f"пакет {key} не совпал с опубликованным")
        for key, (tp, n_syn, fp, n_hum) in EXPECTED_SPEECH.items():
            if key[0] != dataset:
                continue
            row = metric_row(metrics, dataset, "speech", key[1], key[2])
            if (int(row["tp"]), int(row["n_synthetic"]), int(row["fp"]), int(row["n_human"])) != (tp, n_syn, fp, n_hum):
                raise AssertionError(f"речь {key} не совпала с опубликованным")
    ok("базовая комбинация A и отдельные модели совпали с опубликованными TP/FP по дате, пакету и длительности")

    majority_ids = set(COMBOS)
    if "pair_2of2" in majority_ids:
        raise AssertionError("2 из 2 попало в список majority")
    for dataset, frame in frames.items():
        y = frame["y"].to_numpy(dtype=bool)
        pair = frame["pair_2of2"].to_numpy(dtype=bool)
        if not np.array_equal(pair, frame["spectra0"].to_numpy(dtype=bool) & frame["aasist3"].to_numpy(dtype=bool)):
            raise AssertionError("колонка 2 из 2 не равна пересечению Spectra-0 и AASIST3")
        summed = (
            frame["spectra0"].to_numpy(dtype=np.int8)
            + frame["svm_new"].to_numpy(dtype=np.int8)
            + frame["lightgbm_new"].to_numpy(dtype=np.int8)
        )
        if not np.array_equal(frame["A"].to_numpy(dtype=bool), summed >= 2):
            raise AssertionError("колонка A считает не бинарные голоса")
        # Сырой score не может быть входом: проверка на подмене столбца.
        if np.issubdtype(frame["A"].to_numpy().dtype, np.floating):
            raise AssertionError("голос A сохранён как score")
        confusion(y, pair)
    ok("в таблицах majority и 2 из 2 разведены, вход голоса бинарный")
    return passed


def ru_calls(count: int) -> str:
    number = abs(int(count))
    if number % 10 == 1 and number % 100 != 11:
        word = "звонок"
    elif number % 10 in {2, 3, 4} and number % 100 not in {12, 13, 14}:
        word = "звонка"
    else:
        word = "звонков"
    return f"{number} {word}"


def fmt_rate(num: int, den: int) -> str:
    if den == 0:
        return "—"
    return f"{num / den:.3f} ({num}/{den})"


def fmt_pp(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{value:+.2f}"


def fmt_signed_int(value: int) -> str:
    return f"{value:+d}"


def md_table(headers: list[str], rows: list[list[object]], aligns: list[str] | None = None) -> str:
    if aligns is None:
        aligns = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def metric_cells(row: pd.Series) -> list[object]:
    return [
        int(row["tp"]),
        int(row["fp"]),
        int(row["tn"]),
        int(row["fn"]),
        fmt_rate(int(row["tp"]), int(row["n_synthetic"])),
        fmt_rate(int(row["tn"]), int(row["n_human"])),
        fmt_rate(int(row["fp"]), int(row["n_human"])),
        fmt_rate(int(row["tp"]), int(row["tp"]) + int(row["fp"])),
        f"{row['balanced_accuracy']:.3f}",
        int(row["errors"]),
    ]


METRIC_HEADERS = [
    "Правило",
    "TP",
    "FP",
    "TN",
    "FN",
    "Recall",
    "Specificity",
    "FPR",
    "Precision",
    "Balanced accuracy",
    "Ошибки",
]
METRIC_ALIGN = ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]


def style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.6)
    ax.set_axisbelow(True)


def annotate_bars(ax: plt.Axes, bars, labels: list[str], rotation: int = 0) -> None:
    for bar, label in zip(bars, labels):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=rotation,
            clip_on=False,
        )


def finish(ax: plt.Axes, ymin: float, ymax: float) -> None:
    ax.set_ylim(ymin, ymax)
    style_ax(ax)


def plot_rate(
    path: Path,
    metrics: pd.DataFrame,
    field: str,
    numerator: str,
    denominator: str,
    title: str,
    ylabel: str,
    *,
    label_rotation: int = 0,
    headroom: float = 1.28,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), gridspec_kw={"width_ratios": [4.2, 1.6]})
    panel = (
        (axes[0], list(COMBOS), "Majority: минимум 2 из 3"),
        (axes[1], ["pair_2of2"], "Отдельно: оба голоса, 2 из 2"),
    )
    for ax, rules, panel_title in panel:
        x = np.arange(len(rules))
        width = 0.36
        tops = []
        for index, dataset in enumerate(DATE_ORDER):
            values = []
            labels = []
            for rule_id in rules:
                row = metric_row(metrics, dataset, "все", "все", rule_id)
                values.append(float(row[field]))
                labels.append(f"{int(row[numerator])}/{int(row[denominator])}")
            offset = (index - 0.5) * width
            bars = ax.bar(x + offset, values, width, label=dataset + " ноября", color=DATE_COLOR[dataset])
            annotate_bars(ax, bars, labels, rotation=label_rotation)
            tops.extend(values)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[rule_id] + "\n" + COMBO_NAME[rule_id].replace(" + ", "\n") for rule_id in rules], fontsize=8)
        ax.set_title(panel_title)
        ax.set_ylabel(ylabel)
        finish(ax, 0, max(tops) * headroom if tops else 1)
        if label_rotation == 0:
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=12)
    if label_rotation:
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.tight_layout(rect=(0, 0.08, 1, 0.94))
        fig.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_counts(path: Path, metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6))
    for row_index, dataset in enumerate(DATE_ORDER):
        for col_index, (field, label) in enumerate((("fp", "FP"), ("fn", "FN"))):
            ax = axes[row_index, col_index]
            rules = list(COMBOS) + ["pair_2of2"]
            values = [int(metric_row(metrics, dataset, "все", "все", rule_id)[field]) for rule_id in rules]
            colors = [COMBO_COLOR[rule_id] for rule_id in rules]
            bars = ax.bar(np.arange(len(rules)), values, color=colors)
            annotate_bars(ax, bars, [str(value) for value in values])
            ax.set_xticks(np.arange(len(rules)))
            ax.set_xticklabels([SHORT[rule_id] for rule_id in rules])
            sample = metric_row(metrics, dataset, "все", "все", "A")
            ax.set_title(f"{dataset} ноября, {label}. {int(sample['n_synthetic'])} synthetic, {int(sample['n_human'])} human")
            ax.set_ylabel(label)
            finish(ax, 0, max(values) * 1.25 if values else 1)
            ax.axvline(3.5, color="#bbbbbb", linewidth=0.8)
    fig.suptitle("Число FP и FN. A–D — majority 2 из 3. Правый столбец — отдельное правило 2 из 2.", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_slices(path: Path, metrics: pd.DataFrame, slice_type: str, order: tuple[str, ...], title: str) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12.2, 11.2))
    blocks = (
        (0, list(COMBOS), "Majority 2 из 3"),
        (2, ["pair_2of2"], "Отдельно: 2 из 2"),
    )
    # Строки 0 и 1 — две даты для majority. Строка 2 — 2 из 2, обе даты на одних осях.
    for date_index, dataset in enumerate(DATE_ORDER):
        for col_index, (field, numerator, ylabel) in enumerate(
            (("recall", "tp", "Recall"), ("fpr", "fp", "FPR"))
        ):
            ax = axes[date_index, col_index]
            x = np.arange(len(order))
            width = 0.18
            tops = []
            tick_labels = []
            for position, slice_value in enumerate(order):
                sample = metric_row(metrics, dataset, slice_type, slice_value, "A")
                tick_labels.append(f"{slice_value}\n{int(sample['n_synthetic'])} syn / {int(sample['n_human'])} hum")
            for index, rule_id in enumerate(COMBOS):
                values = []
                labels = []
                for slice_value in order:
                    row = metric_row(metrics, dataset, slice_type, slice_value, rule_id)
                    values.append(float(row[field]))
                    labels.append(str(int(row[numerator])))
                offset = (index - 1.5) * width
                bars = ax.bar(x + offset, values, width, label=rule_id, color=COMBO_COLOR[rule_id])
                annotate_bars(ax, bars, labels, rotation=90)
                tops.extend(values)
            ax.set_xticks(x)
            ax.set_xticklabels(tick_labels, fontsize=8)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{dataset} ноября, {ylabel}. Число над столбцом — {numerator.upper()}")
            finish(ax, 0, (max(tops) if tops else 1) * 1.45)
            ax.legend(frameon=False, ncol=4, fontsize=8)
    for col_index, (field, numerator, ylabel) in enumerate((("recall", "tp", "Recall"), ("fpr", "fp", "FPR"))):
        ax = axes[2, col_index]
        x = np.arange(len(order))
        width = 0.36
        tops = []
        tick_labels = []
        for position, slice_value in enumerate(order):
            sample = metric_row(metrics, "11–12", slice_type, slice_value, "A")
            late = metric_row(metrics, "16", slice_type, slice_value, "A")
            tick_labels.append(
                f"{slice_value}\n11–12: {int(sample['n_synthetic'])}/{int(sample['n_human'])}\n16: {int(late['n_synthetic'])}/{int(late['n_human'])}"
            )
        for index, dataset in enumerate(DATE_ORDER):
            values = []
            labels = []
            for slice_value in order:
                row = metric_row(metrics, dataset, slice_type, slice_value, "pair_2of2")
                values.append(float(row[field]))
                labels.append(str(int(row[numerator])))
            bars = ax.bar(x + (index - 0.5) * width, values, width, label=dataset + " ноября", color=DATE_COLOR[dataset])
            annotate_bars(ax, bars, labels)
            tops.extend(values)
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, fontsize=7.5)
        ax.set_ylabel(ylabel)
        ax.set_title(f"2 из 2, {ylabel}. Число над столбцом — {numerator.upper()}")
        finish(ax, 0, (max(tops) if tops else 1) * 1.35)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    del blocks


def plot_overlap(path: Path, frames: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.8), constrained_layout=True)
    for row_index, dataset in enumerate(DATE_ORDER):
        for col_index, kind in enumerate(("FN", "FP")):
            ax = axes[row_index, col_index]
            matrix = overlap_matrix(frames[dataset], kind)
            values = matrix.to_numpy(dtype=float)
            ax.imshow(values, cmap="Blues", vmin=0)
            names = list(matrix.index)
            ax.set_xticks(range(len(names)))
            ax.set_yticks(range(len(names)))
            ax.set_xticklabels(names, rotation=25, ha="right")
            ax.set_yticklabels(names)
            vmax = float(values.max()) if values.size else 1
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    number = int(values[i, j])
                    ax.text(j, i, str(number), ha="center", va="center", color="white" if number > vmax * 0.62 else "black", fontsize=9)
            class_name = "synthetic" if kind == "FN" else "human"
            class_n = int(frames[dataset]["y"].sum() if kind == "FN" else (~frames[dataset]["y"]).sum())
            ax.set_title(f"{dataset} ноября, {kind}\nдиагональ: ошибки из {class_n} {class_name}", fontsize=10)
    fig.suptitle("Пересечение ошибок. Клетка — звонки, на которых ошиблись обе модели.", fontsize=11)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def score_summary(frame: pd.DataFrame) -> list[str]:
    lines = []
    for model in MODELS:
        score = pd.to_numeric(frame[f"{model}_score"], errors="coerce")
        exact = int(np.isclose(score.to_numpy(dtype=float), THRESHOLDS[model], rtol=0, atol=0).sum())
        direction = "synthetic, если score < порога" if model in BELOW else "synthetic, если score ≥ порога"
        lines.append(
            f"| {MODEL_NAME[model]} | {SCORE_NAME[model]} | {THRESHOLD_TEXT[model]} | {direction} | "
            f"{score.min():.3f} | {score.median():.3f} | {score.max():.3f} | {exact} |"
        )
    return lines


def call_rows(frame: pd.DataFrame, mask: pd.Series) -> list[list[object]]:
    part = frame.loc[mask].sort_values(["pack", "call_id"])
    rows = []
    for row in part.itertuples(index=False):
        rows.append(
            [
                f"`{row.call_id}`",
                row.pack,
                f"{float(row.speech_s):.2f}",
                row.speech_bin,
                row.primary_label,
                "synthetic" if row.spectra0 else "human",
                "synthetic" if row.aasist3 else "human",
                "synthetic" if row.svm_new else "human",
                "synthetic" if row.lightgbm_new else "human",
            ]
        )
    return rows


CALL_HEADERS = ["call_id", "Пакет", "Речь, с", "Длина", "Метка", "Spectra-0", "AASIST3", "SVM", "LightGBM"]
CALL_ALIGN = ["---", "---", "---:", "---", "---", "---", "---", "---", "---"]


def section_metrics(metrics: pd.DataFrame, dataset: str, rule_ids: list[str]) -> str:
    rows = []
    for rule_id in rule_ids:
        row = metric_row(metrics, dataset, "все", "все", rule_id)
        label = f"{rule_id}. {COMBO_NAME[rule_id]}" if rule_id in COMBO_NAME else MODEL_NAME[rule_id]
        rows.append([label, *metric_cells(row)])
    return md_table(METRIC_HEADERS, rows, METRIC_ALIGN)


def slice_table(metrics: pd.DataFrame, slice_type: str, order: tuple[str, ...], rule_ids: list[str]) -> str:
    headers = ["Дата", "Срез", "n syn", "n hum", "Правило", "TP", "FP", "TN", "FN", "Recall", "FPR", "Precision", "Ошибки"]
    aligns = ["---", "---", "---:", "---:", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]
    rows = []
    for dataset in DATE_ORDER:
        for slice_value in order:
            for rule_id in rule_ids:
                row = metric_row(metrics, dataset, slice_type, slice_value, rule_id)
                rows.append(
                    [
                        dataset,
                        slice_value,
                        int(row["n_synthetic"]),
                        int(row["n_human"]),
                        SHORT.get(rule_id, MODEL_NAME.get(rule_id, rule_id)),
                        int(row["tp"]),
                        int(row["fp"]),
                        int(row["tn"]),
                        int(row["fn"]),
                        fmt_rate(int(row["tp"]), int(row["n_synthetic"])),
                        fmt_rate(int(row["fp"]), int(row["n_human"])),
                        fmt_rate(int(row["tp"]), int(row["tp"]) + int(row["fp"])),
                        int(row["errors"]),
                    ]
                )
    return md_table(headers, rows, aligns)


def pareto_better(alt: pd.Series, base: pd.Series) -> bool:
    return int(alt["fn"]) <= int(base["fn"]) and int(alt["fp"]) <= int(base["fp"]) and int(alt["errors"]) < int(base["errors"])


def pareto_not_worse(alt: pd.Series, base: pd.Series) -> bool:
    return int(alt["fn"]) <= int(base["fn"]) and int(alt["fp"]) <= int(base["fp"])


def build_report(
    frames: dict[str, pd.DataFrame],
    audits: dict[str, dict],
    metrics: pd.DataFrame,
    pairs: pd.DataFrame,
    flips: pd.DataFrame,
    files: list[Path],
    checks: list[str],
    terciles: dict[str, float],
) -> str:
    early, late = frames["11–12"], frames["16"]
    file_rows = []
    for path in files:
        file_rows.append([f"`{path.relative_to(ROOT).as_posix()}`", sha256(path)[:16], int(path.stat().st_size)])
    dispute_rows = []
    for item in audits["16"]["disputes"]:
        dispute_rows.append([f"`{item['call_id']}`", item["labels"].replace("|", " и ")])

    def count_text(dataset: str, rule_id: str) -> str:
        row = metric_row(metrics, dataset, "все", "все", rule_id)
        return (
            f"TP {int(row['tp'])}, FP {int(row['fp'])}, TN {int(row['tn'])}, FN {int(row['fn'])}, "
            f"recall {fmt_rate(int(row['tp']), int(row['n_synthetic']))}, "
            f"FPR {fmt_rate(int(row['fp']), int(row['n_human']))}, "
            f"precision {fmt_rate(int(row['tp']), int(row['tp']) + int(row['fp']))}, "
            f"ошибок {int(row['errors'])}"
        )

    lines: list[str] = []
    lines.append("# Сравнение комбинаций голосования для детекции синтетической речи")
    lines.append("")
    lines.append("Офлайн-сравнение сохранённых решений. Нового inference, обучения и подбора порогов нет.")
    lines.append("")
    lines.append("## 1. Цель эксперимента")
    lines.append("")
    lines.append(
        "Нужно сравнить правила, которые ставят звонку synthetic, когда за synthetic проголосовали минимум две модели из трёх. "
        "Текущая комбинация A — Spectra-0, SVM new и LightGBM new. Рядом считаются три замены одной из моделей на AASIST3 "
        "и, отдельно, правило «обе модели» для пары Spectra-0 + AASIST3."
    )
    lines.append("")
    lines.append(md_table(
        ["ID", "Состав", "Правило"],
        [
            ["A", COMBO_NAME["A"], "synthetic, если голосов synthetic минимум 2 из 3"],
            ["B", COMBO_NAME["B"], "synthetic, если голосов synthetic минимум 2 из 3"],
            ["C", COMBO_NAME["C"], "synthetic, если голосов synthetic минимум 2 из 3"],
            ["D", COMBO_NAME["D"], "synthetic, если голосов synthetic минимум 2 из 3"],
            ["2 из 2", COMBO_NAME["pair_2of2"], "synthetic, только если synthetic сказали обе модели"],
        ],
        ["---", "---", "---"],
    ))
    lines.append("")
    lines.append(
        "Logistic regression в эти правила не входит. Её порог оставлен прежним и в расчёте голоса не используется. "
        "Подтипы synthetic не склеиваются в новые классы: для бинарного счёта все прежние synthetic-метки остаются положительным классом, "
        "все прежние human-метки — отрицательным."
    )
    lines.append("")
    lines.append(
        "Замена A считается подтверждённой только если альтернатива 2 из 3 на обоих полных днях и на PACK1 и PACK2 каждого дня "
        "даёт не больше FN и не больше FP, чем A, и хотя бы на одном полном дне строго меньше ошибок FP+FN. "
        "Рост precision сам по себе замену не подтверждает. Правило 2 из 2 в этот отбор не входит."
    )
    lines.append("")
    lines.append("## 2. Источники данных и состав выборок")
    lines.append("")
    lines.append(
        "Обе когорты взяты из уже опубликованных разборов. 11–12 ноября — прослушанная приоритетная когорта. "
        "16 ноября — размеченный день. Порог моделей выбирался на валидации 13 ноября и на этих звонках заново не считается."
    )
    lines.append("")
    lines.append(md_table(["Файл", "SHA-256, 16 символов", "Байт"], file_rows, ["---", "---", "---:"]))
    lines.append("")
    lines.append(
        f"11–12 ноября: в файле предсказаний {audits['11–12']['prediction_rows']} звонков, "
        f"в файле меток {audits['11–12']['label_rows']}. Повторов `call_id` в метках нет, расхождений основных меток нет. "
        f"Из бинарного счёта выведены `unknown` ({audits['11–12']['unknown']}) и `mixed` ({audits['11–12']['mixed']}). "
        "Звонков без предсказания одной из четырёх моделей внутри бинарной когорты нет. "
        f"В счёте {len(early)} звонков: {int(early['y'].sum())} synthetic и {int((~early['y']).sum())} human."
    )
    lines.append("")
    subtype_rows = []
    for dataset, frame in frames.items():
        counts = frame["primary_label"].value_counts()
        subtype_rows.append([dataset, int(counts.get("synthetic_tts", 0)), int(counts.get("synthetic_voice_conversion", 0)), int(counts.get("synthetic_playback", 0)), int(counts.get("human_live", 0)), int(counts.get("human_recorded", 0))])
    lines.append(md_table(
        ["Дата", "synthetic_tts", "synthetic_voice_conversion", "synthetic_playback", "human_live", "human_recorded"],
        subtype_rows,
        ["---", "---:", "---:", "---:", "---:", "---:"],
    ))
    lines.append("")
    lines.append(
        "Других бинарных подтипов в этих когортах нет. `unknown` и `mixed` в таблицу не входят."
    )
    lines.append("")
    lines.append(
        f"16 ноября: строк аннотаций {audits['16']['label_rows']}, уникальных звонков меньше из-за повторных строк. "
        f"Одинаковая основная метка на двух строках встретилась у {audits['16']['same_label_duplicates']} звонков: оставлена первая строка, как в прежнем подсчёте. "
        "Три звонка с разными основными метками исключены целиком. Их `call_id` совпадают с `reports/origin_retrain_20261113/test_disputes.csv`."
    )
    lines.append("")
    lines.append(md_table(["call_id", "Основные метки"], dispute_rows, ["---", "---"]))
    lines.append("")
    lines.append(
        f"После снятия этих трёх звонков вне бинарного счёта остаются `unknown` ({audits['16']['unknown']}) и `mixed` ({audits['16']['mixed']}). "
        f"В счёте {len(late)} звонков: {int(late['y'].sum())} synthetic и {int((~late['y']).sum())} human. "
        "Предсказания четырёх моделей есть у каждого из них. Список исключённых `call_id` лежит в `reports/voting_combinations/excluded_calls.csv`."
    )
    lines.append("")
    pack_rows = []
    for dataset, frame in frames.items():
        for pack in PACK_ORDER:
            part = frame[frame["pack"] == pack]
            pack_rows.append([dataset, pack, len(part), int(part["y"].sum()), int((~part["y"]).sum())])
    lines.append(md_table(["Дата", "Пакет", "Звонков", "Synthetic", "Human"], pack_rows, ["---", "---", "---:", "---:", "---:"]))
    lines.append("")
    lines.append(
        "Пакет читается из имени кампании: суффикс PACK1 или PACK2. Звонков бинарной когорты вне этих двух пакетов нет."
    )
    lines.append("")
    speech_rows = []
    for dataset, frame in frames.items():
        for name in SPEECH_ORDER:
            part = frame[frame["speech_bin"] == name]
            speech_rows.append([
                dataset,
                name,
                len(part),
                int(part["y"].sum()),
                int((~part["y"]).sum()),
                f"{part['speech_s'].min():.2f}",
                f"{part['speech_s'].max():.2f}",
            ])
    lines.append(md_table(
        ["Дата", "Речь", "Звонков", "Synthetic", "Human", "Минимум, с", "Максимум, с"],
        speech_rows,
        ["---", "---", "---:", "---:", "---:", "---:", "---:"],
    ))
    lines.append("")
    lines.append(
        "Длительность речи — сумма речевых интервалов Spectra-0, не длительность файла. "
        "На 11–12 это колонка `spectra_speech_duration_s` в `per_call_predictions.csv`. "
        "На 16 ноября это `speech_duration_s` в `evaluation/day_20261116/20261116T152222/call_results/spectra0.csv`. "
        f"Границы прежние и на 16 ноября заново не считались: короткая речь до {SPEECH_LOW:g} с включительно, "
        f"средняя строго после {SPEECH_LOW:g} и до {SPEECH_HIGH:g} с включительно, длинная строго длиннее {SPEECH_HIGH:g} с. "
        f"Неокруглённые терцили речи на бинарных звонках 11–12 ноября равны {terciles['low']:.6f} и {terciles['high']:.6f}. "
        "В группировке используются опубликованные 9.68 и 17.04. Каждый звонок попадает ровно в одну группу, пропусков длительности нет."
    )
    lines.append("")
    on_low = int(np.isclose(pd.concat([early["speech_s"], late["speech_s"]]), SPEECH_LOW).sum())
    on_high = int(np.isclose(pd.concat([early["speech_s"], late["speech_s"]]), SPEECH_HIGH).sum())
    lines.append(
        f"Ровно на границе 9.68 с — {ru_calls(on_low)}, ровно на 17.04 с — {ru_calls(on_high)}. "
        "Граница 9.68 относится к короткой речи, граница 17.04 — к средней."
    )
    lines.append("")
    lines.append("Кампания callback 16 ноября в эти когорты не входит: у неё нет той же полной бинарной разметки происхождения.")
    lines.append("")
    lines.append("## 3. Использованные пороги и направления классификации")
    lines.append("")
    lines.append(
        "Порог применяется к уже сохранённому call-level score функцией `prediction` из `src/research/model_assisted_annotation/rules.py`. "
        "Для каждого звонка анализа пересчитанное решение совпало с сохранённой меткой. Score разных моделей не складываются и не читаются как вероятность."
    )
    lines.append("")
    lines.append("| Модель | Что лежит в score | Порог | Synthetic, если | Минимум | Медиана | Максимум | Ровно на пороге |")
    lines.append("| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |")
    lines.append("| | | | **11–12 ноября** | | | | |")
    lines.extend(score_summary(early))
    lines.append("| | | | **16 ноября** | | | | |")
    lines.extend(score_summary(late))
    lines.append("")
    lines.append(
        "У Spectra-0 и AASIST3 шкалы другие: это margin и средний bonafide logit. Число больше единицы или меньше нуля для них нормально и вероятностью не является. "
        "SVM new — `decision_function`, LightGBM new в сохранённом прогоне получен через `predict_proba`, но в голосовании участвует только бинарное сравнение с порогом 0.346. "
        "Logistic regression остаётся при пороге 0.7354826103642409 и в комбинации A–D не входит."
    )
    lines.append("")
    lines.append(
        "На пороге classical-модель говорит synthetic, потому что правило — «больше или равно». Spectra-0 и AASIST3 на пороге говорят human, потому что правило — «строго меньше». "
        "В этих когортах звонков ровно на пороге нет, так что граничный случай на исход не влияет."
    )
    lines.append("")
    lines.append("## 4. Методика голосования")
    lines.append("")
    lines.append(
        "Сначала каждая модель переводится в human или synthetic. Пустое решение на human не заменяется. "
        "Дальше голос 2 из 3 считает число бинарных synthetic-решений и ставит synthetic при сумме 2 или 3. "
        "Сырой score в сумму не входит. Правило 2 из 2 ставит synthetic только при двух synthetic и считается отдельной колонкой `pair_2of2`. "
        "Оно не подаётся в функцию majority и не стоит в ряду A–D."
    )
    lines.append("")
    lines.append("Перед записью отчёта скрипт проверяет:")
    lines.append("")
    for item in checks:
        lines.append(f"- {item}")
    lines.append("")
    lines.append(
        "Дополнительно на каждой строке метрик проверяется: TP + FN равно числу synthetic, TN + FP равно числу human, "
        "сумма четырёх клеток равна размеру среза, specificity + FPR = 1. Смена голоса относительно A раскладывается на исправленные и новые FN и FP, "
        "и это разложение сходится с ΔFN и ΔFP."
    )
    lines.append("")
    lines.append("## 5. Общие результаты по каждой дате")
    lines.append("")
    lines.append(
        f"11–12 ноября, {len(early)} звонков, {int(early['y'].sum())} synthetic и {int((~early['y']).sum())} human. "
        f"A: {count_text('11–12', 'A')}."
    )
    lines.append("")
    lines.append(section_metrics(metrics, "11–12", list(COMBOS)))
    lines.append("")
    lines.append("Отдельное правило 2 из 2 на тех же 457 звонках:")
    lines.append("")
    lines.append(section_metrics(metrics, "11–12", ["pair_2of2"]))
    lines.append("")
    lines.append(
        f"16 ноября, {len(late)} звонков, {int(late['y'].sum())} synthetic и {int((~late['y']).sum())} human. "
        f"A: {count_text('16', 'A')}."
    )
    lines.append("")
    lines.append(section_metrics(metrics, "16", list(COMBOS)))
    lines.append("")
    lines.append("Отдельное правило 2 из 2 на тех же 361 звонках:")
    lines.append("")
    lines.append(section_metrics(metrics, "16", ["pair_2of2"]))
    lines.append("")
    lines.append("Отдельные модели на тех же звонках, без голосования:")
    lines.append("")
    lines.append("### 11–12 ноября")
    lines.append("")
    lines.append(section_metrics(metrics, "11–12", list(MODELS)))
    lines.append("")
    lines.append("### 16 ноября")
    lines.append("")
    lines.append(section_metrics(metrics, "16", list(MODELS)))
    lines.append("")
    lines.append("![Recall](voting_combinations/figures/recall.png)")
    lines.append("")
    lines.append("Подпись столбца — TP / число synthetic в когорте.")
    lines.append("")
    lines.append("![Precision](voting_combinations/figures/precision.png)")
    lines.append("")
    lines.append("Подпись столбца — TP / (TP + FP).")
    lines.append("")
    lines.append("![FPR](voting_combinations/figures/fpr.png)")
    lines.append("")
    lines.append("Подпись столбца — FP / число human. Ось начинается с нуля.")
    lines.append("")
    lines.append("![FP и FN](voting_combinations/figures/fp_fn.png)")
    lines.append("")
    lines.append("Числа над столбцами — абсолютные FP и FN. Вертикальная черта отделяет правило 2 из 2 от majority.")
    lines.append("")
    lines.append(overall_reading(metrics))
    lines.append("")
    lines.append("## 6. Результаты PACK1/PACK2")
    lines.append("")
    lines.append(
        "PACK1 держит почти всю синтетику: 115 из 134 на 11–12 и 51 из 58 на 16 ноября. "
        "PACK2 — это в основном human: 171 и 220 человек при 19 и 7 synthetic. Доля на семи synthetic меняется шагом около 14 процентных пунктов."
    )
    lines.append("")
    lines.append(slice_table(metrics, "pack", PACK_ORDER, list(COMBOS)))
    lines.append("")
    lines.append("Правило 2 из 2 по пакетам, отдельно от таблицы выше:")
    lines.append("")
    lines.append(slice_table(metrics, "pack", PACK_ORDER, ["pair_2of2"]))
    lines.append("")
    lines.append("![Пакеты](voting_combinations/figures/pack.png)")
    lines.append("")
    lines.append(
        "Число над столбцом recall — TP, над столбцом FPR — FP. Подпись группы показывает число synthetic и human, поэтому доля на PACK2 читается вместе с семью или девятнадцатью положительными звонками."
    )
    lines.append("")
    lines.append(pack_reading(metrics))
    lines.append("")
    lines.append("## 7. Результаты по длительности речи")
    lines.append("")
    lines.append(
        f"Короткая: речь ≤ {SPEECH_LOW:g} с. Средняя: {SPEECH_LOW:g} < речь ≤ {SPEECH_HIGH:g} с. Длинная: речь > {SPEECH_HIGH:g} с. "
        "На 16 ноября короткая группа содержит 7 synthetic и 112 human. Precision и recall в ней стоят на единицах звонков."
    )
    lines.append("")
    lines.append(slice_table(metrics, "speech", SPEECH_ORDER, list(COMBOS)))
    lines.append("")
    lines.append("Правило 2 из 2 по длительности речи:")
    lines.append("")
    lines.append(slice_table(metrics, "speech", SPEECH_ORDER, ["pair_2of2"]))
    lines.append("")
    lines.append("![Длительность речи](voting_combinations/figures/duration.png)")
    lines.append("")
    lines.append(speech_reading(metrics))
    lines.append("")
    lines.append("## 8. Пересечение ошибок")
    lines.append("")
    lines.append(
        "Ошибка модели — бинарное решение. FN: звонок synthetic, модель сказала human. FP: звонок human, модель сказала synthetic. "
        "В паре «обе» значит, что ошиблись обе модели. «Только A» значит, что ошиблась первая и не ошиблась вторая."
    )
    lines.append("")
    pair_view = pairs.copy()
    pair_view["jaccard_text"] = pair_view["jaccard"].map(lambda value: "—" if not np.isfinite(value) else f"{value:.3f}")
    lines.append(md_table(
        ["Дата", "Ошибка", "Модель A", "Модель B", "Обе", "Только A", "Только B", "Ни одна", "Jaccard"],
        [
            [row.dataset, row.error_type, row.model_a, row.model_b, int(row.both), int(row.only_a), int(row.only_b), int(row.neither), row.jaccard_text]
            for row in pair_view.itertuples(index=False)
        ],
        ["---", "---", "---", "---", "---:", "---:", "---:", "---:", "---:"],
    ))
    lines.append("")
    lines.append("![Пересечение ошибок](voting_combinations/figures/error_overlap.png)")
    lines.append("")
    lines.append(
        "Диагональ матрицы — сколько ошибок у самой модели. Вне диагонали — сколько из них общие со второй моделью. "
        "Полная таблица и матрица также лежат в `reports/voting_combinations/pairwise_errors.csv` и `reports/voting_combinations/error_matrix.csv`."
    )
    lines.append("")
    lines.append(diversity_reading(frames, pairs))
    lines.append("")
    lines.append("### Synthetic, которые находит Spectra-0 и пропускает AASIST3")
    lines.append("")
    lines.append(disagreement_block(frames, caught="spectra0", missed="aasist3"))
    lines.append("")
    lines.append("### Synthetic, которые находит AASIST3 и пропускает Spectra-0")
    lines.append("")
    lines.append(disagreement_block(frames, caught="aasist3", missed="spectra0"))
    lines.append("")
    lines.append("### Human, которых модель назвала synthetic")
    lines.append("")
    lines.append(fp_block(frames))
    lines.append("")
    lines.append("### Ошибки, общие для всех четырёх моделей")
    lines.append("")
    lines.append(common_four_block(frames))
    lines.append("")
    lines.append("Решения всех бинарных звонков, пакет, длительность речи и четыре голоса записаны в `reports/voting_combinations/call_decisions.csv`.")
    lines.append("")
    lines.append("## 9. Сравнение с текущей комбинацией")
    lines.append("")
    lines.append(
        "Изменение считается как альтернатива минус A. Положительный recall — больше найденных synthetic. "
        "Положительные FPR и FP — больше ложных synthetic на человеческой речи. Положительный FN — больше пропусков. "
        "Процентные пункты — это разница долей, умноженная на 100. Один звонок на 11–12 — это 0.75 п.п. recall и 0.31 п.п. FPR. "
        "Один звонок 16 ноября — 1.72 п.п. recall и 0.33 п.п. FPR."
    )
    lines.append("")
    lines.append(delta_md(metrics, ("B", "C", "D")))
    lines.append("")
    lines.append("Правило 2 из 2 против A, той же арифметикой, но это не кандидат в ряд majority:")
    lines.append("")
    lines.append(delta_md(metrics, ("pair_2of2",)))
    lines.append("")
    lines.append(flip_reading(metrics, flips))
    lines.append("")
    lines.append("## 10. Ограничения исследования")
    lines.append("")
    lines.append(
        "Обе когорты уже использовались в предыдущих отчётах. Пороги зафиксированы на валидации 13 ноября. "
        "Этот расчёт описывает те же звонки и не является новой независимой проверкой. "
        "Он не показывает, как правило поведёт себя на будущих звонках."
    )
    lines.append("")
    lines.append(
        "11–12 ноября — приоритетная прослушанная когорта, а не случайная выборка всех звонков этих дней. "
        "16 ноября — один размеченный день. Доля synthetic разная: 134/457 и 58/361. "
        "PACK2 16 ноября содержит 7 synthetic, короткая речь этого дня — тоже 7. На таких группах один звонок сильно двигает recall."
    )
    lines.append("")
    lines.append(
        "Score моделей не калибровались и не сравнивались между собой. Голос видит только human/synthetic. "
        "Длительность речи взята из интервалов Spectra-0. Classical-длина речи в срезы не подставлялась: на 16 ноября эти два определения заметно расходятся, и прежние срезы использовали интервалы Spectra-0."
    )
    lines.append("")
    lines.append(
        "Метки ручные. Три спора 16 ноября исключены, а не разрешены большинством. "
        "`unknown` и `mixed` не добавлялись в human или synthetic. Ошибочные звонки из счёта не удалялись. "
        "Callback-кампания 16 ноября не смешивалась с пакетами."
    )
    lines.append("")
    lines.append("## 11. Выводы и рекомендации")
    lines.append("")
    lines.append(conclusions(frames, metrics, pairs))
    lines.append("")
    lines.append(
        "Эти выводы относятся к двум уже размеченным когортам. Они не доказывают, что соотношение сохранится на следующих звонках. "
        "Для окончательного выбора нужна независимая тестовая выборка, на которой порог и состав голоса заранее зафиксированы и не пересматриваются."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def overall_reading(metrics: pd.DataFrame) -> str:
    parts = []
    for dataset in DATE_ORDER:
        base = metric_row(metrics, dataset, "все", "все", "A")
        sentences = [f"На {dataset} ноября A даёт {int(base['tp'])} TP и {int(base['fp'])} FP при {int(base['fn'])} FN."]
        for rule_id in ("B", "C", "D"):
            alt = metric_row(metrics, dataset, "все", "все", rule_id)
            sentences.append(
                f"{rule_id} относительно A: FN {int(alt['fn']) - int(base['fn']):+d}, FP {int(alt['fp']) - int(base['fp']):+d}, ошибок {int(alt['errors']) - int(base['errors']):+d}."
            )
        pair = metric_row(metrics, dataset, "все", "все", "pair_2of2")
        sentences.append(
            f"Правило 2 из 2 относительно A: FN {int(pair['fn']) - int(base['fn']):+d}, FP {int(pair['fp']) - int(base['fp']):+d}."
        )
        parts.append(" ".join(sentences))
    return "\n\n".join(parts)


def pack_reading(metrics: pd.DataFrame) -> str:
    sentences = []
    for dataset in DATE_ORDER:
        for pack in PACK_ORDER:
            base = metric_row(metrics, dataset, "pack", pack, "A")
            moves = []
            for rule_id in ("B", "C", "D"):
                alt = metric_row(metrics, dataset, "pack", pack, rule_id)
                moves.append(f"{rule_id} {int(alt['fn']) - int(base['fn']):+d} FN и {int(alt['fp']) - int(base['fp']):+d} FP")
            sentences.append(
                f"{dataset}, {pack}: A ловит {int(base['tp'])} из {int(base['n_synthetic'])} и даёт {int(base['fp'])} FP на {int(base['n_human'])} human. "
                + ", ".join(moves)
                + "."
            )
    return " ".join(sentences)


def speech_reading(metrics: pd.DataFrame) -> str:
    sentences = []
    for dataset in DATE_ORDER:
        for name in SPEECH_ORDER:
            base = metric_row(metrics, dataset, "speech", name, "A")
            best = min(
                ((rule_id, metric_row(metrics, dataset, "speech", name, rule_id)) for rule_id in COMBOS),
                key=lambda item: (int(item[1]["errors"]), int(item[1]["fn"]), item[0]),
            )
            sentences.append(
                f"{dataset}, {name}: {int(base['n_synthetic'])} synthetic и {int(base['n_human'])} human, "
                f"A даёт FN {int(base['fn'])} и FP {int(base['fp'])}. "
                f"Меньше всего ошибок FP+FN в этом срезе у {best[0]}: FN {int(best[1]['fn'])}, FP {int(best[1]['fp'])}."
            )
    return " ".join(sentences)


def diversity_reading(frames: dict[str, pd.DataFrame], pairs: pd.DataFrame) -> str:
    paragraphs = []
    for dataset, frame in frames.items():
        spectra_fn = frame["y"] & ~frame["spectra0"]
        caught = {
            model: int((spectra_fn & frame[model]).sum())
            for model in ("aasist3", "svm_new", "lightgbm_new")
        }
        spectra_fp = ~frame["y"] & frame["spectra0"]
        shared_fp = {
            model: int((spectra_fp & frame[model]).sum())
            for model in ("aasist3", "svm_new", "lightgbm_new")
        }
        aasist_only = int((frame["y"] & frame["aasist3"] & ~frame["spectra0"]).sum())
        paragraphs.append(
            f"На {dataset} ноября из {int(spectra_fn.sum())} пропусков Spectra-0 SVM находит {caught['svm_new']}, "
            f"LightGBM находит {caught['lightgbm_new']}, AASIST3 находит {caught['aasist3']}. "
            f"Из {int(spectra_fp.sum())} ложных Spectra-0 SVM повторяет {shared_fp['svm_new']}, "
            f"LightGBM повторяет {shared_fp['lightgbm_new']}, AASIST3 повторяет {shared_fp['aasist3']}. "
            f"Synthetic, которые AASIST3 видит, а Spectra-0 нет: {aasist_only}."
        )
    fn_pair = pairs[(pairs["model_a_id"] == "spectra0") & (pairs["model_b_id"] == "aasist3") & (pairs["error_type"] == "FN")]
    texts = []
    for row in fn_pair.itertuples(index=False):
        texts.append(
            f"на {row.dataset} Jaccard пропусков Spectra-0 и AASIST3 равен {row.jaccard:.3f} "
            f"({int(row.both)} общих, {int(row.only_a)} только Spectra-0, {int(row.only_b)} только AASIST3)"
        )
    paragraphs.append("Перекрытие пропусков Spectra-0 и AASIST3 высокое: " + "; ".join(texts) + ".")
    paragraphs.append(
        "SVM и LightGBM закрывают часть пропусков Spectra-0 и почти не повторяют её ложные срабатывания. "
        "AASIST3 эту пару свойств не повторяет: пропуски Spectra-0 он в основном делит, а не закрывает, и часть ложных Spectra-0 подтверждает. "
        "Замена SVM или LightGBM на AASIST3 поэтому уменьшает разнообразие ошибок, на котором держится голос A."
    )
    return "\n\n".join(paragraphs)


def disagreement_block(frames: dict[str, pd.DataFrame], caught: str, missed: str) -> str:
    chunks = []
    headers = CALL_HEADERS
    total = 0
    for dataset, frame in frames.items():
        mask = frame["y"] & frame[caught] & ~frame[missed]
        total += int(mask.sum())
        chunks.append(f"{dataset} ноября: {int(mask.sum())} звонков.")
        chunks.append("")
        rows = call_rows(frame, mask)
        chunks.append(md_table(headers, rows, CALL_ALIGN) if rows else "Таких звонков нет.")
        chunks.append("")
    if total == 0:
        return "Таких звонков нет ни на одной дате."
    return "\n".join(chunks).rstrip()


def fp_block(frames: dict[str, pd.DataFrame]) -> str:
    chunks = []
    for dataset, frame in frames.items():
        mask = (~frame["y"]) & (frame["spectra0"] | frame["aasist3"] | frame["svm_new"] | frame["lightgbm_new"])
        counts = ", ".join(
            f"{MODEL_NAME[model]} {int((~frame['y'] & frame[model]).sum())}"
            for model in MODELS
        )
        chunks.append(f"{dataset} ноября, {counts}. Ниже все human-звонки, где хотя бы одна из четырёх моделей сказала synthetic: {int(mask.sum())}.")
        chunks.append("")
        chunks.append(md_table(CALL_HEADERS, call_rows(frame, mask), CALL_ALIGN))
        chunks.append("")
    return "\n".join(chunks).rstrip()


def common_four_block(frames: dict[str, pd.DataFrame]) -> str:
    chunks = []
    for dataset, frame in frames.items():
        all_fn = frame["y"] & ~frame[list(MODELS)].any(axis=1)
        all_fp = (~frame["y"]) & frame[list(MODELS)].all(axis=1)
        chunks.append(f"{dataset} ноября: общие FN {int(all_fn.sum())}, общие FP {int(all_fp.sum())}.")
        chunks.append("")
        chunks.append("Общие пропуски synthetic:")
        chunks.append("")
        fn_rows = call_rows(frame, all_fn)
        chunks.append(md_table(CALL_HEADERS, fn_rows, CALL_ALIGN) if fn_rows else "Нет звонка, который пропустили все четыре модели.")
        chunks.append("")
        chunks.append("Общие ложные synthetic:")
        chunks.append("")
        fp_rows = call_rows(frame, all_fp)
        chunks.append(md_table(CALL_HEADERS, fp_rows, CALL_ALIGN) if fp_rows else "Нет human-звонка, которого все четыре модели назвали synthetic.")
        chunks.append("")
    return "\n".join(chunks).rstrip()


def delta_md(metrics: pd.DataFrame, rule_ids: tuple[str, ...]) -> str:
    headers = ["Дата", "Вариант", "Δ Recall, п.п.", "Δ Precision, п.п.", "Δ FPR, п.п.", "Δ FP", "Δ FN", "Δ ошибок"]
    aligns = ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:"]
    rows = []
    for dataset in DATE_ORDER:
        base = metric_row(metrics, dataset, "все", "все", "A")
        for rule_id in rule_ids:
            alt = metric_row(metrics, dataset, "все", "все", rule_id)
            rows.append(
                [
                    dataset,
                    f"{SHORT[rule_id]}. {COMBO_NAME[rule_id]}",
                    fmt_pp((alt["recall"] - base["recall"]) * 100),
                    fmt_pp((alt["precision"] - base["precision"]) * 100),
                    fmt_pp((alt["fpr"] - base["fpr"]) * 100),
                    fmt_signed_int(int(alt["fp"] - base["fp"])),
                    fmt_signed_int(int(alt["fn"] - base["fn"])),
                    fmt_signed_int(int(alt["errors"] - base["errors"])),
                ]
            )
    return md_table(headers, rows, aligns)


def flip_reading(metrics: pd.DataFrame, flips: pd.DataFrame) -> str:
    chunks = [
        "Смена итога относительно A разбирается по звонкам. Исправленный FN — synthetic, который A пропустил, а альтернатива взяла. Новый FN — наоборот. Для FP то же на человеческой речи.",
        "",
    ]
    if flips.empty:
        chunks.append("Итог альтернатив совпал с A на всех звонках.")
        return "\n".join(chunks)
    summary_rows = []
    for dataset in DATE_ORDER:
        for rule_id in ("B", "C", "D", "pair_2of2"):
            part = flips[(flips["dataset"] == dataset) & (flips["rule_id"] == rule_id)]
            summary_rows.append(
                [
                    dataset,
                    SHORT[rule_id],
                    int((part["change"] == "исправленный FN").sum()),
                    int((part["change"] == "новый FN").sum()),
                    int((part["change"] == "исправленный FP").sum()),
                    int((part["change"] == "новый FP").sum()),
                    int(len(part)),
                ]
            )
    chunks.append(md_table(
        ["Дата", "Вариант", "Исправленный FN", "Новый FN", "Исправленный FP", "Новый FP", "Звонков с другим итогом"],
        summary_rows,
        ["---", "---", "---:", "---:", "---:", "---:", "---:"],
    ))
    chunks.append("")
    for dataset in DATE_ORDER:
        for rule_id in ("B", "C", "D", "pair_2of2"):
            part = flips[(flips["dataset"] == dataset) & (flips["rule_id"] == rule_id)]
            if part.empty:
                continue
            chunks.append(f"{dataset} ноября, {SHORT[rule_id]}:")
            chunks.append("")
            rows = []
            order = {"исправленный FN": 0, "новый FN": 1, "исправленный FP": 2, "новый FP": 3}
            part = part.sort_values(["change", "call_id"], key=lambda column: column.map(order) if column.name == "change" else column)
            for row in part.itertuples(index=False):
                rows.append(
                    [
                        row.change,
                        f"`{row.call_id}`",
                        row.pack,
                        f"{float(row.speech_s):.2f}",
                        row.primary_label,
                        row.spectra0,
                        row.aasist3,
                        row.svm_new,
                        row.lightgbm_new,
                    ]
                )
            chunks.append(md_table(
                ["Что изменилось", "call_id", "Пакет", "Речь, с", "Метка", "Spectra-0", "AASIST3", "SVM", "LightGBM"],
                rows,
                ["---", "---", "---", "---:", "---", "---", "---", "---", "---"],
            ))
            chunks.append("")
    return "\n".join(chunks).rstrip()


def conclusions(frames: dict[str, pd.DataFrame], metrics: pd.DataFrame, pairs: pd.DataFrame) -> str:
    qualified = []
    for rule_id in ("B", "C", "D"):
        full_ok = True
        strict = False
        for dataset in DATE_ORDER:
            base = metric_row(metrics, dataset, "все", "все", "A")
            alt = metric_row(metrics, dataset, "все", "все", rule_id)
            if not pareto_not_worse(alt, base):
                full_ok = False
            if pareto_better(alt, base):
                strict = True
            for pack in PACK_ORDER:
                if not pareto_not_worse(
                    metric_row(metrics, dataset, "pack", pack, rule_id),
                    metric_row(metrics, dataset, "pack", pack, "A"),
                ):
                    full_ok = False
        if full_ok and strict:
            qualified.append(rule_id)

    recall_leader = {}
    fpr_leader = {}
    for dataset in DATE_ORDER:
        rows = [(rule_id, metric_row(metrics, dataset, "все", "все", rule_id)) for rule_id in COMBOS]
        recall_leader[dataset] = sorted(rows, key=lambda item: (-item[1]["recall"], item[1]["fp"], item[0]))[0]
        fpr_leader[dataset] = sorted(rows, key=lambda item: (item[1]["fpr"], -item[1]["recall"], item[0]))[0]

    paragraphs = []
    fpr_ties = {}
    for dataset in DATE_ORDER:
        rows = [(rule_id, metric_row(metrics, dataset, "все", "все", rule_id)) for rule_id in COMBOS]
        best_fpr = min(row["fpr"] for _, row in rows)
        tied = [rule_id for rule_id, row in rows if row["fpr"] == best_fpr]
        fpr_ties[dataset] = tied
        recall_id, recall_row = recall_leader[dataset]
        fpr_id, fpr_row = fpr_leader[dataset]
        if len(tied) == 1:
            fpr_sentence = (
                f"Наименьший FPR у {fpr_id}: {fmt_rate(int(fpr_row['fp']), int(fpr_row['n_human']))}, "
                f"при recall {fmt_rate(int(fpr_row['tp']), int(fpr_row['n_synthetic']))}."
            )
        else:
            details = []
            for rule_id in tied:
                row = metric_row(metrics, dataset, "все", "все", rule_id)
                details.append(
                    f"{rule_id} recall {fmt_rate(int(row['tp']), int(row['n_synthetic']))}, FN {int(row['fn'])}"
                )
            fpr_sentence = (
                f"Наименьший FPR совпадает у {' и '.join(tied)}: "
                f"{fmt_rate(int(fpr_row['fp']), int(fpr_row['n_human']))}. " + "; ".join(details) + "."
            )
        paragraphs.append(
            f"На {dataset} ноября наибольший recall среди правил 2 из 3 у {recall_id}: "
            f"{fmt_rate(int(recall_row['tp']), int(recall_row['n_synthetic']))}, FP {int(recall_row['fp'])}. "
            + fpr_sentence
        )
    recall_ids = [recall_leader[dataset][0] for dataset in DATE_ORDER]
    if recall_ids == ["A", "A"] and any(len(tied) > 1 for tied in fpr_ties.values()):
        shared = [dataset for dataset, tied in fpr_ties.items() if len(tied) > 1]
        paragraphs.append(
            "На обоих полных днях наибольший recall у A. На "
            + " и ".join(shared)
            + " ноября наименьший FPR у A совпадает с другой комбинацией, но у A меньше пропусков synthetic. "
            "Это описание двух когорт, а не выбор правила на следующий день."
        )
    elif recall_ids[0] != recall_ids[1] or fpr_leader["11–12"][0] != fpr_leader["16"][0]:
        paragraphs.append(
            "Лидер по recall и лидер по FPR не совпадают между датами или между собой, поэтому одна комбинация балансом обоих дней не назначается."
        )
    else:
        paragraphs.append(
            f"На обоих днях наибольший recall и наименьший FPR среди 2 из 3 оказались у {recall_leader['11–12'][0]}. "
            "Это описание двух когорт, а не выбор правила на следующий день."
        )

    aasist_added = []
    for dataset, frame in frames.items():
        added = int((frame["y"] & frame["aasist3"] & ~frame["spectra0"]).sum())
        missed = int((frame["y"] & frame["spectra0"] & ~frame["aasist3"]).sum())
        aasist_added.append(f"{dataset}: AASIST3 добавляет {added} synthetic сверх Spectra-0 и пропускает {missed} synthetic, которые Spectra-0 находит")
    paragraphs.append("Отдельный вклад AASIST3 относительно Spectra-0: " + "; ".join(aasist_added) + ".")
    paragraphs.append(
        "Этого вклада мало для замены SVM или LightGBM. Те закрывают пропуски Spectra-0, которых AASIST3 не видит, и почти не подтверждают её ложные срабатывания. "
        "Голос A как раз использует это расхождение. Когда AASIST3 садится на место SVM или LightGBM, третий голос чаще повторяет Spectra-0, чем дополняет её."
    )
    if qualified:
        paragraphs.append(
            "По заранее заданному правилу — не больше FN и не больше FP на обоих днях и обоих пакетах, и строго меньше ошибок хотя бы на одном полном дне — "
            f"проходит {' и '.join(qualified)}. Даже в этом случае результат нужно проверить на новой выборке."
        )
    else:
        paragraphs.append(
            "По заранее заданному правилу ни B, ни C, ни D не держат одновременно FN и FP не хуже A на обоих днях и на PACK1 с PACK2. "
            "Менять текущую комбинацию Spectra-0 + SVM + LightGBM по этим двум когортам не стоит."
        )
    pair_bits = []
    for dataset in DATE_ORDER:
        base = metric_row(metrics, dataset, "все", "все", "A")
        pair = metric_row(metrics, dataset, "все", "все", "pair_2of2")
        pair_bits.append(
            f"{dataset}: FN {int(pair['fn']) - int(base['fn']):+d}, FP {int(pair['fp']) - int(base['fp']):+d}"
        )
    paragraphs.append(
        "Правило 2 из 2 убирает часть ложных ценой пропусков, которые majority ещё берёт вторым голосом classical-модели. "
        "Относительно A это " + "; ".join(pair_bits) + ". Заменой голоса 2 из 3 оно не является."
    )
    return "\n\n".join(paragraphs)


def decisions_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset, frame in frames.items():
        for row in frame.itertuples(index=False):
            rows.append(
                {
                    "dataset": dataset,
                    "call_id": row.call_id,
                    "pack": row.pack,
                    "primary_label": row.primary_label,
                    "y_synthetic": bool(row.y),
                    "speech_s": row.speech_s,
                    "speech_bin": row.speech_bin,
                    "duration_s": row.duration_s,
                    "spectra0_margin": row.spectra0_score,
                    "aasist3_bonafide_logit": row.aasist3_score,
                    "svm_new_score": row.svm_new_score,
                    "lightgbm_new_score": row.lightgbm_new_score,
                    "spectra0": "synthetic" if row.spectra0 else "human",
                    "aasist3": "synthetic" if row.aasist3 else "human",
                    "svm_new": "synthetic" if row.svm_new else "human",
                    "lightgbm_new": "synthetic" if row.lightgbm_new else "human",
                    "vote_A": "synthetic" if row.A else "human",
                    "vote_B": "synthetic" if row.B else "human",
                    "vote_C": "synthetic" if row.C else "human",
                    "vote_D": "synthetic" if row.D else "human",
                    "pair_2of2": "synthetic" if row.pair_2of2 else "human",
                }
            )
    return pd.DataFrame(rows)


def matrix_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset, frame in frames.items():
        for kind in ("FN", "FP"):
            matrix = overlap_matrix(frame, kind)
            for left in matrix.index:
                for right in matrix.columns:
                    rows.append(
                        {
                            "dataset": dataset,
                            "error_type": kind,
                            "model_a": left,
                            "model_b": right,
                            "n_intersection": int(matrix.loc[left, right]),
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    checks = check_rules()
    excluded: list[dict] = []
    early, early_audit = load_early(excluded)
    late, late_audit = load_late(excluded)
    frames = {"11–12": attach_votes(early), "16": attach_votes(late)}
    audits = {"11–12": early_audit, "16": late_audit}
    metrics = collect_metrics(frames)
    checks.extend(check_frames(frames, audits, metrics))
    pairs = pairwise_rows(frames)
    flips = flip_table(frames)
    deltas = delta_table(metrics)
    speech = pd.to_numeric(frames["11–12"]["speech_s"], errors="coerce").dropna().to_numpy()
    low, high = np.quantile(speech, [1 / 3, 2 / 3])
    terciles = {"low": float(low), "high": float(high)}

    files = [
        audits["11–12"]["predictions"],
        audits["11–12"]["labels"],
        audits["16"]["predictions"],
        audits["16"]["labels"],
        audits["16"]["manifest"],
        audits["16"]["speech"],
    ]
    report = build_report(frames, audits, metrics, pairs, flips, files, checks, terciles)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    deltas.to_csv(OUT / "delta_vs_A.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(OUT / "pairwise_errors.csv", index=False, encoding="utf-8-sig")
    matrix_frame(frames).to_csv(OUT / "error_matrix.csv", index=False, encoding="utf-8-sig")
    flips.to_csv(OUT / "vote_flips.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(excluded).to_csv(OUT / "excluded_calls.csv", index=False, encoding="utf-8-sig")
    decisions_frame(frames).to_csv(OUT / "call_decisions.csv", index=False, encoding="utf-8-sig")
    (OUT / "assertions.txt").write_text("\n".join(checks) + "\n", encoding="utf-8")
    plot_rate(
        OUT / "figures" / "recall.png",
        metrics,
        "recall",
        "tp",
        "n_synthetic",
        "Recall. Подпись столбца — TP / число synthetic.",
        "Recall",
        label_rotation=90,
        headroom=1.55,
    )
    plot_rate(OUT / "figures" / "fpr.png", metrics, "fpr", "fp", "n_human", "FPR. Подпись столбца — FP / число human.", "FPR")
    plot_counts(OUT / "figures" / "fp_fn.png", metrics)
    plot_slices(
        OUT / "figures" / "pack.png",
        metrics,
        "pack",
        PACK_ORDER,
        "PACK1 и PACK2. Верхние ряды — majority 2 из 3, нижний ряд — отдельное правило 2 из 2.",
    )
    plot_slices(
        OUT / "figures" / "duration.png",
        metrics,
        "speech",
        SPEECH_ORDER,
        "Длительность речи Spectra-0. Верхние ряды — majority 2 из 3, нижний ряд — отдельное правило 2 из 2.",
    )
    plot_overlap(OUT / "figures" / "error_overlap.png", frames)
    # Precision labels need TP/(TP+FP), not TP/TP. Rebuild that figure explicitly.
    plot_precision(OUT / "figures" / "precision.png", metrics)
    REPORT.write_text(report, encoding="utf-8")
    print(f"отчёт {REPORT}")
    print(f"таблицы {OUT}")
    for line in checks:
        print("OK", line)


def plot_precision(path: Path, metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), gridspec_kw={"width_ratios": [4.2, 1.6]})
    panel = (
        (axes[0], list(COMBOS), "Majority: минимум 2 из 3"),
        (axes[1], ["pair_2of2"], "Отдельно: оба голоса, 2 из 2"),
    )
    for ax, rules, panel_title in panel:
        x = np.arange(len(rules))
        width = 0.36
        tops = []
        for index, dataset in enumerate(DATE_ORDER):
            values = []
            labels = []
            for rule_id in rules:
                row = metric_row(metrics, dataset, "все", "все", rule_id)
                values.append(float(row["precision"]))
                labels.append(f"{int(row['tp'])}/{int(row['tp']) + int(row['fp'])}")
            bars = ax.bar(x + (index - 0.5) * width, values, width, label=dataset + " ноября", color=DATE_COLOR[dataset])
            annotate_bars(ax, bars, labels, rotation=90)
            tops.extend(values)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[rule_id] + "\n" + COMBO_NAME[rule_id].replace(" + ", "\n") for rule_id in rules], fontsize=8)
        ax.set_title(panel_title)
        ax.set_ylabel("Precision")
        finish(ax, 0, max(tops) * 1.45 if tops else 1)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Precision. Подпись столбца — TP / (TP + FP).", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
