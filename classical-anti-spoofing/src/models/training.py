from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from src.models import lightgbm as lightgbm_model
from src.models import logistic_regression, svm
from src.models.dataset import acoustic_columns, labels_machine, load_call_frame, modeling_rows
from src.models.evaluation import binary_metrics
from src.models.features import (
    columns_for_variant,
    is_log_energy,
    is_mfcc_shape,
    is_shape,
    is_spectral_measure,
    is_voicing,
    select_features,
)
from src.utils.config import resolve_path
from src.utils.reports import next_numbered_dir, write_json


def train_model(cfg: dict, name: str, variant: str = "full") -> Path:
    frame = load_call_frame(cfg)
    train = modeling_rows(frame, "train")
    validation = modeling_rows(frame, "validation")
    _reject_locked_rows(train, "train")
    _reject_locked_rows(validation, "validation")
    y_train = labels_machine(train)
    y_validation = labels_machine(validation)
    columns, excluded = columns_for_variant(acoustic_columns(train.columns.tolist()), variant)
    model_cfg = cfg["models"]
    features = select_features(
        train,
        y_train,
        columns,
        max_features=int(model_cfg["max_features"]),
        max_nan_fraction=float(model_cfg["max_nan_fraction"]),
        min_finite=int(model_cfg["min_finite"]),
        max_pairwise_correlation=float(model_cfg["max_pairwise_correlation"]),
    )
    candidates = _candidates(name, model_cfg, int(cfg["seed"]))
    scored = []
    best = None
    for params, estimator in candidates:
        estimator.fit(train[features], y_train)
        scores = _positive_scores(estimator, validation[features], name)
        threshold = _best_threshold(y_validation.to_numpy(), scores, _default_threshold(name))
        metrics = binary_metrics(y_validation.to_numpy(), scores, _predict_from_scores(scores, threshold))
        record = {
            "params": params,
            "threshold": threshold,
            "validation_balanced_accuracy": metrics["balanced_accuracy"],
            "validation_roc_auc": metrics["roc_auc"],
        }
        scored.append(record)
        rank = (metrics["balanced_accuracy"], metrics["roc_auc"], -_complexity(name, params))
        if best is None or rank > best[0]:
            best = (rank, estimator, params, scores, threshold)
    estimator, params, validation_scores, threshold = best[1], best[2], best[3], best[4]
    validation_metrics = binary_metrics(
        y_validation.to_numpy(),
        validation_scores,
        _predict_from_scores(validation_scores, threshold),
    )
    importance = _importance(name, estimator, validation[features], y_validation.to_numpy(), threshold, features)
    bundle = {
        "name": name,
        "model": estimator,
        "features": features,
        "threshold": threshold,
        "params": params,
        "importance": importance.to_dict(orient="records"),
        "score_note": "score ранжирует machine выше human и не проверялся как калиброванная вероятность",
        "seed": int(cfg["seed"]),
        "variant": variant,
        "excluded_log_energy": len(excluded),
    }
    _reject_foreign_features(variant, features)
    model_path = _bundle_path(cfg, name, variant)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    report_dir = next_numbered_dir(resolve_path(cfg, cfg["paths"]["reports_dir"]), "experiment")
    joblib.dump(bundle, report_dir / "model.joblib")
    write_json(
        report_dir / "metrics.json",
        {
            "model": name,
            "split": "validation",
            "threshold": threshold,
            "params": params,
            "metrics": validation_metrics,
            "candidates": scored,
            "n_features": len(features),
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_used_for_selection": False,
            "variant": variant,
            "excluded_log_energy": len(excluded),
            "exclusion_rule": _exclusion_rule(variant),
        },
    )
    pd.DataFrame({"feature": features}).to_csv(report_dir / "features.csv", index=False)
    importance.to_csv(report_dir / "feature_importance.csv", index=False)
    predictions = validation[["call_id", "label", "speech_duration_s", "n_usable_segments", "snr_db"]].copy()
    predictions["score"] = validation_scores
    predictions["predicted"] = np.where(_predict_from_scores(validation_scores, threshold) == 1, "machine", "human")
    predictions.to_csv(report_dir / "predictions.csv", index=False)
    snapshot = {key: value for key, value in cfg.items() if not str(key).startswith("_")}
    snapshot["selected_model"] = {"name": name, "params": params, "threshold": threshold}
    with (report_dir / "experiment_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(snapshot, handle, allow_unicode=True, sort_keys=False)
    (report_dir / "summary.md").write_text(
        _summary(name, params, threshold, validation_metrics, len(features), len(train), len(validation), importance, variant, len(excluded)),
        encoding="utf-8",
    )
    (report_dir / "selection.json").write_text(json.dumps({"candidates": scored}, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_dir


def load_bundle(cfg: dict, name: str, variant: str = "full") -> dict:
    path = _bundle_path(cfg, name, variant)
    if not path.exists():
        raise FileNotFoundError(f"Нет обученной модели {variant}/{name}. Сначала выполните train --model {name} --variant {variant}.")
    return joblib.load(path)


def _bundle_path(cfg: dict, name: str, variant: str) -> Path:
    root = resolve_path(cfg, "models")
    if variant == "full":
        return root / f"{name}.joblib"
    if variant in {"no_energy", "shape", "voicing", "mfcc", "spectral"}:
        return root / variant / f"{name}.joblib"
    raise ValueError(f"Неизвестный вариант модели: {variant}")


def _reject_foreign_features(variant: str, features: list[str]) -> None:
    if variant == "no_energy" and any(is_log_energy(feature) for feature in features):
        raise RuntimeError("В матрицу варианта no_energy попал нулевой кепстр или RMS")
    if variant == "shape" and any(not is_shape(feature) for feature in features):
        raise RuntimeError("В матрицу варианта shape попал признак вне кепстральной формы и спектра")
    if variant == "voicing" and any(not is_voicing(feature) for feature in features):
        raise RuntimeError("В матрицу варианта voicing попал признак вне F0 и качества голоса")
    if variant == "mfcc" and any(not is_mfcc_shape(feature) for feature in features):
        raise RuntimeError("В матрицу варианта mfcc попал признак вне MFCC начиная с коэффициента 1")
    if variant == "spectral" and any(not is_spectral_measure(feature) for feature in features):
        raise RuntimeError("В матрицу варианта spectral попал признак вне спектральных мер")


def _exclusion_rule(variant: str) -> str:
    if variant == "no_energy":
        return "До отбора убраны energy_rms и нулевой коэффициент LFCC, MFCC, CQCC: static, delta и delta-delta."
    if variant == "shape":
        return "Оставлены коэффициенты LFCC, MFCC и CQCC начиная с 1 и спектральные меры. Энергия, F0 и качество голоса убраны до отбора."
    if variant == "voicing":
        return "Оставлены F0, доля озвученных кадров, автокорреляционная гармоничность, jitter, shimmer и HNR. Кепстр и спектр убраны до отбора."
    if variant == "mfcc":
        return "Оставлены только MFCC начиная с коэффициента 1. Энергия, LFCC, CQCC, спектр, F0 и качество голоса убраны до отбора."
    if variant == "spectral":
        return "Оставлены центроид, полоса, rolloff, flatness и zero-crossing rate. Кепстр, энергия, F0 и качество голоса убраны до отбора."
    return "Энергетические коэффициенты оставлены."


def score_frame(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    return _positive_scores(bundle["model"], frame[bundle["features"]], bundle["name"])


def predict_frame(bundle: dict, scores: np.ndarray) -> np.ndarray:
    return _predict_from_scores(scores, float(bundle["threshold"]))


def _candidates(name: str, model_cfg: dict, seed: int):
    if name == "logistic":
        return [({"C": float(value)}, logistic_regression.build(float(value))) for value in model_cfg["logistic_C"]]
    if name == "svm":
        pairs = []
        for C in model_cfg["svm_C"]:
            for gamma in model_cfg["svm_gamma"]:
                gamma_value = gamma if isinstance(gamma, str) else float(gamma)
                pairs.append(({"C": float(C), "gamma": gamma_value}, svm.build(float(C), gamma_value)))
        return pairs
    if name == "lightgbm":
        pairs = []
        for trees in model_cfg["lightgbm_n_estimators"]:
            for leaves in model_cfg["lightgbm_num_leaves"]:
                for rate in model_cfg["lightgbm_learning_rate"]:
                    for child in model_cfg["lightgbm_min_child_samples"]:
                        params = {
                            "n_estimators": int(trees),
                            "num_leaves": int(leaves),
                            "learning_rate": float(rate),
                            "min_child_samples": int(child),
                        }
                        pairs.append((params, lightgbm_model.build(seed=seed, **params)))
        return pairs
    raise ValueError(name)


def _positive_scores(estimator, matrix: pd.DataFrame, name: str) -> np.ndarray:
    if name == "svm":
        return np.asarray(estimator.decision_function(matrix), dtype=float)
    probabilities = estimator.predict_proba(matrix)
    return np.asarray(probabilities[:, 1], dtype=float)


def _default_threshold(name: str) -> float:
    return 0.0 if name == "svm" else 0.5


def _predict_from_scores(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(scores) >= threshold).astype(int)


def _best_threshold(y_true: np.ndarray, scores: np.ndarray, default: float) -> float:
    candidates = np.unique(np.concatenate([np.quantile(scores, np.linspace(0.02, 0.98, 49)), [default]]))
    best_threshold = float(candidates[0])
    best_rank = (-1.0, -1.0)
    for threshold in candidates:
        predicted = _predict_from_scores(scores, float(threshold))
        accuracy = balanced_accuracy_score(y_true, predicted)
        rank = (accuracy, roc_auc_score(y_true, scores) if len(np.unique(y_true)) == 2 else 0.0)
        if rank > best_rank:
            best_rank = rank
            best_threshold = float(threshold)
    return best_threshold


def _complexity(name: str, params: dict) -> float:
    if name == "logistic":
        return float(params["C"])
    if name == "svm":
        return float(params["C"])
    return float(params["n_estimators"]) * float(params["num_leaves"])


def _importance(name: str, estimator, matrix: pd.DataFrame, y_true: np.ndarray, threshold: float, features: list[str]) -> pd.DataFrame:
    rows = []
    if name == "logistic":
        coefficients = estimator.named_steps["clf"].coef_[0]
        for feature, value in zip(features, coefficients):
            rows.append({"feature": feature, "importance": float(value), "importance_type": "standardized_coefficient"})
    elif name == "lightgbm":
        gain = estimator.booster_.feature_importance(importance_type="gain")
        for feature, value in zip(features, gain):
            rows.append({"feature": feature, "importance": float(value), "importance_type": "gain"})
    result = permutation_importance(
        estimator,
        matrix,
        y_true,
        scoring=_threshold_scorer(threshold),
        n_repeats=5,
        random_state=42,
        n_jobs=1,
    )
    for feature, value in zip(features, result.importances_mean):
        rows.append({"feature": feature, "importance": float(value), "importance_type": "permutation_balanced_accuracy"})
    table = pd.DataFrame(rows)
    table["note"] = _importance_note(name)
    return table.sort_values(["importance_type", "importance"], ascending=[True, False])


def _threshold_scorer(threshold: float):
    def scorer(estimator, matrix, y_true):
        if hasattr(estimator, "decision_function") and estimator.__class__.__name__ == "Pipeline" and "SVC" in estimator.named_steps["clf"].__class__.__name__:
            scores = estimator.decision_function(matrix)
        elif hasattr(estimator, "predict_proba"):
            scores = estimator.predict_proba(matrix)[:, 1]
        else:
            scores = estimator.decision_function(matrix)
        return balanced_accuracy_score(y_true, (np.asarray(scores) >= threshold).astype(int))

    return scorer


def _importance_note(name: str) -> str:
    if name == "logistic":
        return "Коэффициент относится к стандартизованному признаку. Знак не доказывает причину синтетической речи."
    if name == "svm":
        return "У RBF SVM нет одного коэффициента на признак. Permutation importance — ассоциация, не причина."
    return "Gain и permutation importance показывают вклад в эту модель, а не причину синтетической речи."


def _reject_locked_rows(frame: pd.DataFrame, split: str) -> None:
    locked = set(frame["locked_split"].astype(str))
    if split in {"train", "validation"} and locked & {"test", "spectra_threshold"}:
        raise RuntimeError(f"{split} содержит зафиксированный test или n100")


def _summary(name, params, threshold, metrics, n_features, n_train, n_validation, importance, variant, n_excluded) -> str:
    top = importance[importance["importance_type"] == "permutation_balanced_accuracy"].head(15)
    lines = [
        f"# {name}",
        "",
        "Порог и параметры выбраны на validation. Test в этом запуске не использовался.",
        "Score не является калиброванной вероятностью.",
        f"Вариант: {variant}.",
        _exclusion_rule(variant),
        f"Исключено колонок до отбора: {n_excluded}.",
        "",
        f"- train: {n_train}",
        f"- validation: {n_validation}",
        f"- признаков: {n_features}",
        f"- параметры: {params}",
        f"- порог machine: {threshold}",
        "",
        "## Validation",
        "",
    ]
    for key in (
        "human", "machine", "recall_synthetic", "specificity_human", "precision_synthetic",
        "f1_synthetic", "balanced_accuracy", "fpr", "fnr", "roc_auc", "pr_auc",
    ):
        lines.append(f"- {key}: {metrics[key]}")
    lines.extend(["", "## Permutation importance, первые 15", ""])
    for row in top.itertuples(index=False):
        lines.append(f"- {row.feature}: {row.importance:.5f}")
    lines.extend(["", top.iloc[0].note if len(top) else "", ""])
    return "\n".join(lines)
