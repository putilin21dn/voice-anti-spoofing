"""SHAP замороженных классических моделей на validation, test и n100.

Признаки для повторного обучения берутся только из SHAP на validation.
Test и n100 в этот отбор не входят. Файлы models/*.joblib не перезаписываются.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score

from src.models import lightgbm as lightgbm_model
from src.models import logistic_regression, svm
from src.models.dataset import load_call_frame, modeling_rows
from src.models.evaluation import binary_metrics
from src.models.features import is_log_energy, is_spectral_measure, is_voicing
from src.models.training import (
    _best_threshold,
    _default_threshold,
    _positive_scores,
    _predict_from_scores,
    load_bundle,
    score_frame,
)
from src.utils.config import resolve_path
from src.utils.reports import next_numbered_dir, write_json


MODEL_NAMES = ("logistic", "svm", "lightgbm")
SPLIT_NAMES = ("validation", "test", "spectra_threshold")
SPLIT_TITLE = {"validation": "validation", "test": "test", "spectra_threshold": "n100"}
PREFIX_SIZES = (50, 100, 150)
SVM_PERMUTATIONS = 5
_CEP = re.compile(r"^(?:lfcc|mfcc|cqcc)_(static|delta|delta2)_(\d+)_")


def write_shap_report(cfg: dict) -> Path:
    started = time.perf_counter()
    frame = load_call_frame(cfg)
    parts = {name: modeling_rows(frame, name) for name in ("train", *SPLIT_NAMES)}
    _assert_disjoint(parts)
    labels = {name: (part["label"] == "machine").astype(int).to_numpy() for name, part in parts.items()}
    bundles = {name: load_bundle(cfg, name, "full") for name in MODEL_NAMES}
    feature_rows: list[dict] = []
    check_rows: list[dict] = []
    phi_validation: dict[str, tuple[list[str], np.ndarray]] = {}
    for name in MODEL_NAMES:
        features = list(bundles[name]["features"])
        print(f"shap {name} features {len(features)}", flush=True)
        for split in SPLIT_NAMES:
            phi, base, score, method = _shap(name, bundles[name], parts["train"], parts[split], features)
            if phi.shape != (len(parts[split]), len(features)):
                raise RuntimeError(f"Неожиданная форма SHAP {name}/{split}: {phi.shape}")
            error = float(np.max(np.abs(base + phi.sum(axis=1) - score)))
            check_rows.append({
                "model": name,
                "split": split,
                "method": method,
                "rows": int(len(parts[split])),
                "features": len(features),
                "max_additivity_error": error,
            })
            if error > 1e-4:
                raise RuntimeError(f"SHAP {name}/{split} не сходится с выходом модели: {error}")
            feature_rows.extend(_feature_rows(name, split, features, phi, labels[split]))
            if split == "validation":
                phi_validation[name] = (features, phi)
            print(f"  {split} additivity {error:.2e}", flush=True)
    feature_table = pd.DataFrame(feature_rows)
    group_table = _group_table(feature_table)
    ablation_table, retrain_table = _reduction_experiments(cfg, bundles, parts, labels, phi_validation)
    report_dir = next_numbered_dir(resolve_path(cfg, cfg["paths"]["reports_dir"]), "shap")
    feature_table.to_csv(report_dir / "feature_shap.csv", index=False)
    group_table.to_csv(report_dir / "group_shap.csv", index=False)
    ablation_table.to_csv(report_dir / "frozen_ablation.csv", index=False)
    retrain_table.to_csv(report_dir / "retrain.csv", index=False)
    same_features = len({tuple(bundles[name]["features"]) for name in MODEL_NAMES}) == 1
    payload = {
        "variant": "full",
        "models_overwritten": False,
        "feature_lists_identical": same_features,
        "selection_split": "validation",
        "held_out_splits": ["test", "spectra_threshold"],
        "svm_permutations": SVM_PERMUTATIONS,
        "additivity": check_rows,
        "elapsed_s": round(time.perf_counter() - started, 1),
        "note": (
            "Вклад посчитан на марже модели: log-odds у логистической регрессии и LightGBM, "
            "decision_function у SVM. Это не калиброванная вероятность. "
            "Повторное обучение использует параметры уже выбранной модели и заново выбирает порог на validation."
        ),
    }
    write_json(report_dir / "metrics.json", payload)
    text = _summary(feature_table, group_table, ablation_table, retrain_table, payload, parts, labels)
    (report_dir / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    print("report", report_dir)
    return report_dir


def describe_feature(name: str) -> dict[str, str]:
    family = _family(name)
    kind = ""
    index = ""
    match = _CEP.match(name)
    if match is not None:
        kind = match.group(1)
        index = str(int(match.group(2)))
    return {
        "family": family,
        "call_agg": name.rsplit("__", 1)[1] if "__" in name else "",
        "cep_kind": kind,
        "cep_index": index,
        "cep_bucket": _cep_bucket(family, index),
    }


def coverage_count(values: np.ndarray, level: float) -> int:
    ordered = np.sort(np.asarray(values, dtype=float))[::-1]
    total = float(ordered.sum())
    if total <= 0 or ordered.size == 0:
        return 0
    cumulative = np.cumsum(ordered) / total
    return int(np.searchsorted(cumulative, level, side="left") + 1)


def _family(name: str) -> str:
    if is_log_energy(name):
        return "energy"
    if name.startswith("lfcc_"):
        return "lfcc"
    if name.startswith("mfcc_"):
        return "mfcc"
    if name.startswith("cqcc_"):
        return "cqcc"
    if is_voicing(name):
        return "voicing"
    if is_spectral_measure(name):
        return "spectral"
    return "other"


def _cep_bucket(family: str, index: str) -> str:
    if family == "energy":
        return "c0"
    if index == "":
        return ""
    number = int(index)
    if number <= 4:
        return "c1-4"
    if number <= 9:
        return "c5-9"
    return "c10-19"


def _assert_disjoint(parts: dict[str, pd.DataFrame]) -> None:
    seen: dict[str, str] = {}
    for split, part in parts.items():
        for call_id in part["call_id"].astype(str):
            if call_id in seen:
                raise RuntimeError(f"call_id {call_id} одновременно в {seen[call_id]} и {split}")
            seen[call_id] = split


def _shap(name: str, bundle: dict, train: pd.DataFrame, frame: pd.DataFrame, features: list[str]):
    model = bundle["model"]
    matrix = frame[features]
    if name == "logistic":
        return (*_linear_shap(model, train[features], matrix), "linear_scaled_train_mean")
    if name == "lightgbm":
        return (*_tree_shap(model, matrix), "lightgbm_pred_contrib")
    return (*_svm_shap(model, train[features], matrix), f"permutation_{SVM_PERMUTATIONS}_vs_train_median")


def _linear_shap(model, train: pd.DataFrame, frame: pd.DataFrame):
    imputer = model.named_steps["imputer"]
    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["clf"]
    train_z = scaler.transform(imputer.transform(train))
    frame_z = scaler.transform(imputer.transform(frame))
    center = train_z.mean(axis=0)
    coefficient = np.asarray(classifier.coef_[0], dtype=float)
    phi = (frame_z - center) * coefficient
    base = np.full(len(frame), float(classifier.intercept_[0] + center @ coefficient))
    score = np.asarray(model.decision_function(frame), dtype=float)
    return phi, base, score


def _tree_shap(model, frame: pd.DataFrame):
    contrib = np.asarray(model.booster_.predict(frame, pred_contrib=True))
    if contrib.ndim == 3:
        contrib = contrib[:, 1, :]
    phi = contrib[:, :-1]
    base = contrib[:, -1]
    score = np.asarray(model.predict(frame, raw_score=True), dtype=float)
    return phi, base, score


def _svm_shap(model, train: pd.DataFrame, frame: pd.DataFrame):
    imputer = model.named_steps["imputer"]
    columns = list(frame.columns)
    train_i = imputer.transform(train)
    frame_i = imputer.transform(frame)
    background = np.median(train_i, axis=0).reshape(1, -1)

    def predict(data: np.ndarray) -> np.ndarray:
        table = pd.DataFrame(np.asarray(data), columns=columns)
        return np.asarray(model.decision_function(table), dtype=float)

    explainer = shap.PermutationExplainer(predict, background, seed=42)
    width = 2 * len(columns) + 1
    max_evals = SVM_PERMUTATIONS * width
    blocks = []
    bases = []
    for start in range(0, len(frame_i), 50):
        explanation = explainer(frame_i[start:start + 50], max_evals=max_evals, silent=True)
        blocks.append(np.asarray(explanation.values, dtype=float))
        bases.append(np.reshape(np.asarray(explanation.base_values, dtype=float), (-1,)))
        print(f"  svm {min(start + 50, len(frame_i))}/{len(frame_i)}", flush=True)
    phi = np.vstack(blocks)
    base = np.concatenate(bases)
    score = predict(frame_i)
    return phi, base, score


def _feature_rows(model: str, split: str, features: list[str], phi: np.ndarray, y: np.ndarray) -> list[dict]:
    mean_shap = phi.mean(axis=0)
    mean_abs = np.abs(phi).mean(axis=0)
    total = float(mean_abs.sum())
    machine = phi[y == 1].mean(axis=0) if np.any(y == 1) else np.zeros(len(features))
    human = phi[y == 0].mean(axis=0) if np.any(y == 0) else np.zeros(len(features))
    gap = machine - human
    order_abs = np.argsort(-mean_abs, kind="mergesort")
    order_gap = np.argsort(-np.abs(gap), kind="mergesort")
    rank_abs = np.empty(len(features), dtype=int)
    rank_gap = np.empty(len(features), dtype=int)
    rank_abs[order_abs] = np.arange(1, len(features) + 1)
    rank_gap[order_gap] = np.arange(1, len(features) + 1)
    rows = []
    for index, feature in enumerate(features):
        described = describe_feature(feature)
        rows.append({
            "model": model,
            "split": split,
            "feature": feature,
            **described,
            "mean_shap": float(mean_shap[index]),
            "mean_abs": float(mean_abs[index]),
            "share": float(mean_abs[index] / total) if total else 0.0,
            "rank_abs": int(rank_abs[index]),
            "mean_shap_machine": float(machine[index]),
            "mean_shap_human": float(human[index]),
            "class_gap": float(gap[index]),
            "rank_gap": int(rank_gap[index]),
        })
    return rows


def _group_table(feature_table: pd.DataFrame) -> pd.DataFrame:
    totals = feature_table.groupby(["model", "split"])["mean_abs"].sum().rename("total")
    frames = []
    for column in ("family", "call_agg", "cep_kind", "cep_bucket"):
        part = feature_table[feature_table[column].astype(str) != ""].copy()
        grouped = part.groupby(["model", "split", column], as_index=False)["mean_abs"].sum()
        grouped = grouped.rename(columns={column: "group", "mean_abs": "mean_abs_sum"})
        grouped["grouping"] = column
        grouped = grouped.merge(totals.reset_index(), on=["model", "split"], how="left")
        grouped["share"] = grouped["mean_abs_sum"] / grouped["total"]
        frames.append(grouped.drop(columns=["total"]))
    return pd.concat(frames, ignore_index=True)


def _reduction_experiments(cfg, bundles, parts, labels, phi_validation):
    ablation_rows = []
    retrain_rows = []
    for name in MODEL_NAMES:
        features, phi = phi_validation[name]
        y_validation = labels["validation"]
        rankings = {
            "mean_abs": [features[index] for index in np.argsort(-np.abs(phi).mean(axis=0), kind="mergesort")],
            "class_gap": [features[index] for index in np.argsort(-np.abs(_class_gap(phi, y_validation)), kind="mergesort")],
        }
        sizes = set(PREFIX_SIZES)
        for ranked in rankings.values():
            shares = _share_in_order(phi, features, ranked)
            sizes.add(coverage_count(shares, 0.95))
        original_scores = {
            split: score_frame(bundles[name], parts[split])
            for split in SPLIT_NAMES
        }
        original_threshold = float(bundles[name]["threshold"])
        original_metrics = {
            split: binary_metrics(labels[split], original_scores[split], _predict_from_scores(original_scores[split], original_threshold))
            for split in SPLIT_NAMES
        }
        fill = parts["train"][features].median(numeric_only=True)
        for rank_by, ranked in rankings.items():
            for size in sorted(sizes):
                if size >= len(features):
                    continue
                kept = ranked[:size]
                ablation_rows.append(_ablation_row(
                    name, rank_by, size, kept, features, fill, bundles[name], parts, labels, original_scores, original_threshold,
                ))
                retrain_rows.append(_retrain_row(
                    cfg, name, rank_by, size, kept, bundles[name], parts, labels, original_metrics, original_scores,
                ))
        retrain_rows.append(_full_row(name, len(features), original_metrics))
    return pd.DataFrame(ablation_rows), pd.DataFrame(retrain_rows)


def _class_gap(phi: np.ndarray, y: np.ndarray) -> np.ndarray:
    return phi[y == 1].mean(axis=0) - phi[y == 0].mean(axis=0)


def _share_in_order(phi: np.ndarray, features: list[str], ranked: list[str]) -> np.ndarray:
    position = {feature: index for index, feature in enumerate(features)}
    mean_abs = np.abs(phi).mean(axis=0)
    return np.asarray([mean_abs[position[feature]] for feature in ranked], dtype=float)


def _ablation_row(name, rank_by, size, kept, features, fill, bundle, parts, labels, original_scores, threshold):
    row = {"model": name, "rank_by": rank_by, "k": int(size), "threshold": threshold}
    kept_set = set(kept)
    for split in SPLIT_NAMES:
        edited = parts[split][features].copy()
        for feature in features:
            if feature not in kept_set:
                edited[feature] = fill[feature]
        scored = parts[split].copy()
        scored[features] = edited
        scores = score_frame(bundle, scored)
        metrics = binary_metrics(labels[split], scores, _predict_from_scores(scores, threshold))
        original_pred = _predict_from_scores(original_scores[split], threshold)
        title = SPLIT_TITLE[split]
        row[f"{title}_roc_auc"] = metrics["roc_auc"]
        row[f"{title}_balanced_accuracy"] = metrics["balanced_accuracy"]
        row[f"{title}_flips"] = int(np.sum(_predict_from_scores(scores, threshold) != original_pred))
    return row


def _retrain_row(cfg, name, rank_by, size, kept, bundle, parts, labels, original_metrics, original_scores):
    estimator = _rebuild(name, bundle, int(cfg["seed"]))
    estimator.fit(parts["train"][kept], labels["train"])
    validation_scores = _positive_scores(estimator, parts["validation"][kept], name)
    threshold = _best_threshold(labels["validation"], validation_scores, _default_threshold(name))
    row = {
        "model": name,
        "rank_by": rank_by,
        "k": int(size),
        "retrained": True,
        "threshold": threshold,
        "selected_on": "validation_roc_auc",
    }
    for split in SPLIT_NAMES:
        scores = _positive_scores(estimator, parts[split][kept], name)
        metrics = binary_metrics(labels[split], scores, _predict_from_scores(scores, threshold))
        title = SPLIT_TITLE[split]
        row[f"{title}_roc_auc"] = metrics["roc_auc"]
        row[f"{title}_balanced_accuracy"] = metrics["balanced_accuracy"]
        row[f"{title}_recall_synthetic"] = metrics["recall_synthetic"]
        row[f"{title}_specificity_human"] = metrics["specificity_human"]
        row[f"{title}_roc_auc_delta"] = metrics["roc_auc"] - original_metrics[split]["roc_auc"]
        row[f"{title}_balanced_accuracy_delta"] = metrics["balanced_accuracy"] - original_metrics[split]["balanced_accuracy"]
        if split != "validation":
            low, mid, high = _bootstrap_delta(labels[split], scores, original_scores[split])
            row[f"{title}_auc_delta_p2_5"] = low
            row[f"{title}_auc_delta_p50"] = mid
            row[f"{title}_auc_delta_p97_5"] = high
    print(f"retrain {name} {rank_by} k={size} val_auc={row['validation_roc_auc']:.3f}", flush=True)
    return row


def _full_row(name: str, size: int, original_metrics: dict) -> dict:
    row = {
        "model": name,
        "rank_by": "full",
        "k": int(size),
        "retrained": False,
        "threshold": None,
        "selected_on": "validation_roc_auc",
    }
    for split in SPLIT_NAMES:
        title = SPLIT_TITLE[split]
        metrics = original_metrics[split]
        row[f"{title}_roc_auc"] = metrics["roc_auc"]
        row[f"{title}_balanced_accuracy"] = metrics["balanced_accuracy"]
        row[f"{title}_recall_synthetic"] = metrics["recall_synthetic"]
        row[f"{title}_specificity_human"] = metrics["specificity_human"]
        row[f"{title}_roc_auc_delta"] = 0.0
        row[f"{title}_balanced_accuracy_delta"] = 0.0
    return row


def _rebuild(name: str, bundle: dict, seed: int):
    params = bundle["params"]
    if name == "logistic":
        return logistic_regression.build(float(params["C"]))
    if name == "svm":
        gamma = params["gamma"]
        gamma_value = gamma if isinstance(gamma, str) else float(gamma)
        return svm.build(float(params["C"]), gamma_value)
    return lightgbm_model.build(seed=seed, **{key: params[key] for key in params})


def _bootstrap_delta(y: np.ndarray, scores: np.ndarray, baseline: np.ndarray, draws: int = 2000) -> tuple[float, float, float]:
    rng = np.random.default_rng(42)
    deltas = []
    count = len(y)
    for _ in range(draws):
        index = rng.integers(0, count, count)
        if len(np.unique(y[index])) < 2:
            continue
        deltas.append(roc_auc_score(y[index], scores[index]) - roc_auc_score(y[index], baseline[index]))
    quantiles = np.quantile(deltas, [0.025, 0.5, 0.975])
    return (float(quantiles[0]), float(quantiles[1]), float(quantiles[2]))


def _summary(feature_table, group_table, ablation_table, retrain_table, payload, parts, labels) -> str:
    lines = [
        "# SHAP классических моделей",
        "",
        "Замороженные модели `models/{logistic,svm,lightgbm}.joblib`, вариант full. Файлы моделей не перезаписывались.",
        "Вклад посчитан на марже: log-odds у логистической регрессии и LightGBM, decision_function у SVM.",
        "Фон логистической регрессии — среднее train после стандартизации. Фон SVM — медиана train после импутации, 5 перестановок.",
        "LightGBM — TreeSHAP из `pred_contrib`.",
        "Коррелирующие признаки делят один вклад, поэтому маленький SHAP сам по себе ещё не повод удалять колонку. Проверка — повторное обучение ниже.",
        "",
        "Префикс признаков для повторного обучения взят из SHAP на validation. Test и n100 в выбор k не входили. Порог урезанной модели заново выбран на validation, параметры сетки оставлены прежними.",
        "",
        "## Объёмы",
        "",
    ]
    for split in SPLIT_NAMES:
        y = labels[split]
        lines.append(f"- {SPLIT_TITLE[split]}: {len(parts[split])} звонков, human {(y == 0).sum()}, machine {(y == 1).sum()}")
    lines.extend(["", "## Сходимость", ""])
    for row in payload["additivity"]:
        lines.append(
            f"- {row['model']} / {SPLIT_TITLE[row['split']]}: {row['method']}, максимальная ошибка суммы вкладов {row['max_additivity_error']:.2e}"
        )
    lines.extend(["", "## Сколько признаков несут вклад", ""])
    lines.append("Число признаков, которые покрывают долю суммарного среднего |SHAP|.")
    lines.append("")
    lines.append("| Модель | Набор | 50% | 80% | 95% | 99% |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for model in MODEL_NAMES:
        for split in SPLIT_NAMES:
            values = feature_table[(feature_table["model"] == model) & (feature_table["split"] == split)]["mean_abs"].to_numpy()
            counts = [coverage_count(values, level) for level in (0.5, 0.8, 0.95, 0.99)]
            lines.append(f"| {model} | {SPLIT_TITLE[split]} | {counts[0]} | {counts[1]} | {counts[2]} | {counts[3]} |")
    lines.extend([
        "",
        "## Группы на n100",
        "",
        "Доля среднего |SHAP| от суммы по всем признакам модели. n100 здесь устойчивее test: 190 звонков против 43.",
        "Кепстральные таблицы ниже не включают спектральные меры и озвученность, поэтому их доли не складываются в 1.",
        "",
    ])
    lines.extend(_group_markdown(group_table, "family", "n100"))
    lines.extend(["", "## Кепстр по номеру коэффициента, n100", ""])
    lines.extend(_group_markdown(group_table, "cep_bucket", "n100"))
    lines.extend(["", "## Агрегат звонка, n100", ""])
    lines.extend(_group_markdown(group_table, "call_agg", "n100"))
    lines.extend(["", "## Первые 10 признаков на n100", ""])
    for model in MODEL_NAMES:
        lines.append(f"### {model}")
        lines.append("")
        subset = feature_table[(feature_table["model"] == model) & (feature_table["split"] == "spectra_threshold")]
        subset = subset.sort_values("rank_abs").head(10)
        test_share = feature_table[(feature_table["model"] == model) & (feature_table["split"] == "test")].set_index("feature")["share"]
        for row in subset.itertuples(index=False):
            lines.append(f"- {row.feature}: доля {row.share:.3f}, на test {float(test_share.get(row.feature, float('nan'))):.3f}, зазор классов {row.class_gap:.3f}")
        lines.append("")
    lines.extend(["## Согласие рангов |SHAP|", ""])
    for model in MODEL_NAMES:
        left = feature_table[(feature_table["model"] == model) & (feature_table["split"] == "test")].set_index("feature")["mean_abs"]
        right = feature_table[(feature_table["model"] == model) & (feature_table["split"] == "spectra_threshold")].set_index("feature")["mean_abs"]
        aligned = pd.concat([left, right], axis=1, keys=["test", "n100"]).dropna()
        spearman = float(aligned["test"].corr(aligned["n100"], method="spearman"))
        lines.append(f"- {model}, test против n100: Spearman {spearman:.3f}")
    if payload["feature_lists_identical"]:
        lines.append("- Списки 200 признаков у трёх моделей совпадают.")
    lines.extend(["", "## Замороженная модель, хвост заменён медианой train", ""])
    lines.append("Модель не переобучалась. Признаки вне префикса validation-|SHAP| заменены медианой train. Порог прежний.")
    lines.append("")
    lines.append("| Модель | k | test AUC | test BA | test смен | n100 AUC | n100 BA | n100 смен |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    frozen = ablation_table[ablation_table["rank_by"] == "mean_abs"].sort_values(["model", "k"])
    for row in frozen.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.k} | {row.test_roc_auc:.3f} | {row.test_balanced_accuracy:.3f} | {row.test_flips} | "
            f"{row.n100_roc_auc:.3f} | {row.n100_balanced_accuracy:.3f} | {row.n100_flips} |"
        )
    lines.extend(["", "## Повторное обучение на префиксе", ""])
    lines.append("k — сколько первых признаков validation-ранжирования оставлено. `full` — текущая модель без переобучения.")
    lines.append("Разница AUC на test и n100 сопровождается бутстрепом 2000 перевыборок, интервал 2.5–97.5%.")
    lines.append("")
    for model in MODEL_NAMES:
        lines.append(f"### {model}")
        lines.append("")
        lines.append("| Отбор | k | val AUC | test AUC | n100 AUC | test BA | n100 BA | n100 ΔAUC | интервал n100 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        subset = retrain_table[retrain_table["model"] == model].sort_values(["rank_by", "k"])
        for row in subset.itertuples(index=False):
            interval = ""
            if row.rank_by != "full":
                interval = f"[{row.n100_auc_delta_p2_5:.3f}, {row.n100_auc_delta_p97_5:.3f}]"
            lines.append(
                f"| {row.rank_by} | {row.k} | {row.validation_roc_auc:.3f} | {row.test_roc_auc:.3f} | {row.n100_roc_auc:.3f} | "
                f"{row.test_balanced_accuracy:.3f} | {row.n100_balanced_accuracy:.3f} | {row.n100_roc_auc_delta:+.3f} | {interval} |"
            )
        chosen = _choose(subset)
        lines.append("")
        lines.append(
            f"По validation ROC-AUC остаётся `{chosen.rank_by}`, k={int(chosen.k)}: "
            f"test BA {chosen.test_balanced_accuracy:.3f}, n100 BA {chosen.n100_balanced_accuracy:.3f}, "
            f"n100 AUC {chosen.n100_roc_auc:.3f}."
        )
        lines.append("")
    lines.extend([
        "## Вывод",
        "",
        _conclusion(retrain_table),
        "",
    ])
    note = _balanced_accuracy_note(retrain_table)
    if note:
        lines.extend([note, ""])
    lines.extend([
        "Один неверный звонок на test меняет balanced accuracy примерно на 0.02. Для решения о признаках опираться нужно на n100 и на интервал разницы AUC.",
        "Маленький вклад отдельного признака на validation раньше уже был виден в permutation importance: у 200 связанных колонок удаление одной почти не двигает качество.",
        "",
    ])
    return "\n".join(lines)


def _choose(subset: pd.DataFrame) -> pd.Series:
    full_auc = float(subset.loc[subset["rank_by"] == "full", "validation_roc_auc"].iloc[0])
    better = subset[subset["validation_roc_auc"] > full_auc + 1e-12]
    if better.empty:
        return subset[subset["rank_by"] == "full"].iloc[0]
    return better.sort_values(["validation_roc_auc", "k"], ascending=[False, True]).iloc[0]


def _group_markdown(group_table: pd.DataFrame, grouping: str, split_title: str) -> list[str]:
    split = {title: name for name, title in SPLIT_TITLE.items()}[split_title]
    part = group_table[(group_table["grouping"] == grouping) & (group_table["split"] == split)]
    preferred = [
        "energy", "lfcc", "mfcc", "cqcc", "spectral", "voicing", "other",
        "c0", "c1-4", "c5-9", "c10-19",
        "static", "delta", "delta2",
        "mean", "median", "std", "min", "max", "p25", "p75",
    ]
    groups = sorted(part["group"].unique(), key=lambda group: (preferred.index(group) if group in preferred else 99, str(group)))
    lines = ["| Группа | logistic | svm | lightgbm |", "| --- | ---: | ---: | ---: |"]
    for group in groups:
        cells = []
        for model in MODEL_NAMES:
            values = part[(part["group"] == group) & (part["model"] == model)]["share"]
            cells.append(f"{float(values.iloc[0]):.3f}" if len(values) else "")
        lines.append(f"| {group} | {cells[0]} | {cells[1]} | {cells[2]} |")
    return lines


def _balanced_accuracy_note(retrain_table: pd.DataFrame) -> str:
    sentences = []
    for model in MODEL_NAMES:
        subset = retrain_table[retrain_table["model"] == model]
        full = subset[subset["rank_by"] == "full"].iloc[0]
        reduced = subset[subset["rank_by"] != "full"]
        candidates = reduced[
            (reduced["validation_balanced_accuracy"] + 1e-9 >= full["validation_balanced_accuracy"])
            & (reduced["n100_balanced_accuracy"] > full["n100_balanced_accuracy"] + 0.01)
        ]
        if candidates.empty:
            continue
        row = candidates.sort_values(["n100_balanced_accuracy", "k"], ascending=[False, True]).iloc[0]
        sentences.append(
            f"{model}, {row.rank_by} k={int(row.k)}: validation balanced accuracy {row.validation_balanced_accuracy:.3f} "
            f"при полной модели {full.validation_balanced_accuracy:.3f}, на n100 {full.n100_balanced_accuracy:.3f} → {row.n100_balanced_accuracy:.3f}. "
            f"ROC-AUC на n100 остаётся {full.n100_roc_auc:.3f} → {row.n100_roc_auc:.3f}. Сдвигается порог, порядок скоров тот же."
        )
    return " ".join(sentences)


def _conclusion(retrain_table: pd.DataFrame) -> str:
    sentences = []
    any_gain = False
    for model in MODEL_NAMES:
        subset = retrain_table[retrain_table["model"] == model]
        chosen = _choose(subset)
        full = subset[subset["rank_by"] == "full"].iloc[0]
        if chosen.rank_by == "full":
            sentences.append(
                f"{model}: на validation ни один урезанный префикс не обогнал полный набор по ROC-AUC "
                f"({full.validation_roc_auc:.3f})."
            )
            continue
        any_gain = True
        holds = chosen.n100_auc_delta_p2_5 > 0
        sentences.append(
            f"{model}: validation выбирает {chosen.rank_by} k={int(chosen.k)}. "
            f"На n100 AUC {full.n100_roc_auc:.3f} → {chosen.n100_roc_auc:.3f}, "
            f"интервал разницы [{chosen.n100_auc_delta_p2_5:.3f}, {chosen.n100_auc_delta_p97_5:.3f}]. "
            f"На test AUC {full.test_roc_auc:.3f} → {chosen.test_roc_auc:.3f}, "
            f"интервал [{chosen.test_auc_delta_p2_5:.3f}, {chosen.test_auc_delta_p97_5:.3f}]. "
            + (
                "Нижняя граница на n100 выше нуля."
                if holds
                else "Интервал на n100 включает ноль или отрицательные значения, устойчивого прироста нет."
            )
        )
    if not any_gain:
        sentences.append(
            "Убирать хвост признаков, чтобы поднять качество этих трёх моделей, по сетке k=50/100/150 и по префиксу 95% вклада не получается: validation оставляет полный набор."
        )
    return " ".join(sentences)
