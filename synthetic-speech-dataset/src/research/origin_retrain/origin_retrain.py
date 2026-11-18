"""Новые модели происхождения на дне 13 ноября. День 16 ноября в подбор не входит.

Запускать интерпретатором classical-anti-spoofing, чтобы `src` был его кодом:
PYTHONPATH указывает на classical-anti-spoofing, не на этот репозиторий.
Старые bundle и порог Spectra 1.02 не перезаписываются.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from src.aggregation.call_aggregation import aggregate_calls
from src.features.pipeline import _one_call
from src.models import lightgbm as lightgbm_model
from src.models import logistic_regression, svm
from src.models.evaluation import binary_metrics
from src.models.features import select_features
from src.utils.config import load_config


DATASET = Path(__file__).resolve().parents[3]


def _component(name: str) -> Path:
    import importlib.util

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "repo_paths.py"
        if candidate.is_file() and (parent / "pyproject.toml").is_file():
            spec = importlib.util.spec_from_file_location("voice_repo_paths", candidate)
            module = importlib.util.module_from_spec(spec)
            if spec.loader is None:
                break
            spec.loader.exec_module(module)
            return module.component_dir(here, name)
    raise RuntimeError("Код вне репозитория voice-anti-spoofing не используется")


CLASSICAL = _component("classical-anti-spoofing")
OUT = DATASET / "reports" / "origin_retrain_20261113"
FEATURES = OUT / "call_features.csv"
SPLIT = OUT / "split.csv"
HUMAN = {"human_live", "human_recorded"}
SYNTHETIC = {"synthetic_tts", "synthetic_voice_conversion", "synthetic_playback"}
SEED = 42
VAL_FRACTION = 0.2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["extract", "fit", "evaluate"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.step == "extract":
        extract()
        return
    if args.step == "evaluate":
        evaluate()
        return
    fit()


def extract() -> None:
    cfg = load_config(CLASSICAL / "config" / "config.yaml")
    wanted = _wanted_calls()
    stored = pd.DataFrame()
    if FEATURES.is_file():
        stored = pd.read_csv(FEATURES, encoding="utf-8-sig")
    done = set(stored["call_id"]) if not stored.empty else set()
    pending = [row for row in wanted if row["call_id"] not in done]
    print(f"уже есть {len(done)}, осталось {len(pending)}", flush=True)
    for index, row in enumerate(pending, start=1):
        frame = _features_for_call(cfg, row)
        stored = pd.concat([stored, frame], ignore_index=True, sort=False)
        if index % 5 == 0 or index == len(pending):
            stored.to_csv(FEATURES, index=False, encoding="utf-8-sig")
            print(f"признаки {index}/{len(pending)} {row['call_id']}", flush=True)
    print(f"файл: {FEATURES}", flush=True)


def fit() -> None:
    cfg = load_config(CLASSICAL / "config" / "config.yaml")
    discovery = _binary_table("day_20261113")
    split = _ensure_split(discovery)
    features = pd.read_csv(FEATURES, encoding="utf-8-sig")
    features = features.drop_duplicates("call_id", keep="last")
    table = discovery.merge(split[["call_id", "split"]], on="call_id", how="inner").merge(features, on="call_id", how="left", suffixes=("", "_feature"))
    missing = int(table["n_usable_segments"].isna().sum()) if "n_usable_segments" in table else len(table)
    if missing:
        raise SystemExit(f"Нет признаков у {missing} звонков обучения. Сначала extract.")
    usable = table[pd.to_numeric(table["n_usable_segments"], errors="coerce").fillna(0) > 0].copy()
    train = usable[usable["split"] == "train"]
    validation = usable[usable["split"] == "validation"]
    print(f"train {len(train)} validation {len(validation)}", flush=True)
    acoustic = _acoustic_columns(train)
    y_train = train["synthetic"].to_numpy()
    selected = select_features(
        train,
        y_train,
        acoustic,
        max_features=int(cfg["models"]["max_features"]),
        max_nan_fraction=float(cfg["models"]["max_nan_fraction"]),
        min_finite=int(cfg["models"]["min_finite"]),
        max_pairwise_correlation=float(cfg["models"]["max_pairwise_correlation"]),
    )
    pd.DataFrame({"feature": selected}).to_csv(OUT / "features.csv", index=False, encoding="utf-8-sig")
    model_rows = []
    prediction_rows = []
    for name in ("logistic", "svm", "lightgbm"):
        bundle, validation_scores, threshold, record = _fit_one(name, cfg, train, validation, selected)
        bundle_path = OUT / "models" / f"{name}.joblib"
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        import joblib

        joblib.dump(bundle, bundle_path)
        model_rows.append(record)
        prediction_rows.extend(_prediction_rows(validation, name, validation_scores, threshold, "validation"))
        print(f"{name} порог {threshold:.6f} balanced {record['validation_balanced_accuracy']:.3f}", flush=True)

    score_rows = _threshold_models(validation)
    model_rows.extend(score_rows)
    day_scores = _read_scores("day_20261113")
    scored_validation = validation.merge(day_scores, on="call_id", how="left")
    for record in score_rows:
        column = record["score_column"]
        raw = pd.to_numeric(scored_validation[column], errors="coerce")
        mask = raw.notna()
        prediction_rows.extend(
            _prediction_rows(
                scored_validation.loc[mask],
                record["model"],
                -raw[mask].to_numpy(),
                record["threshold_on_inverted_score"],
                "validation",
            )
        )
    pd.DataFrame(model_rows).to_csv(OUT / "validation_selection.csv", index=False, encoding="utf-8-sig")

    test = _current_test()
    test_metrics = []
    if not test.empty:
        test_features = test.merge(features, on="call_id", how="left")
        scored = _score_new_models(test_features, selected)
        test_metrics.extend(scored)
        old = _old_model_metrics(test)
        test_metrics.extend(old)
        pd.DataFrame(test_metrics).to_csv(OUT / "test_metrics_interim.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(prediction_rows).to_csv(OUT / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    _write_report(discovery, split, usable, model_rows, test, test_metrics, len(selected))
    print(f"отчёт: {OUT / 'summary.md'}", flush=True)


def _fit_one(name: str, cfg: dict, train: pd.DataFrame, validation: pd.DataFrame, features: list[str]):
    y_train = train["synthetic"].to_numpy()
    y_validation = validation["synthetic"].to_numpy()
    best = None
    scored = []
    for params, estimator in _candidates(name, cfg["models"], SEED):
        estimator.fit(train[features], y_train)
        scores = _positive_scores(estimator, validation[features], name)
        threshold = _best_threshold(y_validation, scores, 0.0 if name == "svm" else 0.5)
        metrics = binary_metrics(y_validation, scores, (scores >= threshold).astype(int))
        rank = (metrics["balanced_accuracy"], metrics["roc_auc"] if np.isfinite(metrics["roc_auc"]) else -1.0, -_complexity(name, params))
        scored.append({"params": params, "threshold": threshold, "balanced_accuracy": metrics["balanced_accuracy"]})
        if best is None or rank > best[0]:
            best = (rank, estimator, params, scores, threshold, metrics)
    _, estimator, params, scores, threshold, metrics = best
    bundle = {
        "name": name,
        "task": "synthetic_vs_human_voice",
        "positive_class": "synthetic",
        "model": estimator,
        "features": features,
        "threshold": float(threshold),
        "params": params,
        "seed": SEED,
        "score_note": "Больший score значит более синтетический. Это не вероятность.",
    }
    record = {
        "model": name,
        "family": "retrained_classical",
        "threshold": float(threshold),
        "threshold_note": "synthetic, если score >= порога",
        "params": json.dumps(params, ensure_ascii=False),
        "validation_balanced_accuracy": metrics["balanced_accuracy"],
        "validation_recall_synthetic": metrics["recall_synthetic"],
        "validation_specificity_human": metrics["specificity_human"],
        "validation_precision_synthetic": metrics["precision_synthetic"],
        "validation_roc_auc": metrics["roc_auc"],
        "candidates": json.dumps(scored, ensure_ascii=False),
    }
    return bundle, scores, float(threshold), record


def _threshold_models(validation: pd.DataFrame) -> list[dict]:
    scores = _read_scores("day_20261113")
    joined = validation.merge(scores, on="call_id", how="left")
    rows = []
    for model, column, frozen in (
        ("spectra0", "spectra0_score", 1.02),
        ("aasist3", "aasist3_score", None),
    ):
        values = pd.to_numeric(joined[column], errors="coerce")
        mask = values.notna()
        y = joined.loc[mask, "synthetic"].to_numpy()
        raw = values[mask].to_numpy()
        ranking = -raw
        threshold = _best_threshold(y, ranking, float(np.median(ranking)))
        metrics = binary_metrics(y, ranking, (ranking >= threshold).astype(int))
        frozen_metrics = None
        if frozen is not None:
            frozen_pred = (raw < frozen).astype(int)
            frozen_metrics = binary_metrics(y, ranking, frozen_pred)
        rows.append(
            {
                "model": model,
                "family": "threshold_only",
                "threshold": float(-threshold),
                "threshold_on_inverted_score": float(threshold),
                "threshold_note": "synthetic, если исходный score < порога. Веса не менялись.",
                "score_column": column,
                "validation_balanced_accuracy": metrics["balanced_accuracy"],
                "validation_recall_synthetic": metrics["recall_synthetic"],
                "validation_specificity_human": metrics["specificity_human"],
                "validation_precision_synthetic": metrics["precision_synthetic"],
                "validation_roc_auc": metrics["roc_auc"],
                "frozen_threshold": frozen if frozen is not None else "",
                "frozen_validation_recall_synthetic": None if frozen_metrics is None else frozen_metrics["recall_synthetic"],
                "frozen_validation_specificity_human": None if frozen_metrics is None else frozen_metrics["specificity_human"],
                "n": int(mask.sum()),
            }
        )
    return rows


def _score_new_models(test: pd.DataFrame, features: list[str]) -> list[dict]:
    import joblib

    usable = test[pd.to_numeric(test["n_usable_segments"], errors="coerce").fillna(0) > 0].copy()
    rows = []
    for name in ("logistic", "svm", "lightgbm"):
        bundle = joblib.load(OUT / "models" / f"{name}.joblib")
        scores = _positive_scores(bundle["model"], usable[features], name)
        threshold = float(bundle["threshold"])
        metrics = binary_metrics(usable["synthetic"].to_numpy(), scores, (scores >= threshold).astype(int))
        rows.append(_metric_row(name, "retrained_classical", metrics, len(usable), threshold))
    day_scores = _read_scores("day_20261116")
    joined = usable.merge(day_scores, on="call_id", how="left")
    selection = pd.read_csv(OUT / "validation_selection.csv", encoding="utf-8-sig")
    for model, column in (("spectra0", "spectra0_score"), ("aasist3", "aasist3_score")):
        chosen = selection[selection["model"] == model].iloc[0]
        raw = pd.to_numeric(joined[column], errors="coerce")
        mask = raw.notna()
        y = joined.loc[mask, "synthetic"].to_numpy()
        ranking = -raw[mask].to_numpy()
        threshold = float(chosen["threshold_on_inverted_score"])
        metrics = binary_metrics(y, ranking, (ranking >= threshold).astype(int))
        rows.append(_metric_row(model, "threshold_only", metrics, int(mask.sum()), float(chosen["threshold"])))
        if model == "spectra0":
            frozen = binary_metrics(y, ranking, (raw[mask].to_numpy() < 1.02).astype(int))
            rows.append(_metric_row("spectra0_frozen_1.02", "unchanged", frozen, int(mask.sum()), 1.02))
    return rows


def _old_model_metrics(test: pd.DataFrame) -> list[dict]:
    scores = _read_scores("day_20261116")
    joined = test.merge(scores, on="call_id", how="left")
    rows = []
    for name, threshold in (("logistic", None), ("svm", None), ("lightgbm", None)):
        status = joined[f"{name}_status"]
        mask = status == "ok"
        y = joined.loc[mask, "synthetic"].to_numpy()
        raw = pd.to_numeric(joined.loc[mask, f"{name}_score"], errors="coerce").to_numpy()
        predicted = (joined.loc[mask, f"{name}_class"] == "machine").astype(int).to_numpy()
        metrics = binary_metrics(y, raw, predicted)
        rows.append(_metric_row(f"{name}_old_bundle", "frozen_answering_machine_model", metrics, int(mask.sum()), threshold))
    return rows


def _metric_row(model: str, family: str, metrics: dict, n: int, threshold) -> dict:
    return {
        "model": model,
        "family": family,
        "n": n,
        "threshold": threshold if threshold is not None else "",
        "recall_synthetic": metrics["recall_synthetic"],
        "specificity_human": metrics["specificity_human"],
        "precision_synthetic": metrics["precision_synthetic"],
        "f1_synthetic": metrics["f1_synthetic"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "roc_auc": metrics["roc_auc"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "tn": metrics["tn"],
        "fn": metrics["fn"],
    }


def evaluate() -> None:
    """Считает тест 16 ноября уже выбранными порогами. Подбор не повторяет."""

    import joblib

    test, disputes = _resolved_test()
    features = pd.read_csv(FEATURES, encoding="utf-8-sig").drop_duplicates("call_id", keep="last")
    selected = pd.read_csv(OUT / "features.csv", encoding="utf-8-sig")["feature"].tolist()
    selection = pd.read_csv(OUT / "validation_selection.csv", encoding="utf-8-sig")
    bundles = {name: joblib.load(OUT / "models" / f"{name}.joblib") for name in ("logistic", "svm", "lightgbm")}
    joined = test.merge(features, on="call_id", how="left", suffixes=("", "_feature"))
    joined = joined.merge(_read_scores("day_20261116"), on="call_id", how="left", suffixes=("", "_saved"))
    usable = joined[pd.to_numeric(joined["n_usable_segments"], errors="coerce").fillna(0) > 0].copy()
    missing = sorted(set(test["call_id"]) - set(usable["call_id"]))
    if missing:
        raise SystemExit(f"Нет признаков у {len(missing)} тестовых звонков. Сначала extract.")

    rows = []
    comparisons = []
    for name, default in (("logistic", 0.5), ("svm", 0.0), ("lightgbm", 0.5)):
        bundle = bundles[name]
        scores = _positive_scores(bundle["model"], usable[selected], name)
        tuned = float(bundle["threshold"])
        y = usable["synthetic"].to_numpy()
        for label, threshold in (("validation", tuned), ("default", default)):
            metrics = binary_metrics(y, scores, (scores >= threshold).astype(int))
            row = _metric_row(name, f"retrained_{label}_threshold", metrics, len(usable), threshold)
            rows.append(row)
            comparisons.append({**row, "threshold_role": "новый, с валидации" if label == "validation" else "старый, по умолчанию"})
    saved = _read_scores("day_20261116")
    scored = usable.merge(saved, on="call_id", how="left", suffixes=("", "_dup"))
    for model, column, old_threshold in (
        ("spectra0", "spectra0_score", 1.02),
        ("aasist3", "aasist3_score", None),
    ):
        raw = pd.to_numeric(scored[column], errors="coerce")
        mask = raw.notna()
        y = scored.loc[mask, "synthetic"].to_numpy()
        values = raw[mask].to_numpy()
        ranking = -values
        chosen = selection[selection["model"] == model].iloc[0]
        new_threshold = float(chosen["threshold"])
        metrics = binary_metrics(y, ranking, (values < new_threshold).astype(int))
        row = _metric_row(model, "new_threshold", metrics, int(mask.sum()), new_threshold)
        rows.append(row)
        comparisons.append({**row, "threshold_role": "новый, с валидации"})
        if old_threshold is not None:
            frozen = binary_metrics(y, ranking, (values < old_threshold).astype(int))
            old_row = _metric_row(model, "old_threshold", frozen, int(mask.sum()), old_threshold)
            rows.append(old_row)
            comparisons.append({**old_row, "threshold_role": "старый"})
    for name in ("logistic", "svm", "lightgbm"):
        mask = scored[f"{name}_status"] == "ok"
        y = scored.loc[mask, "synthetic"].to_numpy()
        raw = pd.to_numeric(scored.loc[mask, f"{name}_score"], errors="coerce").to_numpy()
        predicted = (scored.loc[mask, f"{name}_class"] == "machine").astype(int).to_numpy()
        metrics = binary_metrics(y, raw, predicted)
        rows.append(_metric_row(f"{name}_old_bundle", "frozen_answering_machine_model", metrics, int(mask.sum()), ""))
    if not disputes.empty:
        disputes.to_csv(OUT / "test_disputes.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(OUT / "test_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(comparisons).to_csv(OUT / "threshold_comparison.csv", index=False, encoding="utf-8-sig")
    _write_test_report(test, usable, disputes, rows, comparisons)
    print(f"тест: {len(test)} бинарных, разногласия {len(disputes)}", flush=True)
    print(f"отчёт: {OUT / 'test_day_20261116.md'}", flush=True)


def _write_test_report(test, usable, disputes, rows, comparisons) -> None:
    human = int((test["synthetic"] == 0).sum())
    synthetic = int((test["synthetic"] == 1).sum())
    lines = [
        "# Тест 16 ноября",
        "",
        "Порог и модели взяты с валидации 13 ноября и здесь не подбирались.",
        "Положительный класс — синтетика. unknown и mixed не входят в знаменатель.",
        "Если два разметчика поставили разные primary, звонок не входит ни в одну сторону.",
        "Score — ранжирующая оценка, не вероятность.",
        "",
        f"- бинарных звонков: {len(test)}, человек {human}, синтетика {synthetic}",
        f"- с речью для новых classical-моделей: {len(usable)}",
        f"- исключены из-за разных меток: {len(disputes)}",
        "",
        "## Старый и новый порог на одном score",
        "",
        "Для Spectra-0 и AASIST3 synthetic, если score ниже порога. Для новых classical-моделей synthetic, если score не ниже порога.",
        "У AASIST3 раньше порога звонка не было, поэтому старой строки нет.",
        "",
    ]
    for row in comparisons:
        lines.append(
            f"- {row['model']}, {row['threshold_role']} {row['threshold']}: "
            f"n {row['n']}, recall {float(row['recall_synthetic']):.3f}, specificity {float(row['specificity_human']):.3f}, "
            f"precision {float(row['precision_synthetic']):.3f}, "
            f"TP {row['tp']} FP {row['fp']} TN {row['tn']} FN {row['fn']}"
        )
    lines.extend(["", "## Старые модели автоответчика, их порог не менялся", ""])
    for row in rows:
        if row["family"] != "frozen_answering_machine_model":
            continue
        lines.append(
            f"- {row['model']}: n {row['n']}, recall {float(row['recall_synthetic']):.3f}, "
            f"specificity {float(row['specificity_human']):.3f}, precision {float(row['precision_synthetic']):.3f}, "
            f"TP {row['tp']} FP {row['fp']} TN {row['tn']} FN {row['fn']}"
        )
    lines.append("")
    (OUT / "test_day_20261116.md").write_text("\n".join(lines), encoding="utf-8")


def _write_report(discovery, split, usable, selection, test, test_metrics, n_features: int) -> None:
    train_n = int((split["split"] == "train").sum())
    val_n = int((split["split"] == "validation").sum())
    lines = [
        "# Переобучение на 13 ноября, тест — размеченная часть 16 ноября",
        "",
        "Положительный класс — синтетический голос: synthetic_tts, synthetic_voice_conversion, synthetic_playback. human_live и human_recorded — человек. unknown и mixed в счёт не входят.",
        "Порог и состав признаков выбирались только на валидации 13 ноября. Метки 16 ноября в этот выбор не входили.",
        "Старые bundle classical-anti-spoofing и веса Spectra-0 / AASIST3 не изменялись. Новые файлы лежат в этом каталоге.",
        "Score — ранжирующая оценка, не вероятность.",
        "",
        f"- бинарных звонков 13 ноября: {len(discovery)}",
        f"- train / validation: {train_n} / {val_n}, доля валидации {VAL_FRACTION}, seed {SEED}",
        f"- с речью, попали в обучение: {len(usable)}",
        f"- признаков после отбора на train: {n_features}",
        "",
        "## Валидация, по ней выбирался порог",
        "",
    ]
    for row in selection:
        frozen = ""
        if row.get("frozen_threshold") not in ("", None) and not _missing(row.get("frozen_validation_recall_synthetic")):
            frozen = (
                f" Старый порог {row['frozen_threshold']}: recall {float(row['frozen_validation_recall_synthetic']):.3f}, "
                f"specificity {float(row['frozen_validation_specificity_human']):.3f}."
            )
        lines.append(
            f"- {row['model']} ({row['family']}): порог {float(row['threshold']):.4f}, "
            f"recall {float(row['validation_recall_synthetic']):.3f}, specificity {float(row['validation_specificity_human']):.3f}, "
            f"balanced accuracy {float(row['validation_balanced_accuracy']):.3f}.{frozen}"
        )
    lines.extend(["", "## Тест 16 ноября, промежуточный срез", ""])
    if test.empty:
        lines.append("Размеченных бинарных звонков ещё нет.")
    else:
        lines.append(
            f"Сейчас в тесте {len(test)} бинарных звонков, и все они из местной половины. "
            f"Вторая половина и ещё не прослушанные звонки сюда не входят. Это не закрытый день."
        )
        lines.append(f"Человек {int((test['synthetic'] == 0).sum())}, синтетика {int((test['synthetic'] == 1).sum())}.")
        lines.append("")
        for row in test_metrics:
            lines.append(
                f"- {row['model']} ({row['family']}): n {row['n']}, recall {float(row['recall_synthetic']):.3f}, "
                f"specificity {float(row['specificity_human']):.3f}, precision {float(row['precision_synthetic']):.3f}, "
                f"TP {row['tp']} FP {row['fp']} TN {row['tn']} FN {row['fn']}."
            )
    lines.extend(
        [
            "",
            "Валидационные числа показывают, какой порог выиграла сетка. Они не являются проверкой на новом дне.",
            "Старые logistic, SVM и LightGBM в тесте — это прежняя задача «живой диалог или автоответчик», их класс machine не равен синтетике.",
            "",
        ]
    )
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "run_config.yaml").write_text(
        yaml.safe_dump(
            {
                "seed": SEED,
                "val_fraction": VAL_FRACTION,
                "positive_class": "synthetic",
                "test_day_used_for_selection": False,
                "weights_overwritten": False,
                "spectra_weights_unchanged": True,
                "aasist_weights_unchanged": True,
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _ensure_split(discovery: pd.DataFrame) -> pd.DataFrame:
    if SPLIT.is_file():
        return pd.read_csv(SPLIT, dtype=str, encoding="utf-8-sig")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SEED)
    index = np.arange(len(discovery))
    train_index, val_index = next(splitter.split(index, discovery["synthetic"].to_numpy()))
    rows = discovery[["call_id", "synthetic", "campaign_name"]].copy()
    rows["split"] = "train"
    rows.iloc[val_index, rows.columns.get_loc("split")] = "validation"
    rows.to_csv(SPLIT, index=False, encoding="utf-8-sig")
    return rows


def _binary_table(batch: str) -> pd.DataFrame:
    annotations = _read_csv(DATASET / "data" / "annotations" / f"{batch}_calls.csv")
    manifest = _read_csv(DATASET / "data" / "manifests" / f"{batch}.csv")
    labels = annotations[["call_id", "primary_label", "review_status"]].drop_duplicates("call_id")
    frame = manifest.merge(labels, on="call_id", how="inner")
    frame = frame[frame["primary_label"].isin(HUMAN | SYNTHETIC)].copy()
    frame["synthetic"] = frame["primary_label"].isin(SYNTHETIC).astype(int)
    return frame


def _current_test() -> pd.DataFrame:
    test, _disputes = _resolved_test()
    return test


def _resolved_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Одна метка на звонок. Расхождение двух разметчиков выводится из бинарного счёта."""

    annotations = _read_csv(DATASET / "data" / "annotations" / "day_20261116_calls.csv")
    manifest = _read_csv(DATASET / "data" / "manifests" / "day_20261116.csv")
    kept = []
    disputes = []
    for call_id, group in annotations.groupby("call_id", sort=False):
        labels = sorted(set(group["primary_label"]))
        if len(labels) > 1:
            disputes.append(
                {
                    "call_id": call_id,
                    "labels": "|".join(f"{row.annotator_id or 'local'}:{row.primary_label}" for row in group.itertuples(index=False)),
                }
            )
            continue
        kept.append(group.iloc[0])
    labels = pd.DataFrame(kept)
    frame = manifest.merge(labels[["call_id", "primary_label", "review_status", "annotator_id"]], on="call_id", how="inner")
    binary = frame[frame["primary_label"].isin(HUMAN | SYNTHETIC)].copy()
    binary["synthetic"] = binary["primary_label"].isin(SYNTHETIC).astype(int)
    assignment = _read_csv(DATASET / "data" / "annotations" / "day_20261116_assignment.csv")
    if not assignment.empty:
        binary = binary.merge(assignment[["call_id", "share"]], on="call_id", how="left")
    return binary, pd.DataFrame(disputes)


def _wanted_calls() -> list[dict]:
    frames = [_binary_table("day_20261113"), _current_test()]
    rows = []
    seen = set()
    for frame in frames:
        for row in frame.itertuples(index=False):
            if row.call_id in seen:
                continue
            seen.add(row.call_id)
            rows.append(
                {
                    "call_id": row.call_id,
                    "wav_path": str((DATASET / row.source_file).resolve()),
                    "client_channel": row.client_channel,
                    "channel_count": row.channel_count,
                    "campaign_name": row.campaign_name,
                    "outcome": getattr(row, "outcome", ""),
                    "prior_classification": getattr(row, "prior_classification", ""),
                    "timestamp": getattr(row, "timestamp", ""),
                }
            )
    return rows


def _features_for_call(cfg: dict, row: dict) -> pd.DataFrame:
    from types import SimpleNamespace

    call = SimpleNamespace(
        file_exists=True,
        resolved_path=Path(row["wav_path"]),
        client_channel=row.get("client_channel", ""),
        call_id=row["call_id"],
        label="",
        locked_split="",
        source_channels=row.get("channel_count", ""),
        classification_result=row.get("prior_classification", ""),
        outcome=row.get("outcome", ""),
        campaign_name=row.get("campaign_name", ""),
        timestamp=row.get("timestamp", ""),
    )
    try:
        outcome = _one_call(call, cfg)
    except Exception as exc:
        return pd.DataFrame([{"call_id": row["call_id"], "n_usable_segments": 0, "extract_error": exc.__class__.__name__}])
    if outcome["n_usable"] <= 0 or not outcome["segments"]:
        return pd.DataFrame([{"call_id": row["call_id"], "n_usable_segments": 0, "extract_error": outcome["status"]}])
    frame = aggregate_calls(pd.DataFrame(outcome["segments"]))
    frame["extract_error"] = ""
    return frame


def _acoustic_columns(frame: pd.DataFrame) -> list[str]:
    reserved = {"call_id", "synthetic", "primary_label", "review_status", "split", "campaign_name", "extract_error", "label", "share"}
    columns = []
    for name in frame.columns:
        if name in reserved or name.startswith("flag_"):
            continue
        numeric = pd.to_numeric(frame[name], errors="coerce")
        if numeric.notna().any():
            frame[name] = numeric
            columns.append(name)
    return columns


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


def _positive_scores(estimator, matrix: pd.DataFrame, name: str) -> np.ndarray:
    if name == "svm":
        return np.asarray(estimator.decision_function(matrix), dtype=float)
    return np.asarray(estimator.predict_proba(matrix)[:, 1], dtype=float)


def _best_threshold(y_true: np.ndarray, scores: np.ndarray, default: float) -> float:
    finite = scores[np.isfinite(scores)]
    if len(finite) == 0:
        return float(default)
    candidates = np.unique(np.concatenate([np.quantile(finite, np.linspace(0.02, 0.98, 49)), [default]]))
    best_threshold = float(candidates[0])
    best_rank = (-1.0, -1.0)
    for threshold in candidates:
        predicted = (scores >= float(threshold)).astype(int)
        accuracy = balanced_accuracy_score(y_true, predicted)
        auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) == 2 else 0.0
        rank = (accuracy, auc)
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


def _prediction_rows(frame: pd.DataFrame, model: str, scores: np.ndarray, threshold: float, split: str) -> list[dict]:
    rows = []
    for call_id, synthetic, score in zip(frame["call_id"], frame["synthetic"], scores):
        if not np.isfinite(score):
            continue
        rows.append(
            {
                "call_id": call_id,
                "split": split,
                "model": model,
                "synthetic": int(synthetic),
                "score": float(score),
                "predicted_synthetic": int(score >= threshold),
            }
        )
    return rows


def _read_scores(batch: str) -> pd.DataFrame:
    return _read_csv(DATASET / "data" / "manifests" / f"{batch}_scores.csv")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def _missing(value) -> bool:
    return value is None or value == "" or (isinstance(value, float) and np.isnan(value))


if __name__ == "__main__":
    main()
