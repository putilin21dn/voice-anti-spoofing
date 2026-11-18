from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.comparison.complementarity import compare
from src.comparison.spectra0_comparison import baseline_tables
from src.data.call_audit import audit_calls
from src.data.outcome_check import check_outcomes
from src.data.dataset_loader import load_dataset
from src.data.feature_store import FeatureStore
from src.data.split import create_call_splits, select_even, write_splits
from src.features.pipeline import extract_dataset
from src.models.evaluate import evaluate_models
from src.models.features import VARIANTS
from src.models.shap_report import write_shap_report
from src.models.training import train_model
from src.utils.config import load_config, resolve_path
from src.utils.feature_audit import audit_feature_table
from src.utils.logging_config import setup_logging
from src.utils.reports import next_numbered_dir, write_json
from src.utils.validation import run_synthetic_checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Классический анализ синтетической речи. Локальный эксперимент.")
    parser.add_argument("--config", default="config/config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("inspect-data", help="Показать схему выборки, метки и доступность WAV")
    extract = sub.add_parser("extract-features", help="Извлечь признаки речевых сегментов")
    extract.add_argument("--splits", choices=["all", "pool", "test", "spectra_threshold"], default="all")
    extract.add_argument("--per-class", type=int, default=0)
    extract.add_argument("--call-id", action="append", default=[])
    sub.add_parser("validate-features", help="Проверить признаки и записать отчёт")
    sub.add_parser("audit-calls", help="Проверить утечки, метки и связь признаков с длительностью")
    sub.add_parser("audit-outcomes", help="Сравнить метку sample с полем outcome, не меняя модели")
    sub.add_parser("create-splits", help="Разделить звонки на train, validation и зафиксированный test")
    train = sub.add_parser("train", help="Обучить классификатор")
    train.add_argument("--model", choices=["logistic", "svm", "lightgbm"], required=True)
    train.add_argument("--variant", choices=list(VARIANTS), default="full")
    evaluate = sub.add_parser("evaluate", help="Оценить обученные модели")
    evaluate.add_argument("--variant", choices=list(VARIANTS), default="full")
    spectra = sub.add_parser("compare-spectra0", help="Посчитать зафиксированный baseline Spectra-0")
    spectra.add_argument("--variant", choices=list(VARIANTS), default="full")
    sub.add_parser("shap", help="SHAP замороженных моделей full на validation, test и n100")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(resolve_path(cfg, cfg["paths"]["logs_dir"]) / "pipeline.log")
    if args.command == "inspect-data":
        _inspect(cfg)
    elif args.command == "extract-features":
        _extract(cfg, args, logger)
    elif args.command == "validate-features":
        _validate(cfg)
    elif args.command == "audit-calls":
        report = audit_calls(cfg)
        print(report["report_dir"])
        print(f"duplicates {report['duplicate_call_ids']} label_mismatch {report['label_mismatch']} unreadable {report['unreadable']}")
    elif args.command == "audit-outcomes":
        result = check_outcomes(cfg)
        print(result["summary"])
        print("report", result["report_dir"])
    elif args.command == "create-splits":
        _splits(cfg)
    elif args.command == "train":
        report_dir = train_model(cfg, args.model, args.variant)
        print("report", report_dir)
    elif args.command == "evaluate":
        result = evaluate_models(cfg, args.variant)
        print(result["summary"])
        print("report", result["report_dir"])
    elif args.command == "compare-spectra0":
        _spectra(cfg, args.variant)
    elif args.command == "shap":
        report_dir = write_shap_report(cfg)
        print("report", report_dir)


def _inspect(cfg: dict) -> None:
    frame = load_dataset(cfg)
    manifest_dir = resolve_path(cfg, "data/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "call_id", "label", "locked_split", "wav_ok", "file_exists", "client_channel",
        "source_channels", "sample_rate", "duration_ms", "classification_result",
        "outcome", "campaign_name", "timestamp", "resolved_path", "label_folder_mismatch",
    ]
    frame[columns].to_csv(manifest_dir / "dataset_manifest.csv", index=False, encoding="utf-8")
    print(f"rows {len(frame)}")
    print("labels", frame["label"].value_counts().to_dict())
    print("splits", frame["locked_split"].value_counts().to_dict())
    print("files", frame["file_exists"].value_counts().to_dict())
    print("sample_rate", frame["sample_rate"].value_counts().to_dict())
    print("source_channels", frame["source_channels"].value_counts().to_dict())
    print("client_channel", frame["client_channel"].value_counts().to_dict())
    print("folder_mismatch", int(frame["label_folder_mismatch"].sum()))
    missing = frame[~frame["file_exists"]]
    print("missing", len(missing))
    for row in missing.itertuples(index=False):
        print(f"  {row.call_id} {row.label} {row.locked_split}")
    print("manifest", manifest_dir / "dataset_manifest.csv")


def _extract(cfg: dict, args, logger) -> None:
    frame = load_dataset(cfg)
    if args.splits != "all":
        frame = frame[frame["locked_split"] == args.splits]
    if args.call_id:
        frame = frame[frame["call_id"].isin(args.call_id)]
    if args.per_class:
        frame = select_even(frame, args.per_class)
    store = FeatureStore(resolve_path(cfg, cfg["paths"]["features_dir"]), cfg)
    selected = frame[["call_id", "label", "locked_split"]].copy()
    selected_path = store.directory / "selected_call_ids.csv"
    if selected_path.exists():
        previous = pd.read_csv(selected_path, dtype=str)
        selected = pd.concat([previous, selected], ignore_index=True).drop_duplicates("call_id")
    selected.to_csv(selected_path, index=False, encoding="utf-8")
    stats = extract_dataset(frame, cfg, store, logger)
    print(stats)


def _validate(cfg: dict) -> None:
    features_dir = resolve_path(cfg, cfg["paths"]["features_dir"])
    segments_path = features_dir / "segments.csv"
    if not segments_path.exists():
        raise SystemExit("Нет segments.csv. Сначала выполните extract-features.")
    segments = pd.read_csv(segments_path, dtype={"call_id": str})
    status = pd.read_csv(features_dir / "call_status.csv", dtype={"call_id": str}) if (features_dir / "call_status.csv").exists() else pd.DataFrame()
    audit = audit_feature_table(segments)
    checks = run_synthetic_checks(cfg)
    audit["synthetic_checks"] = checks
    if not status.empty:
        audit["status"] = status["status"].value_counts().to_dict()
        audit["processing_ms_sum"] = float(pd.to_numeric(status["processing_ms"], errors="coerce").sum())
        audit["processing_ms_median"] = float(pd.to_numeric(status["processing_ms"], errors="coerce").median())
    errors_path = features_dir / "errors.csv"
    audit["errors"] = int(len(pd.read_csv(errors_path))) if errors_path.exists() else 0
    report_dir = next_numbered_dir(resolve_path(cfg, cfg["paths"]["reports_dir"]), "feature_check")
    write_json(report_dir / "feature_audit.json", audit)
    (report_dir / "summary.md").write_text(_summary(audit), encoding="utf-8")
    print(_summary(audit))
    print("report", report_dir)


def _splits(cfg: dict) -> None:
    frame = load_dataset(cfg)
    parts = create_call_splits(frame, val_fraction=float(cfg["split"]["val_fraction"]), seed=int(cfg["seed"]))
    directory = resolve_path(cfg, cfg["paths"]["splits_dir"])
    write_splits(directory, parts, seed=int(cfg["seed"]), val_fraction=float(cfg["split"]["val_fraction"]))
    for name, part in parts.items():
        print(name, len(part), part["label"].value_counts().to_dict() if "label" in part.columns else {})
    print("splits", directory)


def _spectra(cfg: dict, variant: str = "full") -> None:
    result = compare(cfg, variant)
    if result.get("summary"):
        print(result["summary"])
        print("report", result["report_dir"])
        return
    reports = baseline_tables(cfg)
    report_dir = next_numbered_dir(resolve_path(cfg, cfg["paths"]["reports_dir"]), "spectra0_baseline")
    write_json(report_dir / "metrics.json", reports)
    lines = ["# Spectra-0 baseline", "", "Порог 1.02 не подбирался заново. Score — средний зазор logit, не вероятность.", ""]
    for name, metrics in reports.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(metrics["caution"])
        lines.append("")
        for key in (
            "calls", "human", "machine", "recall_synthetic", "specificity_human",
            "precision_synthetic", "f1_synthetic", "balanced_accuracy", "fpr", "fnr", "roc_auc", "pr_auc",
        ):
            lines.append(f"- {key}: {metrics[key]}")
        lines.append("")
    text = "\n".join(lines)
    (report_dir / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    print("report", report_dir)


def _summary(audit: dict) -> str:
    lines = ["# Проверка признаков", ""]
    lines.append(f"- сегментов: {audit.get('segments')}")
    lines.append(f"- пригодных сегментов: {audit.get('usable_segments')}")
    lines.append(f"- звонков: {audit.get('calls')}")
    lines.append(f"- ошибки файлов: {audit.get('errors')}")
    lines.append(f"- частоты дискретизации: {audit.get('sample_rates')}")
    lines.append(f"- ресемплинг: {audit.get('resample_actions')}")
    if "processing_ms_sum" in audit:
        lines.append(f"- суммарное время обработки, мс: {audit['processing_ms_sum']:.0f}")
        lines.append(f"- медианное время звонка, мс: {audit['processing_ms_median']:.0f}")
    lines.append(f"- статусы: {audit.get('status')}")
    lines.append("")
    for label, info in audit.get("by_label", {}).items():
        lines.append(f"## {label}")
        lines.append("")
        for key, value in info.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    lines.append("## Покрытие групп")
    lines.append("")
    for name, info in audit.get("coverage", {}).items():
        lines.append(f"- {name}: колонок {info['columns']}, доля конечных значений {info['finite_fraction']}")
    lines.append("")
    problems = audit.get("physical", {}).get("problems", [])
    lines.append("## Физические границы 8 кГц")
    lines.append("")
    lines.append("Нарушений нет." if not problems else "Нарушения: " + "; ".join(problems))
    for name, info in audit.get("physical", {}).items():
        if name == "problems":
            continue
        lines.append(f"- {name}: {info}")
    lines.append("")
    lines.append("## Синтетический контроль")
    lines.append("")
    for check in audit.get("synthetic_checks", []):
        state = "ok" if check["passed"] else "fail"
        lines.append(f"- {check['name']}: {state}. {check['detail']}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
