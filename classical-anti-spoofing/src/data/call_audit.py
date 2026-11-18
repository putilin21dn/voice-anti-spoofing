from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.dataset_loader import load_dataset
from src.models.dataset import acoustic_columns, load_call_frame, modeling_rows
from src.utils.config import resolve_path
from src.utils.reports import next_numbered_dir, write_json


DISTRIBUTION_COLUMNS = (
    "speech_duration_s",
    "recording_duration_s",
    "speech_ratio",
    "snr_db",
    "n_usable_segments",
    "spectral_centroid_hz_mean__mean",
    "spectral_flatness_mean__mean",
    "f0_voiced_fraction__mean",
    "mfcc_static_00_mean__mean",
    "mfcc_delta_06_max__mean",
    "acf_harmonicity_mean__mean",
)
ACOUSTIC_KEYS = (
    "spectral_centroid_hz_mean__mean",
    "spectral_flatness_mean__mean",
    "f0_voiced_fraction__mean",
    "mfcc_static_00_mean__mean",
    "mfcc_delta_06_max__mean",
    "acf_harmonicity_mean__mean",
)


def audit_calls(cfg: dict) -> dict:
    frame = load_call_frame(cfg)
    status = pd.read_csv(resolve_path(cfg, cfg["paths"]["features_dir"]) / "call_status.csv", dtype={"call_id": str})
    errors = pd.read_csv(resolve_path(cfg, cfg["paths"]["features_dir"]) / "errors.csv", dtype={"call_id": str})
    report = {
        "calls": int(len(frame)),
        "duplicate_call_ids": int(frame["call_id"].duplicated().sum()),
        "label_mismatch": int(
            (
                frame["label_split"].notna()
                & (frame["label"].astype(str) != frame["label_split"].astype(str))
            ).sum()
        ),
        "missing_split": int(frame["split"].isna().sum()),
        "split_counts": {str(name): int(count) for name, count in frame["split"].value_counts(dropna=False).items()},
        "unreadable": int(len(errors)),
        "unknown": int((status["label"] == "unknown").sum()) if "label" in status.columns else 0,
        "status": {str(name): int(count) for name, count in status["status"].value_counts().items()},
        "locked_overlap": _locked_overlap(frame),
    }
    usable = pd.to_numeric(frame["n_usable_segments"], errors="coerce").fillna(0)
    report["calls_without_usable_segments"] = int((usable <= 0).sum())
    train = modeling_rows(frame, "train")
    columns = [name for name in acoustic_columns(train.columns.tolist()) if name.endswith("__mean")]
    target = (train["label"] == "machine").astype(float)
    association = train[columns].corrwith(target).dropna()
    duration = pd.to_numeric(train["speech_duration_s"], errors="coerce")
    recording = pd.to_numeric(train["recording_duration_s"], errors="coerce")
    duration_corr = train[columns].corrwith(duration)
    recording_corr = train[columns].corrwith(recording)
    ranked = association.abs().sort_values(ascending=False).head(30)
    separation = []
    quality_like = []
    for name in ranked.index:
        label_corr = float(association[name])
        speech_corr = float(duration_corr.get(name, np.nan))
        record_corr = float(recording_corr.get(name, np.nan))
        row = {
            "feature": name,
            "corr_machine": label_corr,
            "corr_speech_duration": speech_corr,
            "corr_recording_duration": record_corr,
        }
        separation.append(row)
        if np.isfinite(speech_corr) and abs(speech_corr) > abs(label_corr):
            quality_like.append(row)
    report["top_label_association_train"] = separation
    report["stronger_duration_than_label"] = quality_like
    report["nan_fraction_train_mean_features"] = float(train[columns].isna().mean().mean())
    present = [name for name in DISTRIBUTION_COLUMNS if name in train.columns]
    report["distributions_train"] = _distributions(train, present)
    report["outlier_counts_train"] = _outlier_counts(train, present)
    report["missing_by_group_train"] = _missing_by_group(train)
    report["outcomes"] = _outcomes(cfg)
    report["note"] = (
        "Ассоциации посчитаны только на train. Длительность и SNR не входят в признаки модели. "
        "Высокая корреляция не означает, что признак вызван синтезом."
    )
    report_dir = next_numbered_dir(resolve_path(cfg, cfg["paths"]["reports_dir"]), "data_audit")
    write_json(report_dir / "audit.json", report)
    pd.DataFrame(separation).to_csv(report_dir / "label_association.csv", index=False)
    pd.DataFrame(report["distributions_train"]).to_csv(report_dir / "distributions.csv", index=False)
    pd.DataFrame(report["outlier_counts_train"]).to_csv(report_dir / "outliers.csv", index=False)
    pd.DataFrame(report["missing_by_group_train"]).to_csv(report_dir / "missingness.csv", index=False)
    pd.DataFrame(report["outcomes"]).to_csv(report_dir / "outcomes.csv", index=False)
    _distribution_plots(train, present, report_dir)
    (report_dir / "summary.md").write_text(_summary(report), encoding="utf-8")
    report["report_dir"] = str(report_dir)
    return report


def _locked_overlap(frame: pd.DataFrame) -> dict:
    problems = {}
    for split, allowed in {
        "train": {"pool"},
        "validation": {"pool"},
        "test": {"test"},
        "spectra_threshold": {"spectra_threshold"},
    }.items():
        rows = frame[frame["split"] == split]
        foreign = sorted(set(rows["locked_split"].astype(str)) - allowed)
        problems[split] = foreign
    return problems


def _distributions(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    rows = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        human = values[frame["label"] == "human"].dropna()
        machine = values[frame["label"] == "machine"].dropna()
        if human.empty or machine.empty:
            continue
        low, high = human.quantile([0.25, 0.75]).tolist()
        inside = float(((machine >= low) & (machine <= high)).mean())
        rows.append(
            {
                "feature": column,
                "human_n": int(len(human)),
                "human_p25": float(human.quantile(0.25)),
                "human_median": float(human.median()),
                "human_p75": float(human.quantile(0.75)),
                "machine_n": int(len(machine)),
                "machine_p25": float(machine.quantile(0.25)),
                "machine_median": float(machine.median()),
                "machine_p75": float(machine.quantile(0.75)),
                "machine_inside_human_iqr": inside,
            }
        )
    return rows


def _outlier_counts(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    rows = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        finite = values.dropna()
        if len(finite) < 20:
            continue
        low, high = finite.quantile([0.25, 0.75]).tolist()
        width = high - low
        if width <= 0:
            continue
        mask = values.notna() & ((values < low - 1.5 * width) | (values > high + 1.5 * width))
        labels = frame.loc[mask, "label"]
        rows.append(
            {
                "feature": column,
                "outlier_calls": int(mask.sum()),
                "human": int((labels == "human").sum()),
                "machine": int((labels == "machine").sum()),
                "fence_low": float(low - 1.5 * width),
                "fence_high": float(high + 1.5 * width),
            }
        )
    return rows


def _missing_by_group(frame: pd.DataFrame) -> list[dict]:
    groups = {
        "lfcc": "lfcc_",
        "mfcc": "mfcc_",
        "cqcc": "cqcc_",
        "f0": "f0_",
        "voice_quality": ("jitter_local", "shimmer_local", "hnr_praat_db", "acf_harmonicity_mean"),
        "spectral": ("spectral_", "zero_crossing_rate"),
        "energy": "energy_rms",
    }
    rows = []
    for name, prefix in groups.items():
        prefixes = prefix if isinstance(prefix, tuple) else (prefix,)
        columns = [column for column in acoustic_columns(frame.columns.tolist()) if column.startswith(prefixes)]
        if not columns:
            continue
        rows.append(
            {
                "group": name,
                "columns": len(columns),
                "nan_fraction": float(frame[columns].isna().mean().mean()),
            }
        )
    return rows


def _outcomes(cfg: dict) -> list[dict]:
    dataset = load_dataset(cfg)[["call_id", "label", "outcome", "classification_result"]]
    rows = []
    for column in ("outcome", "classification_result"):
        counts = dataset.groupby(["label", column], dropna=False).size().reset_index(name="calls")
        for record in counts.itertuples(index=False):
            rows.append({"field": column, "label": record.label, "value": record[1], "calls": int(record.calls)})
    return rows


def _distribution_plots(frame: pd.DataFrame, columns: list[str], report_dir) -> None:
    shown = [name for name in ACOUSTIC_KEYS if name in columns]
    if not shown:
        return
    figure, axes = plt.subplots(len(shown), 1, figsize=(7.2, 2.2 * len(shown)), squeeze=False)
    for axis, column in zip(axes.ravel(), shown):
        values = pd.to_numeric(frame[column], errors="coerce")
        for label, color in (("human", "#4c78a8"), ("machine", "#f58518")):
            sample = values[frame["label"] == label].dropna()
            axis.hist(sample, bins=30, density=True, alpha=0.55, label=label, color=color)
        axis.set_title(column)
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(report_dir / "distributions_train.png", dpi=120)
    plt.close(figure)


def _summary(report: dict) -> str:
    lines = [
        "# Проверка таблицы звонков",
        "",
        report["note"],
        "",
        f"- звонков с признаками: {report['calls']}",
        f"- дубли call_id: {report['duplicate_call_ids']}",
        f"- расхождение метки и split: {report['label_mismatch']}",
        f"- без split: {report['missing_split']}",
        f"- нечитаемые файлы: {report['unreadable']}",
        f"- unknown: {report['unknown']}",
        f"- без пригодных сегментов: {report['calls_without_usable_segments']}",
        f"- статусы: {report['status']}",
        f"- split: {report['split_counts']}",
        f"- чужие locked_split: {report['locked_overlap']}",
        "",
        "## Признаки, сильнее всего связанные с machine на train",
        "",
    ]
    for row in report["top_label_association_train"][:15]:
        lines.append(
            f"- {row['feature']}: corr={row['corr_machine']:.3f}, "
            f"с длительностью речи {row['corr_speech_duration']:.3f}"
        )
    lines.extend(["", "## Среди них сильнее связаны с длительностью, чем с меткой", ""])
    if not report["stronger_duration_than_label"]:
        lines.append("Таких среди первых 30 нет.")
    for row in report["stronger_duration_than_label"]:
        lines.append(f"- {row['feature']}: метка {row['corr_machine']:.3f}, речь {row['corr_speech_duration']:.3f}")
    lines.extend(["", "## Распределения на train", "", "Доля machine внутри межквартильного размаха human. Границы взяты только с train.", ""])
    for row in report["distributions_train"]:
        lines.append(
            f"- {row['feature']}: human median {row['human_median']:.3f}, "
            f"machine median {row['machine_median']:.3f}, "
            f"machine внутри human IQR {row['machine_inside_human_iqr']:.3f}"
        )
    lines.extend(["", "## Выбросы на train", "", "Граница Тьюки посчитана по всему train, без разреза по метке.", ""])
    for row in report["outlier_counts_train"]:
        lines.append(f"- {row['feature']}: {row['outlier_calls']} звонков, human {row['human']}, machine {row['machine']}")
    lines.extend(["", "## Пропуски по группам на train", ""])
    for row in report["missing_by_group_train"]:
        lines.append(f"- {row['group']}: колонок {row['columns']}, доля NaN {row['nan_fraction']:.3f}")
    lines.extend(["", "## Исходы кампании", "", "Колонки с телефоном не читались. Исход в модель не входит.", ""])
    for row in report["outcomes"]:
        if row["calls"] < 5:
            continue
        lines.append(f"- {row['label']} {row['field']}={row['value']}: {row['calls']}")
    lines.append("")
    return "\n".join(lines)
