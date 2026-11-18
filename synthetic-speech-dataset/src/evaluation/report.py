from __future__ import annotations

import csv
from pathlib import Path

from src.evaluation.dataset import ORIGIN_LABELS, PreparedCall, SYNTHETIC_ORIGINS
from src.evaluation.metrics import binary_metrics, category_counts, disagreement_flags, legacy_task_counts


MODELS = ("spectra0", "aasist3", "logistic", "svm", "lightgbm")
CLASSICAL = ("logistic", "svm", "lightgbm")


def write_outputs(
    run_dir: Path,
    calls: list[PreparedCall],
    spectra_windows: list[dict],
    aasist_windows: list[dict],
    model_rows: list[dict],
) -> None:
    joined = join_calls(calls, model_rows)
    _csv(run_dir / "manifest_snapshot.csv", [_public_row(call) for call in calls])
    _csv(run_dir / "exclusions.csv", [_exclusion_row(call) for call in calls if call.exclusion])
    _csv(run_dir / "window_results" / "spectra0.csv", spectra_windows)
    _csv(run_dir / "window_results" / "aasist3.csv", aasist_windows)
    for model in MODELS:
        _csv(run_dir / "call_results" / f"{_file_model(model)}.csv", [row for row in model_rows if row["model"] == model])
    for category in ORIGIN_LABELS:
        if category == "unknown":
            continue
        _csv(
            run_dir / "category_reports" / f"{category}.csv",
            [row for row in joined if row["category"] == category],
        )
    _csv(run_dir / "category_reports" / "unclassified_robots.csv", [row for row in joined if row["category"] == "unclassified_robot"])
    _csv(run_dir / "category_reports" / "legacy_test.csv", [row for row in joined if row["category"] == "legacy_test"])
    _special(run_dir, joined)
    _csv(run_dir / "disagreements" / "model_disagreements.csv", [_disagreement(row) for row in joined if row["scored"]])
    metric_rows = [binary_metrics(joined, model) for model in MODELS]
    _csv(run_dir / "summary" / "metrics.csv", metric_rows)
    for model in MODELS:
        _csv(run_dir / "summary" / f"{_file_model(model)}_categories.csv", category_counts(joined, model))
    legacy_rows = [item for model in MODELS for item in legacy_task_counts(joined, model)]
    _csv(run_dir / "summary" / "legacy_test.csv", legacy_rows)
    (run_dir / "summary" / "summary.md").write_text(_summary(joined, metric_rows), encoding="utf-8")
    _chart(run_dir / "summary" / "charts" / "human_live_spoof_share.svg", joined)


def join_calls(calls: list[PreparedCall], model_rows: list[dict]) -> list[dict]:
    by_model: dict[tuple[str, str], dict] = {}
    for row in model_rows:
        by_model[(row["call_id"], row["model"])] = row
    joined = []
    for call in calls:
        item = {
            "call_id": call.call_id,
            "category": call.category,
            "primary_label": call.primary_label,
            "listener_label": call.listener_label,
            "cohort": call.cohort,
            "legacy_label": call.legacy_label,
            "exclusion": call.exclusion,
            "in_category_metrics": call.in_category_metrics,
            "in_binary_metrics": call.in_binary_metrics,
            "outcome": call.outcome,
            "scored": False,
        }
        for model in MODELS:
            result = by_model.get((call.call_id, model), {})
            item[f"{model}_status"] = result.get("status", "")
            item[f"{model}_class"] = result.get("predicted_class", "")
            item[f"{model}_score"] = result.get("score", "")
            item[f"{model}_error"] = result.get("error", "")
            if result.get("status") == "ok":
                item["scored"] = True
        joined.append(item)
    return joined


def _special(run_dir: Path, joined: list[dict]) -> None:
    recorded = [row for row in joined if row["category"] == "human_recorded" and row["in_category_metrics"]]
    synthetic = [row for row in joined if row["category"] in SYNTHETIC_ORIGINS and row["in_category_metrics"]]
    playback = [row for row in joined if row["category"] == "synthetic_playback" and row["in_category_metrics"]]
    live = [row for row in joined if row["category"] == "human_live" and row["in_category_metrics"]]
    _reason_or_rows(
        run_dir / "special_comparisons" / "human_recorded_vs_synthetic.csv",
        recorded + synthetic,
        "human_recorded в закрытой пачке нет. Human Live на его место не подставляется."
        if not recorded
        else "Подтверждённой синтетики для сравнения с human_recorded нет.",
    )
    _reason_or_rows(
        run_dir / "special_comparisons" / "synthetic_playback_vs_human_live.csv",
        playback + live if playback and live else [],
        "synthetic_playback в закрытой пачке нет." if not playback else "human_live нет.",
    )
    _reason_or_rows(
        run_dir / "special_comparisons" / "ai_secretary_vs_human_live.csv",
        [],
        "Подтверждённой метки AI-секретаря в этой пачке нет. По исходу кампании она не угадывается.",
    )


def _reason_or_rows(path: Path, rows: list[dict], reason: str) -> None:
    if rows:
        _csv(path, rows)
        return
    _csv(path, [{"reason": reason}])


def _summary(joined: list[dict], metric_rows: list[dict]) -> str:
    counted = [row for row in joined if row["in_category_metrics"]]
    lines = [
        "# Сравнение пяти готовых моделей",
        "",
        "Нового обучения не было. Порог Spectra-0 остался 1.02. Пороги классических моделей взяты из сохранённых bundle.",
        "Классические human и machine — это старая задача «живой диалог против автоответчика», не детектор синтеза.",
        "AASIST3 не получил нового порога звонка: сохранены logits и класс окна, класс звонка не назначался.",
        "",
        f"- звонков в снимке: {len(joined)}",
        f"- в категориальных таблицах происхождения: {len(counted)}",
        f"- прежний test20: {sum(row['category'] == 'legacy_test' for row in joined)}",
        f"- unclassified_robots: {sum(row['category'] == 'unclassified_robot' for row in joined)}",
        f"- исключены по другим причинам: {sum(bool(row['exclusion']) and row['category'] not in {'unclassified_robot', 'legacy_test'} for row in joined)}",
        "",
        "## Категории",
        "",
    ]
    for category in (*ORIGIN_LABELS, "unclassified_robot", "legacy_test"):
        count = sum(row["category"] == category for row in joined)
        lines.append(f"- {category}: {count}")
    lines.extend(["", "## Spectra-0 на human_live", ""])
    live = [row for row in joined if row["category"] == "human_live" and row.get("spectra0_status") == "ok"]
    spoof = sum(row.get("spectra0_class") == "spoof" for row in live)
    lines.append(f"- звонков: {len(live)}")
    lines.append(f"- bonafide: {len(live) - spoof}")
    lines.append(f"- spoof: {spoof}")
    lines.extend(["", "## Классические модели на human_live", ""])
    for model in CLASSICAL:
        ok = [row for row in live if row.get(f"{model}_status") == "ok"]
        machine = sum(row.get(f"{model}_class") == "machine" for row in ok)
        lines.append(f"- {model}: human {len(ok) - machine}, machine {machine}")
    lines.extend(["", "## Прежний test20", ""])
    lines.append(
        "Метки human и machine здесь — живой диалог и автоответчик. В классы происхождения они не переводятся."
    )
    legacy = [row for row in joined if row["category"] == "legacy_test"]
    for label in ("human", "machine"):
        lines.append(f"- старая метка {label}: {sum(row.get('legacy_label') == label for row in legacy)}")
    for model in ("spectra0", *CLASSICAL):
        for label in ("human", "machine"):
            ok = [
                row
                for row in legacy
                if row.get("legacy_label") == label and row.get(f"{model}_status") == "ok"
            ]
            if model == "spectra0":
                spoof = sum(row.get("spectra0_class") == "spoof" for row in ok)
                lines.append(f"- spectra0 / {label}: bonafide {len(ok) - spoof}, spoof {spoof}")
            else:
                matched = sum(row.get(f"{model}_class") == label for row in ok)
                lines.append(f"- {model} / {label}: совпало со старой меткой {matched} из {len(ok)}")
    lines.extend(["", "## Бинарные метрики происхождения", ""])
    for row in metric_rows:
        reason = row.get("reason") or "посчитаны"
        lines.append(f"- {row['model']}: {reason}")
    lines.extend(
        [
            "",
            "Пустые категории происхождения не заполнялись.",
            "Прежний test20 добавлен отдельной таблицей. n100 не добавлен: порог Spectra-0 1.02 выбирался на нём и в этом прогоне не менялся.",
            "batch_001 в эту оценку не входил.",
            "",
        ]
    )
    return "\n".join(lines)


def _chart(path: Path, joined: list[dict]) -> None:
    live = [row for row in joined if row["category"] == "human_live"]
    bars = []
    for model in ("spectra0", *CLASSICAL):
        ok = [row for row in live if row.get(f"{model}_status") == "ok"]
        positive = "spoof" if model == "spectra0" else "machine"
        share = (sum(row.get(f"{model}_class") == positive for row in ok) / len(ok)) if ok else 0
        bars.append((model, share, len(ok)))
    width = 480
    height = 40 + 36 * len(bars)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        "<text x='12' y='20' font-family='sans-serif' font-size='14'>Доля spoof или machine на human_live</text>",
    ]
    for index, (model, share, count) in enumerate(bars):
        y = 36 + index * 36
        parts.append(f"<text x='12' y='{y + 16}' font-family='sans-serif' font-size='12'>{model} ({count})</text>")
        parts.append(f"<rect x='170' y='{y}' width='{max(1, int(share * 280))}' height='18' fill='#355'/>")
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts), encoding="utf-8")


def _disagreement(row: dict) -> dict:
    item = {
        "call_id": row["call_id"],
        "category": row["category"],
        "cohort": row.get("cohort", ""),
        "legacy_label": row.get("legacy_label", ""),
        "in_category_metrics": row["in_category_metrics"],
        "spectra0_class": row.get("spectra0_class", ""),
        "spectra0_score": row.get("spectra0_score", ""),
        "aasist3_mean_bonafide_logit": row.get("aasist3_score", ""),
        "aasist3_class": "",
        "logistic_class": row.get("logistic_class", ""),
        "logistic_score": row.get("logistic_score", ""),
        "svm_class": row.get("svm_class", ""),
        "svm_score": row.get("svm_score", ""),
        "lightgbm_class": row.get("lightgbm_class", ""),
        "lightgbm_score": row.get("lightgbm_score", ""),
        "flags": "|".join(disagreement_flags(row)),
    }
    return item


def _public_row(call: PreparedCall) -> dict:
    return {
        "call_id": call.call_id,
        "source_file": call.source_file,
        "listener_label": call.listener_label,
        "primary_label": call.primary_label,
        "category": call.category,
        "cohort": call.cohort,
        "legacy_label": call.legacy_label,
        "client_channel": call.client_channel,
        "sample_rate": call.sample_rate,
        "duration_seconds": call.duration_seconds,
        "channel_count": call.channel_count,
        "campaign_name": call.campaign_name,
        "outcome": call.outcome,
        "training_role": call.training_role,
        "exclusion": call.exclusion,
        "exclusion_reason": call.exclusion_reason,
        "in_category_metrics": call.in_category_metrics,
        "in_binary_metrics": call.in_binary_metrics,
    }


def _exclusion_row(call: PreparedCall) -> dict:
    return {"call_id": call.call_id, "exclusion": call.exclusion, "reason": call.exclusion_reason}


def _file_model(model: str) -> str:
    return "logistic_regression" if model == "logistic" else model


def _csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def quality_report(calls: list[PreparedCall], errors: list[str]) -> str:
    lines = [
        "# Контроль данных перед inference",
        "",
        f"- всего звонков: {len(calls)}",
        f"- подтверждённых категорий в основных таблицах: {sum(call.in_category_metrics for call in calls)}",
        f"- unknown: {sum(call.category == 'unknown' for call in calls)}",
        f"- mixed: {sum(call.category == 'mixed' for call in calls)}",
        f"- unclassified_robots: {sum(call.category == 'unclassified_robot' for call in calls)}",
        f"- прежний test20: {sum(call.category == 'legacy_test' for call in calls)}",
        f"- not_for_training / batch_001: {sum(call.exclusion == 'batch_001' for call in calls)}",
        f"- locked_baseline: {sum(call.exclusion == 'locked_baseline' for call in calls)}",
        f"- нет WAV: {sum(call.exclusion == 'missing_wav' for call in calls)}",
        f"- повреждён WAV: {sum(call.exclusion == 'broken_wav' for call in calls)}",
        f"- канал не определён: {sum(call.exclusion == 'channel_uncertain' for call in calls)}",
        "",
        "## Критические ошибки",
        "",
    ]
    lines.extend(f"- {item}" for item in errors) if errors else lines.append("Критических ошибок нет.")
    lines.append("")
    return "\n".join(lines)
