"""Собирает таблицы приоритизации из уже посчитанных score. Модели заново не запускает."""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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

from src.annotation.schema import CALL_FIELDS, SEGMENT_FIELDS  # noqa: E402
from src.research.model_assisted_annotation.rules import (  # noqa: E402
    ANTISPOOF,
    THRESHOLDS,
    WIDTHS,
    WINDOW_SECONDS,
    borderline,
    campaign_amd,
    disagreement_codes,
    duration_band,
    finite_score,
    flag,
    prediction,
    priority_of,
    public_reasons,
    sample_control,
    technical_codes,
)

OUT = ROOT / "reports" / "model_assisted_annotation_20261111_20261112"
MOSCOW = ZoneInfo("Europe/Moscow")
FORBIDDEN = {"phone", "phone_number", "external_id", "s3_key", "s3_bucket", "msisdn"}
WINDOW = WINDOW_SECONDS


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [{key: value or "" for key, value in row.items() if key} for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    leaked = FORBIDDEN & set(fields)
    if leaked:
        raise RuntimeError("В таблицу попали запрещённые колонки: " + ", ".join(sorted(leaked)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def latest_run(batch: str) -> Path:
    root = ROOT / "evaluation" / batch
    runs = [path for path in root.iterdir() if path.is_dir() and (path / "call_scores.csv").is_file()]
    if not runs:
        raise SystemExit(f"Нет прогона в {root}")
    return max(runs, key=lambda path: path.name)


def padding_of(windows: list[dict]) -> tuple[str, str]:
    valid = 0.0
    repeated = 0.0
    full = 0
    for window in windows:
        try:
            speech = float(window.get("valid_speech_audio_sec") or 0)
        except ValueError:
            continue
        padded = str(window.get("padded_to_model_length") or "").lower() == "true"
        extra = max(WINDOW - speech, 0.0) if padded else 0.0
        valid += speech
        repeated += extra
        if speech >= WINDOW - 0.001 and not padded:
            full += 1
    if valid + repeated <= 0:
        return "", "0" if windows else ""
    return f"{repeated / (valid + repeated):.6f}", str(full)


def moscow_day(timestamp: str, batch_id: str) -> str:
    try:
        moment = datetime.fromisoformat(timestamp)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=MOSCOW)
        return moment.astimezone(MOSCOW).date().isoformat()
    except ValueError:
        return "2026-11-11" if "20261111" in batch_id else "2026-11-12"


def live_flags(call_ids: list[str]) -> dict[str, str]:
    """Читает только наличие ключа живого детектора. Текст реплики не запрашивается."""

    import os
    import subprocess

    collector = _component("voice-dataset-collector")
    python = Path(sys.executable)
    script = """
import sys
from config.settings import load_settings
from src.db.connection import connect
ids = [line.strip() for line in sys.stdin if line.strip()]
sql = '''
SELECT c.id::text, (c.collected_data ? 'answering_machine_text')
FROM calls AS c
WHERE c.id::text = ANY(%(ids)s)
'''
with connect(load_settings()) as connection:
    for call_id, live in connection.execute(sql, {"ids": ids}).fetchall():
        print(f"{call_id}\\t{'yes' if live else 'no'}")
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(collector)
    completed = subprocess.run(
        [str(python), "-c", script],
        input="\n".join(call_ids),
        text=True,
        capture_output=True,
        cwd=str(collector),
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "запрос живого детектора не выполнился").strip()
        raise SystemExit(message.splitlines()[-1][:300])
    flags = {}
    for line in completed.stdout.splitlines():
        call_id, _, value = line.partition("\t")
        if call_id:
            flags[call_id] = value
    return flags


def _combined_status(left: str, right: str) -> str:
    for status in (left, right):
        if status and status != "ok":
            return status
    return left or right or ""


def index_by_call(rows: list[dict]) -> dict[str, dict]:
    return {row["call_id"]: row for row in rows if row.get("call_id")}


def main() -> None:
    manifests = []
    score_files = {}
    runs = {}
    for batch in ("day_20261111", "day_20261112"):
        manifests.extend(read_csv(ROOT / "data" / "manifests" / f"{batch}.csv"))
        score_files.update(index_by_call(read_csv(ROOT / "data" / "manifests" / f"{batch}_scores.csv")))
        runs[batch] = latest_run(batch)
    new_path = OUT / "new_classical_scores.csv"
    if not new_path.is_file():
        raise SystemExit("Нет new_classical_scores.csv. Сначала нужен прогон сохранённых joblib.")
    new_scores = index_by_call(read_csv(new_path))
    spectra_calls = {}
    windows: dict[str, list[dict]] = defaultdict(list)
    for run in runs.values():
        for row in read_csv(run / "call_results" / "spectra0.csv"):
            spectra_calls[row["call_id"]] = row
        for row in read_csv(run / "window_results" / "spectra0.csv"):
            windows[row["call_id"]].append(row)
    labels = {}
    annotation_path = ROOT / "data" / "annotations" / "calls.csv"
    if annotation_path.is_file():
        for row in read_csv(annotation_path):
            if row.get("review_status") == "reviewed" and row.get("primary_label"):
                labels[row["call_id"]] = row
    live = live_flags([row["call_id"] for row in manifests])

    predictions = []
    agreed_residual: dict[tuple, list[str]] = defaultdict(list)
    prepared = []
    for manifest in manifests:
        call_id = manifest["call_id"]
        flat = score_files.get(call_id, {})
        fresh = new_scores.get(call_id, {})
        spectra = spectra_calls.get(call_id, {})
        ratio, full_windows = padding_of(windows.get(call_id, []))
        values = {
            "spectra0": (flat.get("spectra0_score", ""), flat.get("spectra0_status", "") or spectra.get("status", "")),
            "aasist3": (flat.get("aasist3_score", ""), flat.get("aasist3_status", "")),
            "logistic_old": (flat.get("logistic_score", ""), flat.get("logistic_status", "")),
            "svm_old": (flat.get("svm_score", ""), flat.get("svm_status", "")),
            "lightgbm_old": (flat.get("lightgbm_score", ""), flat.get("lightgbm_status", "")),
            "logistic_new": (fresh.get("logistic_new_score", ""), "ok" if fresh.get("logistic_new_score") else fresh.get("status", "")),
            "svm_new": (fresh.get("svm_new_score", ""), "ok" if fresh.get("svm_new_score") else fresh.get("status", "")),
            "lightgbm_new": (fresh.get("lightgbm_new_score", ""), "ok" if fresh.get("lightgbm_new_score") else fresh.get("status", "")),
        }
        predicted = {name: prediction(name, score, status) for name, (score, status) in values.items()}
        campaign = campaign_amd(manifest.get("prior_classification", ""))
        live_flag = live.get(call_id, "")
        codes = disagreement_codes(predicted, campaign, live_flag)
        technical_row = {
            "spectra_status": values["spectra0"][1],
            "classical_status": _combined_status(fresh.get("status", ""), values["logistic_old"][1]),
            "spectra_n_segments": spectra.get("speech_segments", ""),
            "spectra_speech_duration_s": spectra.get("speech_duration_s", ""),
            "spectra_n_windows": spectra.get("n_windows", ""),
            "spectra_full_windows": full_windows,
            "duration_seconds": manifest.get("duration_seconds", ""),
            "client_channel": manifest.get("client_channel", ""),
        }
        codes.extend(technical_codes(technical_row))
        for name in ANTISPOOF:
            if borderline(name, values[name][0], values[name][1]):
                codes.append(f"borderline_{name}")
        available = [predicted[name] for name in ANTISPOOF if predicted[name]]
        agreed = len(available) >= 2 and len(set(available)) == 1
        existing = labels.get(call_id)
        record = {
            "call_id": call_id,
            "date": moscow_day(manifest.get("timestamp", ""), manifest.get("batch_id", "")),
            "campaign": manifest.get("campaign_name", ""),
            "human_annotation": existing.get("primary_label", "") if existing else "",
            "human_annotation_source": "batch_001" if existing else "",
            "amd_campaign": campaign,
            "amd_live": live_flag,
            "amd_outcome": "yes" if manifest.get("outcome") == "answering_machine" else "no",
            "duration_seconds": manifest.get("duration_seconds", ""),
            "client_channel": manifest.get("client_channel", ""),
            "audio_reference": manifest.get("source_file", ""),
            "spectra_status": values["spectra0"][1],
            "spectra_n_segments": spectra.get("speech_segments", ""),
            "spectra_speech_duration_s": spectra.get("speech_duration_s", ""),
            "spectra_n_windows": spectra.get("n_windows", ""),
            "padding_ratio": ratio,
            "classical_status": _combined_status(fresh.get("status", ""), values["logistic_old"][1]),
            "classical_n_segments": fresh.get("n_segments", ""),
            "classical_n_usable_segments": fresh.get("n_usable_segments", ""),
            "classical_speech_duration_s": fresh.get("speech_duration_s", ""),
            "agreed_antispoof": "yes" if agreed else "no",
            "agreed_class": available[0] if agreed else "",
        }
        for name, (score, status) in values.items():
            record[f"{name}_score"] = "" if finite_score(score) is None else f"{finite_score(score):.17g}"
            record[f"{name}_prediction"] = predicted[name]
            record[f"{name}_status"] = status
        prepared.append((record, codes, agreed, existing))
        if agreed and not existing:
            blocked = any(
                code.startswith("spectra_")
                or code.startswith("lightgbm_")
                or code == "antispoof_models"
                or code.startswith("old_new_")
                or code.startswith("borderline_")
                or code.startswith("technical_")
                for code in codes
            )
            if not blocked:
                key = (
                    record["date"],
                    record["campaign"],
                    record["agreed_class"],
                    duration_band(record["duration_seconds"]),
                )
                agreed_residual[key].append(call_id)

    weights = sample_control(dict(agreed_residual))
    queue_rows = []
    frame_rows = []
    for record, codes, agreed, existing in prepared:
        queues: set[str] = set()
        if not existing:
            if any(
                code in {"spectra_lightgbm", "antispoof_models"}
                or code.endswith("_amd")
                or code.startswith("old_new_")
                for code in codes
            ):
                queues.add("A")
            if any(code.startswith("borderline_") for code in codes):
                queues.add("B")
            if any(code.startswith("technical_") for code in codes):
                queues.add("C")
            if record["call_id"] in weights:
                queues.add("D")
                codes = [*codes, "control_sample"]
        priority = priority_of(queues)
        weight = ""
        inclusion = "0"
        mechanism = "not_selected"
        if existing:
            mechanism = "already_labeled"
            inclusion = ""
        elif queues & {"A", "B", "C"}:
            mechanism = "certainty"
            inclusion = "1"
            weight = "1"
        elif "D" in queues:
            mechanism = "control"
            info = weights[record["call_id"]]
            inclusion = f"{info['stratum_sampled'] / info['stratum_population']:.6f}"
            weight = f"{info['sampling_weight']:.6f}"
        frame_rows.append(
            {
                "call_id": record["call_id"],
                "date": record["date"],
                "campaign": record["campaign"],
                "agreed_class": record["agreed_class"],
                "duration_band": duration_band(record["duration_seconds"]),
                "queues": "|".join(sorted(queues, key=lambda name: {"A": 1, "C": 2, "B": 3, "D": 4}[name])),
                "inclusion_probability": inclusion,
                "sampling_weight": weight,
                "mechanism": mechanism,
            }
        )
        predictions.append(record)
        if not queues:
            continue
        info = weights.get(record["call_id"], {})
        queue_rows.append(
            {
                **record,
                "review_priority": priority,
                "queues": "|".join(sorted(queues, key=lambda name: {"A": 1, "C": 2, "B": 3, "D": 4}[name])),
                "reason_codes": "|".join(codes),
                "public_reasons": public_reasons(codes),
                "sampling_weight": weight,
                "stratum": info.get("stratum", ""),
                "stratum_population": info.get("stratum_population", ""),
                "stratum_sampled": info.get("stratum_sampled", ""),
            }
        )
    queue_rows.sort(key=lambda row: ({"A": 1, "C": 2, "B": 3, "D": 4}[row["review_priority"]], row["date"], row["campaign"], row["call_id"]))
    for index, row in enumerate(queue_rows, start=1):
        row["sort_index"] = str(index)

    prediction_fields = list(predictions[0].keys()) if predictions else ["call_id"]
    write_csv(OUT / "per_call_predictions.csv", predictions, prediction_fields)
    queue_fields = [
        "sort_index",
        "call_id",
        "date",
        "campaign",
        "review_priority",
        "queues",
        "public_reasons",
        "reason_codes",
        "audio_reference",
        "duration_seconds",
        "client_channel",
        "spectra_speech_duration_s",
        "spectra_n_segments",
        "spectra_n_windows",
        "padding_ratio",
        "spectra_status",
        "classical_status",
        "sampling_weight",
        "stratum",
        "stratum_population",
        "stratum_sampled",
    ]
    write_csv(OUT / "review_queue_all.csv", queue_rows, queue_fields)
    for name, letter in (
        ("review_queue_disagreements.csv", "A"),
        ("review_queue_borderline.csv", "B"),
        ("review_queue_technical.csv", "C"),
        ("review_queue_random_control.csv", "D"),
    ):
        chosen = [row for row in queue_rows if letter in row["queues"].split("|")]
        write_csv(OUT / name, chosen, queue_fields)
    template = []
    for row in queue_rows:
        template.append(
            {
                "call_id": row["call_id"],
                "review_priority": row["review_priority"],
                "review_reasons": row["public_reasons"],
                "audio_reference": row["audio_reference"],
                "date": row["date"],
                "campaign": row["campaign"],
                "human_label": "",
                "reviewer_comment": "",
                "review_status": "unreviewed",
                "reviewed_at": "",
            }
        )
    write_csv(
        OUT / "manual_annotation_template.csv",
        template,
        ["call_id", "review_priority", "review_reasons", "audio_reference", "date", "campaign", "human_label", "reviewer_comment", "review_status", "reviewed_at"],
    )
    write_csv(
        OUT / "sampling_frame.csv",
        frame_rows,
        ["call_id", "date", "campaign", "agreed_class", "duration_band", "queues", "inclusion_probability", "sampling_weight", "mechanism"],
    )

    membership = Counter()
    pair = Counter()
    for row in queue_rows:
        letters = set(row["queues"].split("|"))
        for letter in letters:
            membership[letter] += 1
        for left, right in (("A", "B"), ("A", "C"), ("B", "C"), ("A", "D"), ("B", "D"), ("C", "D")):
            if left in letters and right in letters:
                pair[f"{left}_{right}"] += 1
    summary_rows = [
        {"metric": "calls_in_manifests", "value": str(len(predictions))},
        {"metric": "unique_review_calls", "value": str(len(queue_rows))},
        {"metric": "queue_A", "value": str(membership["A"])},
        {"metric": "queue_B", "value": str(membership["B"])},
        {"metric": "queue_C", "value": str(membership["C"])},
        {"metric": "queue_D", "value": str(membership["D"])},
        {"metric": "intersection_A_B", "value": str(pair["A_B"])},
        {"metric": "intersection_A_C", "value": str(pair["A_C"])},
        {"metric": "intersection_B_C", "value": str(pair["B_C"])},
        {"metric": "intersection_A_D", "value": str(pair["A_D"])},
        {"metric": "intersection_B_D", "value": str(pair["B_D"])},
        {"metric": "intersection_C_D", "value": str(pair["C_D"])},
        {"metric": "already_labeled_batch_001", "value": str(sum(1 for row in predictions if row["human_annotation"]))},
        {"metric": "not_selected", "value": str(sum(1 for row in frame_rows if row["mechanism"] == "not_selected"))},
        {"metric": "control_seed", "value": "20261111"},
        {"metric": "control_target", "value": "150"},
    ]
    for name in ANTISPOOF:
        present = sum(1 for row in predictions if row[f"{name}_prediction"])
        summary_rows.append({"metric": f"predictions_{name}", "value": str(present)})
    write_csv(OUT / "queue_summary.csv", summary_rows, ["metric", "value"])

    availability = []
    for name in ANTISPOOF:
        present = sum(1 for row in predictions if row[f"{name}_prediction"])
        availability.append(
            {
                "signal": name,
                "available": str(present),
                "missing": str(len(predictions) - present),
                "threshold": str(THRESHOLDS[name]),
                "borderline_half_width": str(WIDTHS[name]),
            }
        )
    for signal, column in (("amd_campaign", "amd_campaign"), ("amd_live", "amd_live"), ("amd_outcome", "amd_outcome")):
        present = sum(1 for row in predictions if row[column] in {"yes", "no"})
        availability.append(
            {
                "signal": signal,
                "available": str(present),
                "missing": str(len(predictions) - present),
                "threshold": "",
                "borderline_half_width": "",
            }
        )
    availability.append(
        {
            "signal": "human_annotation_batch_001",
            "available": str(sum(1 for row in predictions if row["human_annotation"])),
            "missing": str(sum(1 for row in predictions if not row["human_annotation"])),
            "threshold": "",
            "borderline_half_width": "",
        }
    )
    write_csv(OUT / "model_availability.csv", availability, ["signal", "available", "missing", "threshold", "borderline_half_width"])

    manifest_out = ROOT / "data" / "manifests" / "priority_20261111_20261112.csv"
    write_csv(manifest_out, manifests, list(manifests[0].keys()))
    calls_out = ROOT / "data" / "annotations" / "priority_20261111_20261112_calls.csv"
    segments_out = ROOT / "data" / "annotations" / "priority_20261111_20261112_segments.csv"
    if not calls_out.exists():
        write_csv(calls_out, [], list(CALL_FIELDS))
    if not segments_out.exists():
        write_csv(segments_out, [], list(SEGMENT_FIELDS))
    print(f"звонков: {len(predictions)}")
    print(f"к прослушиванию: {len(queue_rows)}")
    for row in summary_rows:
        print(f"{row['metric']}: {row['value']}")


if __name__ == "__main__":
    main()
