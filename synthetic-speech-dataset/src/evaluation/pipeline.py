from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.evaluation.dataset import batch_status, load_evaluation_config, prepare_dataset, readiness
from src.evaluation.report import quality_report, write_outputs
from src.utils.config import resolve_path


def inspect_dataset(config_path: str = "config/evaluation.yaml") -> dict:
    cfg = load_evaluation_config(config_path)
    calls, errors = prepare_dataset(cfg)
    status = batch_status(cfg)
    return {
        "batch_id": cfg["batch_id"],
        "status": status.get("status"),
        "calls": len(calls),
        "in_category_metrics": sum(call.in_category_metrics for call in calls),
        "unclassified_robots": sum(call.category == "unclassified_robot" for call in calls),
        "errors": errors,
        "categories": _counts(calls),
    }


def validate_readiness(config_path: str = "config/evaluation.yaml") -> tuple[dict, list[str]]:
    cfg = load_evaluation_config(config_path)
    calls, prepare_errors = prepare_dataset(cfg)
    errors = readiness(cfg, calls, prepare_errors)
    check = _classical_check(cfg)
    errors.extend(check.get("errors") or [])
    if not check.get("ok", False) and not check.get("errors"):
        errors.append("Проверка классических моделей не вернула ok")
    return {"calls": calls, "check": check, "status": batch_status(cfg)}, errors


def run_evaluation(
    config_path: str = "config/evaluation.yaml",
    *,
    dry_run: bool = False,
    call_ids: list[str] | None = None,
    run_name: str = "",
    reuse_from: str = "",
) -> Path:
    cfg = load_evaluation_config(config_path)
    calls, prepare_errors = prepare_dataset(cfg)
    errors = readiness(cfg, calls, prepare_errors)
    check = _classical_check(cfg)
    errors.extend(check.get("errors") or [])
    run_dir = _run_dir(cfg, run_name or ("dry" if dry_run else _stamp()))
    _write_preflight(cfg, run_dir, calls, errors, check, dry_run, reuse_from)
    if errors:
        raise SystemExit(1)
    if dry_run:
        return run_dir
    selected = [call for call in calls if not call.exclusion or call.category == "unclassified_robot"]
    if call_ids:
        wanted = set(call_ids)
        selected = [call for call in selected if call.call_id in wanted]
        missing = wanted - {call.call_id for call in selected}
        if missing:
            raise SystemExit("Нет разрешённых звонков: " + ", ".join(sorted(missing)))
    if not selected:
        raise SystemExit("Нет звонков для inference")
    reused_rows: list[dict] = []
    reused_spectra: list[dict] = []
    reused_aasist: list[dict] = []
    pending = selected
    if reuse_from:
        reuse_dir = Path(reuse_from)
        if not reuse_dir.is_dir():
            reuse_dir = _run_dir(cfg, reuse_from)
        reused_rows, reused_spectra, reused_aasist, reused_ids = _load_reuse(reuse_dir, {call.call_id for call in selected})
        pending = [call for call in selected if call.call_id not in reused_ids]
        print(f"Уже посчитаны: {len(reused_ids)}. Новые: {len(pending)}", flush=True)
    classical_rows: list[dict] = []
    spectra_windows: list[dict] = []
    aasist_windows: list[dict] = []
    spectra_rows: list[dict] = []
    if pending:
        print(f"Классические модели: {len(pending)} звонков", flush=True)
        classical_rows = _run_classical(cfg, run_dir, pending)
        print("Spectra-0 и AASIST3", flush=True)
        spectra_windows, aasist_windows, spectra_rows = _run_spectra(cfg, pending)
    write_outputs(
        run_dir,
        calls,
        reused_spectra + spectra_windows,
        reused_aasist + aasist_windows,
        reused_rows + classical_rows + spectra_rows,
    )
    return run_dir


def run_day_scores(
    reuse_from: str = "confirmed_001/20261113T155919",
    config_path: str = "config/day_20261113.yaml",
) -> Path:
    """Считает пять моделей по партии дня. Меток ещё нет, метрики происхождения не пишет."""

    from types import SimpleNamespace

    from src.utils.config import load_config

    day_cfg = load_config(config_path)
    eval_cfg = load_evaluation_config()
    manifest = resolve_path(day_cfg, day_cfg["paths"]["manifest_out"])
    rows = _read_csv(manifest)
    calls = []
    for row in rows:
        wav_path = resolve_path(day_cfg, row.get("source_file", ""))
        if not wav_path.is_file():
            raise SystemExit(f"Нет WAV для {row.get('call_id', '')}")
        calls.append(
            SimpleNamespace(
                call_id=row["call_id"],
                wav_path=wav_path,
                client_channel=row.get("client_channel", "0"),
                duration_seconds=row.get("duration_seconds", ""),
                channel_count=row.get("channel_count", ""),
                primary_label="",
                listener_label="",
                prior_classification=row.get("prior_classification", ""),
                outcome=row.get("outcome", ""),
                campaign_name=row.get("campaign_name", ""),
                timestamp=row.get("timestamp", ""),
            )
        )
    if not calls:
        raise SystemExit("Партия дня пуста. Сначала нужна выгрузка.")
    check = _classical_check(eval_cfg)
    if check.get("errors"):
        raise SystemExit(check["errors"][0])
    run_dir = resolve_path(eval_cfg, eval_cfg["paths"]["evaluation_root"]) / day_cfg["batch_id"] / _stamp()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    reused_rows: list[dict] = []
    reused_spectra: list[dict] = []
    reused_aasist: list[dict] = []
    pending = calls
    if reuse_from:
        reuse_dir = Path(reuse_from)
        if not reuse_dir.is_dir():
            reuse_dir = resolve_path(eval_cfg, eval_cfg["paths"]["evaluation_root"]) / reuse_from
        wanted = {call.call_id for call in calls}
        reused_rows, reused_spectra, reused_aasist, reused_ids = _load_reuse(reuse_dir, wanted)
        pending = [call for call in calls if call.call_id not in reused_ids]
        print(f"Уже посчитаны раньше: {len(reused_ids)}. Новые: {len(pending)}", flush=True)
    classical_rows: list[dict] = []
    spectra_windows: list[dict] = []
    aasist_windows: list[dict] = []
    spectra_rows: list[dict] = []
    if pending:
        print(f"Классические модели: {len(pending)} звонков", flush=True)
        classical_rows = _run_classical(eval_cfg, run_dir, pending)
        print("Spectra-0 и AASIST3", flush=True)
        spectra_windows, aasist_windows, spectra_rows = _run_spectra(eval_cfg, pending)
    model_rows = reused_rows + classical_rows + spectra_rows
    from src.evaluation.report import _csv

    _csv(run_dir / "window_results" / "spectra0.csv", reused_spectra + spectra_windows)
    _csv(run_dir / "window_results" / "aasist3.csv", reused_aasist + aasist_windows)
    for model, filename in (
        ("spectra0", "spectra0.csv"),
        ("aasist3", "aasist3.csv"),
        ("logistic", "logistic_regression.csv"),
        ("svm", "svm.csv"),
        ("lightgbm", "lightgbm.csv"),
    ):
        _csv(run_dir / "call_results" / filename, [row for row in model_rows if row.get("model") == model])
    flat = _flat_scores(model_rows)
    scores_path = resolve_path(day_cfg, day_cfg["paths"]["scores_csv"])
    _csv(scores_path, flat)
    _csv(run_dir / "call_scores.csv", flat)
    (run_dir / "summary.md").write_text(
        "\n".join(
            [
                f"# Прогон записей {day_cfg['batch_id']}",
                "",
                "Метки происхождения ещё не стоят. Метрики после ручной проверки, этот файл их не содержит.",
                "Порог Spectra-0 остался 1.02. Нового обучения не было.",
                "Классические human и machine — прогноз старой задачи «живой диалог или автоответчик».",
                "AASIST3 без порога звонка: в таблице только средний bonafide logit.",
                "",
                f"- звонков: {len(calls)}",
                f"- посчитаны сейчас: {len(pending)}",
                f"- взяты из прошлого прогона: {len(calls) - len(pending)}",
                f"- минимальная длительность отбора: {day_cfg.get('min_duration_seconds')} с",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"оценки: {scores_path}", flush=True)
    return run_dir


def _flat_scores(model_rows: list[dict]) -> list[dict]:
    by_call: dict[str, dict[str, str]] = {}
    for row in model_rows:
        item = by_call.setdefault(row["call_id"], {"call_id": row["call_id"]})
        model = row.get("model", "")
        item[f"{model}_status"] = row.get("status", "")
        item[f"{model}_class"] = row.get("predicted_class", "")
        item[f"{model}_score"] = row.get("score", "")
    return [by_call[call_id] for call_id in sorted(by_call)]


def _run_classical(cfg: dict, run_dir: Path, calls: list) -> list[dict]:
    request = run_dir / "logs" / "classical_request.json"
    output = run_dir / "logs" / "classical_scores.csv"
    request.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "calls": [
            {
                "call_id": call.call_id,
                "wav_path": str(call.wav_path),
                "client_channel": call.client_channel,
                "duration_seconds": call.duration_seconds,
                "channel_count": call.channel_count,
                "primary_label": call.primary_label,
                "listener_label": call.listener_label,
                "prior_classification": call.prior_classification,
                "outcome": call.outcome,
                "campaign_name": call.campaign_name,
                "timestamp": call.timestamp,
            }
            for call in calls
        ]
    }
    request.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    worker = Path(__file__).with_name("classical_worker.py")
    python = resolve_path(cfg, cfg["paths"]["classical_python"])
    root = resolve_path(cfg, cfg["paths"]["classical_root"])
    completed = subprocess.run(
        [str(python), str(worker), "--output", str(output), "--request", str(request)],
        cwd=str(root),
        env={**dict(**_environ()), "PYTHONPATH": str(root)},
        check=False,
        capture_output=True,
        text=True,
    )
    (run_dir / "logs" / "classical_worker.log").write_text(
        (completed.stdout or "") + (completed.stderr or ""), encoding="utf-8"
    )
    if completed.returncode != 0 or not output.exists():
        raise RuntimeError("Классические модели не посчитались. См. logs/classical_worker.log")
    import pandas as pd

    frame = pd.read_csv(output, dtype=str, keep_default_na=False)
    return frame.to_dict(orient="records")


def _run_spectra(cfg: dict, calls: list) -> tuple[list[dict], list[dict], list[dict]]:
    import torch

    from src.evaluation.spectra_runner import load_modules, score_files

    spectra_root = resolve_path(cfg, cfg["paths"]["spectra_root"])
    webrtc_site = resolve_path(cfg, cfg["paths"]["classical_root"]) / ".venv" / "Lib" / "site-packages"
    inference, aasist = load_modules(spectra_root, webrtc_site)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = [
        {"call_id": call.call_id, "wav_path": str(call.wav_path), "client_channel": call.client_channel}
        for call in calls
    ]
    return score_files(inference, aasist, payload, device)


def _load_reuse(run_dir: Path, wanted: set[str]) -> tuple[list[dict], list[dict], list[dict], set[str]]:
    from src.evaluation.report import MODELS, _file_model

    model_rows: list[dict] = []
    by_call: dict[str, set[str]] = {}
    for model in MODELS:
        path = run_dir / "call_results" / f"{_file_model(model)}.csv"
        if not path.is_file():
            return [], [], [], set()
        for row in _read_csv(path):
            if row.get("call_id") not in wanted or row.get("status") != "ok" or row.get("model") != model:
                continue
            model_rows.append(row)
            by_call.setdefault(row["call_id"], set()).add(model)
    complete = {call_id for call_id, models in by_call.items() if models == set(MODELS)}
    model_rows = [row for row in model_rows if row["call_id"] in complete]
    spectra = [row for row in _read_csv(run_dir / "window_results" / "spectra0.csv") if row.get("call_id") in complete]
    aasist = [row for row in _read_csv(run_dir / "window_results" / "aasist3.csv") if row.get("call_id") in complete]
    return model_rows, spectra, aasist, complete


def _read_csv(path: Path) -> list[dict]:
    import csv

    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [{key: value or "" for key, value in row.items() if key} for row in csv.DictReader(handle)]


def _classical_check(cfg: dict) -> dict:
    run_root = resolve_path(cfg, cfg["paths"]["evaluation_root"]) / "_check"
    destination = run_root / "classical_check.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    worker = Path(__file__).with_name("classical_worker.py")
    python = resolve_path(cfg, cfg["paths"]["classical_python"])
    root = resolve_path(cfg, cfg["paths"]["classical_root"])
    if not python.is_file():
        return {"ok": False, "errors": [f"Нет интерпретатора {python}"]}
    completed = subprocess.run(
        [str(python), str(worker), "--check", "--output", str(destination)],
        cwd=str(root),
        env={**_environ(), "PYTHONPATH": str(root)},
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not destination.exists():
        message = (completed.stderr or completed.stdout or "проверка моделей не выполнилась").strip()
        return {"ok": False, "errors": [message.splitlines()[-1][:300]]}
    return json.loads(destination.read_text(encoding="utf-8"))


def _write_preflight(cfg: dict, run_dir: Path, calls: list, errors: list[str], check: dict, dry_run: bool, reuse_from: str = "") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "data_quality_report.md").write_text(quality_report(calls, errors), encoding="utf-8")
    from src.evaluation.report import _csv, _exclusion_row, _public_row

    _csv(run_dir / "manifest_snapshot.csv", [_public_row(call) for call in calls])
    _csv(run_dir / "exclusions.csv", [_exclusion_row(call) for call in calls if call.exclusion])
    config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "reuse_from": reuse_from,
        "batch_id": cfg["batch_id"],
        "batch_status": batch_status(cfg),
        "spectra0_threshold": cfg["spectra0_threshold"],
        "spectra0_rule": cfg["spectra0_rule"],
        "window": cfg["window"],
        "aasist3_call_threshold": None,
        "aasist3_note": "Класс окна — argmax logit. Порог звонка не зафиксирован и не подбирался.",
        "unused_model_classify_threshold": "Spectra0Model.classify содержит -1.0625009. Этот прогон его не использует.",
        "classical_models": check.get("models"),
        "classical_versions": check.get("versions"),
        "python": sys.version,
        "platform": platform.platform(),
        "git": _git_revision(),
        "call_ids": [call.call_id for call in calls],
        "errors": errors,
    }
    (run_dir / "run_config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _run_dir(cfg: dict, name: str) -> Path:
    return resolve_path(cfg, cfg["paths"]["evaluation_root"]) / cfg["batch_id"] / name


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def _counts(calls: list) -> dict[str, int]:
    counts: dict[str, int] = {}
    for call in calls:
        counts[call.category] = counts.get(call.category, 0) + 1
    return counts


def _environ() -> dict[str, str]:
    import os

    return dict(os.environ)


def _git_revision() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unavailable"
    text = (completed.stdout or "").strip()
    return text or "unavailable"
