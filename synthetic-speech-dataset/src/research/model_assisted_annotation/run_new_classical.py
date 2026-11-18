"""Запускает сохранённые новые classical-модели. Обучения нет."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


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


def _python(name: str) -> Path:
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
            return module.component_python(here, name)
    raise RuntimeError("Код вне репозитория voice-anti-spoofing не используется")


PYTHON = _python("classical-anti-spoofing")
WORKER = ROOT / "src" / "evaluation" / "new_classical_scores.py"
MODELS = ROOT / "reports" / "origin_retrain_20261113" / "models"
OUT = ROOT / "reports" / "model_assisted_annotation_20261111_20261112"


def _calls(manifest: Path) -> list[dict]:
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    calls = []
    for row in rows:
        wav = (ROOT / row["source_file"]).resolve()
        calls.append(
            {
                "call_id": row["call_id"],
                "wav_path": str(wav),
                "client_channel": row.get("client_channel", ""),
                "duration_seconds": row.get("duration_seconds", ""),
                "channel_count": row.get("channel_count", ""),
                "prior_classification": row.get("prior_classification", ""),
                "outcome": row.get("outcome", ""),
                "campaign_name": row.get("campaign_name", ""),
                "timestamp": row.get("timestamp", ""),
            }
        )
    return calls


def main() -> None:
    if not PYTHON.is_file():
        raise SystemExit(f"Нет интерпретатора {PYTHON}")
    OUT.mkdir(parents=True, exist_ok=True)
    calls = _calls(ROOT / "data" / "manifests" / "day_20261111.csv")
    calls.extend(_calls(ROOT / "data" / "manifests" / "day_20261112.csv"))
    request = OUT / "logs" / "new_classical_request.json"
    request.parent.mkdir(parents=True, exist_ok=True)
    request.write_text(json.dumps({"calls": calls}, ensure_ascii=False), encoding="utf-8")
    completed = subprocess.run(
        [
            str(PYTHON),
            str(WORKER),
            "--request",
            str(request),
            "--models",
            str(MODELS),
            "--output",
            str(OUT / "new_classical_scores.csv"),
            "--features-out",
            str(OUT / "new_classical_features.csv"),
        ],
        cwd=str(CLASSICAL),
        env={**os.environ, "PYTHONPATH": str(CLASSICAL)},
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
