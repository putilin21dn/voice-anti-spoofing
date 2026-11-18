"""Голос 2 из 3 на B2X-16-11-2026-callback. Пороги не двигаются, модели не обучаются."""

from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

import pandas as pd

from src.research.model_assisted_annotation.build_report import live_flags
from src.research.model_assisted_annotation.rules import prediction

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


CLASSICAL = _component("classical-anti-spoofing")
PYTHON = _python("classical-anti-spoofing")
WORKER = ROOT / "src" / "evaluation" / "new_classical_scores.py"
MODELS = ROOT / "reports" / "origin_retrain_20261113" / "models"
OUT = ROOT / "reports" / "callback_20261116"
MANIFEST = ROOT / "data" / "manifests" / "callback_20261116.csv"
SCORES = ROOT / "data" / "manifests" / "callback_20261116_scores.csv"
SPECTRA = ROOT / "evaluation" / "callback_20261116" / "20261117T024359" / "call_results" / "spectra0.csv"


def _calls() -> list[dict]:
    with MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    calls = []
    for row in rows:
        calls.append(
            {
                "call_id": row["call_id"],
                "wav_path": str((ROOT / row["source_file"]).resolve()),
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


def _new_scores(calls: list[dict]) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / "new_classical_scores.csv"
    if destination.is_file():
        ready = pd.read_csv(destination, encoding="utf-8-sig")
        if len(ready) == len(calls):
            return ready
    request = OUT / "new_classical_request.json"
    request.write_text(json.dumps({"calls": calls}, ensure_ascii=False), encoding="utf-8")
    destination = OUT / "new_classical_scores.csv"
    completed = subprocess.run(
        [
            str(PYTHON),
            str(WORKER),
            "--request",
            str(request),
            "--models",
            str(MODELS),
            "--output",
            str(destination),
            "--features-out",
            str(OUT / "new_classical_features.csv"),
        ],
        cwd=str(CLASSICAL),
        env={**os.environ, "PYTHONPATH": str(CLASSICAL)},
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return pd.read_csv(destination, encoding="utf-8-sig")


def _group(label: str) -> str:
    if label == "автоответчик":
        return "класс автоответчик"
    if label in {"отказ_лпр", "перезвон", "не_лпр", "не_та_организация", "асинхронная_связь"}:
        return "живой диалог по классу"
    return "прочее"


def main() -> None:
    calls = _calls()
    fresh = _new_scores(calls).rename(columns={"status": "classical_status"})
    manifest = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    saved = pd.read_csv(SCORES, encoding="utf-8-sig")
    spectra = pd.read_csv(SPECTRA, usecols=["call_id", "speech_duration_s", "speech_segments", "n_windows"]).rename(
        columns={"speech_duration_s": "spectra_speech_s", "n_windows": "spectra_windows"}
    )
    frame = manifest.merge(saved, on="call_id").merge(fresh, on="call_id").merge(spectra, on="call_id")
    flags = live_flags(frame["call_id"].tolist())
    frame["live_amd"] = frame["call_id"].map(flags)
    frame["spectra_pred"] = [
        prediction("spectra0", score, status) for score, status in zip(frame["spectra0_score"], frame["spectra0_status"])
    ]
    frame["svm_pred"] = [
        prediction("svm_new", score, status) for score, status in zip(frame["svm_new_score"], frame["classical_status"])
    ]
    frame["lgbm_pred"] = [
        prediction("lightgbm_new", score, status) for score, status in zip(frame["lightgbm_new_score"], frame["classical_status"])
    ]
    votes = frame[["spectra_pred", "svm_pred", "lgbm_pred"]]
    ready = votes.apply(lambda column: column.isin(["human", "synthetic"])).all(axis=1)
    frame["vote"] = ""
    frame.loc[ready, "vote"] = (
        frame.loc[ready, ["spectra_pred", "svm_pred", "lgbm_pred"]].eq("synthetic").sum(axis=1) >= 2
    ).map({True: "synthetic", False: "human"})
    frame["class_group"] = frame["prior_classification"].map(_group)
    keep = [
        "call_id",
        "prior_classification",
        "outcome",
        "live_amd",
        "class_group",
        "duration_seconds",
        "spectra_speech_s",
        "spectra_windows",
        "spectra0_score",
        "spectra_pred",
        "svm_new_score",
        "svm_pred",
        "lightgbm_new_score",
        "lgbm_pred",
        "vote",
    ]
    frame[keep].to_csv(OUT / "vote.csv", index=False, encoding="utf-8-sig")
    print("n", len(frame), "с тремя оценками", int(ready.sum()))
    print(frame["class_group"].value_counts().to_string())
    print("--- класс x голос ---")
    print(pd.crosstab(frame["class_group"], frame["vote"], dropna=False).to_string())
    print("--- класс x модели ---")
    for column in ("spectra_pred", "svm_pred", "lgbm_pred", "vote"):
        print(column)
        print(pd.crosstab(frame["prior_classification"], frame[column], dropna=False).to_string())
    print("--- исход x голос ---")
    print(pd.crosstab(frame["outcome"].fillna(""), frame["vote"], dropna=False).to_string())
    print("--- живой ключ x голос ---")
    print(pd.crosstab(frame["live_amd"].fillna(""), frame["vote"], dropna=False).to_string())
    human_vote = frame[frame["vote"] == "human"]
    print("--- голос human ---")
    for row in human_vote.itertuples(index=False):
        print(
            row.call_id,
            row.prior_classification,
            row.outcome,
            "live",
            row.live_amd,
            "speech",
            row.spectra_speech_s,
            "spectra",
            row.spectra_pred,
            f"{float(row.spectra0_score):.2f}",
            "svm",
            row.svm_pred,
            "lgbm",
            row.lgbm_pred,
        )


if __name__ == "__main__":
    main()
