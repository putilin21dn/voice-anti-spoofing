"""Считает новые logistic, SVM и LightGBM сохранёнными joblib.

Запускается интерпретатором classical-anti-spoofing: там же собраны признаки.
Обучения нет. Пороги не подбираются.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from src.aggregation.call_aggregation import aggregate_calls
from src.features.pipeline import _one_call
from src.utils.config import load_config


def _namespace(call: dict):
    from types import SimpleNamespace

    return SimpleNamespace(
        file_exists=True,
        resolved_path=Path(call["wav_path"]),
        client_channel=call.get("client_channel", ""),
        call_id=call["call_id"],
        label="",
        locked_split="",
        source_channels=call.get("channel_count", ""),
        classification_result=call.get("prior_classification", ""),
        outcome=call.get("outcome", ""),
        campaign_name=call.get("campaign_name", ""),
        timestamp=call.get("timestamp", ""),
    )


def _speech(segments: list[dict]) -> tuple[str, str, str]:
    if not segments:
        return "", "", ""
    first = segments[0]
    speech = first.get("speech_duration_s") or ""
    duration = first.get("recording_duration_s") or ""
    try:
        speech_value = float(speech)
        duration_value = float(duration)
        ratio = f"{(speech_value / duration_value):.6f}" if duration_value else ""
        return f"{speech_value:.3f}", ratio, f"{duration_value:.3f}" if duration else ""
    except (TypeError, ValueError):
        return str(speech), "", str(duration)


def _scores(bundle: dict, frame: pd.DataFrame, name: str) -> str:
    features = list(bundle["features"])
    matrix = frame.reindex(columns=features)
    model = bundle["model"]
    if name == "svm":
        value = model.decision_function(matrix)[0]
    else:
        value = model.predict_proba(matrix)[0, 1]
    return f"{float(value):.17g}"


def score_request(cfg: dict, calls: list[dict], model_dir: Path) -> tuple[list[dict], list[dict]]:
    bundles = {name: joblib.load(model_dir / f"{name}.joblib") for name in ("logistic", "svm", "lightgbm")}
    score_rows: list[dict] = []
    feature_rows: list[dict] = []
    for index, call in enumerate(calls, start=1):
        base = {
            "call_id": call["call_id"],
            "n_segments": 0,
            "n_usable_segments": 0,
            "speech_duration_s": "",
            "speech_ratio": "",
            "status": "ok",
            "error": "",
            "logistic_new_score": "",
            "svm_new_score": "",
            "lightgbm_new_score": "",
        }
        try:
            outcome = _one_call(_namespace(call), cfg)
            base["n_segments"] = outcome["n_segments"]
            base["n_usable_segments"] = outcome["n_usable"]
            speech, ratio, _duration = _speech(outcome["segments"])
            base["speech_duration_s"] = speech
            base["speech_ratio"] = ratio
            if outcome["n_usable"] <= 0 or not outcome["segments"]:
                base["status"] = outcome["status"]
                base["error"] = "Нет пригодных речевых сегментов"
            else:
                frame = aggregate_calls(pd.DataFrame(outcome["segments"]))
                feature_rows.append(frame.iloc[0].to_dict())
                for name, bundle in bundles.items():
                    base[f"{name}_new_score"] = _scores(bundle, frame, name)
        except Exception as exc:
            base["status"] = "error"
            base["error"] = exc.__class__.__name__
        score_rows.append(base)
        if index % 25 == 0 or index == len(calls):
            print(f"новые classical {index}/{len(calls)}", flush=True)
    return score_rows, feature_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--features-out", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    cfg = load_config(None)
    scores, features = score_request(cfg, request["calls"], Path(args.models))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scores).to_csv(destination, index=False, encoding="utf-8-sig")
    pd.DataFrame(features).to_csv(Path(args.features_out), index=False, encoding="utf-8-sig")
    print(f"новые оценки: {destination}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"NEW_CLASSICAL {exc.__class__.__name__}: {exc}", file=sys.stderr)
        raise
