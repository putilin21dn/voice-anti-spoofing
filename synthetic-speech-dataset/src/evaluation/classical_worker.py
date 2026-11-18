"""Запускается Python-ом classical-anti-spoofing. Свой `src` туда не подключать.

Обучение не выполняется. Берутся сохранённые bundle и существующий расчёт признаков.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.aggregation.call_aggregation import aggregate_calls
from src.features.extractor import feature_names
from src.features.pipeline import _one_call
from src.models.training import load_bundle, predict_frame, score_frame
from src.utils.config import load_config


NAMES = ("logistic", "svm", "lightgbm")
AGGREGATIONS = {"mean", "median", "std", "min", "max", "p25", "p75"}


def check_models(cfg: dict) -> dict:
    import lightgbm
    import numpy
    import sklearn

    stems = set(feature_names(cfg))
    models = {}
    errors: list[str] = []
    for name in NAMES:
        bundle = load_bundle(cfg, name, "full")
        features = list(bundle["features"])
        for feature in features:
            if "__" not in feature:
                errors.append(f"{name}: признак без агрегации {feature}")
                continue
            stem, aggregation = feature.rsplit("__", 1)
            if aggregation not in AGGREGATIONS or stem not in stems:
                errors.append(f"{name}: нет входа для {feature}")
        models[name] = {
            "threshold": float(bundle["threshold"]),
            "n_features": len(features),
            "params": bundle.get("params"),
            "score": "decision_function" if name == "svm" else "predict_proba_class1_machine",
            "type": type(bundle["model"]).__name__,
        }
    return {
        "ok": not errors,
        "errors": errors,
        "models": models,
        "versions": {
            "sklearn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "numpy": numpy.__version__,
        },
    }


def score_calls(cfg: dict, calls: list[dict], extra_dir: Path | None = None) -> list[dict]:
    bundles = {name: load_bundle(cfg, name, "full") for name in NAMES}
    if extra_dir is not None:
        import joblib

        for name in ("svm", "lightgbm"):
            path = extra_dir / f"{name}.joblib"
            if not path.is_file():
                raise FileNotFoundError(f"Нет bundle {path.name}")
            bundles[f"{name}_new"] = joblib.load(path)
    rows: list[dict] = []
    total = len(calls)
    for index, call in enumerate(calls, start=1):
        base = {
            "call_id": call["call_id"],
            "channel_index": call.get("client_channel", ""),
            "channel_reason": "",
            "sample_rate": "",
            "duration_s": call.get("duration_seconds", ""),
            "n_segments": 0,
            "n_usable_segments": 0,
            "speech_duration_s": "",
            "speech_ratio": "",
            "status": "ok",
            "error": "",
        }
        try:
            outcome = _one_call(_namespace(call), cfg)
            segments = outcome["segments"]
            base["n_segments"] = outcome["n_segments"]
            base["n_usable_segments"] = outcome["n_usable"]
            if segments:
                first = segments[0]
                base["channel_index"] = first.get("channel_index", base["channel_index"])
                base["channel_reason"] = first.get("channel_reason", "")
                base["sample_rate"] = first.get("analysis_sample_rate_hz", "")
                speech = float(first.get("speech_duration_s") or 0)
                duration = float(first.get("recording_duration_s") or 0)
                base["speech_duration_s"] = f"{speech:.3f}"
                base["speech_ratio"] = f"{(speech / duration):.6f}" if duration else ""
                base["duration_s"] = f"{duration:.3f}" if duration else base["duration_s"]
            if outcome["n_usable"] <= 0:
                base["status"] = outcome["status"]
                base["error"] = "Нет пригодных речевых сегментов"
                rows.extend(_blank_models(base, bundles))
                continue
            frame = aggregate_calls(pd.DataFrame(segments))
            for name, bundle in bundles.items():
                scores = score_frame(bundle, frame)
                predicted = int(predict_frame(bundle, scores)[0])
                rows.append(
                    {
                        **base,
                        "model": name,
                        "score": float(scores[0]),
                        "threshold": float(bundle["threshold"]),
                        "predicted_class": "machine" if predicted == 1 else "human",
                        "score_kind": _score_kind(name, bundle),
                    }
                )
        except Exception as exc:
            base["status"] = "error"
            base["error"] = f"{exc.__class__.__name__}"
            rows.extend(_blank_models(base, bundles))
        if index == 1 or index % 25 == 0 or index == total:
            print(f"классика {index}/{total}", flush=True)
    return rows


def _score_kind(name: str, bundle: dict) -> str:
    model_name = str(bundle.get("name") or name)
    if model_name == "svm" or name.startswith("svm"):
        return "decision_function"
    return "predict_proba_class1_machine"


def _blank_models(base: dict, bundles: dict) -> list[dict]:
    return [
        {
            **base,
            "model": name,
            "score": "",
            "threshold": float(bundle["threshold"]),
            "predicted_class": "",
            "score_kind": _score_kind(name, bundle),
        }
        for name, bundle in bundles.items()
    ]


def _namespace(call: dict) -> SimpleNamespace:
    return SimpleNamespace(
        file_exists=True,
        resolved_path=Path(call["wav_path"]),
        client_channel=call.get("client_channel", ""),
        call_id=call["call_id"],
        label=call.get("primary_label") or call.get("listener_label") or "",
        locked_split="",
        source_channels=call.get("channel_count", ""),
        classification_result=call.get("prior_classification", ""),
        outcome=call.get("outcome", ""),
        campaign_name=call.get("campaign_name", ""),
        timestamp=call.get("timestamp", ""),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--request", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--extra-dir", default="", help="Дополнительные svm.joblib и lightgbm.joblib, имена svm_new и lightgbm_new")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config or None)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if args.check:
        destination.write_text(json.dumps(check_models(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
        return
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    extra = Path(args.extra_dir) if args.extra_dir else None
    rows = score_calls(cfg, request["calls"], extra)
    pd.DataFrame(rows).to_csv(destination, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"CLASSICAL_WORKER {exc.__class__.__name__}: {exc}", file=sys.stderr)
        raise
