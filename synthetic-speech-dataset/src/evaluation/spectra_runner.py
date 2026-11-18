from __future__ import annotations

import importlib.util
import sys
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from src.evaluation.metrics import SPECTRA0_THRESHOLD, spectra0_class


def load_modules(spectra_root: Path, webrtc_site: Path):
    if str(webrtc_site) not in sys.path:
        sys.path.append(str(webrtc_site))
    inference_path = spectra_root / "src" / "run_inference.py"
    spec = importlib.util.spec_from_file_location("spectra_run_inference", inference_path)
    inference = importlib.util.module_from_spec(spec)
    sys.modules["spectra_run_inference"] = inference
    sys.modules["run_inference"] = inference
    spec.loader.exec_module(inference)
    aasist_path = spectra_root / "src" / "run_aasist3.py"
    spec = importlib.util.spec_from_file_location("spectra_run_aasist3", aasist_path)
    aasist = importlib.util.module_from_spec(spec)
    sys.modules["spectra_run_aasist3"] = aasist
    spec.loader.exec_module(aasist)
    inserted = str(aasist_path.parent)
    if sys.path and sys.path[0] == inserted:
        sys.path.pop(0)
    return inference, aasist


def score_files(inference, aasist, calls: list[dict], device: str) -> tuple[list[dict], list[dict], list[dict]]:
    import torch

    torch.set_num_threads(min(4, os_cpu()))
    print(f"Spectra-0: {len(calls)} звонков, устройство {device}", flush=True)
    spectra_model = inference.load_model(device)
    aasist_model = None
    windows: list[dict] = []
    aasist_windows: list[dict] = []
    calls_out: list[dict] = []
    try:
        prepared = [_prepare(inference, call) for call in calls]
        for index, item in enumerate(prepared, start=1):
            print(f"Spectra-0 {index}/{len(prepared)} {item['call']['call_id']}", flush=True)
            calls_out.append(_spectra_call(inference, spectra_model, item, device, windows))
    finally:
        del spectra_model
    print("AASIST3: загрузка готовой модели", flush=True)
    aasist_model = aasist.load_model(device)
    try:
        prepared = [_prepare(inference, call) for call in calls]
        for index, item in enumerate(prepared, start=1):
            print(f"AASIST3 {index}/{len(prepared)} {item['call']['call_id']}", flush=True)
            _aasist_call(inference, aasist, aasist_model, item, device, aasist_windows, calls_out)
    finally:
        del aasist_model
    return windows, aasist_windows, calls_out


def _prepare(inference, call: dict) -> dict:
    data, sample_rate = inference.read_stereo(call["wav_path"])
    channel_index = int(call["client_channel"])
    if channel_index >= data.shape[1]:
        raise ValueError("Канал абонента отсутствует")
    mono = data[:, channel_index]
    intervals = inference.detect_speech(mono, sample_rate, 2, 300, 500)
    divisor = gcd(inference.TARGET_SR, sample_rate)
    audio16 = mono if sample_rate == inference.TARGET_SR else resample_poly(
        mono, inference.TARGET_SR // divisor, sample_rate // divisor
    ).astype(np.float32)
    duration = len(mono) / sample_rate
    speech = sum(end - start for start, end in intervals)
    return {
        "call": call,
        "sample_rate": sample_rate,
        "channels": int(data.shape[1]),
        "duration": duration,
        "intervals": intervals,
        "audio16": np.ascontiguousarray(audio16, dtype=np.float32),
        "speech": speech,
    }


def _spectra_call(inference, model, item: dict, device: str, windows: list[dict]) -> dict:
    call = item["call"]
    margins = []
    spoof_windows = 0
    bonafide_windows = 0
    try:
        for segment_index, (start, end) in enumerate(item["intervals"]):
            for window_index, (window_start, chunk, valid_sec, padded) in enumerate(
                inference.make_segment_windows(item["audio16"], start, end)
            ):
                spoof, bonafide, spoof_probability, bonafide_probability, _elapsed = inference.infer_one(
                    model, chunk, device
                )
                margin = bonafide - spoof
                window_class = "spoof" if spoof > bonafide else "bonafide"
                spoof_windows += int(window_class == "spoof")
                bonafide_windows += int(window_class == "bonafide")
                margins.append(margin)
                windows.append(
                    _window_row(
                        call, item, segment_index, window_index, window_start, valid_sec, padded,
                        spoof, bonafide, spoof_probability, bonafide_probability, margin, window_class,
                    )
                )
        if not margins:
            return _call_row(call, item, "no_speech", "Нет речевых окон Spectra-0", 0, 0, 0, "", "")
        score = float(np.mean(margins))
        return _call_row(
            call, item, "ok", "", len(margins), spoof_windows, bonafide_windows, score, spectra0_class(score)
        )
    except Exception as exc:
        return _call_row(call, item, "error", exc.__class__.__name__, len(margins), spoof_windows, bonafide_windows, "", "")


def _aasist_call(inference, aasist, model, item: dict, device: str, windows: list[dict], calls_out: list[dict]) -> None:
    call = item["call"]
    audio = aasist.preemphasis_wave(item["audio16"])
    pending = []
    collected: list[dict] = []

    def flush() -> None:
        if not pending:
            return
        logits, probabilities, _elapsed = aasist.infer_windows(model, [row["chunk"] for row in pending], device)
        for row, logit, probability in zip(pending, logits, probabilities):
            spoof, bonafide = float(logit[0]), float(logit[1])
            window_class = "spoof" if spoof > bonafide else "bonafide"
            stored = {key: value for key, value in row.items() if key != "chunk"}
            stored.update(
                {
                    "spoof_logit": spoof,
                    "bonafide_logit": bonafide,
                    "spoof_softmax": float(probability[0]),
                    "bonafide_softmax": float(probability[1]),
                    "margin": bonafide - spoof,
                    "predicted_class": window_class,
                    "status": "ok",
                }
            )
            windows.append(stored)
            collected.append(stored)
        pending.clear()

    try:
        for segment_index, (start, end) in enumerate(item["intervals"]):
            for window_index, (window_start, chunk, valid_sec, padded) in enumerate(
                inference.make_segment_windows(audio, start, end)
            ):
                pending.append(
                    {
                        **_window_row(
                            call, item, segment_index, window_index, window_start, valid_sec, padded,
                            "", "", "", "", "", "",
                        ),
                        "chunk": np.ascontiguousarray(chunk, dtype=np.float32),
                    }
                )
                if len(pending) >= 2:
                    flush()
        flush()
        if not collected:
            calls_out.append(_aasist_row(call, item, "no_speech", "Нет речевых окон AASIST3", 0, 0, 0, "", "", ""))
            return
        calls_out.append(
            _aasist_row(
                call,
                item,
                "ok",
                "",
                len(collected),
                sum(row["predicted_class"] == "spoof" for row in collected),
                sum(row["predicted_class"] == "bonafide" for row in collected),
                float(np.mean([row["bonafide_logit"] for row in collected])),
                float(np.mean([row["spoof_logit"] for row in collected])),
                float(np.mean([row["margin"] for row in collected])),
            )
        )
    except Exception as exc:
        calls_out.append(_aasist_row(call, item, "error", exc.__class__.__name__, 0, 0, 0, "", "", ""))


def _window_row(call, item, segment_index, window_index, window_start, valid_sec, padded, spoof, bonafide, spoof_probability, bonafide_probability, margin, window_class) -> dict:
    return {
        "call_id": call["call_id"],
        "window_id": f"{call['call_id']}:{segment_index}:{window_index}",
        "segment_index": segment_index,
        "window_index": window_index,
        "start_time": round(float(window_start), 3),
        "end_time": round(float(window_start) + float(valid_sec), 3),
        "valid_speech_audio_sec": round(float(valid_sec), 3),
        "padded_to_model_length": bool(padded),
        "spoof_logit": spoof,
        "bonafide_logit": bonafide,
        "spoof_softmax": spoof_probability,
        "bonafide_softmax": bonafide_probability,
        "margin": margin,
        "predicted_class": window_class,
        "channel_index": call["client_channel"],
        "sample_rate": item["sample_rate"],
        "threshold_used_for_window": "",
    }


def _call_row(call, item, status, error, windows, spoof_windows, bonafide_windows, score, predicted) -> dict:
    duration = item["duration"]
    speech = item["speech"]
    return {
        "model": "spectra0",
        "call_id": call["call_id"],
        "status": status,
        "error": error,
        "n_windows": windows,
        "n_spoof_windows": spoof_windows,
        "n_bonafide_windows": bonafide_windows,
        "score": score,
        "predicted_class": predicted,
        "threshold": SPECTRA0_THRESHOLD,
        "threshold_applied": True,
        "speech_segments": len(item["intervals"]),
        "speech_duration_s": f"{speech:.3f}",
        "speech_ratio": f"{(speech / duration):.6f}" if duration else "",
        "duration_s": f"{duration:.3f}",
        "sample_rate": item["sample_rate"],
        "channel_count": item["channels"],
        "channel_index": call["client_channel"],
        "mean_bonafide_logit": "",
        "mean_spoof_logit": "",
    }


def _aasist_row(call, item, status, error, windows, spoof_windows, bonafide_windows, bonafide, spoof, margin) -> dict:
    row = _call_row(call, item, status, error, windows, spoof_windows, bonafide_windows, margin, "")
    row["model"] = "aasist3"
    row["threshold"] = ""
    row["threshold_applied"] = False
    row["score"] = bonafide
    row["mean_bonafide_logit"] = bonafide
    row["mean_spoof_logit"] = spoof
    row["predicted_class"] = ""
    return row


def os_cpu() -> int:
    import os

    return os.cpu_count() or 1
