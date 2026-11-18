from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.aggregation.call_aggregation import aggregate_calls
from src.audio.loader import read_wav, select_client_channel
from src.audio.preprocessing import prepare_analysis_audio
from src.audio.quality import recording_quality, rms
from src.audio.segmentation import detect_speech
from src.data.feature_store import FeatureStore
from src.features.extractor import empty_features, extract_segment_features
from src.features.schema import FLAG_COLUMNS, METADATA_COLUMNS


def extract_dataset(frame: pd.DataFrame, cfg: dict, store: FeatureStore, logger) -> dict[str, int]:
    done = store.completed()
    stats = {"seen": 0, "skipped": 0, "ok": 0, "no_speech": 0, "errors": 0}
    for row in frame.itertuples(index=False):
        call_id = str(row.call_id)
        stats["seen"] += 1
        if call_id in done:
            stats["skipped"] += 1
            continue
        started = time.perf_counter()
        try:
            outcome = _one_call(row, cfg)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            message = f"{exc.__class__.__name__}: {exc}"[:300]
            store.append_error({"call_id": call_id, "label": getattr(row, "label", ""), "error": message})
            store.append_status(_status(row, "error", 0, 0, elapsed_ms, message))
            stats["errors"] += 1
            logger.info("error call_id=%s %s", call_id, message)
            continue
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if outcome["segments"]:
            store.append_segments(outcome["segments"])
        store.append_status(_status(row, outcome["status"], outcome["n_segments"], outcome["n_usable"], elapsed_ms, ""))
        store.mark_completed(call_id)
        stats[outcome["status"]] = stats.get(outcome["status"], 0) + 1
        logger.info(
            "call_id=%s label=%s status=%s segments=%s usable=%s ms=%.0f",
            call_id,
            row.label,
            outcome["status"],
            outcome["n_segments"],
            outcome["n_usable"],
            elapsed_ms,
        )
    _write_calls(store)
    return stats


def _one_call(row, cfg: dict) -> dict:
    if not bool(row.file_exists):
        raise FileNotFoundError("WAV не найден")
    data, sample_rate = read_wav(row.resolved_path)
    override = cfg["audio"]["channel_override"]
    channel, channel_index, channel_reason = select_client_channel(
        data,
        manifest_channel=getattr(row, "client_channel", ""),
        config_channel=int(cfg["audio"]["client_channel"]),
        channel_override=None if override is None else int(override),
    )
    audio, analysis_rate, resample_action = prepare_analysis_audio(
        channel,
        sample_rate,
        target_sample_rate=int(cfg["audio"]["target_sample_rate"]),
        allow_upsample=bool(cfg["audio"]["allow_upsample"]),
    )
    segments = detect_speech(
        audio,
        analysis_rate,
        vad_mode=int(cfg["vad"]["mode"]),
        frame_ms=int(cfg["vad"]["frame_ms"]),
        merge_gap_ms=int(cfg["vad"]["merge_gap_ms"]),
    )
    quality = recording_quality(
        audio,
        analysis_rate,
        segments,
        min_noise_ms=float(cfg["quality"]["min_noise_ms"]),
    )
    rows = []
    min_duration = float(cfg["vad"]["min_feature_ms"]) / 1000.0
    min_rms = float(cfg["quality"]["min_segment_rms"])
    low_snr = float(cfg["quality"]["low_snr_db"])
    n_usable = 0
    for index, (start, end) in enumerate(segments):
        left = max(0, int(round(start * analysis_rate)))
        right = min(len(audio), int(round(end * analysis_rate)))
        piece = audio[left:right]
        duration = len(piece) / float(analysis_rate) if analysis_rate else 0.0
        energy = rms(piece)
        insufficient = duration < min_duration
        low_energy = (not np.isfinite(energy)) or energy < min_rms
        usable = not insufficient and not low_energy
        if usable:
            features, flags = extract_segment_features(piece, analysis_rate, cfg)
            n_usable += 1
        else:
            features = empty_features(cfg)
            flags = {name: 0 for name in FLAG_COLUMNS}
            flags["flag_insufficient_duration"] = int(insufficient)
            flags["flag_low_energy"] = int(low_energy)
            flags["voice_quality_reliability"] = "not_extracted"
        flags["flag_low_snr"] = int(np.isfinite(quality["snr_db"]) and float(quality["snr_db"]) < low_snr)
        flags["flag_snr_unavailable"] = int(quality["snr_method"] == "unavailable")
        if any(np.isinf(value) for value in features.values() if isinstance(value, float)):
            flags["flag_nonfinite"] = 1
        reliability = flags.pop("voice_quality_reliability", "")
        record = {column: "" for column in METADATA_COLUMNS}
        record.update(features)
        record.update(flags)
        record.update(
            {
                "call_id": str(row.call_id),
                "label": row.label,
                "locked_split": row.locked_split,
                "segment_id": f"{row.call_id}:{index}",
                "start_time": round(start, 3),
                "end_time": round(end, 3),
                "duration": round(duration, 3),
                "speech_ratio": quality["speech_ratio"],
                "silence_ratio": quality["silence_ratio"],
                "energy_rms": energy,
                "snr_db": quality["snr_db"],
                "snr_method": quality["snr_method"],
                "recording_duration_s": quality["recording_duration_s"],
                "speech_duration_s": quality["speech_duration_s"],
                "sample_rate_hz": sample_rate,
                "analysis_sample_rate_hz": analysis_rate,
                "resample_action": resample_action,
                "source_channels": getattr(row, "source_channels", ""),
                "channel_index": channel_index,
                "channel_reason": channel_reason,
                "usable_for_features": int(usable),
                "classification_result": getattr(row, "classification_result", ""),
                "outcome": getattr(row, "outcome", ""),
                "campaign_name": getattr(row, "campaign_name", ""),
                "timestamp": getattr(row, "timestamp", ""),
                "voice_quality_reliability": reliability,
            }
        )
        rows.append(record)
    if not segments:
        status = "no_speech"
    elif n_usable == 0:
        status = "insufficient_speech"
    else:
        status = "ok"
    return {"segments": rows, "status": status, "n_segments": len(rows), "n_usable": n_usable}


def _status(row, status: str, n_segments: int, n_usable: int, elapsed_ms: float, error: str) -> dict:
    return {
        "call_id": str(row.call_id),
        "label": row.label,
        "locked_split": row.locked_split,
        "status": status,
        "n_segments": n_segments,
        "n_usable": n_usable,
        "processing_ms": round(elapsed_ms, 1),
        "error": error,
    }


def _write_calls(store: FeatureStore) -> None:
    if not store.segments_path.exists():
        segments = pd.DataFrame()
    else:
        segments = pd.read_csv(store.segments_path, dtype={"call_id": str}, keep_default_na=False)
    calls = aggregate_calls(segments) if not segments.empty else pd.DataFrame()
    if not store.status_path.exists():
        store.write_calls(calls)
        return
    status = pd.read_csv(store.status_path, dtype={"call_id": str}, keep_default_na=False)
    status = status.drop_duplicates("call_id", keep="last")
    if calls.empty:
        store.write_calls(status)
        return
    overlap = (set(status.columns) & set(calls.columns)) - {"call_id"}
    merged = status.merge(calls.drop(columns=list(overlap)), on="call_id", how="left")
    store.write_calls(merged)
