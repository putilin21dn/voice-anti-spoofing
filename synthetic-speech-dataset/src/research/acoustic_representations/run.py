"""Run the offline acoustic-representation study. Does not call Spectra-0 or AASIST3."""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.acoustic_representations.analyze import write_analysis
from src.research.acoustic_representations.config import (
    FORBIDDEN_COLUMNS,
    HUMAN_LABELS,
    SAMPLE_RATE,
    SYNTH_LABELS,
    config_document,
)
from src.research.acoustic_representations.extract import (
    call_feature_columns,
    comparison_columns,
    extract_call,
    segment_columns,
)
from src.research.acoustic_representations.features import (
    cqt_top_hz,
    cqcc_librosa,
    cqcc_spafe,
    lfcc_matrix,
    log_safe,
    mel_power,
    spafe_effective_band,
    spectral_shape,
    stft_power,
)
from src.research.acoustic_representations.plots import plot_protocol, plot_summaries
from src.research.acoustic_representations.signal import (
    binary_role,
    campaign_code,
    load_subscriber,
    nonspeech_rms,
    padding_stats,
    peak_normalize,
    preemphasis,
    read_csv,
    reconstruct_intervals,
    write_csv,
)

warnings.filterwarnings("ignore", message=r"n_fft=.*too large for input signal")

ROOT = Path(__file__).resolve().parents[3]
META_COLUMNS = [
    "call_id",
    "status",
    "primary_label",
    "secondary_label",
    "binary_group",
    "campaign",
    "spectra_error_role",
    "spectra0_predicted_class_frozen",
    "duration_s",
    "n_speech_segments",
    "speech_duration_s",
    "speech_ratio",
    "silence_ratio",
    "raw_speech_rms",
    "raw_speech_peak",
    "raw_clip_fraction",
    "nonspeech_rms",
    "speech_to_nonspeech_db",
    "explanatory_f0_median_hz",
    "explanatory_f0_iqr_hz",
    "explanatory_f0_voiced_fraction",
    "spectra_n_windows",
    "spectra_n_padded_windows",
    "spectra_valid_audio_s",
    "spectra_repeated_audio_s",
    "spectra_padding_ratio",
    "spectra_min_tail_s",
    "spectra_n_tails_below_0_1s",
    "spectra_n_tails_below_0_5s",
]


def cell(value) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return "1" if bool(value) else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return ""
    return f"{number:.8g}"


def self_check() -> None:
    time_axis = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    raw = 0.2 * np.sin(2 * np.pi * 200 * time_axis) + 0.2 * np.sin(2 * np.pi * 1000 * time_axis)
    emphasized = preemphasis(raw)
    raw_centroid = spectral_shape(stft_power(raw))["centroid_hz_mean"]
    pre_centroid = spectral_shape(stft_power(emphasized))["centroid_hz_mean"]
    if not pre_centroid > raw_centroid + 20:
        raise RuntimeError(f"pre-emphasis не поднял центроид: {raw_centroid:.1f} -> {pre_centroid:.1f}")
    normalized, gain = peak_normalize(raw)
    if abs(float(np.max(np.abs(normalized))) - 0.99) > 1e-6 or not gain > 1:
        raise RuntimeError("peak-norm не привёл пик к 0.99")
    silence = log_safe(stft_power(np.zeros(SAMPLE_RATE)))
    if not np.isfinite(silence).all():
        raise RuntimeError("log(0) не защищён")
    power = stft_power(emphasized)
    if mel_power(power).shape[1] != 26 or lfcc_matrix(power).shape[1] != 20:
        raise RuntimeError("неверная ширина log-mel или LFCC")
    band = spafe_effective_band()
    if not (500 < band["effective_fmin_hz"] < 600 and band["effective_fmax_hz"] < 4000):
        raise RuntimeError(f"неожиданная полоса spafe: {band}")
    if cqt_top_hz() >= 4000:
        raise RuntimeError("CQCC librosa вышел за Найквист")
    if cqcc_librosa(emphasized).shape[1] != 13 or cqcc_spafe(emphasized).shape[1] != 13:
        raise RuntimeError("CQCC вернул не 13 коэффициентов")
    features, segments, comparison, _quality = extract_call("self-check", raw, [(0, 0.0, 1.0)])
    if segments[0]["status"] != "ok" or not np.isfinite(features["logmel_mean"]):
        raise RuntimeError("extract_call не посчитал сегмент в 1 с")
    if not comparison["preemph_centroid_hz_mean"] > comparison["raw_centroid_hz_mean"]:
        raise RuntimeError("сравнение raw и pre-emphasis не разделилось")
    short = extract_call("self-check-short", raw, [(0, 0.0, 0.01)])
    if short[1][0]["status"] != "too_short":
        raise RuntimeError("короткий сегмент был проглочен")
    print(
        "self_check ok",
        f"spafe {band['effective_fmin_hz']:.1f}-{band['effective_fmax_hz']:.1f} Hz",
        f"cqt_top {cqt_top_hz():.1f} Hz",
    )


def load_records() -> tuple[list[dict], dict, dict]:
    annotations = read_csv(ROOT / "data" / "annotations" / "day_20261113_calls.csv")
    scores = {row["call_id"]: row for row in read_csv(ROOT / "data" / "manifests" / "day_20261113_scores.csv")}
    spectra = {
        row["call_id"]: row
        for row in read_csv(ROOT / "evaluation" / "day_20261113" / "20261113T175850" / "call_results" / "spectra0.csv")
    }
    windows = read_csv(ROOT / "evaluation" / "day_20261113" / "20261113T175850" / "window_results" / "spectra0.csv")
    if len(annotations) != 279 or len(scores) != 279 or len(spectra) != 279:
        raise RuntimeError("День 2026-11-13 должен содержать 279 звонков")
    intervals = reconstruct_intervals(windows)
    padding = padding_stats(windows)
    counts = {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "excluded": 0}
    records = []
    for row in annotations:
        if row["client_channel"] != "0" or row["sample_rate"] != "8000":
            raise RuntimeError(f"{row['call_id']}: ожидались канал 0 и 8000 Гц")
        margin = scores[row["call_id"]]["spectra0_score"]
        role, predicted = binary_role(row["primary_label"], margin)
        counts[role] += 1
        spectra_row = spectra[row["call_id"]]
        if spectra_row["channel_index"] != "0":
            raise RuntimeError("Spectra-0 считала не канал абонента")
        spans = intervals.get(row["call_id"], [])
        reconstructed = sum(end - start for _, start, end in spans)
        stored = spectra_row["speech_duration_s"]
        if stored != "" and abs(reconstructed - float(stored)) > 0.002:
            raise RuntimeError(f"{row['call_id']}: речевые границы не совпали с Spectra-0")
        if row["primary_label"] in HUMAN_LABELS:
            binary = "human_voice"
        elif row["primary_label"] in SYNTH_LABELS:
            binary = "synthetic"
        else:
            binary = "excluded"
        records.append(
            {
                "call_id": row["call_id"],
                "primary_label": row["primary_label"],
                "secondary_label": row["secondary_label"],
                "binary_group": binary,
                "campaign": campaign_code(row["campaign_name"]),
                "role": role,
                "predicted": predicted,
                "margin": float(margin) if margin else np.nan,
                "duration": float(row["duration_seconds"]),
                "speech_duration": float(stored) if stored else reconstructed,
                "speech_ratio": float(spectra_row["speech_ratio"]) if spectra_row["speech_ratio"] else np.nan,
                "status": "no_speech" if not spans else "ok",
                "source_file": ROOT / row["source_file"],
                "intervals": spans,
                "padding": padding.get(row["call_id"], {}),
            }
        )
    if counts != {"TP": 63, "FP": 32, "TN": 151, "FN": 5, "excluded": 28}:
        raise RuntimeError(f"Контрольная точка Spectra-0 не сошлась: {counts}")
    return records, intervals, {}


def select_protocol(records: list[dict]) -> list[dict]:
    by_id = {row["call_id"]: row for row in records}
    reasons: dict[str, list[str]] = {row["call_id"]: [] for row in records}

    def add(row: dict | None, reason: str, scale: bool = False) -> None:
        if row is None:
            return
        reasons[row["call_id"]].append(reason)
        if scale:
            reasons[row["call_id"]].append("scale_source")

    def closest(pool: list[dict], duration: float, margin: float) -> dict:
        return min(pool, key=lambda row: (abs(row["duration"] - duration), abs(row["margin"] - margin), row["call_id"]))

    humans = [row for row in records if row["primary_label"] == "human_live" and row["role"] == "TN"]
    synths = [row for row in records if row["primary_label"] == "synthetic_tts" and row["role"] == "TP"]
    human_duration = float(np.median([row["duration"] for row in humans]))
    human_margin = float(np.median([row["margin"] for row in humans]))
    synth_duration = float(np.median([row["duration"] for row in synths]))
    synth_margin = float(np.median([row["margin"] for row in synths]))
    typical_human = closest(humans, human_duration, human_margin)
    typical_synth = closest(synths, synth_duration, synth_margin)
    add(typical_human, "typical_human_live_tn", scale=True)
    add(typical_synth, "typical_synthetic_tts_tp", scale=True)
    other_humans = [row for row in humans if row["campaign"] != typical_human["campaign"]]
    other_synths = [row for row in synths if row["campaign"] != typical_synth["campaign"]]
    add(closest(other_humans, human_duration, human_margin) if other_humans else None, "typical_human_other_campaign")
    add(closest(other_synths, synth_duration, synth_margin) if other_synths else None, "typical_synthetic_other_campaign")
    deep_fp = [row for row in records if row["role"] == "FP" and row["margin"] < -2]
    if len(deep_fp) != 3:
        raise RuntimeError("Ожидались три глубоких FP с margin < -2")
    for row in deep_fp:
        add(row, "deep_fp_margin_below_-2")
    fns = [row for row in records if row["role"] == "FN"]
    if len(fns) != 5 or not any(row["call_id"].startswith("8b34a750") for row in fns):
        raise RuntimeError("Пять FN, включая 8b34a750, не найдены")
    for row in fns:
        add(row, "false_negative")
    for binary, label in (("human_voice", "human"), ("synthetic", "synthetic")):
        pool = [row for row in records if row["binary_group"] == binary and row["status"] == "ok"]
        add(min(pool, key=lambda row: (row["duration"], row["call_id"])), f"shortest_{label}")
        add(max(pool, key=lambda row: (row["duration"], row["call_id"])), f"longest_{label}")
        add(min(pool, key=lambda row: (row["speech_ratio"], row["call_id"])), f"lowest_speech_ratio_{label}")
        add(max(pool, key=lambda row: (row["speech_ratio"], row["call_id"])), f"highest_speech_ratio_{label}")
    for row in records:
        if row["primary_label"] == "human_recorded":
            add(row, "human_recorded")
        if row["primary_label"] == "synthetic_playback":
            add(row, "only_synthetic_playback")
    selected = []
    for call_id, tags in reasons.items():
        if not tags:
            continue
        row = by_id[call_id]
        selected.append(
            {
                "call_id": call_id,
                "reasons": "|".join(dict.fromkeys(tags)),
                "primary_label": row["primary_label"],
                "secondary_label": row["secondary_label"],
                "campaign": row["campaign"],
                "spectra_error_role": row["role"],
                "duration_s": row["duration"],
                "speech_ratio": row["speech_ratio"],
                "spectra0_margin_frozen": row["margin"],
            }
        )
    return selected


def _empty_numbers(columns: list[str]) -> dict:
    return {column: np.nan for column in columns}


def build_call_row(record: dict, features: dict | None, quality: dict | None, outside: float) -> dict:
    rms = quality["raw_speech_rms"] if quality else np.nan
    ratio = record["speech_ratio"]
    row = {
        "call_id": record["call_id"],
        "status": record["status"],
        "primary_label": record["primary_label"],
        "secondary_label": record["secondary_label"],
        "binary_group": record["binary_group"],
        "campaign": record["campaign"],
        "spectra_error_role": record["role"],
        "spectra0_predicted_class_frozen": record["predicted"],
        "duration_s": record["duration"],
        "n_speech_segments": len(record["intervals"]),
        "speech_duration_s": record["speech_duration"],
        "speech_ratio": ratio,
        "silence_ratio": 1.0 - ratio if np.isfinite(ratio) else np.nan,
        "nonspeech_rms": outside,
        "speech_to_nonspeech_db": (
            20.0 * np.log10(rms / outside) if np.isfinite(rms) and np.isfinite(outside) and rms > 0 and outside > 0 else np.nan
        ),
    }
    row.update(_empty_numbers([column for column in META_COLUMNS if column not in row]))
    if quality:
        row.update(quality)
    row.update(record["padding"])
    if features:
        row.update(features)
    else:
        row.update(_empty_numbers(call_feature_columns()))
    return row


def flush(output_dir: Path, calls: list[dict], segments: list[dict], comparisons: list[dict]) -> None:
    columns = META_COLUMNS + call_feature_columns()
    forbidden = set(columns) & set(FORBIDDEN_COLUMNS)
    if forbidden:
        raise RuntimeError(f"Запрещённые колонки: {sorted(forbidden)}")
    write_csv(output_dir / "call_level_features.csv", [{key: cell(row.get(key, "")) for key in columns} for row in calls], columns)
    write_csv(
        output_dir / "segment_level_features.csv",
        [{key: cell(row.get(key, "")) for key in segment_columns()} for row in segments],
        segment_columns(),
    )
    write_csv(
        output_dir / "log_normalization_comparison.csv",
        [{key: cell(row.get(key, "")) for key in comparison_columns()} for row in comparisons],
        comparison_columns(),
    )


def extract(output_dir: Path, records: list[dict], limit: int | None, resume: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    calls: list[dict] = []
    segments: list[dict] = []
    comparisons: list[dict] = []
    if resume and (output_dir / "call_level_features.csv").exists():
        calls = read_csv(output_dir / "call_level_features.csv")
        done = {row["call_id"] for row in calls}
        if (output_dir / "segment_level_features.csv").exists():
            segments = [row for row in read_csv(output_dir / "segment_level_features.csv") if row["call_id"] in done]
        if (output_dir / "log_normalization_comparison.csv").exists():
            comparisons = [row for row in read_csv(output_dir / "log_normalization_comparison.csv") if row["call_id"] in done]
        print(f"resume: {len(done)} звонков уже есть", flush=True)
    pending = [row for row in records if row["call_id"] not in done]
    if limit is not None:
        pending = pending[:limit]
    started = time.perf_counter()
    for index, record in enumerate(pending, start=1):
        if record["status"] != "ok":
            calls.append(build_call_row(record, None, None, np.nan))
            comparisons.append({"call_id": record["call_id"]})
        else:
            audio = load_subscriber(record["source_file"])
            outside = nonspeech_rms(audio, record["intervals"])
            features, segment_rows, comparison, quality = extract_call(record["call_id"], audio, record["intervals"])
            calls.append(build_call_row(record, features, quality, outside))
            segments.extend(segment_rows)
            comparisons.append(comparison)
        if index == 1 or index % 10 == 0 or index == len(pending):
            elapsed = time.perf_counter() - started
            print(f"{index}/{len(pending)} {record['call_id'][:8]} {elapsed / index:.1f}s/call", flush=True)
        if index % 25 == 0:
            flush(output_dir, calls, segments, comparisons)
    flush(output_dir, calls, segments, comparisons)
    padding_columns = [
        "call_id",
        "status",
        "primary_label",
        "binary_group",
        "campaign",
        "spectra_error_role",
        "duration_s",
        "n_speech_segments",
        "speech_duration_s",
        "speech_ratio",
        "silence_ratio",
        "spectra_n_windows",
        "spectra_n_padded_windows",
        "spectra_valid_audio_s",
        "spectra_repeated_audio_s",
        "spectra_padding_ratio",
        "spectra_min_tail_s",
        "spectra_n_tails_below_0_1s",
        "spectra_n_tails_below_0_5s",
        "n_stft_frames",
        "stft_discarded_samples",
        "n_segments_shorter_than_cqt_window",
    ]
    write_csv(
        output_dir / "padding_analysis.csv",
        [{key: cell(row.get(key, "")) for key in padding_columns} for row in calls],
        padding_columns,
    )


def write_dataset_summary(output_dir: Path, records: list[dict]) -> None:
    audio_dir = ROOT / "data" / "audio" / "day_20261113"
    extra = [path for path in audio_dir.glob("*.wav") if path.stem not in {row["call_id"] for row in records}]
    rows = [
        ("n_annotated_calls", 279),
        ("n_binary_calls", 251),
        ("n_human_voice", 183),
        ("n_synthetic", 68),
        ("n_excluded_unknown_or_mixed", 28),
        ("spectra0_tp", 63),
        ("spectra0_fp", 32),
        ("spectra0_tn", 151),
        ("spectra0_fn", 5),
        ("n_no_speech", sum(row["status"] == "no_speech" for row in records)),
        ("n_wav_on_disk", len(list(audio_dir.glob("*.wav")))),
        ("n_wav_excluded_shorter_than_15s", len(extra)),
        ("client_channel", 0),
        ("sample_rate_hz", 8000),
        ("spectra0_threshold", 1.02),
        ("model_inference_rerun", 0),
    ]
    write_csv(output_dir / "dataset_summary.csv", [{"metric": key, "value": value} for key, value in rows], ["metric", "value"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "day_20261113_acoustic_representations")
    parser.add_argument("--steps", default="selfcheck,extract,analyze,plots")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    steps = set(args.steps.split(","))
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    if "selfcheck" in steps:
        self_check()
    document = config_document()
    document["measured_spafe_band"] = spafe_effective_band()
    document["measured_librosa_cqt_top_hz"] = cqt_top_hz()
    (output_dir / "feature_extraction_config.json").write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    if steps & {"extract", "analyze", "plots"}:
        records, intervals, _unused = load_records()
        protocol = select_protocol(records)
        write_csv(output_dir / "protocol_selection.csv", protocol, list(protocol[0]))
        write_dataset_summary(output_dir, records)
    else:
        records, intervals, protocol = [], {}, []
    if "extract" in steps:
        extract(output_dir, records, args.limit, args.resume)
    if "analyze" in steps:
        calls = pd.read_csv(output_dir / "call_level_features.csv", encoding="utf-8-sig")
        comparison = pd.read_csv(output_dir / "log_normalization_comparison.csv", encoding="utf-8-sig")
        write_analysis(output_dir, calls, comparison)
    if "plots" in steps:
        calls = pd.read_csv(output_dir / "call_level_features.csv", encoding="utf-8-sig")
        protocol_frame = pd.read_csv(output_dir / "protocol_selection.csv", encoding="utf-8-sig")
        audio_paths = {row["call_id"]: row["source_file"] for row in records}
        plot_protocol(ROOT, output_dir, protocol_frame, intervals, audio_paths)
        plot_summaries(output_dir, calls)


if __name__ == "__main__":
    main()
