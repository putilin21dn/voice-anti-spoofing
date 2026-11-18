from __future__ import annotations

import numpy as np

from src.audio.preprocessing import prepare_analysis_audio, resolve_fmax
from src.audio.segmentation import merge_voice_frames
from src.features.extractor import extract_segment_features
from src.features.lfcc import linear_filterbank


def run_synthetic_checks(cfg: dict) -> list[dict]:
    checks = [
        _check_nyquist(),
        _check_no_upsample(),
        _check_merge_gap(),
        _check_short_segment_kept(),
        _check_sine(cfg),
        _check_filterbank_band(),
    ]
    return checks


def _check_nyquist() -> dict:
    try:
        resolve_fmax(8000, 4001)
        return {"name": "fmax_above_nyquist", "passed": False, "detail": "исключение не возникло"}
    except ValueError:
        return {"name": "fmax_above_nyquist", "passed": True, "detail": "частота выше Найквиста отклонена"}


def _check_no_upsample() -> dict:
    audio = np.zeros(8000, dtype=np.float64)
    signal, rate, action = prepare_analysis_audio(audio, 8000, target_sample_rate=16000, allow_upsample=False)
    passed = rate == 8000 and action == "upsample_refused_native_kept" and len(signal) == 8000
    return {"name": "no_upsample", "passed": passed, "detail": action}


def _check_merge_gap() -> dict:
    flags = [(0.0, True), (0.02, True), (0.50, False), (0.70, True), (0.72, True)]
    segments = merge_voice_frames(flags, 0.02, 0.30)
    passed = len(segments) == 2
    return {"name": "vad_does_not_bridge_gap", "passed": passed, "detail": str(segments)}


def _check_short_segment_kept() -> dict:
    flags = [(0.0, True), (0.02, False)]
    segments = merge_voice_frames(flags, 0.02, 0.30)
    passed = len(segments) == 1 and segments[0][1] - segments[0][0] < 0.1
    return {"name": "short_segment_kept", "passed": passed, "detail": str(segments)}


def _check_filterbank_band() -> dict:
    weights = linear_filterbank(256, 8000, 20, 0.0, 4000.0)
    freqs = np.linspace(0.0, 4000.0, weights.shape[1])
    passed = weights.shape[1] == 129 and float(freqs[-1]) <= 4000.0 and np.all(weights >= 0)
    return {"name": "lfcc_filterbank_band", "passed": bool(passed), "detail": f"bins={weights.shape[1]}"}


def _check_sine(cfg: dict) -> dict:
    sample_rate = 8000
    duration = np.arange(sample_rate, dtype=np.float64) / sample_rate
    audio = 0.5 * np.sin(2 * np.pi * 200.0 * duration)
    features, flags = extract_segment_features(audio, sample_rate, cfg)
    f0 = features.get("f0_median", float("nan"))
    centroid = features.get("spectral_centroid_hz_mean", float("nan"))
    zcr = features.get("zero_crossing_rate_mean", float("nan"))
    spectrum = np.abs(np.fft.rfft(audio * np.hamming(len(audio)))) ** 2
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sample_rate)
    high = float(spectrum[freqs > 1000].sum() / spectrum.sum())
    problems = []
    if not (180 <= f0 <= 220):
        problems.append(f"f0_median={f0}")
    if not (150 <= centroid <= 280):
        problems.append(f"centroid={centroid}")
    if not (0.03 <= zcr <= 0.08):
        problems.append(f"zcr={zcr}")
    if high > 0.05:
        problems.append(f"energy_above_1khz={high}")
    if flags.get("flag_voice_quality_8khz_limited") != 1:
        problems.append("нет флага ограниченной надёжности voice quality на 8 кГц")
    lfcc = [value for key, value in features.items() if key.startswith("lfcc_static_") and key.endswith("_mean")]
    if not lfcc or not any(np.isfinite(value) for value in lfcc):
        problems.append("lfcc не посчитан")
    return {"name": "sine_200hz", "passed": not problems, "detail": "; ".join(problems) if problems else "ok"}
