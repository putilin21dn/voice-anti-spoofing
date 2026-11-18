import argparse
import importlib.util
import json
import sys
import time
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import webrtcvad
from scipy.signal import resample_poly
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "spectra_0"
OUTPUT_DIR = ROOT / "data" / "output"
TARGET_SR = 16000
MODEL_WINDOW = 64600  # 4.0375 sec at 16 kHz
PREEMPHASIS = 0.97
VAD_SR = 8000
VAD_FRAME_MS = 20
VAD_FRAME_SAMPLES = VAD_SR * VAD_FRAME_MS // 1000


def load_model(device):
    model_py = MODEL_DIR / "model.py"
    if not model_py.exists() or not (MODEL_DIR / "model.safetensors").exists():
        raise FileNotFoundError("Model missing. Run: python src/download_model.py")
    spec = importlib.util.spec_from_file_location("spectra0_impl", model_py)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.spectra_0.from_pretrained(str(MODEL_DIR)).eval().to(device)


def read_stereo(path):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if data.shape[0] == 0:
        raise ValueError("Empty audio")
    return data, int(sr)


def to_pcm16_8k(audio, sr):
    x = audio.astype(np.float32)
    if sr != VAD_SR:
        g = gcd(VAD_SR, sr)
        x = resample_poly(x, VAD_SR // g, sr // g).astype(np.float32)
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype("<i2")


def detect_speech(audio, sr, vad_mode=2, merge_gap_ms=300, min_speech_ms=500):
    """WebRTC VAD at 8 kHz; returns speech intervals in seconds on original timeline."""
    pcm = to_pcm16_8k(audio, sr)
    vad = webrtcvad.Vad(vad_mode)
    frame_n = VAD_FRAME_SAMPLES
    flags = []
    for pos in range(0, len(pcm) - frame_n + 1, frame_n):
        frame = pcm[pos:pos + frame_n].tobytes()
        flags.append((pos / VAD_SR, vad.is_speech(frame, VAD_SR)))

    raw = []
    active_start = None
    last_voice_end = None
    for start, is_voice in flags:
        end = start + VAD_FRAME_MS / 1000.0
        if is_voice:
            if active_start is None:
                active_start = start
            last_voice_end = end
        elif active_start is not None and (start - last_voice_end) * 1000 > merge_gap_ms:
            raw.append((active_start, last_voice_end))
            active_start = None
            last_voice_end = None
    if active_start is not None:
        raw.append((active_start, last_voice_end))

    min_s = min_speech_ms / 1000.0
    duration = len(audio) / sr
    return [(max(0.0, s), min(duration, e)) for s, e in raw if e - s >= min_s]


def make_segment_windows(audio16, start_sec, end_sec):
    start = max(0, int(round(start_sec * TARGET_SR)))
    end = min(len(audio16), int(round(end_sec * TARGET_SR)))
    segment = audio16[start:end]
    if len(segment) == 0:
        return []
    chunks = []
    for offset in range(0, len(segment), MODEL_WINDOW):
        chunk = segment[offset:offset + MODEL_WINDOW]
        valid = len(chunk)
        if valid < MODEL_WINDOW:
            # Repeat only within this same speech segment; never bridge separate segments.
            chunk = np.tile(chunk, MODEL_WINDOW // valid + 1)[:MODEL_WINDOW]
        chunk_start = start_sec + offset / TARGET_SR
        chunks.append((chunk_start, chunk, valid / TARGET_SR, valid < MODEL_WINDOW))
    return chunks


def preemphasis(x):
    return torch.cat([x[:, :1], x[:, 1:] - PREEMPHASIS * x[:, :-1]], dim=1)


def infer_one(model, chunk, device):
    x = preemphasis(torch.from_numpy(chunk).unsqueeze(0)).to(device)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) * 1000
    s_spoof, s_real = float(logits[0, 0]), float(logits[0, 1])
    return s_spoof, s_real, float(probs[0, 0]), float(probs[0, 1]), elapsed


def main():
    ap = argparse.ArgumentParser(description="Spectra-0 inference on VAD speech, independently per channel")
    ap.add_argument("--input", default=str(ROOT / "data" / "input"))
    ap.add_argument("--output", default=str(OUTPUT_DIR / "spectra0_results.csv"))
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--channels", choices=["both", "0", "1"], default="both")
    ap.add_argument("--swap-channels", action="store_true", help="Swap channel labels (e.g. Excel imports)")
    ap.add_argument("--vad-mode", type=int, choices=[0, 1, 2, 3], default=2)
    ap.add_argument("--merge-gap-ms", type=int, default=300)
    ap.add_argument("--min-speech-ms", type=int, default=500)
    args = ap.parse_args()

    src = Path(args.input)
    wavs = [src] if src.is_file() and src.suffix.lower() == ".wav" else (
        sorted(p for p in src.rglob("*.wav")) if src.is_dir() else [])
    if not wavs:
        raise FileNotFoundError(f"No WAV files found: {src}")

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    print("Device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    model = load_model(device)
    rows = []
    for path in tqdm(wavs, desc="WAV files"):
        try:
            data, original_sr = read_stereo(path)
            channel_indices = list(range(data.shape[1])) if args.channels == "both" else [int(args.channels)]
            for channel_index in channel_indices:
                if channel_index >= data.shape[1]:
                    rows.append({"file": path.name, "channel_index": channel_index, "status": "error",
                                 "error": f"File has only {data.shape[1]} channel(s)"})
                    continue
                logical_channel = 1 - channel_index if args.swap_channels and data.shape[1] == 2 else channel_index
                channel_name = "client" if logical_channel == 0 else "bot" if logical_channel == 1 else f"channel_{logical_channel}"
                mono = data[:, channel_index]
                intervals = detect_speech(mono, original_sr, args.vad_mode, args.merge_gap_ms, args.min_speech_ms)
                g = gcd(TARGET_SR, original_sr)
                audio16 = mono if original_sr == TARGET_SR else resample_poly(mono, TARGET_SR // g, original_sr // g).astype(np.float32)
                if not intervals:
                    rows.append({"file": path.name, "channel_index": channel_index, "channel": channel_name,
                                 "original_sample_rate": original_sr, "duration_sec": round(len(mono)/original_sr, 3),
                                 "speech_segments": 0, "status": "no_speech", "error": "No VAD speech intervals"})
                    continue
                for segment_index, (seg_start, seg_end) in enumerate(intervals):
                    for window_index, (window_start, chunk, valid_sec, padded) in enumerate(
                            make_segment_windows(audio16, seg_start, seg_end)):
                        s0, s1, p0, p1, elapsed = infer_one(model, chunk, device)
                        rows.append({
                            "file": path.name, "channel_index": channel_index, "channel": channel_name,
                            "original_sample_rate": original_sr, "channels": data.shape[1],
                            "duration_sec": round(len(mono) / original_sr, 3),
                            "speech_segment_index": segment_index,
                            "speech_segment_start_sec": round(seg_start, 3),
                            "speech_segment_end_sec": round(seg_end, 3),
                            "speech_segment_duration_sec": round(seg_end - seg_start, 3),
                            "window_index_in_segment": window_index,
                            "window_start_sec": round(window_start, 3),
                            "window_end_sec": round(window_start + valid_sec, 3),
                            "model_input_sec": round(MODEL_WINDOW / TARGET_SR, 4),
                            "valid_speech_audio_sec": round(valid_sec, 3),
                            "padded_to_model_length": padded,
                            "short_fragment_under_1s": valid_sec < 1.0,
                            "spoof_logit": s0, "bonafide_logit": s1,
                            "spoof_softmax": p0, "bonafide_softmax": p1,
                            "higher_logit_class": "spoof" if s0 > s1 else "bonafide",
                            "inference_ms": round(elapsed, 2), "status": "ok", "error": ""
                        })
        except Exception as e:
            rows.append({"file": path.name, "status": "error", "error": repr(e)})
            print(f"ERROR {path}: {e}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    meta = {
        "model_repo": "lab260/spectra_0", "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "channels": args.channels, "swap_channels": args.swap_channels,
        "channel_mapping": "default: channel 0=client, channel 1=bot; --swap-channels reverses labels",
        "vad": "WebRTC VAD on original audio resampled to 8 kHz; 20 ms frames",
        "vad_mode": args.vad_mode, "merge_gap_ms": args.merge_gap_ms,
        "min_speech_ms": args.min_speech_ms, "target_sample_rate": TARGET_SR,
        "model_window_samples": MODEL_WINDOW, "model_window_seconds": MODEL_WINDOW / TARGET_SR,
        "preemphasis": PREEMPHASIS,
        "note": "Only VAD speech intervals are scored. Short segment tails are repeated within that segment to model length. Softmax is not calibrated probability; no decision threshold is applied. Verify channel order for each source."
    }
    meta_path = out.with_name(out.stem + "_metadata.json") if out.name != "spectra0_results.csv" else OUTPUT_DIR / "run_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Results:", out)


if __name__ == "__main__":
    main()
