import argparse
import importlib.util
import json
import sys
import time
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_inference import (  # noqa: E402
    MODEL_WINDOW,
    PREEMPHASIS,
    TARGET_SR,
    detect_speech,
    make_segment_windows,
    read_stereo,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "spectra_aasist3"
OUTPUT_DIR = ROOT / "data" / "output"


def load_model(device):
    model_py = MODEL_DIR / "model.py"
    weights = MODEL_DIR / "model.safetensors"
    if not model_py.exists() or not weights.exists():
        raise FileNotFoundError(
            "Spectra-AASIST3 weights missing. Download model.py and model.safetensors "
            "from lab260/Spectra-AASIST3 into models/spectra_aasist3."
        )
    spec = importlib.util.spec_from_file_location("spectra_aasist3_impl", model_py)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.spectra_aasist3.from_pretrained(str(MODEL_DIR)).eval().to(device)


def preemphasis_wave(audio):
    """Pre-emphasis on the whole waveform, then windows are cut from it."""
    y = np.empty_like(audio, dtype=np.float32)
    y[0] = audio[0]
    y[1:] = audio[1:] - PREEMPHASIS * audio[:-1]
    return y


def infer_windows(model, chunks, device):
    x = torch.from_numpy(np.stack(chunks)).to(device)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) * 1000
    return logits.detach().float().cpu().numpy(), probs.detach().float().cpu().numpy(), elapsed


def summarize(frame):
    rows = []
    for sample, group in frame.groupby("sample"):
        per_call = group.groupby("file").agg(
            windows=("bonafide_softmax", "size"),
            mean_bonafide=("bonafide_softmax", "mean"),
            mean_spoof=("spoof_softmax", "mean"),
            mean_bonafide_logit=("bonafide_logit", "mean"),
            bonafide_share=("higher_logit_class", lambda s: float((s == "bonafide").mean())),
        )
        rows.append({
            "sample": sample,
            "files": int(per_call.shape[0]),
            "windows": int(len(group)),
            "window_mean_bonafide_softmax": round(float(group["bonafide_softmax"].mean()), 4),
            "window_mean_spoof_softmax": round(float(group["spoof_softmax"].mean()), 4),
            "window_mean_bonafide_logit": round(float(group["bonafide_logit"].mean()), 4),
            "call_mean_bonafide_softmax": round(float(per_call["mean_bonafide"].mean()), 4),
            "call_mean_spoof_softmax": round(float(per_call["mean_spoof"].mean()), 4),
            "call_mean_bonafide_logit": round(float(per_call["mean_bonafide_logit"].mean()), 4),
            "mean_bonafide_window_share": round(float(per_call["bonafide_share"].mean()), 4),
            "calls_mostly_bonafide": int((per_call["bonafide_share"] >= 0.5).sum()),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description="Spectra-AASIST3 on the same VAD speech windows as Spectra-0")
    ap.add_argument("--input", required=True)
    ap.add_argument("--labels", default="")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--channels", choices=["both", "0", "1"], default="0")
    ap.add_argument("--batch-size", type=int, default=4)
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
    print("Files:", len(wavs))

    model = load_model(device)
    pending = []
    rows = []

    def flush():
        if not pending:
            return
        chunks = [item["chunk"] for item in pending]
        logits, probs, elapsed = infer_windows(model, chunks, device)
        each_ms = elapsed / len(pending)
        for item, logit, prob in zip(pending, logits, probs):
            s0, s1 = float(logit[0]), float(logit[1])
            row = {k: v for k, v in item.items() if k != "chunk"}
            row.update({
                "spoof_logit": s0,
                "bonafide_logit": s1,
                "spoof_softmax": float(prob[0]),
                "bonafide_softmax": float(prob[1]),
                "higher_logit_class": "spoof" if s0 > s1 else "bonafide",
                "inference_ms": round(each_ms, 2),
                "status": "ok",
                "error": "",
            })
            rows.append(row)
        pending.clear()

    for path in tqdm(wavs, desc="WAV files"):
        try:
            data, original_sr = read_stereo(path)
            channel_indices = list(range(data.shape[1])) if args.channels == "both" else [int(args.channels)]
            for channel_index in channel_indices:
                if channel_index >= data.shape[1]:
                    rows.append({
                        "file": path.name, "channel_index": channel_index, "status": "error",
                        "error": f"File has only {data.shape[1]} channel(s)",
                    })
                    continue
                channel_name = "client" if channel_index == 0 else "bot" if channel_index == 1 else f"channel_{channel_index}"
                mono = data[:, channel_index]
                intervals = detect_speech(mono, original_sr, args.vad_mode, args.merge_gap_ms, args.min_speech_ms)
                g = gcd(TARGET_SR, original_sr)
                audio16 = mono if original_sr == TARGET_SR else resample_poly(
                    mono, TARGET_SR // g, original_sr // g).astype(np.float32)
                audio16 = preemphasis_wave(np.ascontiguousarray(audio16, dtype=np.float32))
                base = {
                    "file": path.name,
                    "channel_index": channel_index,
                    "channel": channel_name,
                    "original_sample_rate": original_sr,
                    "channels": data.shape[1],
                    "duration_sec": round(len(mono) / original_sr, 3),
                }
                if not intervals:
                    rows.append({**base, "speech_segments": 0, "status": "no_speech", "error": "No VAD speech intervals"})
                    continue
                for segment_index, (seg_start, seg_end) in enumerate(intervals):
                    for window_index, (window_start, chunk, valid_sec, padded) in enumerate(
                            make_segment_windows(audio16, seg_start, seg_end)):
                        pending.append({
                            **base,
                            "chunk": np.ascontiguousarray(chunk, dtype=np.float32),
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
                        })
                        if len(pending) >= args.batch_size:
                            flush()
        except Exception as e:
            rows.append({"file": path.name, "status": "error", "error": repr(e)})
            print(f"ERROR {path}: {e}")
    flush()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "spectra_aasist3_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    meta = {
        "model_repo": "lab260/Spectra-AASIST3",
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "channels": args.channels,
        "batch_size": args.batch_size,
        "input": str(src),
        "vad": "WebRTC VAD on original audio resampled to 8 kHz; 20 ms frames",
        "vad_mode": args.vad_mode,
        "merge_gap_ms": args.merge_gap_ms,
        "min_speech_ms": args.min_speech_ms,
        "target_sample_rate": TARGET_SR,
        "model_window_samples": MODEL_WINDOW,
        "model_window_seconds": MODEL_WINDOW / TARGET_SR,
        "preemphasis": PREEMPHASIS,
        "preemphasis_order": "applied to the full 16 kHz waveform before speech windows are cut",
        "score": "logit index 0 = spoof, index 1 = bonafide. Published score is the bonafide logit; softmax is not a calibrated probability.",
        "note": "Same VAD speech windows as the Spectra-0 run. Silence outside speech segments is not scored. Short tails are repeated to 64600 samples. The model card's own eval uses only the first 64600 samples of a file, which on these calls is often silence.",
    }
    (OUTPUT_DIR / "spectra_aasist3_run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Results:", out)

    if args.labels:
        labels = pd.read_csv(args.labels)
        labels["file"] = labels["call_id"].astype(str) + ".wav"
        scored = pd.DataFrame(rows)
        scored = scored.merge(labels[["file", "sample"]], on="file", how="left")
        ok = scored[scored["status"] == "ok"].copy()
        full = ok[~ok["padded_to_model_length"].astype(bool)].copy()
        summary = {
            "files_scored_ok": int(ok["file"].nunique()) if len(ok) else 0,
            "windows": int(len(ok)),
            "full_windows": int(len(full)),
            "unlabeled_windows": int(ok["sample"].isna().sum()) if len(ok) else 0,
            "all_windows": summarize(ok) if len(ok) else [],
            "full_windows_only": summarize(full) if len(full) else [],
        }
        summary_path = OUTPUT_DIR / "spectra_aasist3_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
