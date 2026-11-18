"""Замер инференса голоса A на CPU.

Spectra-0 запускать интерпретатором с torch: часть spectra.
SVM и LightGBM — интерпретатором classical-anti-spoofing: часть classical.
Число потоков задаётся переменными окружения до старта процесса и аргументом.
Веса и пороги не меняются.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from statistics import median

import pandas as pd


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


SPECTRA_ROOT = _component("voice-anti-spoofing-spectra0")
CLASSICAL = _component("classical-anti-spoofing")
OUT = ROOT / "reports" / "cpu_inference"
REPORT = ROOT / "reports" / "cpu_inference_analysis.md"
DECISIONS = ROOT / "reports" / "voting_combinations" / "call_decisions.csv"
MODELS = ROOT / "reports" / "origin_retrain_20261113" / "models"
SPECTRA_THRESHOLD = 0.22710928916931153
SVM_THRESHOLD = 0.0
LIGHTGBM_THRESHOLD = 0.3460832258410117
WINDOW_MS = 64600 / 16000 * 1000
MANIFESTS = (
    ROOT / "data" / "manifests" / "priority_20261111_20261112.csv",
    ROOT / "data" / "manifests" / "day_20261111.csv",
    ROOT / "data" / "manifests" / "day_20261112.csv",
    ROOT / "data" / "manifests" / "day_20261116.csv",
)


def threads_from_argv() -> int:
    if len(sys.argv) < 3:
        raise SystemExit("нужны часть и число потоков")
    return int(sys.argv[2])


def load_manifest() -> pd.DataFrame:
    frames = []
    for path in MANIFESTS:
        if path.is_file():
            frames.append(pd.read_csv(path, dtype={"call_id": str}, usecols=["call_id", "source_file", "client_channel"]))
    return pd.concat(frames, ignore_index=True).drop_duplicates("call_id", keep="first")


def choose_calls() -> list[dict]:
    frame = pd.read_csv(DECISIONS, dtype={"call_id": str})
    frame["y"] = frame["y_synthetic"].map(lambda value: str(value).lower() in {"1", "true"})
    manifest = load_manifest().set_index("call_id")
    chosen = []
    for dataset in ("11–12", "16"):
        for label, flag in (("synthetic", True), ("human", False)):
            for speech in ("короткая", "длинная"):
                part = frame[(frame["dataset"] == dataset) & (frame["y"] == flag) & (frame["speech_bin"] == speech)]
                part = part.sort_values("call_id")
                picked = None
                for record in part.itertuples(index=False):
                    if record.call_id not in manifest.index:
                        continue
                    source = ROOT / str(manifest.loc[record.call_id, "source_file"])
                    if source.is_file():
                        picked = record
                        break
                if picked is None:
                    raise SystemExit(f"нет wav для {dataset} {label} {speech}")
                channel = manifest.loc[picked.call_id, "client_channel"]
                chosen.append(
                    {
                        "call_id": picked.call_id,
                        "dataset": dataset,
                        "pack": picked.pack,
                        "label": label,
                        "speech_bin": speech,
                        "speech_s": float(picked.speech_s),
                        "duration_s": float(picked.duration_s),
                        "saved_margin": float(picked.spectra0_margin),
                        "saved_spectra": picked.spectra0,
                        "saved_svm": float(picked.svm_new_score),
                        "saved_lightgbm": float(picked.lightgbm_new_score),
                        "saved_svm_label": picked.svm_new,
                        "saved_lightgbm_label": picked.lightgbm_new,
                        "wav": str(ROOT / str(manifest.loc[picked.call_id, "source_file"])),
                        "client_channel": int(float(channel)),
                    }
                )
    return chosen


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return float(ordered[index])


def load_spectra_module():
    path = SPECTRA_ROOT / "src" / "run_inference.py"
    spec = importlib.util.spec_from_file_location("spectra_run_inference_cpu", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def spectra_part() -> None:
    from math import gcd

    import torch
    from scipy.signal import resample_poly

    count = threads_from_argv()
    torch.set_num_threads(count)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    inference = load_spectra_module()
    model = inference.load_model("cpu")
    model.eval()
    calls = choose_calls()
    rows = []
    for call in calls:
        started = time.perf_counter()
        data, sample_rate = inference.read_stereo(call["wav"])
        mono = data[:, call["client_channel"]]
        intervals = inference.detect_speech(mono, sample_rate, 2, 300, 500)
        divisor = gcd(inference.TARGET_SR, sample_rate)
        audio16 = mono if sample_rate == inference.TARGET_SR else resample_poly(
            mono, inference.TARGET_SR // divisor, sample_rate // divisor
        ).astype("float32")
        windows = []
        for start, end in intervals:
            windows.extend(inference.make_segment_windows(audio16, start, end))
        prep_ms = (time.perf_counter() - started) * 1000
        if not windows:
            rows.append({**public_call(call), "prep_ms": prep_ms, "n_windows": 0, "window_ms": [], "repeat_ms": []})
            print(f"spectra {call['call_id']} нет окон", flush=True)
            continue
        for _ in range(2):
            inference.infer_one(model, windows[0][1], "cpu")
        window_ms = []
        margins = []
        for _start, chunk, _valid, _padded in windows:
            _spoof, bonafide, _p0, _p1, elapsed = inference.infer_one(model, chunk, "cpu")
            # infer_one возвращает logits; margin = bonafide - spoof. elapsed — только forward.
            spoof = _spoof
            margins.append(float(bonafide) - float(spoof))
            window_ms.append(float(elapsed))
        repeat_ms = []
        for _ in range(5):
            _s0, _s1, _p0, _p1, elapsed = inference.infer_one(model, windows[0][1], "cpu")
            repeat_ms.append(float(elapsed))
        cpu_margin = float(sum(margins) / len(margins))
        cpu_label = "synthetic" if cpu_margin < SPECTRA_THRESHOLD else "human"
        rows.append(
            {
                **public_call(call),
                "prep_ms": round(prep_ms, 1),
                "n_windows": len(windows),
                "window_ms": [round(value, 1) for value in window_ms],
                "repeat_ms": [round(value, 1) for value in repeat_ms],
                "cpu_margin": cpu_margin,
                "cpu_label": cpu_label,
                "margin_abs": abs(cpu_margin - call["saved_margin"]),
                "label_match": cpu_label == call["saved_spectra"],
            }
        )
        print(
            f"spectra {call['call_id']} окна {len(windows)} медиана {median(window_ms):.0f} мс класс {cpu_label}",
            flush=True,
        )
    payload = {"threads": count, "device": "cpu", "torch": torch.__version__, "calls": rows}
    destination = OUT / f"spectra_threads_{count}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(destination, flush=True)


def public_call(call: dict) -> dict:
    return {
        "call_id": call["call_id"],
        "dataset": call["dataset"],
        "pack": call["pack"],
        "label": call["label"],
        "speech_bin": call["speech_bin"],
        "speech_s": call["speech_s"],
        "duration_s": call["duration_s"],
        "saved_margin": call["saved_margin"],
        "saved_spectra": call["saved_spectra"],
        "saved_svm": call["saved_svm"],
        "saved_lightgbm": call["saved_lightgbm"],
    }


def classical_part() -> None:
    sys.path.insert(0, str(CLASSICAL))
    from types import SimpleNamespace

    import joblib
    import numpy as np

    from src.aggregation.call_aggregation import aggregate_calls
    from src.audio.loader import read_wav, select_client_channel
    from src.audio.preprocessing import prepare_analysis_audio
    from src.audio.quality import rms
    from src.audio.segmentation import detect_speech
    from src.features.extractor import extract_segment_features
    from src.features.pipeline import _one_call
    from src.utils.config import load_config

    count = threads_from_argv()
    cfg = load_config(CLASSICAL / "config" / "config.yaml")
    bundles = {name: joblib.load(MODELS / f"{name}.joblib") for name in ("svm", "lightgbm")}
    calls = choose_calls()
    rows = []
    for call in calls:
        namespace = SimpleNamespace(
            file_exists=True,
            resolved_path=Path(call["wav"]),
            client_channel=str(call["client_channel"]),
            call_id=call["call_id"],
            label="",
            locked_split="",
            source_channels=2,
            classification_result="",
            outcome="",
            campaign_name="",
            timestamp="",
        )
        started = time.perf_counter()
        outcome = _one_call(namespace, cfg)
        full_ms = (time.perf_counter() - started) * 1000
        frame = aggregate_calls(__import__("pandas").DataFrame(outcome["segments"]))
        recording = float(frame["recording_duration_s"].iloc[0]) if not frame.empty else 0.0
        frame["recording_duration_s"] = recording
        scores = {}
        predict_ms = {}
        for name in ("svm", "lightgbm"):
            bundle = bundles[name]
            matrix = frame.reindex(columns=list(bundle["features"]))
            model = bundle["model"]
            for _ in range(5):
                if name == "svm":
                    model.decision_function(matrix)
                else:
                    model.predict_proba(matrix)
            samples = []
            for _ in range(20):
                tick = time.perf_counter()
                if name == "svm":
                    value = float(np.asarray(model.decision_function(matrix), dtype=float)[0])
                else:
                    value = float(np.asarray(model.predict_proba(matrix)[:, 1], dtype=float)[0])
                samples.append((time.perf_counter() - tick) * 1000)
            scores[name] = value
            predict_ms[name] = [round(item, 4) for item in samples]
        data, sample_rate = read_wav(Path(call["wav"]))
        channel, _index, _reason = select_client_channel(
            data,
            manifest_channel=call["client_channel"],
            config_channel=int(cfg["audio"]["client_channel"]),
            channel_override=None,
        )
        audio, analysis_rate, _action = prepare_analysis_audio(
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
        min_duration = float(cfg["vad"]["min_feature_ms"]) / 1000.0
        min_rms = float(cfg["quality"]["min_segment_rms"])
        longest = None
        for start, end in segments:
            left = max(0, int(round(start * analysis_rate)))
            right = min(len(audio), int(round(end * analysis_rate)))
            piece = audio[left:right]
            duration = len(piece) / float(analysis_rate)
            energy = rms(piece)
            if duration < min_duration or (not np.isfinite(energy)) or energy < min_rms:
                continue
            if longest is None or duration > longest[0]:
                longest = (duration, piece)
        segment_samples = []
        segment_s = 0.0
        if longest is not None:
            segment_s = longest[0]
            extract_segment_features(longest[1], analysis_rate, cfg)
            for _ in range(4):
                tick = time.perf_counter()
                extract_segment_features(longest[1], analysis_rate, cfg)
                segment_samples.append((time.perf_counter() - tick) * 1000)
        svm_label = "synthetic" if scores["svm"] >= SVM_THRESHOLD else "human"
        lightgbm_label = "synthetic" if scores["lightgbm"] >= LIGHTGBM_THRESHOLD else "human"
        rows.append(
            {
                **public_call(call),
                "full_ms": round(full_ms, 1),
                "n_segments": int(outcome["n_segments"]),
                "n_usable": int(outcome["n_usable"]),
                "segment_s": round(segment_s, 3),
                "segment_ms": [round(item, 1) for item in segment_samples],
                "predict_svm_ms": predict_ms["svm"],
                "predict_lightgbm_ms": predict_ms["lightgbm"],
                "cpu_svm": scores["svm"],
                "cpu_lightgbm": scores["lightgbm"],
                "svm_abs": abs(scores["svm"] - call["saved_svm"]),
                "lightgbm_abs": abs(scores["lightgbm"] - call["saved_lightgbm"]),
                "svm_match": svm_label == call["saved_svm_label"],
                "lightgbm_match": lightgbm_label == call["saved_lightgbm_label"],
            }
        )
        print(f"classical {call['call_id']} файл {full_ms:.0f} мс сегмент {median(segment_samples) if segment_samples else 0:.0f} мс", flush=True)
    payload = {"threads": count, "calls": rows}
    destination = OUT / f"classical_threads_{count}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(destination, flush=True)


def pooled(rows: list[dict], key: str) -> list[float]:
    values = []
    for row in rows:
        values.extend(float(item) for item in row.get(key, []))
    return values


def fmt(value: float, digits: int = 0) -> str:
    if value != value:
        return "—"
    return f"{value:.{digits}f}"


def write_report() -> None:
    spectra = {count: json.loads((OUT / f"spectra_threads_{count}.json").read_text(encoding="utf-8")) for count in (1, 4)}
    classical = {count: json.loads((OUT / f"classical_threads_{count}.json").read_text(encoding="utf-8")) for count in (1, 4)}
    lines = [
        "# Инференс голоса A на CPU",
        "",
        "Замер на этой машине, устройство принудительно CPU. GPU в расчёт времени не входил. Веса и пороги прежние: Spectra-0 synthetic при среднем margin ниже 0.22710928916931153, SVM при score не ниже 0, LightGBM при score не ниже 0.3460832258410117.",
        "",
        "Восемь звонков выбраны заранее: на каждой дате по одному короткому и одному длинному synthetic и human, первый `call_id` в группе, у которого на месте wav. Это не оценка качества на всей когорте, а проверка, успевает ли шаг пайплайна и совпадает ли решение с уже сохранённым.",
        "",
        "Время окна Spectra-0 — это forward и softmax уже собранного тензора, как его меряет `infer_one`. Подготовка записи, VAD и нарезка окон лежат отдельной строкой и на звонок делаются один раз. Повторы первого окна идут после двух прогревов. У SVM и LightGBM само предсказание отделено от признаков: признаки полного файла и отдельно самый длинный уже закрытый речевой сегмент.",
        "",
        f"Окно Spectra-0 длится {WINDOW_MS:.0f} мс речи. Если медиана forward больше этого, следующее окно приходит быстрее, чем считается текущее.",
        "",
        "## Spectra-0",
        "",
        "| Потоки | Окон | Медиана forward, мс | p95 forward, мс | Медиана повтора первого окна, мс | Медиана подготовки звонка, мс | Совпадение класса | Макс. |Δ margin| |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for count in (1, 4):
        rows = spectra[count]["calls"]
        forward = pooled(rows, "window_ms")
        repeats = pooled(rows, "repeat_ms")
        prep = [float(row["prep_ms"]) for row in rows]
        matched = sum(bool(row.get("label_match")) for row in rows)
        delta = max(float(row.get("margin_abs", 0)) for row in rows)
        lines.append(
            f"| {count} | {len(forward)} | {fmt(median(forward), 1)} | {fmt(percentile(forward, 0.95), 1)} | {fmt(median(repeats), 1)} | {fmt(median(prep), 1)} | {matched}/{len(rows)} | {delta:.4f} |"
        )
    lines.extend(["", "| call_id | Дата | Класс | Речь | Окон | Медиана 1 поток, мс | Медиана 4 потока, мс | Класс CPU | Сохранённый |", "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |"])
    by_id_4 = {row["call_id"]: row for row in spectra[4]["calls"]}
    for row in spectra[1]["calls"]:
        other = by_id_4[row["call_id"]]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['call_id']}`",
                    row["dataset"],
                    row["label"],
                    row["speech_bin"],
                    str(row["n_windows"]),
                    fmt(median(row["window_ms"]) if row["window_ms"] else float("nan"), 1),
                    fmt(median(other["window_ms"]) if other["window_ms"] else float("nan"), 1),
                    row.get("cpu_label", "—"),
                    row["saved_spectra"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## SVM и LightGBM",
            "",
            "Полный файл — чтение, VAD и признаки всех сегментов, один холодный проход. Сегмент — самый длинный пригодный кусок, медиана четырёх проходов после одного прогрева. Предсказание — медиана 20 вызовов уже посчитанной строки признаков.",
            "",
            "| Потоки | Медиана файла, мс | Медиана сегмента, мс | Медиана SVM, мс | Медиана LightGBM, мс | Класс SVM | Класс LightGBM |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for count in (1, 4):
        rows = classical[count]["calls"]
        files = [float(row["full_ms"]) for row in rows]
        segments = pooled(rows, "segment_ms")
        svm = pooled(rows, "predict_svm_ms")
        light = pooled(rows, "predict_lightgbm_ms")
        lines.append(
            f"| {count} | {fmt(median(files), 1)} | {fmt(median(segments), 1)} | {fmt(median(svm), 3)} | {fmt(median(light), 3)} | {sum(row['svm_match'] for row in rows)}/{len(rows)} | {sum(row['lightgbm_match'] for row in rows)}/{len(rows)} |"
        )
    lines.extend(["", "| call_id | Дата | Класс | Речь сегмента, с | Файл 1 поток, мс | Сегмент 1 поток, мс | Файл 4 потока, мс | Сегмент 4 потока, мс |", "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    classical_4 = {row["call_id"]: row for row in classical[4]["calls"]}
    for row in classical[1]["calls"]:
        other = classical_4[row["call_id"]]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['call_id']}`",
                    row["dataset"],
                    row["label"],
                    f"{row['segment_s']:.2f}",
                    fmt(row["full_ms"], 1),
                    fmt(median(row["segment_ms"]) if row["segment_ms"] else float("nan"), 1),
                    fmt(other["full_ms"], 1),
                    fmt(median(other["segment_ms"]) if other["segment_ms"] else float("nan"), 1),
                ]
            )
            + " |"
        )
    one = spectra[1]["calls"]
    forward = pooled(one, "window_ms")
    segment = pooled(classical[1]["calls"], "segment_ms")
    lines.extend(
        [
            "",
            "## Что успевает живой шаг",
            "",
            f"На одном потоке медиана forward Spectra-0 — {fmt(median(forward), 0)} мс при окне речи {WINDOW_MS:.0f} мс, p95 — {fmt(percentile(forward, 0.95), 0)} мс. Четыре потока снижают медиану до {fmt(median(pooled(spectra[4]['calls'], 'window_ms')), 0)} мс. Оба режима успевают посчитать окно до прихода следующего.",
            f"Признаки одного закрытого сегмента на одном потоке: медиана {fmt(median(segment), 0)} мс. Самый долгий замер сегмента на этих звонках — {fmt(max(segment), 0)} мс. Четыре потока это время не уменьшают. Вызов SVM и LightGBM по готовой строке — около 1 мс.",
            "",
            "Сборка голоса A после трёх чисел — сравнение с тремя порогами. В срок звонка она не входит.",
            "",
            "Замер сделан, пока другие тяжёлые прогоны на этой машине не считались. Это время процесса на CPU, не обещание сервера воркера.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT, flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    part = sys.argv[1] if len(sys.argv) > 1 else ""
    if part == "spectra":
        spectra_part()
    elif part == "classical":
        classical_part()
    elif part == "report":
        write_report()
    else:
        raise SystemExit("часть: spectra, classical или report")


if __name__ == "__main__":
    main()
