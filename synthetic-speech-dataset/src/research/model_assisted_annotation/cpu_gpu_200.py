"""Spectra-0 на 200 звонках: CPU против GPU, по одному окну.

Выборка фиксируется в reports/cpu_gpu_200/calls.csv. Порог не меняется.
SVM и LightGBM здесь не участвуют: у них нет GPU-пути.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from math import gcd
from pathlib import Path

import numpy as np
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
OUT = ROOT / "reports" / "cpu_gpu_200"
REPORT = ROOT / "reports" / "cpu_gpu_inference_200.md"
DECISIONS = ROOT / "reports" / "voting_combinations" / "call_decisions.csv"
SPECTRA_THRESHOLD = 0.22710928916931153
WINDOW_MS = 64600 / 16000 * 1000
TARGET = 200
MANIFESTS = (
    ROOT / "data" / "manifests" / "priority_20261111_20261112.csv",
    ROOT / "data" / "manifests" / "day_20261111.csv",
    ROOT / "data" / "manifests" / "day_20261112.csv",
    ROOT / "data" / "manifests" / "day_20261116.csv",
)


def load_manifest() -> pd.DataFrame:
    frames = [pd.read_csv(path, dtype={"call_id": str}, usecols=["call_id", "source_file", "client_channel"]) for path in MANIFESTS if path.is_file()]
    return pd.concat(frames, ignore_index=True).drop_duplicates("call_id", keep="first")


def quotas(sizes: list[int], total: int) -> list[int]:
    weight = sum(sizes)
    raw = [total * size / weight for size in sizes]
    base = [min(size, int(value)) for size, value in zip(sizes, raw)]
    left = total - sum(base)
    order = sorted(range(len(sizes)), key=lambda index: (raw[index] - base[index], sizes[index]), reverse=True)
    cursor = 0
    while left > 0 and any(base[index] < sizes[index] for index in range(len(sizes))):
        index = order[cursor % len(order)]
        cursor += 1
        if base[index] < sizes[index]:
            base[index] += 1
            left -= 1
    return base


def choose_calls() -> pd.DataFrame:
    saved = OUT / "calls.csv"
    if saved.is_file():
        return pd.read_csv(saved, dtype={"call_id": str})
    frame = pd.read_csv(DECISIONS, dtype={"call_id": str})
    frame["y"] = frame["y_synthetic"].map(lambda value: str(value).lower() in {"1", "true"})
    manifest = load_manifest().set_index("call_id")
    cells = []
    for dataset in ("11–12", "16"):
        for label, flag in (("synthetic", True), ("human", False)):
            for speech in ("короткая", "средняя", "длинная"):
                part = frame[(frame["dataset"] == dataset) & (frame["y"] == flag) & (frame["speech_bin"] == speech)].sort_values("call_id")
                ready = []
                for record in part.itertuples(index=False):
                    if record.call_id not in manifest.index:
                        continue
                    wav = ROOT / str(manifest.loc[record.call_id, "source_file"])
                    if not wav.is_file():
                        continue
                    ready.append(record)
                cells.append((dataset, label, speech, ready))
    counts = quotas([len(ready) for *_rest, ready in cells], TARGET)
    rows = []
    for (dataset, label, speech, ready), count in zip(cells, counts):
        for record in ready[:count]:
            info = manifest.loc[record.call_id]
            rows.append(
                {
                    "call_id": record.call_id,
                    "dataset": dataset,
                    "pack": record.pack,
                    "label": label,
                    "speech_bin": speech,
                    "speech_s": float(record.speech_s),
                    "duration_s": float(record.duration_s),
                    "saved_margin": float(record.spectra0_margin),
                    "saved_spectra": record.spectra0,
                    "wav": str(ROOT / str(info.source_file)),
                    "client_channel": int(float(info.client_channel)),
                }
            )
    table = pd.DataFrame(rows)
    if len(table) != TARGET:
        raise SystemExit(f"выборка {len(table)}, нужно {TARGET}")
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(saved, index=False)
    return table


def load_spectra():
    path = SPECTRA_ROOT / "src" / "run_inference.py"
    spec = importlib.util.spec_from_file_location("spectra_run_inference_cmp", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return float(ordered[index])


def time_window(inference, model, chunk, device: str) -> tuple[float, float, float]:
    import torch

    array = np.ascontiguousarray(chunk, dtype=np.float32)
    if device == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    tensor = inference.preemphasis(torch.from_numpy(array).unsqueeze(0)).to(device)
    if device == "cuda":
        torch.cuda.synchronize()
    copied = time.perf_counter()
    with torch.inference_mode():
        logits = model(tensor)
        torch.softmax(logits, dim=1)
    if device == "cuda":
        torch.cuda.synchronize()
    finished = time.perf_counter()
    margin = float(logits[0, 1]) - float(logits[0, 0])
    return margin, (copied - started) * 1000, (finished - copied) * 1000


def run_device(device: str, threads: int) -> None:
    import torch
    from scipy.signal import resample_poly

    if device == "cpu":
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA недоступна")
    inference = load_spectra()
    model = inference.load_model(device)
    model.eval()
    calls = choose_calls()
    tag = device if device == "cuda" else f"cpu{threads}"
    summary_path = OUT / f"{tag}_calls.csv"
    done = set()
    if summary_path.is_file():
        done = set(pd.read_csv(summary_path, dtype={"call_id": str})["call_id"])
    windows_path = OUT / f"{tag}_windows.csv"
    warmed = False
    print(f"{tag}: уже есть {len(done)} из {len(calls)}", flush=True)
    for index, call in enumerate(calls.itertuples(index=False), start=1):
        if call.call_id in done:
            continue
        started = time.perf_counter()
        data, sample_rate = inference.read_stereo(call.wav)
        mono = data[:, int(call.client_channel)]
        intervals = inference.detect_speech(mono, sample_rate, 2, 300, 500)
        divisor = gcd(inference.TARGET_SR, sample_rate)
        audio16 = mono if sample_rate == inference.TARGET_SR else resample_poly(
            mono, inference.TARGET_SR // divisor, sample_rate // divisor
        ).astype(np.float32)
        windows = []
        for start, end in intervals:
            windows.extend(inference.make_segment_windows(audio16, start, end))
        prep_ms = (time.perf_counter() - started) * 1000
        margins = []
        copy_ms = []
        forward_ms = []
        if windows and not warmed:
            time_window(inference, model, windows[0][1], device)
            time_window(inference, model, windows[0][1], device)
            warmed = True
        for _start, chunk, _valid, _padded in windows:
            margin, copied, forward = time_window(inference, model, chunk, device)
            margins.append(margin)
            copy_ms.append(copied)
            forward_ms.append(forward)
        margin = float(np.mean(margins)) if margins else float("nan")
        label = "" if not margins else ("synthetic" if margin < SPECTRA_THRESHOLD else "human")
        total = [copied + forward for copied, forward in zip(copy_ms, forward_ms)]
        row = pd.DataFrame(
            [
                {
                    "call_id": call.call_id,
                    "dataset": call.dataset,
                    "label": call.label,
                    "speech_bin": call.speech_bin,
                    "n_windows": len(windows),
                    "prep_ms": round(prep_ms, 1),
                    "copy_median_ms": round(percentile(copy_ms, 0.5), 3),
                    "forward_median_ms": round(percentile(forward_ms, 0.5), 3),
                    "total_median_ms": round(percentile(total, 0.5), 3),
                    "total_sum_ms": round(sum(total), 1),
                    "margin": margin,
                    "cpu_or_gpu_label": label,
                    "saved_margin": float(call.saved_margin),
                    "saved_spectra": call.saved_spectra,
                    "margin_abs": abs(margin - float(call.saved_margin)) if margins else float("nan"),
                    "label_match": label == call.saved_spectra if margins else False,
                }
            ]
        )
        row.to_csv(summary_path, mode="a", header=not summary_path.is_file(), index=False)
        if total:
            pd.DataFrame({"call_id": call.call_id, "total_ms": [round(value, 3) for value in total]}).to_csv(
                windows_path, mode="a", header=not windows_path.is_file(), index=False
            )
        done.add(call.call_id)
        if index % 10 == 0 or index == len(calls):
            med = percentile(total, 0.5) if total else float("nan")
            print(f"{tag} {index}/{len(calls)} окна {len(windows)} медиана {med:.1f} мс", flush=True)
    meta = {"device": device, "threads": threads, "torch": torch.__version__}
    if device == "cuda":
        meta["gpu"] = torch.cuda.get_device_name(0)
    (OUT / f"{tag}_meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(summary_path, flush=True)


def fmt(value: float, digits: int = 1) -> str:
    if value != value:
        return "—"
    return f"{value:.{digits}f}"


def write_report() -> None:
    calls = pd.read_csv(OUT / "calls.csv", dtype={"call_id": str})
    tags = ("cpu1", "cpu4", "cuda")
    frames = {tag: pd.read_csv(OUT / f"{tag}_calls.csv", dtype={"call_id": str}) for tag in tags}
    windows = {tag: pd.read_csv(OUT / f"{tag}_windows.csv") for tag in tags}
    meta = {tag: json.loads((OUT / f"{tag}_meta.json").read_text(encoding="utf-8")) for tag in tags}
    lines = [
        "# Spectra-0: CPU и GPU на 200 звонках",
        "",
        f"Одни и те же {len(calls)} звонков бинарной когорты 11–12 и 16 ноября. Состав по дате, классу и длине речи пропорционален когорте, внутри группы берутся первые `call_id`. Порог прежний: synthetic, если средний margin ниже {SPECTRA_THRESHOLD}. Окно — {WINDOW_MS:.0f} мс речи. Считается по одному окну, как в живом звонке, без пакета окон.",
        "",
        f"GPU: {meta['cuda'].get('gpu', '')}. CPU — {meta['cpu1']['threads']} и {meta['cpu4']['threads']} потока PyTorch. Время окна включает preemphasis, перенос тензора и forward со softmax. Два прогрева до первого звонка в цифры не входят.",
        "",
        "SVM и LightGBM в этом сравнении нет: они и так считаются на CPU, отдельного GPU-пути у них нет.",
        "",
        "## Время одного окна",
        "",
        "| Устройство | Окон | Медиана, мс | p95, мс | Сумма окон, с | Совпадение класса с сохранённым | Макс. |Δ margin| |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for tag, title in (("cpu1", "CPU, 1 поток"), ("cpu4", "CPU, 4 потока"), ("cuda", "GPU")):
        values = windows[tag]["total_ms"].tolist()
        frame = frames[tag]
        lines.append(
            "| "
            + " | ".join(
                [
                    title,
                    str(len(values)),
                    fmt(percentile(values, 0.5)),
                    fmt(percentile(values, 0.95)),
                    fmt(sum(values) / 1000, 1),
                    f"{int(frame['label_match'].sum())}/{len(frame)}",
                    fmt(float(frame["margin_abs"].max()), 4),
                ]
            )
            + " |"
        )
    cpu = frames["cpu1"].set_index("call_id")
    gpu = frames["cuda"].set_index("call_id")
    joined = cpu.join(gpu, lsuffix="_cpu", rsuffix="_gpu")
    delta = (joined["margin_cpu"] - joined["margin_gpu"]).abs()
    disagree = int((joined["cpu_or_gpu_label_cpu"] != joined["cpu_or_gpu_label_gpu"]).sum())
    speed = float(np.median(windows["cpu1"]["total_ms"]) / np.median(windows["cuda"]["total_ms"]))
    p95 = {tag: percentile(windows[tag]["total_ms"].tolist(), 0.95) for tag in tags}
    names = {"cpu1": "CPU, 1 поток", "cpu4": "CPU, 4 потока", "cuda": "GPU"}
    late = [names[tag] for tag in tags if p95[tag] >= WINDOW_MS]
    if late:
        pace = "p95 выше длины окна у: " + ", ".join(late) + "."
    else:
        pace = "p95 всех трёх устройств ниже 4037 мс, одно окно успевает досчитаться до прихода следующего."
    safe = []
    for tag in tags:
        reserve = (frames[tag]["margin"] - SPECTRA_THRESHOLD).abs()
        if bool((reserve >= frames[tag]["margin_abs"]).all()):
            safe.append(names[tag])
    lines.extend(
        [
            "",
            "## CPU против GPU на том же звонке",
            "",
            f"Медиана окна на 1 потоке CPU больше медианы GPU в {speed:.1f} раза. Класс CPU и GPU разошёлся у {disagree} из {len(joined)}. Максимальная разница средних margin CPU и GPU — {float(delta.max()):.4f}.",
            "",
            "Перенос на GPU по звонкам: медиана "
            f"{fmt(float(frames['cuda']['copy_median_ms'].median()), 3)} мс, forward "
            f"{fmt(float(frames['cuda']['forward_median_ms'].median()), 1)} мс. На CPU почти всё время окна — сам forward.",
            "",
            "Подготовка wav и VAD в это время окна не входит. Её медиана на CPU, 1 поток — "
            f"{fmt(float(frames['cpu1']['prep_ms'].median()))} мс, на GPU — {fmt(float(frames['cuda']['prep_ms'].median()))} мс. Это один проход на звонок.",
            "",
            f"Окно речи длится {WINDOW_MS:.0f} мс. {pace} Четыре потока PyTorch — это один звонок на четырёх ядрах, не четыре звонка сразу. Замер одного процесса на этой машине, не время сервера воркера.",
            "",
            "Разница с сохранённым margin меньше расстояния до порога у: " + (", ".join(safe) if safe else "ни одного устройства") + ".",
            "",
        ]
    )
    composition = calls.groupby(["dataset", "label", "speech_bin"]).size().reset_index(name="n")
    lines.extend(["## Состав 200", "", "| Дата | Класс | Речь | N |", "| --- | --- | --- | ---: |"])
    for row in composition.itertuples(index=False):
        lines.append(f"| {row.dataset} | {row.label} | {row.speech_bin} | {int(row.n)} |")
    lines.append("")
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT, flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    part = sys.argv[1] if len(sys.argv) > 1 else ""
    if part == "select":
        table = choose_calls()
        print(f"звонков {len(table)}", flush=True)
    elif part == "cpu":
        run_device("cpu", int(sys.argv[2]))
    elif part == "gpu":
        run_device("cuda", 1)
    elif part == "report":
        write_report()
    else:
        raise SystemExit("часть: select, cpu N, gpu, report")


if __name__ == "__main__":
    main()
