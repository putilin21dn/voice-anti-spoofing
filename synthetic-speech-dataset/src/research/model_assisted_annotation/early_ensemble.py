"""Как голос A меняется, пока копится речь абонента.

Пороги и веса прежние. Обучение не запускается. Worker и backend не меняются.
Spectra-0 берётся из уже сохранённых окон. SVM и LightGBM считаются заново
по уже готовому правилу агрегации, но только на речевых сегментах,
которые к этому моменту уже закончились.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


CLASSICAL = _component("classical-anti-spoofing")
sys.path.insert(0, str(CLASSICAL))

from src.aggregation.call_aggregation import aggregate_calls  # noqa: E402
from src.features.pipeline import _one_call  # noqa: E402
from src.utils.config import load_config  # noqa: E402


OUT = ROOT / "reports" / "early_ensemble"
REPORT = ROOT / "reports" / "early_ensemble_analysis.md"
SEGMENTS = OUT / "segments.csv"
OLD_SEGMENTS = CLASSICAL / "data" / "features" / "segments.csv"
MODELS = ROOT / "reports" / "origin_retrain_20261113" / "models"
DECISIONS = ROOT / "reports" / "voting_combinations" / "call_decisions.csv"
WINDOWS = {
    "11–12": (
        ROOT / "evaluation" / "day_20261111" / "20261116T195124" / "window_results" / "spectra0.csv",
        ROOT / "evaluation" / "day_20261112" / "20261116T195754" / "window_results" / "spectra0.csv",
    ),
    "16": (ROOT / "evaluation" / "day_20261116" / "20261116T152222" / "window_results" / "spectra0.csv",),
}
MANIFESTS = (
    ROOT / "data" / "manifests" / "priority_20261111_20261112.csv",
    ROOT / "data" / "manifests" / "day_20261111.csv",
    ROOT / "data" / "manifests" / "day_20261112.csv",
    ROOT / "data" / "manifests" / "day_20261116.csv",
)
SPECTRA_THRESHOLD = 0.22710928916931153
SVM_THRESHOLD = 0.0
LIGHTGBM_THRESHOLD = 0.3460832258410117
TIMES = (5, 10, 15, 20, 30, 45, 60, 90, 120)
DATE_ORDER = ("11–12", "16")


def load_cohort() -> pd.DataFrame:
    frame = pd.read_csv(DECISIONS, dtype={"call_id": str})
    frame["y"] = frame["y_synthetic"].map(lambda value: str(value).lower() in {"1", "true"})
    return frame


def load_manifest() -> pd.DataFrame:
    frames = []
    for path in MANIFESTS:
        if not path.is_file():
            continue
        part = pd.read_csv(path, dtype={"call_id": str}, usecols=["call_id", "source_file", "client_channel", "channel_count"])
        frames.append(part)
    manifest = pd.concat(frames, ignore_index=True).drop_duplicates("call_id", keep="first")
    return manifest


def load_windows(cohort: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for dataset, paths in WINDOWS.items():
        for path in paths:
            part = pd.read_csv(path, dtype={"call_id": str})
            part["dataset"] = dataset
            frames.append(part)
    windows = pd.concat(frames, ignore_index=True)
    windows = windows[windows["call_id"].isin(set(cohort["call_id"]))].copy()
    windows = windows.drop_duplicates("window_id", keep="first")
    for column in ("start_time", "end_time", "valid_speech_audio_sec", "margin"):
        windows[column] = pd.to_numeric(windows[column], errors="coerce")
    windows = windows[np.isfinite(windows["margin"]) & np.isfinite(windows["end_time"])].copy()
    return windows


def ensure_segments(cohort: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    wanted = set(cohort["call_id"])
    stored = pd.DataFrame()
    if SEGMENTS.is_file() and SEGMENTS.stat().st_size > 0:
        stored = pd.read_csv(SEGMENTS, dtype={"call_id": str})
    have = set(stored["call_id"]) if not stored.empty else set()
    if not have:
        old = pd.read_csv(OLD_SEGMENTS, dtype={"call_id": str})
        stored = old[old["call_id"].isin(wanted)].copy()
        have = set(stored["call_id"])
        if not stored.empty:
            stored.to_csv(SEGMENTS, index=False)
            print(f"из прежнего файла сегментов взято {len(have)} звонков", flush=True)
    pending = [call_id for call_id in cohort["call_id"] if call_id not in have]
    if not pending:
        return stored[stored["call_id"].isin(wanted)].copy()
    info = manifest.set_index("call_id")
    cfg = load_config(CLASSICAL / "config" / "config.yaml")
    from types import SimpleNamespace

    fresh_rows: list[pd.DataFrame] = []
    failures: list[dict] = []
    for index, call_id in enumerate(pending, start=1):
        if call_id not in info.index:
            failures.append({"call_id": call_id, "error": "нет строки manifest"})
            continue
        row = info.loc[call_id]
        wav = ROOT / str(row.source_file)
        call = SimpleNamespace(
            file_exists=wav.is_file(),
            resolved_path=wav,
            client_channel="" if pd.isna(row.client_channel) else str(int(float(row.client_channel))),
            call_id=call_id,
            label="",
            locked_split="",
            source_channels="" if pd.isna(row.channel_count) else int(float(row.channel_count)),
            classification_result="",
            outcome="",
            campaign_name="",
            timestamp="",
        )
        try:
            if not wav.is_file():
                raise FileNotFoundError(wav.name)
            outcome = _one_call(call, cfg)
        except Exception as exc:
            failures.append({"call_id": call_id, "error": exc.__class__.__name__})
            print(f"сегменты {index}/{len(pending)} {call_id} {exc.__class__.__name__}", flush=True)
            continue
        if outcome["segments"]:
            fresh_rows.append(pd.DataFrame(outcome["segments"]))
        else:
            failures.append({"call_id": call_id, "error": outcome["status"]})
        if index % 10 == 0 or index == len(pending):
            if fresh_rows:
                stored = pd.concat([stored, *fresh_rows], ignore_index=True, sort=False)
                fresh_rows = []
                stored.to_csv(SEGMENTS, index=False)
            print(f"сегменты {index}/{len(pending)}", flush=True)
    if fresh_rows:
        stored = pd.concat([stored, *fresh_rows], ignore_index=True, sort=False)
        stored.to_csv(SEGMENTS, index=False)
    if failures:
        pd.DataFrame(failures).to_csv(OUT / "segment_failures.csv", index=False)
    return stored[stored["call_id"].isin(wanted)].copy()


def load_bundles() -> dict:
    import joblib

    return {name: joblib.load(MODELS / f"{name}.joblib") for name in ("svm", "lightgbm")}


def predict_frame(frame: pd.DataFrame, bundles: dict) -> dict[str, float]:
    scores = {}
    for name, key in (("svm", "svm"), ("lightgbm", "lightgbm")):
        bundle = bundles[name]
        matrix = frame.reindex(columns=list(bundle["features"]))
        model = bundle["model"]
        if name == "svm":
            value = float(np.asarray(model.decision_function(matrix), dtype=float)[0])
        else:
            value = float(np.asarray(model.predict_proba(matrix)[:, 1], dtype=float)[0])
        scores[key] = value
    return scores


def classical_scores(rows: list[dict], recording_s: float, bundles: dict) -> dict[str, float] | None:
    if not rows:
        return None
    frame = aggregate_calls(pd.DataFrame(rows))
    if frame.empty:
        return None
    frame["recording_duration_s"] = float(recording_s)
    return predict_frame(frame, bundles)


def vote_of(spectra: float | None, svm: float | None, lightgbm: float | None) -> tuple[str, str]:
    named = []
    if spectra is not None:
        named.append(("Spectra-0", spectra < SPECTRA_THRESHOLD))
    if svm is not None:
        named.append(("SVM", svm >= SVM_THRESHOLD))
    if lightgbm is not None:
        named.append(("LightGBM", lightgbm >= LIGHTGBM_THRESHOLD))
    synthetic = [name for name, flag in named if flag]
    decision = "synthetic" if len(synthetic) >= 2 else "human"
    if not synthetic:
        pattern = "нет synthetic"
    elif len(synthetic) == 3:
        pattern = "все три"
    else:
        pattern = " и ".join(synthetic)
    return decision, pattern


def replay_call(segments: pd.DataFrame, windows: pd.DataFrame, bundles: dict, recording_s: float) -> dict:
    segment_rows = segments.sort_values(["end_time", "start_time"]).to_dict("records")
    window_rows = windows.sort_values(["end_time", "start_time", "window_index"]).to_dict("records")
    times = sorted({float(row["end_time"]) for row in segment_rows} | {float(row["end_time"]) for row in window_rows})
    taken: list[dict] = []
    margins: list[float] = []
    valid_speech = 0.0
    segment_index = 0
    window_index = 0
    path = []
    first = None
    cached_frame: pd.DataFrame | None = None
    cached_count = -1
    for moment in times:
        while segment_index < len(segment_rows) and float(segment_rows[segment_index]["end_time"]) <= moment + 1e-9:
            taken.append(segment_rows[segment_index])
            segment_index += 1
        while window_index < len(window_rows) and float(window_rows[window_index]["end_time"]) <= moment + 1e-9:
            margins.append(float(window_rows[window_index]["margin"]))
            valid_speech += float(window_rows[window_index]["valid_speech_audio_sec"])
            window_index += 1
        spectra = float(np.mean(margins)) if margins else None
        speech_s = float(sum(float(row["duration"]) for row in taken))
        usable = [row for row in taken if int(float(row["usable_for_features"])) == 1]
        if not usable:
            scores = None
        else:
            if len(taken) != cached_count:
                cached_frame = aggregate_calls(pd.DataFrame(taken))
                cached_count = len(taken)
            frame = cached_frame.copy()
            frame["recording_duration_s"] = moment
            scores = None if frame.empty else predict_frame(frame, bundles)
        svm = None if scores is None else scores["svm"]
        lightgbm = None if scores is None else scores["lightgbm"]
        decision, pattern = vote_of(spectra, svm, lightgbm)
        point = {
            "t": moment,
            "speech_s": speech_s,
            "valid_speech_s": valid_speech,
            "decision": decision,
            "pattern": pattern,
            "n_segments": len(taken),
            "n_windows": len(margins),
        }
        path.append(point)
        if first is None and decision == "synthetic":
            first = point
    full_spectra = float(np.mean(margins)) if margins else None
    full_scores = classical_scores(segment_rows, recording_s, bundles)
    full_svm = None if full_scores is None else full_scores["svm"]
    full_lightgbm = None if full_scores is None else full_scores["lightgbm"]
    full_decision, _full_pattern = vote_of(full_spectra, full_svm, full_lightgbm)
    return {
        "path": path,
        "first": first,
        "full_spectra": full_spectra,
        "full_svm": full_svm,
        "full_lightgbm": full_lightgbm,
        "full_decision": full_decision,
        "time_first_segment": float(segment_rows[0]["end_time"]) if segment_rows else None,
        "time_first_window": float(window_rows[0]["end_time"]) if window_rows else None,
    }


def close(saved: float, fresh: float | None, tol: float) -> bool:
    if fresh is None or not np.isfinite(saved):
        return False
    return abs(float(saved) - float(fresh)) <= tol


def ever_by(path: list[dict], limit: float) -> bool:
    return any(point["decision"] == "synthetic" and point["t"] <= limit + 1e-9 for point in path)


def state_at(path: list[dict], limit: float) -> str:
    current = "human"
    for point in path:
        if point["t"] > limit + 1e-9:
            break
        current = point["decision"]
    return current


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.quantile(np.asarray(values, dtype=float), q))


def fmt(value: float, digits: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def ru_calls(count: int) -> str:
    rest = abs(count) % 100
    if 11 <= rest <= 14:
        return f"{count} звонков"
    tail = rest % 10
    if tail == 1:
        return f"{count} звонок"
    if 2 <= tail <= 4:
        return f"{count} звонка"
    return f"{count} звонков"


def analyze(cohort: pd.DataFrame, segments: pd.DataFrame, windows: pd.DataFrame, bundles: dict) -> tuple[pd.DataFrame, list[str]]:
    for column in ("start_time", "end_time", "duration", "recording_duration_s", "usable_for_features"):
        segments[column] = pd.to_numeric(segments[column], errors="coerce")
    segments = segments.dropna(subset=["end_time", "duration"])
    grouped_segments = {call_id: group for call_id, group in segments.groupby("call_id", sort=False)}
    grouped_windows = {call_id: group for call_id, group in windows.groupby("call_id", sort=False)}
    rows = []
    paths = {}
    for index, record in enumerate(cohort.itertuples(index=False), start=1):
        call_segments = grouped_segments.get(record.call_id, pd.DataFrame())
        call_windows = grouped_windows.get(record.call_id, pd.DataFrame())
        if call_segments.empty or call_windows.empty:
            rows.append({"call_id": record.call_id, "dataset": record.dataset, "ready": False})
            continue
        recording_s = float(call_segments["recording_duration_s"].iloc[0])
        played = replay_call(call_segments, call_windows, bundles, recording_s)
        first = played["first"]
        rows.append(
            {
                "call_id": record.call_id,
                "dataset": record.dataset,
                "pack": record.pack,
                "y": bool(record.y),
                "ready": True,
                "published_vote": record.vote_A,
                "published_spectra": float(record.spectra0_margin),
                "published_svm": float(record.svm_new_score),
                "published_lightgbm": float(record.lightgbm_new_score),
                "full_spectra": played["full_spectra"],
                "full_svm": played["full_svm"],
                "full_lightgbm": played["full_lightgbm"],
                "full_decision": played["full_decision"],
                "spectra_match": close(float(record.spectra0_margin), played["full_spectra"], 1e-4),
                "svm_match": close(float(record.svm_new_score), played["full_svm"], 1e-5),
                "lightgbm_match": close(float(record.lightgbm_new_score), played["full_lightgbm"], 1e-5),
                "vote_match": played["full_decision"] == record.vote_A,
                "time_first_segment": played["time_first_segment"],
                "time_first_window": played["time_first_window"],
                "first_t": None if first is None else first["t"],
                "first_speech_s": None if first is None else first["speech_s"],
                "first_valid_speech_s": None if first is None else first["valid_speech_s"],
                "first_pattern": "" if first is None else first["pattern"],
                "first_n_segments": 0 if first is None else first["n_segments"],
                "first_n_windows": 0 if first is None else first["n_windows"],
            }
        )
        paths[record.call_id] = played["path"]
        if index % 50 == 0 or index == len(cohort):
            print(f"проигрыш {index}/{len(cohort)}", flush=True)
    calls = pd.DataFrame(rows)
    ready = calls[calls["ready"]].copy()
    checks = [
        f"к проигрышу готовы {len(ready)} из {len(cohort)}",
        f"средний margin Spectra-0 совпал у {int(ready['spectra_match'].sum())} из {len(ready)}",
        f"SVM совпал у {int(ready['svm_match'].sum())} из {len(ready)}",
        f"LightGBM совпал у {int(ready['lightgbm_match'].sum())} из {len(ready)}",
        f"итог A на полном файле совпал у {int(ready['vote_match'].sum())} из {len(ready)}",
    ]
    for dataset, expected in (("11–12", 457), ("16", 361)):
        actual = int((cohort["dataset"] == dataset).sum())
        checks.append(f"{dataset}: в когорте {actual}, ожидалось {expected}")
    curve_rows = []
    for dataset in DATE_ORDER:
        part = ready[ready["dataset"] == dataset]
        for limit in TIMES:
            for label, mask in (("synthetic", part["y"]), ("human", ~part["y"])):
                chosen = part[mask]
                hit = sum(ever_by(paths[call_id], limit) for call_id in chosen["call_id"])
                now = sum(state_at(paths[call_id], limit) == "synthetic" for call_id in chosen["call_id"])
                curve_rows.append(
                    {
                        "dataset": dataset,
                        "t": limit,
                        "label": label,
                        "n": int(len(chosen)),
                        "ever": int(hit),
                        "now": int(now),
                    }
                )
    curves = pd.DataFrame(curve_rows)
    calls.to_csv(OUT / "calls.csv", index=False)
    curves.to_csv(OUT / "by_time.csv", index=False)
    plot_curves(curves)
    plot_patterns(ready)
    report(ready, curves, checks)
    return ready, checks


def plot_curves(curves: pd.DataFrame) -> None:
    figures = OUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True, constrained_layout=True)
    for axis, dataset in zip(axes, DATE_ORDER):
        part = curves[curves["dataset"] == dataset]
        for label, color in (("synthetic", "#c2410c"), ("human", "#1d4ed8")):
            rows = part[part["label"] == label]
            rate = rows["ever"] / rows["n"]
            axis.plot(rows["t"], rate, marker="o", color=color, label="synthetic" if label == "synthetic" else "human")
            for moment, share, count, total in zip(rows["t"], rate, rows["ever"], rows["n"]):
                if moment in {10, 30, 60}:
                    axis.annotate(f"{int(count)}/{int(total)}", (moment, share), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        axis.set_title("11–12 ноября" if dataset == "11–12" else "16 ноября")
        axis.set_xlabel("Секунды от начала записи")
        axis.set_xlim(0, 125)
        axis.set_ylim(0, 1)
        axis.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Доля, где A уже сказал synthetic")
    fig.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Первый synthetic-голос A к моменту времени. Трубка кладётся и дальше не слушаем.")
    fig.savefig(figures / "ever_by_time.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_patterns(calls: pd.DataFrame) -> None:
    figures = OUT / "figures"
    fired = calls[calls["first_t"].notna()]
    patterns = ["SVM и LightGBM", "Spectra-0 и SVM", "Spectra-0 и LightGBM", "все три"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)
    for axis, dataset in zip(axes, DATE_ORDER):
        part = fired[fired["dataset"] == dataset]
        labels = []
        synthetic_counts = []
        human_counts = []
        for pattern in patterns:
            rows = part[part["first_pattern"] == pattern]
            labels.append(pattern)
            synthetic_counts.append(int(rows["y"].sum()))
            human_counts.append(int((~rows["y"]).sum()))
        position = np.arange(len(labels))
        axis.bar(position, synthetic_counts, color="#c2410c", label="synthetic")
        axis.bar(position, human_counts, bottom=synthetic_counts, color="#1d4ed8", label="human")
        for index, (syn, hum) in enumerate(zip(synthetic_counts, human_counts)):
            axis.text(index, syn + hum + 0.4, str(syn + hum), ha="center", fontsize=8)
        axis.set_xticks(position, labels, rotation=15, ha="right")
        axis.set_title("11–12 ноября" if dataset == "11–12" else "16 ноября")
        axis.set_ylabel("Звонки")
    fig.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Кто дал два голоса в первый момент, когда A сказал synthetic")
    fig.savefig(figures / "first_pattern.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def subset_line(calls: pd.DataFrame, dataset: str) -> str:
    part = calls[calls["dataset"] == dataset]
    synthetic = part[part["y"]]
    human = part[~part["y"]]
    caught = synthetic[synthetic["first_t"].notna()]
    false = human[human["first_t"].notna()]
    published_hit = synthetic[synthetic["published_vote"] == "synthetic"]
    early_and_final = caught[caught["published_vote"] == "synthetic"]
    early_only = caught[caught["published_vote"] != "synthetic"]
    final_only = synthetic[(synthetic["published_vote"] == "synthetic") & synthetic["first_t"].isna()]
    pieces = [
        f"На {dataset} synthetic {len(synthetic)}, human {len(human)}.",
        f"Полный голос A ловит {len(published_hit)} из {len(synthetic)}.",
        f"Если класть трубку в первый synthetic, ловится {len(caught)} из {len(synthetic)}: медиана {fmt(quantile(caught['first_t'].tolist(), 0.5))} с от начала записи, речь к этому моменту {fmt(quantile(caught['first_speech_s'].tolist(), 0.5))} с.",
        f"Из них полный голос тоже говорит synthetic у {len(early_and_final)}. Ранний synthetic при итоговом human — {len(early_only)}.",
        f"Итог synthetic, но по ходу звонка A так и не сказал synthetic — {len(final_only)}.",
        f"Ложных ранних среди human — {len(false)} из {len(human)}. Медиана времени {fmt(quantile(false['first_t'].tolist(), 0.5))} с.",
    ]
    return " ".join(pieces)


def time_table(curves: pd.DataFrame) -> str:
    header = ["Секунда", "11–12 synthetic", "11–12 human", "16 synthetic", "16 human"]
    rows = []
    for limit in TIMES:
        row = [str(limit)]
        for dataset in DATE_ORDER:
            for label in ("synthetic", "human"):
                item = curves[(curves["dataset"] == dataset) & (curves["t"] == limit) & (curves["label"] == label)].iloc[0]
                row.append(f"{int(item.ever)}/{int(item.n)}")
        rows.append(row)
    lines = ["| " + " | ".join(header) + " |", "| --- | ---: | ---: | ---: | ---: |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def pattern_table(calls: pd.DataFrame) -> str:
    fired = calls[calls["first_t"].notna()]
    patterns = ["SVM и LightGBM", "Spectra-0 и SVM", "Spectra-0 и LightGBM", "все три"]
    lines = [
        "| Дата | Состав первого synthetic | Synthetic | Human | Медиана секунды | Медиана речи, с |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset in DATE_ORDER:
        for pattern in patterns:
            rows = fired[(fired["dataset"] == dataset) & (fired["first_pattern"] == pattern)]
            lines.append(
                "| "
                + " | ".join(
                    [
                        dataset,
                        pattern,
                        str(int(rows["y"].sum())),
                        str(int((~rows["y"]).sum())),
                        fmt(quantile(rows["first_t"].tolist(), 0.5)),
                        fmt(quantile(rows["first_speech_s"].tolist(), 0.5)),
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def wait_table(calls: pd.DataFrame) -> str:
    lines = [
        "| Дата | Класс | Медиана до первого сегмента, с | Медиана до первого окна Spectra-0, с |",
        "| --- | --- | ---: | ---: |",
    ]
    for dataset in DATE_ORDER:
        part = calls[calls["dataset"] == dataset]
        for label, mask in (("synthetic", part["y"]), ("human", ~part["y"])):
            chosen = part[mask]
            lines.append(
                f"| {dataset} | {label} | {fmt(quantile(chosen['time_first_segment'].dropna().tolist(), 0.5))} | {fmt(quantile(chosen['time_first_window'].dropna().tolist(), 0.5))} |"
            )
    return "\n".join(lines)


def report(calls: pd.DataFrame, curves: pd.DataFrame, checks: list[str]) -> None:
    ready = calls
    text = [
        "# Голос A по мере накопления речи абонента",
        "",
        "Вопрос: в какой момент записи голос A впервые говорит synthetic, если считать его не по всему файлу, а по той речи абонента, которая уже прозвучала. Это нужно, чтобы понять, можно ли класть трубку автоответчику раньше, а не говорить с ним несколько минут.",
        "",
        "Голос A прежний: Spectra-0, SVM и LightGBM, synthetic если так сказали минимум две модели. Пороги прежние: Spectra-0 synthetic при среднем margin ниже 0.22710928916931153, SVM при score не ниже 0, LightGBM при score не ниже 0.3460832258410117. Веса не менялись.",
        "",
        "Момент решения — конец уже закрытого куска, а не каждая секунда внутри него. SVM и LightGBM обновляются, когда VAD закрыл речевой сегмент: пауза до 300 мс его завершает, кусок короче 0.5 с в признаки не идёт. Пока абонент говорит без паузы, classical-оценки стоят на прошлом сегменте. Spectra-0 обновляется по своим окнам около 4.04 с речи, включая добитый хвост. Окна взяты из уже сохранённого прогона, заново сеть не считалась.",
        "",
        "В классических признаках есть длина записи. На полном файле она равна длине файла, и с ней полный пересчёт совпадает с сохранёнными score. По ходу звонка подставляется время, которое уже прошло от начала записи: будущую длину живой пайплайн не знает.",
        "",
        "Когорта та же, что у сравнения комбинаций: бинарные звонки 11–12 и 16 ноября, без unknown и mixed. На 16 ноября спорные метки по-прежнему исключены.",
        "",
        "## 1. Совпадает ли полный файл с уже посчитанным голосом A",
        "",
    ]
    for item in checks:
        text.append(f"- {item}")
    text.extend(
        [
            "",
            "Дальше в знаменателе только звонки, у которых есть и сегменты, и окна. Пропуски не заполняются human.",
            "",
            "## 2. Когда вообще появляется первая оценка",
            "",
            "Секунды здесь — от начала записи, не сумма речи. Первое окно Spectra-0 часто позже первого сегмента: окну нужно около 4 с речи.",
            "",
            wait_table(ready),
            "",
            "## 3. Если класть трубку на первом synthetic",
            "",
            "Правило для этого прогона одно и заранее фиксировано: как только A впервые сказал synthetic, звонок считается сброшенным. Обратный переход в human уже не слушаем. Порог под это правило не подбирался.",
            "",
            subset_line(ready, "11–12"),
            "",
            subset_line(ready, "16"),
            "",
            "Доля таких ранних решений к фиксированной секунде записи. В ячейке — сколько звонков уже получили synthetic хотя бы раз, из всех звонков класса.",
            "",
            time_table(curves),
            "",
            "![Ранний synthetic по времени записи](early_ensemble/figures/ever_by_time.png)",
            "",
            "## 4. Какая пара голосов срабатывает первой",
            "",
            pattern_table(ready),
            "",
            "![Состав первого synthetic](early_ensemble/figures/first_pattern.png)",
            "",
            "## 5. Что из этого следует для пайплайна",
            "",
            "Ранний сброс по первому synthetic ловит часть автоответчиков до конца записи, но забирает и тех human, которых полный A в итоге оставляет людьми: по ходу звонка голос успевает сказать synthetic, а потом вернуться. Эти звонки видны в строке «ранний synthetic при итоговом human».",
            "",
            "Classical-голос не обновляется внутри длинного монолога. Если у автоответчика нет паузы, SVM и LightGBM молчат до конца этого куска, и двух голосов нет, пока не закроется сегмент или пока Spectra-0 не совпадёт с уже готовым classical-голосом.",
            "",
            "Это описание тех же двух дней. Оно не выбирает новую секунду сброса и не переносит порог. Для живого пайплайна нужен отдельный контур, который кормит модели тем же VAD и теми же окнами по мере аудио, а не этот офлайн-пересчёт.",
            "",
        ]
    )
    REPORT.write_text("\n".join(text), encoding="utf-8")
    print(REPORT, flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False, "figure.facecolor": "white"})
    cohort = load_cohort()
    manifest = load_manifest()
    windows = load_windows(cohort)
    means = windows.groupby("call_id")["margin"].mean()
    joined = cohort.set_index("call_id")["spectra0_margin"].astype(float)
    delta = (means - joined.reindex(means.index)).abs()
    matched = int((delta <= 1e-3).sum())
    print(f"окна {len(windows)} звонков {windows['call_id'].nunique()} margin совпал {matched}", flush=True)
    if matched < int(0.95 * len(means)):
        raise SystemExit(f"Окна Spectra-0 не сходятся с сохранённым margin: {matched} из {len(means)}")
    segments = ensure_segments(cohort, manifest)
    print(f"сегменты звонков {segments['call_id'].nunique()}", flush=True)
    bundles = load_bundles()
    ready, checks = analyze(cohort, segments, windows, bundles)
    for item in checks:
        print(item, flush=True)
    print(f"готово {ru_calls(int(ready['ready'].sum()) if 'ready' in ready else len(ready))}", flush=True)


if __name__ == "__main__":
    main()
