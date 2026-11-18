"""Когда голос A по ходу звонка совпадает с итогом по всему файлу.

Пороги и веса прежние. Сеть заново не считается: окна Spectra-0 уже сохранены,
сегменты тоже. Worker и backend не меняются.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from early_ensemble import (  # noqa: E402
    DATE_ORDER,
    LIGHTGBM_THRESHOLD,
    SPECTRA_THRESHOLD,
    SVM_THRESHOLD,
    classical_scores,
    close,
    ensure_segments,
    load_bundles,
    load_cohort,
    load_manifest,
    load_windows,
    predict_frame,
    vote_of,
)
from src.aggregation.call_aggregation import aggregate_calls  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "reports" / "window_stability"
REPORT = ROOT / "reports" / "window_stability_analysis.md"
WINDOW_COUNTS = (1, 2, 3, 4, 5, 6, 8, 10)
SPEECH_MARKS = (4, 8, 12, 16, 20, 30)
TIME_MARKS = (10, 15, 20, 30, 45, 60)
MODELS = ("Spectra-0", "SVM", "LightGBM")


def replay(segments: pd.DataFrame, windows: pd.DataFrame, bundles: dict, recording_s: float) -> dict:
    segment_rows = segments.sort_values(["end_time", "start_time"]).to_dict("records")
    window_rows = windows.sort_values(["end_time", "start_time", "window_index"]).to_dict("records")
    times = sorted({float(row["end_time"]) for row in segment_rows} | {float(row["end_time"]) for row in window_rows})
    taken: list[dict] = []
    margins: list[float] = []
    segment_index = 0
    window_index = 0
    path = []
    cached_frame: pd.DataFrame | None = None
    cached_count = -1
    for moment in times:
        while segment_index < len(segment_rows) and float(segment_rows[segment_index]["end_time"]) <= moment + 1e-9:
            taken.append(segment_rows[segment_index])
            segment_index += 1
        while window_index < len(window_rows) and float(window_rows[window_index]["end_time"]) <= moment + 1e-9:
            margins.append(float(window_rows[window_index]["margin"]))
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
        decision, _pattern = vote_of(spectra, svm, lightgbm)
        path.append(
            {
                "t": moment,
                "speech_s": speech_s,
                "n_windows": len(margins),
                "n_scored": sum(value is not None for value in (spectra, svm, lightgbm)),
                "decision": decision,
                "spectra": spectra,
                "svm": svm,
                "lightgbm": lightgbm,
            }
        )
    full_spectra = float(np.mean(margins)) if margins else None
    full_scores = classical_scores(segment_rows, recording_s, bundles)
    full_svm = None if full_scores is None else full_scores["svm"]
    full_lightgbm = None if full_scores is None else full_scores["lightgbm"]
    full_decision, _full_pattern = vote_of(full_spectra, full_svm, full_lightgbm)
    return {
        "path": path,
        "n_windows": len(margins),
        "full_spectra": full_spectra,
        "full_svm": full_svm,
        "full_lightgbm": full_lightgbm,
        "full_decision": full_decision,
    }


def label_of(model: str, value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return ""
    if model == "Spectra-0":
        return "synthetic" if value < SPECTRA_THRESHOLD else "human"
    if model == "SVM":
        return "synthetic" if value >= SVM_THRESHOLD else "human"
    return "synthetic" if value >= LIGHTGBM_THRESHOLD else "human"


def first_reaching(path: list[dict], key: str, limit: float) -> dict | None:
    for point in path:
        if point["n_scored"] > 0 and point[key] + 1e-9 >= limit:
            return point
    return None


def state_at_time(path: list[dict], limit: float) -> dict | None:
    chosen = None
    for point in path:
        if point["t"] > limit + 1e-9:
            break
        if point["n_scored"] > 0:
            chosen = point
    return chosen


def stable_point(path: list[dict], final: str) -> dict | None:
    if not path or path[-1]["decision"] != final:
        return None
    index = len(path) - 1
    while index > 0 and path[index - 1]["decision"] == final:
        index -= 1
    while index < len(path) and path[index]["n_scored"] <= 0:
        index += 1
    if index >= len(path) or path[index]["decision"] != final:
        return None
    return path[index]


def errors(frame: pd.DataFrame, column: str) -> dict[str, int]:
    synthetic = frame["y"].astype(bool)
    said = frame[column] == "synthetic"
    tp = int((synthetic & said).sum())
    fp = int((~synthetic & said).sum())
    fn = int((synthetic & ~said).sum())
    tn = int((~synthetic & ~said).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "errors": fp + fn}


def checkpoint_rows(calls: list[dict], kind: str, marks: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    for dataset in DATE_ORDER:
        part = [call for call in calls if call["dataset"] == dataset]
        for mark in marks:
            chosen = []
            for call in part:
                if kind == "windows":
                    point = first_reaching(call["path"], "n_windows", mark)
                    if point is None or point["n_windows"] < mark:
                        continue
                elif kind == "speech":
                    point = first_reaching(call["path"], "speech_s", mark)
                else:
                    point = state_at_time(call["path"], mark)
                if point is None:
                    continue
                chosen.append(
                    {
                        "y": call["y"],
                        "now": point["decision"],
                        "final": call["full_decision"],
                        "spectra_now": label_of("Spectra-0", point["spectra"]),
                        "svm_now": label_of("SVM", point["svm"]),
                        "lightgbm_now": label_of("LightGBM", point["lightgbm"]),
                        "spectra_final": call["spectra_final"],
                        "svm_final": call["svm_final"],
                        "lightgbm_final": call["lightgbm_final"],
                    }
                )
            frame = pd.DataFrame(chosen)
            row = {"dataset": dataset, "kind": kind, "mark": mark, "n": int(len(frame))}
            if frame.empty:
                rows.append(row)
                continue
            same = frame["now"] == frame["final"]
            synthetic = frame["y"].astype(bool)
            now_err = errors(frame, "now")
            full_err = errors(frame, "final")
            row.update(
                {
                    "agree": int(same.sum()),
                    "syn_n": int(synthetic.sum()),
                    "syn_agree": int((same & synthetic).sum()),
                    "hum_n": int((~synthetic).sum()),
                    "hum_agree": int((same & ~synthetic).sum()),
                    "tp": now_err["tp"],
                    "fp": now_err["fp"],
                    "fn": now_err["fn"],
                    "full_tp": full_err["tp"],
                    "full_fp": full_err["fp"],
                    "full_fn": full_err["fn"],
                    "errors": now_err["errors"],
                    "full_errors": full_err["errors"],
                }
            )
            for model, now_col, final_col in (
                ("Spectra-0", "spectra_now", "spectra_final"),
                ("SVM", "svm_now", "svm_final"),
                ("LightGBM", "lightgbm_now", "lightgbm_final"),
            ):
                scored = frame[now_col] != ""
                row[f"{model}_n"] = int(scored.sum())
                row[f"{model}_agree"] = int(((frame[now_col] == frame[final_col]) & scored).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def analyze() -> None:
    cohort = load_cohort()
    manifest = load_manifest()
    windows = load_windows(cohort)
    means = windows.groupby("call_id")["margin"].mean()
    joined = cohort.set_index("call_id")["spectra0_margin"].astype(float)
    matched = int(((means - joined.reindex(means.index)).abs() <= 1e-3).sum())
    print(f"окна {len(windows)} звонков {windows['call_id'].nunique()} margin совпал {matched}", flush=True)
    if matched < int(0.95 * len(means)):
        raise SystemExit(f"Окна Spectra-0 не сходятся с сохранённым margin: {matched} из {len(means)}")
    segments = ensure_segments(cohort, manifest)
    for column in ("start_time", "end_time", "duration", "recording_duration_s", "usable_for_features"):
        segments[column] = pd.to_numeric(segments[column], errors="coerce")
    segments = segments.dropna(subset=["end_time", "duration"])
    grouped_segments = {call_id: group for call_id, group in segments.groupby("call_id", sort=False)}
    grouped_windows = {call_id: group for call_id, group in windows.groupby("call_id", sort=False)}
    bundles = load_bundles()
    calls = []
    ready = 0
    vote_match = 0
    for index, record in enumerate(cohort.itertuples(index=False), start=1):
        call_segments = grouped_segments.get(record.call_id, pd.DataFrame())
        call_windows = grouped_windows.get(record.call_id, pd.DataFrame())
        if call_segments.empty or call_windows.empty:
            continue
        recording_s = float(call_segments["recording_duration_s"].iloc[0])
        played = replay(call_segments, call_windows, bundles, recording_s)
        ready += 1
        vote_match += int(played["full_decision"] == record.vote_A)
        if not close(float(record.spectra0_margin), played["full_spectra"], 1e-4):
            raise SystemExit(f"Spectra не совпала {record.call_id}")
        if not close(float(record.svm_new_score), played["full_svm"], 1e-5):
            raise SystemExit(f"SVM не совпал {record.call_id}")
        if not close(float(record.lightgbm_new_score), played["full_lightgbm"], 1e-5):
            raise SystemExit(f"LightGBM не совпал {record.call_id}")
        point = stable_point(played["path"], played["full_decision"])
        tail = played["path"][-1]
        calls.append(
            {
                "call_id": record.call_id,
                "dataset": record.dataset,
                "y": bool(record.y),
                "n_windows": played["n_windows"],
                "full_decision": played["full_decision"],
                "published_vote": record.vote_A,
                "spectra_final": label_of("Spectra-0", played["full_spectra"]),
                "svm_final": label_of("SVM", played["full_svm"]),
                "lightgbm_final": label_of("LightGBM", played["full_lightgbm"]),
                "tail_decision": tail["decision"],
                "tail_match": tail["decision"] == played["full_decision"],
                "stable_t": None if point is None else point["t"],
                "stable_speech": None if point is None else point["speech_s"],
                "stable_windows": None if point is None else point["n_windows"],
                "path": played["path"],
            }
        )
        if index % 50 == 0 or index == len(cohort):
            print(f"проигрыш {index}/{len(cohort)}", flush=True)
    print(f"готовы {ready} итог A совпал {vote_match}", flush=True)
    if ready != 818 or vote_match != 818:
        raise SystemExit(f"проигрыш {ready}, совпадение голоса A {vote_match}")
    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([{key: value for key, value in call.items() if key != "path"} for call in calls])
    summary.to_csv(OUT / "calls.csv", index=False)
    tables = pd.concat(
        [
            checkpoint_rows(calls, "windows", WINDOW_COUNTS),
            checkpoint_rows(calls, "speech", SPEECH_MARKS),
            checkpoint_rows(calls, "time", TIME_MARKS),
        ],
        ignore_index=True,
    )
    tables.to_csv(OUT / "checkpoints.csv", index=False)
    plot_agreement(tables)
    write_report(summary, tables)


def plot_agreement(tables: pd.DataFrame) -> None:
    figures = OUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    part = tables[tables["kind"] == "windows"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True, constrained_layout=True)
    for axis, dataset in zip(axes, DATE_ORDER):
        rows = part[part["dataset"] == dataset].sort_values("mark")
        for column, label, color in (
            ("agree", "все", "#111827"),
            ("syn_agree", "synthetic", "#c2410c"),
            ("hum_agree", "human", "#1d4ed8"),
        ):
            base = {"agree": "n", "syn_agree": "syn_n", "hum_agree": "hum_n"}[column]
            share = rows[column] / rows[base]
            axis.plot(rows["mark"], share, marker="o", color=color, label=label)
        axis.set_title("11–12 ноября" if dataset == "11–12" else "16 ноября")
        axis.set_xlabel("Закрытых окон Spectra-0")
        axis.set_xticks(list(WINDOW_COUNTS))
        axis.set_ylim(0, 1)
        axis.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Доля, где текущий голос A равен итогу по файлу")
    fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Голос A по первым окнам против решения по всему звонку")
    fig.savefig(figures / "agree_by_window.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fmt(value: float, digits: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def quantile(values: pd.Series, q: float) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(np.quantile(clean.to_numpy(dtype=float), q))


def fraction(done: int, total: int) -> str:
    return f"{int(done)}/{int(total)}"


def window_table(tables: pd.DataFrame, dataset: str) -> str:
    rows = tables[(tables["kind"] == "windows") & (tables["dataset"] == dataset)].sort_values("mark")
    lines = [
        "| Окон | Дошли | Совпадает с итогом A | Synthetic | Human | TP сейчас | TP итога | FP сейчас | FP итога | Ошибки сейчас | Ошибки итога |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(row.mark)),
                    str(int(row.n)),
                    fraction(row.agree, row.n),
                    fraction(row.syn_agree, row.syn_n),
                    fraction(row.hum_agree, row.hum_n),
                    str(int(row.tp)),
                    str(int(row.full_tp)),
                    str(int(row.fp)),
                    str(int(row.full_fp)),
                    str(int(row.errors)),
                    str(int(row.full_errors)),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def time_table(tables: pd.DataFrame, kind: str, dataset: str, title: str) -> str:
    rows = tables[(tables["kind"] == kind) & (tables["dataset"] == dataset)].sort_values("mark")
    lines = [
        f"| {title} | Уже есть оценка | Совпадает с итогом A | Synthetic | Human | TP сейчас/итог | FP сейчас/итог | Ошибки сейчас/итог |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(row.mark)),
                    str(int(row.n)),
                    fraction(row.agree, row.n),
                    fraction(row.syn_agree, row.syn_n),
                    fraction(row.hum_agree, row.hum_n),
                    f"{int(row.tp)}/{int(row.full_tp)}",
                    f"{int(row.fp)}/{int(row.full_fp)}",
                    f"{int(row.errors)}/{int(row.full_errors)}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def model_table(tables: pd.DataFrame, dataset: str) -> str:
    rows = tables[(tables["kind"] == "windows") & (tables["dataset"] == dataset)].sort_values("mark")
    lines = [
        "| Окон | Spectra-0 | SVM | LightGBM |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for _, row in rows.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(row["mark"])),
                    fraction(row["Spectra-0_agree"], row["Spectra-0_n"]),
                    fraction(row["SVM_agree"], row["SVM_n"]),
                    fraction(row["LightGBM_agree"], row["LightGBM_n"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def stable_table(calls: pd.DataFrame) -> str:
    lines = [
        "| Дата | Кто | Звонков | Медиана окон | p75 окон | p90 окон | Медиана речи, с | Медиана от начала, с | p90 от начала, с |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    groups = (
        ("human, итог human", lambda frame: ~frame["y"] & (frame["full_decision"] == "human")),
        ("synthetic, итог synthetic", lambda frame: frame["y"] & (frame["full_decision"] == "synthetic")),
        ("synthetic, итог human", lambda frame: frame["y"] & (frame["full_decision"] == "human")),
        ("human, итог synthetic", lambda frame: ~frame["y"] & (frame["full_decision"] == "synthetic")),
    )
    truth = calls["y"].astype(str).str.lower().isin({"1", "true"})
    calls = calls.copy()
    calls["y"] = truth
    for dataset in DATE_ORDER:
        part = calls[calls["dataset"] == dataset]
        for label, mask in groups:
            chosen = part[mask(part)]
            stable = chosen[chosen["stable_windows"].notna()]
            lines.append(
                "| "
                + " | ".join(
                    [
                        dataset,
                        label,
                        str(len(chosen)),
                        fmt(quantile(stable["stable_windows"], 0.5), 0),
                        fmt(quantile(stable["stable_windows"], 0.75), 0),
                        fmt(quantile(stable["stable_windows"], 0.9), 0),
                        fmt(quantile(stable["stable_speech"], 0.5)),
                        fmt(quantile(stable["stable_t"], 0.5)),
                        fmt(quantile(stable["stable_t"], 0.9)),
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def reading(tables: pd.DataFrame) -> list[str]:
    windows = tables[tables["kind"] == "windows"]
    times = tables[tables["kind"] == "time"]
    lines = []
    for mark in (1, 3):
        bits = []
        for dataset in DATE_ORDER:
            row = windows[(windows["dataset"] == dataset) & (windows["mark"] == mark)].iloc[0]
            bits.append(f"{dataset} TP {int(row.tp)} из {int(row.full_tp)}, FP {int(row.fp)} при {int(row.full_fp)} у итога")
        lines.append(f"На {mark}-м окне " + "; ".join(bits) + ".")
    for mark in (30, 60):
        bits = []
        for dataset in DATE_ORDER:
            row = times[(times["dataset"] == dataset) & (times["mark"] == mark)].iloc[0]
            bits.append(f"{dataset} TP {int(row.tp)}/{int(row.full_tp)}, ошибки {int(row.errors)}/{int(row.full_errors)}")
        lines.append(f"На {mark}-й секунде от начала записи " + "; ".join(bits) + ".")
    lines.append(
        "Human, которых итог A оставляет людьми, застывает на первом окне: p90 тоже одно окно, около 9–10 с от начала записи. "
        "Synthetic, которых итог A ловит, доходит до этого ответа позже. Медиана — 3-е окно, около 5 с речи и 17–19 с от начала записи. "
        "Четверть таких звонков всё ещё меняет голос после 4–5-го окна, p90 — 6–8-е окно и около 50–55 с от начала записи: между окнами есть паузы."
    )
    lines.append(
        "Три звонка 11–12 совпадают с итогом только когда в длину записи подставляется длина файла, а не прошедшее время. "
        "Это описание тех же двух дней. Оно не назначает секунду, на которой звонок надо обрывать."
    )
    return lines


def write_report(calls: pd.DataFrame, tables: pd.DataFrame) -> None:
    tail = int(calls["tail_match"].sum())
    text = [
        "# Сколько окон нужно, чтобы голос A совпал с итогом по всему звонку",
        "",
        "Вопрос: пока копится речь абонента, после какого окна текущий голос A уже такой же, как решение по всему файлу, и с этого момента не меняется. Пока этого нет, раннее решение ещё не то, которое модель даст в конце.",
        "",
        "Голос A прежний: Spectra-0, SVM и LightGBM, synthetic если так сказали минимум две модели. Пороги прежние. Окна Spectra-0 взяты из сохранённого прогона, сеть заново не считалась. Classical-оценки пересчитаны по сегментам, которые к этому моменту уже закрылись. В длину записи подставлено прошедшее время, не будущая длина файла. Модель без скора не голосует. Пока скора нет ни у одной модели, предсказания нет: такой звонок не считается human.",
        "",
        "Когорта та же: бинарные звонки 11–12 и 16 ноября. Спорные метки 16 ноября исключены. В знаменателе строки «дошли» только звонки, у которых к этой отметке уже есть оценка. Остальные не добираются до неё.",
        "",
        "## 1. Полный файл",
        "",
        f"К проигрышу готовы {len(calls)} из 818. Итог A по всему файлу совпал с уже опубликованным голосом у {int((calls['published_vote'] == calls['full_decision']).sum())} из {len(calls)}. Последняя оценка по ходу звонка, ещё с прошедшим временем вместо длины файла, совпала с итогом у {tail} из {len(calls)}.",
        "",
        "## 2. После N окон Spectra-0",
        "",
        "Окно — около 4.04 с речи. Строка показывает решение в момент, когда закрылось N-е окно. TP и FP считаются по метке. Рядом — те же ошибки итога A на этих же звонках.",
        "",
        "### 11–12 ноября",
        "",
        window_table(tables, "11–12"),
        "",
        "### 16 ноября",
        "",
        window_table(tables, "16"),
        "",
        "![Совпадение с итогом по числу окон](window_stability/figures/agree_by_window.png)",
        "",
        "## 3. Секунды от начала записи",
        "",
        "Здесь секунды включают паузы до речи. В ячейке TP, FP и ошибки записаны как «сейчас/итог A» на тех же звонках.",
        "",
        "### 11–12 ноября",
        "",
        time_table(tables, "time", "11–12", "Секунда"),
        "",
        "### 16 ноября",
        "",
        time_table(tables, "time", "16", "Секунда"),
        "",
        "## 4. Секунды речи абонента",
        "",
        "Речь — сумма уже закрытых сегментов VAD, не длина окна с добивкой.",
        "",
        "### 11–12 ноября",
        "",
        time_table(tables, "speech", "11–12", "Речь, с"),
        "",
        "### 16 ноября",
        "",
        time_table(tables, "speech", "16", "Речь, с"),
        "",
        "## 5. С какого момента голос больше не меняется",
        "",
        "Для каждого звонка взят первый момент, когда голос A уже равен итогу по всему файлу и все следующие обновления оставляют тот же класс. Строки разделены по метке и по итогу A: synthetic, которых итог пропускает, застывают как human рано, и в одной медиане со всеми synthetic они прячут тех, кого модель в итоге ловит.",
        "",
        stable_table(calls),
        "",
        "## 6. Каждая модель против своего итога",
        "",
        "В ячейке — сколько звонков к этому окну уже имеют скор модели и этот скор даёт тот же класс, что модель даст по всему файлу. Знаменатель меньше числа дошедших, если к окну у модели ещё нет скора.",
        "",
        "### 11–12 ноября",
        "",
        model_table(tables, "11–12"),
        "",
        "### 16 ноября",
        "",
        model_table(tables, "16"),
        "",
        "## 7. Что из этого видно",
        "",
        *reading(tables),
        "",
    ]
    REPORT.write_text("\n".join(text), encoding="utf-8")
    print(REPORT, flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False, "figure.facecolor": "white"})
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        write_report(pd.read_csv(OUT / "calls.csv", dtype={"call_id": str}), pd.read_csv(OUT / "checkpoints.csv"))
        return
    analyze()


if __name__ == "__main__":
    main()
