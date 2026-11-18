"""Пропуски и ложные срабатывания голоса A и сдвиг score между датами.

Нового прогона и подбора порогов нет. Расстояние каждой модели считается
только до её собственного зафиксированного порога. Сырые score разных
моделей между собой не вычитаются: шкалы не общие.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.research.model_assisted_annotation.rules import BELOW, THRESHOLDS, WIDTHS
from src.research.model_assisted_annotation.voting_combinations import (
    DATE_COLOR,
    DATE_ORDER,
    MODEL_NAME,
    PACK_ORDER,
    SPEECH_ORDER,
    THRESHOLD_TEXT,
    attach_votes,
    load_early,
    load_late,
    md_table,
)

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "reports" / "ensemble_a"
REPORT = ROOT / "reports" / "ensemble_a_error_analysis.md"
TRIO = ("spectra0", "svm_new", "lightgbm_new")
FN_PATTERNS = (
    "все три human",
    "synthetic только Spectra-0",
    "synthetic только SVM",
    "synthetic только LightGBM",
)
FP_PATTERNS = (
    "SVM и LightGBM",
    "Spectra-0 и LightGBM",
    "Spectra-0 и SVM",
    "все три",
)
REQUESTED_SLICES = ("human", "synthetic", "PACK1", "PACK2", "короткая", "средняя", "длинная")
CLASS_SLICES = (
    "human PACK1",
    "human PACK2",
    "synthetic PACK1",
    "synthetic PACK2",
    "human короткая",
    "human средняя",
    "human длинная",
    "synthetic короткая",
    "synthetic средняя",
    "synthetic длинная",
)
EXPECTED_FN = {
    "11–12": {"все три human": 7, "synthetic только Spectra-0": 13, "synthetic только SVM": 4, "synthetic только LightGBM": 1},
    "16": {"все три human": 6, "synthetic только Spectra-0": 8, "synthetic только SVM": 1, "synthetic только LightGBM": 0},
}
EXPECTED_FP = {
    "11–12": {"SVM и LightGBM": 7, "Spectra-0 и LightGBM": 1, "Spectra-0 и SVM": 0, "все три": 0},
    "16": {"SVM и LightGBM": 3, "Spectra-0 и LightGBM": 0, "Spectra-0 и SVM": 0, "все три": 0},
}


def reserve(model: str, score: float) -> float:
    """Положительное значение — модель ещё на стороне human. Ноль — ровно порог.

    Для Spectra-0 порог проходится вниз, для SVM и LightGBM — вверх.
    Знак приведён к одному смыслу и не делает score одной шкалой.
    """

    threshold = THRESHOLDS[model]
    if model in BELOW:
        return float(score) - threshold
    return threshold - float(score)


def on_synthetic_side(model: str, score: float) -> bool:
    if model in BELOW:
        return float(score) < THRESHOLDS[model]
    return float(score) >= THRESHOLDS[model]


def cliffs_delta(later: np.ndarray, earlier: np.ndarray) -> float:
    """Положительное значение: на 16 ноября score чаще больше, чем на 11–12."""

    left = np.asarray(later, dtype=float)
    right = np.asarray(earlier, dtype=float)
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if left.size == 0 or right.size == 0:
        return float("nan")
    diff = left[:, None] - right[None, :]
    return float((diff > 0).mean() - (diff < 0).mean())


def prepare() -> pd.DataFrame:
    early, _ = load_early([])
    late, _ = load_late([])
    frames = [attach_votes(early), attach_votes(late)]
    frame = pd.concat(frames, ignore_index=True)
    for model in TRIO:
        score = pd.to_numeric(frame[f"{model}_score"], errors="coerce")
        frame[f"{model}_reserve"] = [reserve(model, value) for value in score]
        frame[f"{model}_near"] = frame[f"{model}_reserve"].abs() <= WIDTHS[model]
    return frame


def vote_pattern(row: pd.Series, kind: str) -> str:
    synthetic = [model for model in TRIO if bool(row[model])]
    if kind == "FN":
        if not synthetic:
            return "все три human"
        if synthetic == ["spectra0"]:
            return "synthetic только Spectra-0"
        if synthetic == ["svm_new"]:
            return "synthetic только SVM"
        if synthetic == ["lightgbm_new"]:
            return "synthetic только LightGBM"
        raise AssertionError(f"у пропуска A больше одного голоса synthetic: {row.call_id}")
    names = {model: MODEL_NAME[model] for model in synthetic}
    if len(synthetic) == 3:
        return "все три"
    if set(synthetic) == {"svm_new", "lightgbm_new"}:
        return "SVM и LightGBM"
    if set(synthetic) == {"spectra0", "lightgbm_new"}:
        return "Spectra-0 и LightGBM"
    if set(synthetic) == {"spectra0", "svm_new"}:
        return "Spectra-0 и SVM"
    raise AssertionError(f"неожиданный состав FP {row.call_id}: {names}")


def closest_human(row: pd.Series) -> tuple[str, float]:
    human = [model for model in TRIO if not bool(row[model])]
    if not human:
        return "", float("nan")
    model = min(human, key=lambda name: row[f"{name}_reserve"])
    return MODEL_NAME[model], float(row[f"{model}_reserve"])


def error_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for kind, mask in (("FN", frame["y"] & ~frame["A"]), ("FP", ~frame["y"] & frame["A"])):
        part = frame.loc[mask]
        for row in part.itertuples(index=False):
            series = part.loc[part["call_id"] == row.call_id].iloc[0]
            pattern = vote_pattern(series, kind)
            human_name, human_gap = closest_human(series)
            synthetic_models = [MODEL_NAME[model] for model in TRIO if bool(series[model])]
            human_models = [MODEL_NAME[model] for model in TRIO if not bool(series[model])]
            near_models = [MODEL_NAME[model] for model in TRIO if bool(series[f"{model}_near"])]
            classical = {bool(series["svm_new"]), bool(series["lightgbm_new"])}
            if classical == {True}:
                classical_text = "обе synthetic"
            elif classical == {False}:
                classical_text = "обе human"
            else:
                classical_text = "разошлись"
            span = max(series[f"{model}_reserve"] for model in TRIO) - min(series[f"{model}_reserve"] for model in TRIO)
            rows.append(
                {
                    "kind": kind,
                    "dataset": series["dataset"],
                    "call_id": series["call_id"],
                    "pack": series["pack"],
                    "speech_s": float(series["speech_s"]),
                    "speech_bin": series["speech_bin"],
                    "primary_label": series["primary_label"],
                    "pattern": pattern,
                    "synthetic_voters": ", ".join(synthetic_models) if synthetic_models else "нет",
                    "human_voters": ", ".join(human_models) if human_models else "нет",
                    "classical_agreement": classical_text,
                    "closest_human_model": human_name,
                    "closest_human_reserve": human_gap,
                    "reserve_span": float(span),
                    "near_models": ", ".join(near_models) if near_models else "нет",
                    "spectra0": "synthetic" if series["spectra0"] else "human",
                    "svm_new": "synthetic" if series["svm_new"] else "human",
                    "lightgbm_new": "synthetic" if series["lightgbm_new"] else "human",
                    "spectra0_score": float(series["spectra0_score"]),
                    "svm_new_score": float(series["svm_new_score"]),
                    "lightgbm_new_score": float(series["lightgbm_new_score"]),
                    "spectra0_reserve": float(series["spectra0_reserve"]),
                    "svm_new_reserve": float(series["svm_new_reserve"]),
                    "lightgbm_new_reserve": float(series["lightgbm_new_reserve"]),
                }
            )
    return pd.DataFrame(rows)


def pattern_summary(errors: pd.DataFrame, kind: str) -> pd.DataFrame:
    rows = []
    patterns = FN_PATTERNS if kind == "FN" else FP_PATTERNS
    for dataset in DATE_ORDER:
        part = errors[(errors["kind"] == kind) & (errors["dataset"] == dataset)]
        for pattern in patterns:
            group = part[part["pattern"] == pattern]
            row = {
                "dataset": dataset,
                "pattern": pattern,
                "n": int(len(group)),
                "PACK1": int((group["pack"] == "PACK1").sum()),
                "PACK2": int((group["pack"] == "PACK2").sum()),
                "короткая": int((group["speech_bin"] == "короткая").sum()),
                "средняя": int((group["speech_bin"] == "средняя").sum()),
                "длинная": int((group["speech_bin"] == "длинная").sum()),
            }
            if group.empty:
                row.update(
                    {
                        "median_speech": float("nan"),
                        "median_spectra": float("nan"),
                        "median_svm": float("nan"),
                        "median_lgbm": float("nan"),
                        "near_spectra": 0,
                        "near_svm": 0,
                        "near_lgbm": 0,
                    }
                )
            else:
                row.update(
                    {
                        "median_speech": float(group["speech_s"].median()),
                        "median_spectra": float(group["spectra0_reserve"].median()),
                        "median_svm": float(group["svm_new_reserve"].median()),
                        "median_lgbm": float(group["lightgbm_new_reserve"].median()),
                        "near_spectra": int(group["spectra0_reserve"].abs().le(WIDTHS["spectra0"]).sum()),
                        "near_svm": int(group["svm_new_reserve"].abs().le(WIDTHS["svm_new"]).sum()),
                        "near_lgbm": int(group["lightgbm_new_reserve"].abs().le(WIDTHS["lightgbm_new"]).sum()),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def slice_mask(frame: pd.DataFrame, name: str) -> pd.Series:
    if name == "human":
        return ~frame["y"]
    if name == "synthetic":
        return frame["y"]
    if name in PACK_ORDER:
        return frame["pack"].eq(name)
    if name in SPEECH_ORDER:
        return frame["speech_bin"].eq(name)
    class_name, rest = name.split(" ", 1)
    mask = frame["y"] if class_name == "synthetic" else ~frame["y"]
    if rest in PACK_ORDER:
        return mask & frame["pack"].eq(rest)
    return mask & frame["speech_bin"].eq(rest)


def shift_table(frame: pd.DataFrame, slices: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for name in slices:
        mask = slice_mask(frame, name)
        for model in TRIO:
            earlier = frame.loc[mask & frame["dataset"].eq("11–12"), f"{model}_score"].to_numpy(dtype=float)
            later = frame.loc[mask & frame["dataset"].eq("16"), f"{model}_score"].to_numpy(dtype=float)
            rate_early = float(np.mean([on_synthetic_side(model, value) for value in earlier])) if len(earlier) else float("nan")
            rate_late = float(np.mean([on_synthetic_side(model, value) for value in later])) if len(later) else float("nan")
            rows.append(
                {
                    "slice": name,
                    "model": MODEL_NAME[model],
                    "model_id": model,
                    "n_early": int(len(earlier)),
                    "n_late": int(len(later)),
                    "median_early": float(np.median(earlier)) if len(earlier) else float("nan"),
                    "median_late": float(np.median(later)) if len(later) else float("nan"),
                    "delta_median": float(np.median(later) - np.median(earlier)) if len(earlier) and len(later) else float("nan"),
                    "synth_rate_early": rate_early,
                    "synth_rate_late": rate_late,
                    "delta_synth_pp": (rate_late - rate_early) * 100 if np.isfinite(rate_early) and np.isfinite(rate_late) else float("nan"),
                    "cliff": cliffs_delta(later, earlier),
                }
            )
    return pd.DataFrame(rows)


def check(frame: pd.DataFrame, errors: pd.DataFrame) -> list[str]:
    passed = []
    if set(frame.loc[frame["dataset"].eq("11–12"), "call_id"]).intersection(frame.loc[frame["dataset"].eq("16"), "call_id"]):
        raise AssertionError("когорты пересекаются")
    counts = frame.groupby("dataset")["y"].agg(["size", "sum"])
    if int(counts.loc["11–12", "size"]) != 457 or int(counts.loc["11–12", "sum"]) != 134:
        raise AssertionError("когорта 11–12 изменилась")
    if int(counts.loc["16", "size"]) != 361 or int(counts.loc["16", "sum"]) != 58:
        raise AssertionError("когорта 16 ноября изменилась")
    fn = errors[errors["kind"] == "FN"]
    fp = errors[errors["kind"] == "FP"]
    if len(fn) != 40 or len(fp) != 11:
        raise AssertionError(f"ожидались 40 FN и 11 FP, получено {len(fn)} и {len(fp)}")
    for dataset, expected in EXPECTED_FN.items():
        got = fn[fn["dataset"] == dataset]["pattern"].value_counts().to_dict()
        for pattern, count in expected.items():
            if int(got.get(pattern, 0)) != count:
                raise AssertionError(f"FN {dataset} {pattern}: {got.get(pattern, 0)} != {count}")
        votes = frame.loc[frame["dataset"].eq(dataset) & frame["y"] & ~frame["A"], TRIO].sum(axis=1)
        if not (votes < 2).all():
            raise AssertionError("в пропуски A попал звонок с двумя голосами synthetic")
    for dataset, expected in EXPECTED_FP.items():
        got = fp[fp["dataset"] == dataset]["pattern"].value_counts().to_dict()
        for pattern, count in expected.items():
            if int(got.get(pattern, 0)) != count:
                raise AssertionError(f"FP {dataset} {pattern}: {got.get(pattern, 0)} != {count}")
        votes = frame.loc[frame["dataset"].eq(dataset) & ~frame["y"] & frame["A"], TRIO].sum(axis=1)
        if not (votes >= 2).all():
            raise AssertionError("в ложные A попал звонок меньше чем с двумя голосами")
    for model in TRIO:
        score = pd.to_numeric(frame[f"{model}_score"], errors="coerce")
        gap = frame[f"{model}_reserve"]
        decision = frame[model].astype(bool)
        if model in BELOW:
            if not ((gap < 0) == decision).all() or not ((gap >= 0) == ~decision).all():
                raise AssertionError(f"запас {model} не совпал со стороной порога")
        else:
            if not ((gap <= 0) == decision).all() or not ((gap > 0) == ~decision).all():
                raise AssertionError(f"запас {model} не совпал со стороной порога")
        if not np.allclose(gap, [reserve(model, value) for value in score]):
            raise AssertionError(f"запас {model} пересчитался иначе")
    if THRESHOLDS["spectra0"] != 0.22710928916931153 or THRESHOLDS["svm_new"] != 0.0:
        raise AssertionError("порог сдвинулся")
    if THRESHOLDS["lightgbm_new"] != 0.3460832258410117:
        raise AssertionError("порог LightGBM сдвинулся")
    passed.append("40 пропусков и 11 ложных A разобраны по тем же когортам, порог не сдвигался")
    passed.append("запас до synthetic имеет один смысл для трёх шкал и совпадает с бинарным решением")
    passed.append("состав FN и FP совпал с уже проверенным разложением тройки")
    return passed


def fmt_num(value: float, digits: int = 2) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def fmt_pp(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{value:+.1f}"


def style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.6)
    ax.set_axisbelow(True)


def plot_patterns(path: Path, summary: pd.DataFrame, patterns: tuple[str, ...], title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    x = np.arange(len(patterns))
    width = 0.36
    for index, dataset in enumerate(DATE_ORDER):
        part = summary[summary["dataset"] == dataset].set_index("pattern")
        values = [int(part.loc[pattern, "n"]) if pattern in part.index else 0 for pattern in patterns]
        bars = ax.bar(x + (index - 0.5) * width, values, width, label=dataset + " ноября", color=DATE_COLOR[dataset])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(patterns, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    style_ax(ax)
    ax.legend(frameon=False)
    ax.set_ylim(0, max(int(summary["n"].max()) * 1.25, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_fn_reserve(path: Path, errors: pd.DataFrame) -> None:
    fn = errors[errors["kind"] == "FN"]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.4), sharey=False)
    columns = ("spectra0_reserve", "svm_new_reserve", "lightgbm_new_reserve")
    for ax, model, column in zip(axes, TRIO, columns):
        for index, dataset in enumerate(DATE_ORDER):
            values = fn.loc[fn["dataset"].eq(dataset), column].to_numpy(dtype=float)
            jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) else []
            ax.scatter(np.full(len(values), index) + jitter, values, s=22, color=DATE_COLOR[dataset], alpha=0.85, label=dataset + " ноября")
        ax.axhline(0, color="#222", lw=0.8, ls="--")
        ax.axhline(WIDTHS[model], color="#888", lw=0.7, ls=":")
        ax.axhline(-WIDTHS[model], color="#888", lw=0.7, ls=":")
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"{dataset}\nn={int((fn['dataset'] == dataset).sum())}" for dataset in DATE_ORDER])
        ax.set_title(MODEL_NAME[model])
        ax.set_ylabel("Запас до synthetic" if model == "spectra0" else "")
        style_ax(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Пропуски A. Выше нуля модель ещё говорит human. Пунктир — прежняя пограничная зона.", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_fp_reserve(path: Path, frame: pd.DataFrame, errors: pd.DataFrame) -> None:
    fp = errors[errors["kind"] == "FP"]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.4))
    for ax, model in zip(axes, TRIO):
        positions = []
        labels = []
        slot = 0
        for dataset in DATE_ORDER:
            humans = frame[(frame["dataset"] == dataset) & ~frame["y"]]
            tn = humans.loc[~humans["A"], f"{model}_reserve"].to_numpy(dtype=float)
            wrong = fp.loc[fp["dataset"].eq(dataset), f"{model}_reserve"].to_numpy(dtype=float)
            ax.boxplot([tn], positions=[slot], widths=0.55, showfliers=False, patch_artist=True,
                       boxprops={"facecolor": DATE_COLOR[dataset], "alpha": 0.35})
            ax.scatter(np.full(len(wrong), slot), wrong, s=28, color="#8d3b3b", zorder=3)
            labels.append(f"{dataset}\nTN {len(tn)}, FP {len(wrong)}")
            positions.append(slot)
            slot += 1
        ax.axhline(0, color="#222", lw=0.8, ls="--")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(MODEL_NAME[model])
        style_ax(ax)
    axes[0].set_ylabel("Запас до synthetic")
    fig.suptitle("Ложные A на фоне верных human. Точки — 11 FP. Ящик — TN без выбросов, чтобы 11 точек было видно.", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_short(path: Path, frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
    for ax, dataset in zip(axes, DATE_ORDER):
        humans = frame[(frame["dataset"] == dataset) & ~frame["y"]]
        rates = []
        labels = []
        for name in SPEECH_ORDER:
            part = humans[humans["speech_bin"] == name]
            fp = int(part["A"].sum())
            rates.append(fp / len(part) if len(part) else 0)
            labels.append(f"{name}\n{fp}/{len(part)}")
        bars = ax.bar(np.arange(3), rates, color=DATE_COLOR[dataset])
        for bar, label in zip(bars, labels):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label.split("\n")[1], ha="center", va="bottom", fontsize=8)
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels([name for name in SPEECH_ORDER])
        ax.set_title(f"{dataset} ноября, human")
        style_ax(ax)
        ax.set_ylabel("Доля FP голоса A" if dataset == "11–12" else "")
    fig.suptitle("Ложные A среди human по длительности речи. Подпись — FP / human группы.", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_shift(path: Path, frame: pd.DataFrame, row_names: tuple[str, ...], title: str) -> None:
    fig, axes = plt.subplots(
        len(row_names),
        3,
        figsize=(11.6, 3.15 * len(row_names)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, slice_name in enumerate(row_names):
        mask = slice_mask(frame, slice_name)
        for col_index, model in enumerate(TRIO):
            ax = axes[row_index, col_index]
            groups = []
            labels = []
            for dataset in DATE_ORDER:
                values = frame.loc[mask & frame["dataset"].eq(dataset), f"{model}_score"].to_numpy(dtype=float)
                groups.append(values)
                labels.append(f"{dataset}\nn={len(values)}")
            drawn = [values if len(values) else [np.nan] for values in groups]
            box = ax.boxplot(drawn, positions=[0, 1], widths=0.55, patch_artist=True, showfliers=True)
            for patch, dataset in zip(box["boxes"], DATE_ORDER):
                patch.set_facecolor(DATE_COLOR[dataset])
                patch.set_alpha(0.55)
            for index, values in enumerate(groups):
                if 0 < len(values) <= 40:
                    jitter = np.linspace(-0.12, 0.12, len(values))
                    ax.scatter(np.full(len(values), index) + jitter, values, s=12, color="#222", alpha=0.45, zorder=3)
            ax.axhline(THRESHOLDS[model], color="#222", lw=0.8, ls="--")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(labels, fontsize=8)
            if row_index == 0:
                ax.set_title(MODEL_NAME[model])
            if col_index == 0:
                ax.set_ylabel(slice_name)
            style_ax(ax)
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def call_md(errors: pd.DataFrame, kind: str) -> str:
    part = errors[errors["kind"] == kind].sort_values(["dataset", "pattern", "pack", "call_id"])
    rows = []
    for row in part.itertuples(index=False):
        rows.append(
            [
                row.dataset,
                f"`{row.call_id}`",
                row.pack,
                f"{row.speech_s:.2f}",
                row.speech_bin,
                row.pattern,
                fmt_num(row.spectra0_reserve, 3),
                fmt_num(row.svm_new_reserve, 3),
                fmt_num(row.lightgbm_new_reserve, 3),
                row.closest_human_model or "—",
                row.near_models,
            ]
        )
    return md_table(
        ["Дата", "call_id", "Пакет", "Речь, с", "Длина", "Кто сказал synthetic", "Запас Spectra-0", "Запас SVM", "Запас LightGBM", "Ближайший human", "Внутри зоны"],
        rows,
        ["---", "---", "---", "---:", "---", "---", "---:", "---:", "---:", "---", "---"],
    )


def summary_md(summary: pd.DataFrame) -> str:
    rows = []
    for row in summary.itertuples(index=False):
        rows.append(
            [
                row.dataset,
                row.pattern,
                int(row.n),
                int(row.PACK1),
                int(row.PACK2),
                int(row.короткая),
                int(row.средняя),
                int(row.длинная),
                fmt_num(row.median_speech, 1),
                fmt_num(row.median_spectra, 3),
                fmt_num(row.median_svm, 3),
                fmt_num(row.median_lgbm, 3),
            ]
        )
    return md_table(
        ["Дата", "Состав", "N", "PACK1", "PACK2", "Короткая", "Средняя", "Длинная", "Медиана речи, с", "Медиана запаса Spectra-0", "SVM", "LightGBM"],
        rows,
        ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
    )


def shift_md(shifts: pd.DataFrame, slices: tuple[str, ...]) -> str:
    rows = []
    for name in slices:
        for model in TRIO:
            row = shifts[(shifts["slice"] == name) & (shifts["model_id"] == model)].iloc[0]
            rows.append(
                [
                    name,
                    MODEL_NAME[model],
                    int(row.n_early),
                    int(row.n_late),
                    fmt_num(row.median_early, 3),
                    fmt_num(row.median_late, 3),
                    fmt_num(row.delta_median, 3),
                    f"{row.synth_rate_early:.3f}" if np.isfinite(row.synth_rate_early) else "—",
                    f"{row.synth_rate_late:.3f}" if np.isfinite(row.synth_rate_late) else "—",
                    fmt_pp(row.delta_synth_pp),
                    fmt_num(row.cliff, 2),
                ]
            )
    return md_table(
        ["Срез", "Модель", "n 11–12", "n 16", "Медиана 11–12", "Медиана 16", "Δ медианы", "Доля synthetic 11–12", "Доля synthetic 16", "Δ доли, п.п.", "Cliff"],
        rows,
        ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
    )


def fn_reading(summary: pd.DataFrame, errors: pd.DataFrame) -> str:
    pieces = []
    for dataset in DATE_ORDER:
        part = summary[summary["dataset"] == dataset].set_index("pattern")
        total = int(part["n"].sum())
        only_spectra = part.loc["synthetic только Spectra-0"]
        nobody = part.loc["все три human"]
        pieces.append(
            f"На {dataset} ноября голос A пропускает {total} synthetic. "
            f"{int(only_spectra['n'])} из них synthetic называет только Spectra-0: обе classical-модели говорят human, и второго голоса нет. "
            f"{int(nobody['n'])} не называет synthetic ни одна модель тройки. "
            f"Только SVM говорит synthetic: {int(part.loc['synthetic только SVM', 'n'])}. Только LightGBM: {int(part.loc['synthetic только LightGBM', 'n'])}."
        )
    only = errors[(errors["kind"] == "FN") & (errors["pattern"] == "synthetic только Spectra-0")]
    near_svm = int(only["svm_new_reserve"].abs().le(WIDTHS["svm_new"]).sum())
    near_lgbm = int(only["lightgbm_new_reserve"].abs().le(WIDTHS["lightgbm_new"]).sum())
    deep = only[(only["svm_new_reserve"] > WIDTHS["svm_new"]) & (only["lightgbm_new_reserve"] > WIDTHS["lightgbm_new"])]
    pieces.append(
        f"В этих {len(only)} звонках, которые видит одна Spectra-0, внутри прежней пограничной зоны SVM лежит {near_svm}, "
        f"LightGBM — {near_lgbm}. У {len(deep)} обе classical-модели дальше своей зоны: пропуск не упирается в порог. "
        "Медианы запаса в таблице ниже положительны у SVM и LightGBM и отрицательны у Spectra-0. "
        "Это разница положения относительно своих порогов, не разность сырых score."
    )
    nobody = errors[(errors["kind"] == "FN") & (errors["pattern"] == "все три human")]
    near_any = int(nobody["near_models"].ne("нет").sum())
    pieces.append(
        f"Среди {len(nobody)} общих пропусков хотя бы одна модель попадает в свою пограничную зону у {near_any} звонков. "
        "Остальные лежат по human-сторону всех трёх порогов дальше этой зоны. Голос 2 из 3 их не достаёт ни заменой порога внутри зоны, ни перестановкой тех же трёх голосов."
    )
    return "\n\n".join(pieces)


def fp_reading(frame: pd.DataFrame, errors: pd.DataFrame) -> str:
    fp = errors[errors["kind"] == "FP"]
    lines = []
    for dataset in DATE_ORDER:
        part = fp[fp["dataset"] == dataset]
        pair = int((part["pattern"] == "SVM и LightGBM").sum())
        other = int(len(part) - pair)
        lines.append(
            f"На {dataset} ноября ложных A — {len(part)}. Решающая пара SVM и LightGBM, при human у Spectra-0, даёт {pair}. "
            f"Другой состав — {other}."
        )
    classical_yes = int((fp["classical_agreement"] == "обе synthetic").sum())
    lines.append(
        f"Из всех {len(fp)} ложных classical-модели согласны на synthetic в {classical_yes}. "
        "Spectra-0 в этих звонках почти всегда остаётся human и итог не удерживает: двух classical-голосов уже достаточно."
    )
    bits = []
    for dataset in DATE_ORDER:
        humans = frame[(frame["dataset"] == dataset) & ~frame["y"]]
        for name in SPEECH_ORDER:
            group = humans[humans["speech_bin"] == name]
            bits.append(f"{dataset}, {name}: {int(group['A'].sum())}/{len(group)}")
    short_fp = int((fp["speech_bin"] == "короткая").sum())
    short_human = int((~frame["y"] & frame["speech_bin"].eq("короткая")).sum())
    human_n = int((~frame["y"]).sum())
    lines.append(
        "Доля FP среди human по длительности: " + "; ".join(bits) + ". "
        f"Короткой речи среди {len(fp)} ложных — {short_fp}. Среди всех human коротких {short_human} из {human_n}. "
        "Одиночные ложные Spectra-0 на короткой речи в этот список не входят: SVM и LightGBM их не подтверждают, и голос A оставляет human."
    )
    return "\n\n".join(lines)


def shift_reading(shifts: pd.DataFrame) -> str:
    class_rows = shifts[shifts["slice"].isin(("human", "synthetic"))]
    within = shifts[shifts["slice"].isin(CLASS_SLICES)].copy()
    within = within[(within["n_early"] >= 15) & (within["n_late"] >= 15)]
    by_rate = within.reindex(within["delta_synth_pp"].abs().sort_values(ascending=False).index)
    by_cliff = within.reindex(within["cliff"].abs().sort_values(ascending=False).index)
    sentences = [
        "Доля synthetic в таблице — доля звонков по уже зафиксированному порогу, не новый порог. "
        "Cliff сравнивает сами score: положительное значение значит, что на 16 ноября число чаще больше, чем на 11–12. "
        "У Spectra-0 больший score — дальше в сторону human. У SVM и LightGBM больший score — дальше в сторону synthetic."
    ]
    for slice_name in ("human", "synthetic"):
        part = class_rows[class_rows["slice"] == slice_name]
        bits = []
        for row in part.itertuples(index=False):
            bits.append(
                f"{row.model} медиана {row.delta_median:+.3f}, доля synthetic {row.delta_synth_pp:+.1f} п.п., Cliff {row.cliff:+.2f}"
            )
        sentences.append(f"На срезе {slice_name}: " + "; ".join(bits) + ".")
    top_rate = ", ".join(
        f"{row.slice} / {row.model} {row.delta_synth_pp:+.1f} п.п. (n {int(row.n_early)} и {int(row.n_late)})"
        for row in by_rate.head(3).itertuples(index=False)
    )
    top_cliff = by_cliff.iloc[0]
    sentences.append(
        "Смешанные срезы PACK и длительности здесь не используются для выбора самого сильного сдвига: "
        "на 11–12 в длинной речи 57 synthetic из 152, на 16 ноября 21 из 126, и падение доли synthetic в смешанной группе повторяет этот состав. "
        f"Внутри класса, при не меньше чем 15 звонках в каждой дате, крупнейшие сдвиги доли по порогу: {top_rate}. "
        f"Крупнейший сдвиг самого распределения — {top_cliff['slice']}, {top_cliff['model']}, Cliff {top_cliff['cliff']:+.2f}."
    )
    sentences.append(
        "На human доля по порогу меняется в пределах нескольких пунктов и в пакетах, и по длине. "
        "На synthetic общий Cliff Spectra-0 близок к нулю, потому что средняя речь уходит в сторону synthetic, а длинная — в сторону human. "
        "Группы меньше 15 звонков, в том числе 7 synthetic PACK2 и 7 коротких synthetic 16 ноября, в этот выбор самой сильной группы не входят."
    )
    return "\n\n".join(sentences)


def build_report(frame: pd.DataFrame, errors: pd.DataFrame, fn_summary: pd.DataFrame, fp_summary: pd.DataFrame, shifts: pd.DataFrame, checks: list[str]) -> str:
    lines = [
        "# Пропуски, ложные срабатывания и устойчивость score голоса A",
        "",
        "Голос A — Spectra-0, SVM new и LightGBM new. Synthetic, если так сказали минимум две модели. Пороги прежние. Нового прогона нет.",
        "",
        "Запас до synthetic у каждой модели — расстояние до её собственного порога. Положительный запас значит, что модель ещё говорит human. Отрицательный — уже говорит synthetic. Сырой margin Spectra-0, score SVM и score LightGBM между собой не вычитаются.",
        "",
        "Пограничная зона взята из `rules.WIDTHS` и на этих днях не пересчитывалась: Spectra-0 0.716, SVM 0.160, LightGBM 0.083. Это 0.10 IQR на валидации 13 ноября, не новый порог.",
        "",
        "## 1. Пропуски A",
        "",
        "Пропуск — synthetic-звонок, которому голос A поставил human. На 11–12 таких 25 из 134. На 16 ноября — 15 из 58.",
        "",
        fn_reading(fn_summary, errors),
        "",
        summary_md(fn_summary),
        "",
        "![Состав пропусков](ensemble_a/figures/fn_patterns.png)",
        "",
        "![Запас моделей на пропусках](ensemble_a/figures/fn_reserve.png)",
        "",
        "Каждый пропуск:",
        "",
        call_md(errors, "FN"),
        "",
        "Колонка «Ближайший human» — модель, которая сказала human и стоит ближе остальных таких моделей к своему порогу. Если она перейдёт порог, при одном уже имеющемся голосе synthetic итог A сменится. При составе «все три human» одной такой модели недостаточно: нужны два перехода.",
        "",
        "## 2. Ложные срабатывания A",
        "",
        "Ложное — human-звонок, которому A поставил synthetic. На 11–12 таких 8 из 323. На 16 ноября — 3 из 303. В обеих когортах все human имеют метку `human_live`, `human_recorded` в бинарной части нет. Отдельного сравнения подтипов human эти звонки не дают.",
        "",
        fp_reading(frame, errors),
        "",
        summary_md(fp_summary),
        "",
        "![Состав ложных](ensemble_a/figures/fp_patterns.png)",
        "",
        "![Ложные на фоне верных human](ensemble_a/figures/fp_reserve.png)",
        "",
        "![Доля ложных по длительности речи](ensemble_a/figures/fp_by_speech.png)",
        "",
        "Каждое ложное. Решающий состав — две или три модели, которые сказали synthetic. При двух голосах оба необходимы: снятие любого возвращает human.",
        "",
        call_md(errors, "FP"),
        "",
        "## 3. Устойчивость score между датами",
        "",
        shift_reading(shifts),
        "",
        "### Запрошенные срезы",
        "",
        shift_md(shifts, REQUESTED_SLICES),
        "",
        "![Human и synthetic](ensemble_a/figures/score_class.png)",
        "",
        "Пунктир — зафиксированный порог модели. У Spectra-0 synthetic ниже линии. У SVM и LightGBM synthetic на линии и выше. Оси у моделей разные.",
        "",
        "![PACK1 и PACK2 без разреза по классу](ensemble_a/figures/score_pack_pooled.png)",
        "",
        "![Длительность речи без разреза по классу](ensemble_a/figures/score_speech_pooled.png)",
        "",
        "### Те же срезы внутри класса",
        "",
        "Этот разрез нужен, потому что PACK2 и короткая речь содержат мало synthetic. Смешанный срез тогда в основном описывает human.",
        "",
        shift_md(shifts, CLASS_SLICES),
        "",
        "![Пакеты внутри класса](ensemble_a/figures/score_pack_class.png)",
        "",
        "![Длительность внутри класса](ensemble_a/figures/score_speech_class.png)",
        "",
        "Точки поверх ящика показаны там, где в группе не больше 40 звонков. У synthetic PACK2 на 16 ноября таких 7, у короткой synthetic этого дня тоже 7.",
        "",
        "## Что из этого следует для голоса A",
        "",
        "Пропуски A в основном двух видов. Либо synthetic видит только Spectra-0, а SVM и LightGBM остаются на стороне human часто дальше своей пограничной зоны. Либо synthetic не видит никто из трёх. Первые теряются самим правилом 2 из 3: оно для того и собрано, чтобы одиночный голос Spectra-0 не проходил. Вторые этим правилом не лечатся.",
        "",
        "Ложные A почти все держатся на согласии SVM и LightGBM. Spectra-0 в этих звонках обычно говорит human. Короткая речь поднимает ложные у одной Spectra-0, но не у голоса A: classical-модели их не подтверждают. Одиннадцать ложных — мало для отдельного акустического типа, и других полей, кроме пакета и длительности речи, в этот разбор не добавлялось.",
        "",
        "На всех human вместе доля по порогу меняется меньше чем на 3 п.п., Cliff от −0.06 до −0.08. В длинной human-речи SVM и LightGBM снижают эту долю примерно на 5 п.п. "
        "На synthetic Spectra-0 сохраняет долю около 0.74–0.76, хотя медиана margin сдвигается на 0.65. "
        "SVM и LightGBM на synthetic 16 ноября реже переходят свой порог, сильнее всего на длинной речи: там при 57 и 21 звонке доля падает примерно на 20 п.п. у всех трёх моделей. "
        "Смешанный срез длинной речи и PACK2 этот сдвиг преувеличивает, потому что вместе с ним меняется доля synthetic в группе.",
        "",
        "Проверки перед записью отчёта:",
        "",
    ]
    for item in checks:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Списки `call_id` и таблицы сдвига лежат в `reports/ensemble_a/`. Пороги и метки не менялись.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False, "figure.facecolor": "white"})
    frame = prepare()
    errors = error_table(frame)
    fn_summary = pattern_summary(errors, "FN")
    fp_summary = pattern_summary(errors, "FP")
    shifts = pd.concat(
        [shift_table(frame, REQUESTED_SLICES), shift_table(frame, CLASS_SLICES)],
        ignore_index=True,
    )
    checks = check(frame, errors)
    figures = OUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    errors.to_csv(OUT / "errors.csv", index=False, encoding="utf-8-sig")
    fn_summary.to_csv(OUT / "fn_patterns.csv", index=False, encoding="utf-8-sig")
    fp_summary.to_csv(OUT / "fp_patterns.csv", index=False, encoding="utf-8-sig")
    shifts.to_csv(OUT / "score_shift.csv", index=False, encoding="utf-8-sig")
    plot_patterns(figures / "fn_patterns.png", fn_summary, FN_PATTERNS, "Пропуски A: сколько synthetic-голосов не хватило до двух", "Звонков")
    plot_patterns(figures / "fp_patterns.png", fp_summary, FP_PATTERNS, "Ложные A: какая пара дала два голоса synthetic", "Звонков")
    plot_fn_reserve(figures / "fn_reserve.png", errors)
    plot_fp_reserve(figures / "fp_reserve.png", frame, errors)
    plot_short(figures / "fp_by_speech.png", frame)
    plot_shift(figures / "score_class.png", frame, ("human", "synthetic"), "Score human и synthetic. Пунктир — зафиксированный порог.")
    plot_shift(figures / "score_pack_pooled.png", frame, ("PACK1", "PACK2"), "Score пакетов без разреза по классу. Пунктир — порог.")
    plot_shift(figures / "score_speech_pooled.png", frame, SPEECH_ORDER, "Score по длительности речи без разреза по классу. Пунктир — порог.")
    plot_shift(
        figures / "score_pack_class.png",
        frame,
        ("human PACK1", "human PACK2", "synthetic PACK1", "synthetic PACK2"),
        "Score пакетов внутри класса. Пунктир — порог.",
    )
    plot_shift(
        figures / "score_speech_class.png",
        frame,
        ("human короткая", "human средняя", "human длинная", "synthetic короткая", "synthetic средняя", "synthetic длинная"),
        "Score по длительности речи внутри класса. Пунктир — порог.",
    )
    REPORT.write_text(build_report(frame, errors, fn_summary, fp_summary, shifts, checks), encoding="utf-8")
    print(REPORT)
    for line in checks:
        print("OK", line)


if __name__ == "__main__":
    main()
