"""Пересечение ошибок Spectra-0, SVM new и LightGBM new.

Пороги уже заморожены в сохранённых предсказаниях. Нового ensemble нет.
"""

from __future__ import annotations

import pandas as pd

from src.research.model_assisted_annotation.common_misses import REPORT, ROOT, read_csv

MODELS = ("spectra0", "svm_new", "lightgbm_new")
NAMES = {"spectra0": "Spectra-0", "svm_new": "SVM new", "lightgbm_new": "LightGBM new"}
HUMAN = {"human_live", "human_recorded"}
SYNTHETIC = {"synthetic_tts", "synthetic_voice_conversion", "synthetic_playback"}
PAIRS = (("spectra0", "svm_new"), ("spectra0", "lightgbm_new"), ("svm_new", "lightgbm_new"))


def _frame_1112() -> pd.DataFrame:
    pred = pd.DataFrame(read_csv(REPORT / "per_call_predictions.csv")).set_index("call_id")
    labels = pd.DataFrame(read_csv(ROOT / "data" / "annotations" / "priority_20261111_20261112_calls.csv"))
    labels = labels.drop_duplicates("call_id").set_index("call_id")
    binary = labels[labels["primary_label"].isin(HUMAN | SYNTHETIC)]
    frame = binary[["primary_label"]].join(pred[[f"{name}_prediction" for name in MODELS]])
    frame["synthetic"] = frame["primary_label"].isin(SYNTHETIC)
    return frame


def _frame_16() -> pd.DataFrame:
    path = ROOT / "reports" / "day_20261116_model_error_overlap" / "per_call_predictions.csv"
    long = pd.read_csv(path, encoding="utf-8-sig")
    long = long[long["model"].isin(MODELS)]
    wide = long.pivot(index="call_id", columns="model", values="predicted_label")
    wide.columns = [f"{name}_prediction" for name in wide.columns]
    truth = long.drop_duplicates("call_id").set_index("call_id")[["ground_truth", "primary_label"]]
    frame = truth.join(wide)
    frame["synthetic"] = frame["ground_truth"].eq("synthetic")
    return frame


def _flags(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    ready = frame.copy()
    for name in MODELS:
        column = f"{name}_prediction"
        ready[column] = ready[column].fillna("").astype(str)
        ready = ready[ready[column].isin(["human", "synthetic"])]
    dropped = len(frame) - len(ready)
    for name in MODELS:
        ready[name] = ready[f"{name}_prediction"].eq("synthetic")
    return ready, dropped


def _region(row: pd.Series, models: tuple[str, ...]) -> str:
    hit = [NAMES[name] for name in models if bool(row[name])]
    return "+".join(hit) if hit else "никто"


def _errors(frame: pd.DataFrame, kind: str) -> pd.Series:
    if kind == "fn":
        part = frame[frame["synthetic"]].copy()
        # ошибка модели — она сказала human
        for name in MODELS:
            part[name] = ~part[name]
    else:
        part = frame[~frame["synthetic"]].copy()
    return part.apply(lambda row: _region(row, MODELS), axis=1).value_counts()


def _metrics(y: pd.Series, pred: pd.Series) -> dict:
    tp = int((y & pred).sum())
    fn = int((y & ~pred).sum())
    fp = int((~y & pred).sum())
    tn = int((~y & ~pred).sum())
    return {
        "n": int(len(y)),
        "n_synthetic": int(y.sum()),
        "n_human": int((~y).sum()),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "miss": fn / int(y.sum()) if int(y.sum()) else float("nan"),
        "fpr": fp / int((~y).sum()) if int((~y).sum()) else float("nan"),
    }


def _strategies(frame: pd.DataFrame) -> list[tuple[str, dict]]:
    y = frame["synthetic"]
    rows = []
    for name in MODELS:
        rows.append((NAMES[name], _metrics(y, frame[name])))
    for left, right in PAIRS:
        label = f"{NAMES[left]} + {NAMES[right]}"
        rows.append((f"OR {label}", _metrics(y, frame[left] | frame[right])))
        rows.append((f"AND {label}", _metrics(y, frame[left] & frame[right])))
    any_three = frame[list(MODELS)].any(axis=1)
    votes = frame[list(MODELS)].sum(axis=1) >= 2
    rows.append(("OR трёх", _metrics(y, any_three)))
    rows.append(("Голос 2 из 3", _metrics(y, votes)))
    return rows


def _print_day(title: str, source: pd.DataFrame) -> None:
    frame, dropped = _flags(source)
    print(f"\n=== {title} ===")
    print("бинарных", len(source), "с тремя оценками", len(frame), "без полной тройки", dropped)
    print("synthetic", int(frame["synthetic"].sum()), "human", int((~frame["synthetic"]).sum()))
    print("-- FN, кто пропустил TTS --")
    print(_errors(frame, "fn").to_string())
    print("-- FP, кто назвал живого synthetic --")
    print(_errors(frame, "fp").to_string())
    print("-- стратегии --")
    for name, metrics in _strategies(frame):
        print(
            f"{name:42} miss {metrics['miss']:.3f} ({metrics['fn']}/{metrics['n_synthetic']})"
            f"  fpr {metrics['fpr']:.3f} ({metrics['fp']}/{metrics['n_human']})"
            f"  tp {metrics['tp']} fp {metrics['fp']}"
        )


def main() -> None:
    _print_day("11-12", _frame_1112())
    _print_day("16", _frame_16())


if __name__ == "__main__":
    main()
