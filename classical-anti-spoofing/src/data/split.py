from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SPLIT_COLUMNS = ["call_id", "label", "locked_split", "timestamp", "classification_result", "campaign_name"]


def select_even(frame: pd.DataFrame, per_class: int) -> pd.DataFrame:
    chosen: list[pd.DataFrame] = []
    usable = frame[frame["file_exists"] & frame["label"].isin(["human", "machine"])].copy()
    for label in ("human", "machine"):
        group = usable[usable["label"] == label].sort_values(["timestamp", "call_id"], kind="mergesort")
        if group.empty or per_class < 1:
            continue
        if len(group) <= per_class:
            chosen.append(group)
            continue
        indexes = np.unique(np.linspace(0, len(group) - 1, per_class).astype(int))
        chosen.append(group.iloc[indexes])
    if not chosen:
        return usable.iloc[0:0]
    return pd.concat(chosen, ignore_index=True)


def create_call_splits(frame: pd.DataFrame, *, val_fraction: float, seed: int) -> dict[str, pd.DataFrame]:
    usable = frame[frame["file_exists"] & frame["label"].isin(["human", "machine"])].copy()
    unknown = frame[(frame["label"] == "unknown") & frame["file_exists"]].copy()
    unreadable = frame[~frame["file_exists"]].copy()
    test = usable[usable["locked_split"] == "test"]
    threshold = usable[usable["locked_split"] == "spectra_threshold"]
    pool = usable[usable["locked_split"] == "pool"]
    if pool["label"].nunique() < 2:
        raise RuntimeError("В пуле обучения нужны оба класса")
    train, validation = train_test_split(
        pool,
        test_size=val_fraction,
        random_state=seed,
        stratify=pool["label"],
    )
    parts = {
        "train": train,
        "validation": validation,
        "test": test,
        "spectra_threshold": threshold,
        "unknown": unknown,
        "unreadable": unreadable,
    }
    _assert_disjoint(parts)
    return parts


def write_splits(directory: Path, parts: dict[str, pd.DataFrame], *, seed: int, val_fraction: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, frame in parts.items():
        columns = [column for column in SPLIT_COLUMNS if column in frame.columns]
        frame.loc[:, columns].to_csv(directory / f"{name}.csv", index=False, encoding="utf-8")
        counts[name] = {
            "rows": int(len(frame)),
            "by_label": {str(label): int(count) for label, count in frame["label"].value_counts().items()} if "label" in frame.columns else {},
        }
    note = {
        "seed": seed,
        "val_fraction": val_fraction,
        "unit": "call_id",
        "counts": counts,
        "speaker_split": "невозможен: идентификатора говорящего в разметке нет",
        "generator_split": "невозможен: поле TTS-генератора в разметке нет",
        "test_rule": "test — это зафиксированный test20. Он не участвует в обучении и выборе порога.",
        "spectra_threshold_rule": "n100 использовался для порога Spectra-0 1.02. Классические модели на нём не обучаются.",
    }
    (directory / "split_metadata.json").write_text(json.dumps(note, ensure_ascii=False, indent=2), encoding="utf-8")


def _assert_disjoint(parts: dict[str, pd.DataFrame]) -> None:
    seen: dict[str, str] = {}
    for name, frame in parts.items():
        if name == "unreadable":
            continue
        for call_id in frame["call_id"].astype(str):
            previous = seen.get(call_id)
            if previous is not None:
                raise RuntimeError(f"call_id {call_id} попал в {previous} и {name}")
            seen[call_id] = name
