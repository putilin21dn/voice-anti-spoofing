from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import resolve_path


META_COLUMNS = (
    "call_id",
    "label",
    "locked_split",
    "n_segments",
    "n_usable_segments",
    "speech_duration_s",
    "recording_duration_s",
    "speech_ratio",
    "silence_ratio",
    "snr_db",
    "snr_method",
    "sample_rate_hz",
    "classification_result",
    "campaign_name",
    "status",
    "processing_ms",
    "error",
)

EXCLUDED_FROM_MODEL = {
    "speech_duration_s",
    "recording_duration_s",
    "speech_ratio",
    "silence_ratio",
    "snr_db",
    "n_segments",
    "n_usable_segments",
    "sample_rate_hz",
    "processing_ms",
}


def acoustic_columns(columns: list[str]) -> list[str]:
    """Агрегированные акустические признаки. Длительность, SNR и флаги сюда не входят."""

    selected = []
    for name in columns:
        if "__" not in name:
            continue
        stem = name.split("__", 1)[0]
        if stem in EXCLUDED_FROM_MODEL or stem.startswith("flag_"):
            continue
        selected.append(name)
    return selected


def load_call_frame(cfg: dict) -> pd.DataFrame:
    path = resolve_path(cfg, cfg["paths"]["features_dir"]) / "calls.csv"
    header = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [name for name in META_COLUMNS if name in header] + acoustic_columns(header)
    frame = pd.read_csv(path, usecols=usecols, low_memory=False)
    frame["call_id"] = frame["call_id"].astype(str)
    splits = _split_table(cfg)
    merged = frame.merge(splits, on="call_id", how="left", suffixes=("", "_split"))
    if merged["call_id"].duplicated().any():
        raise RuntimeError("В таблице звонков повторяется call_id")
    if splits["call_id"].duplicated().any():
        raise RuntimeError("Один call_id попал в несколько split")
    return merged


def modeling_rows(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = frame[frame["split"] == split].copy()
    usable = pd.to_numeric(rows["n_usable_segments"], errors="coerce").fillna(0).astype(int)
    return rows[usable > 0].reset_index(drop=True)


def labels_machine(frame: pd.DataFrame) -> pd.Series:
    unknown = ~frame["label"].isin(["human", "machine"])
    if unknown.any():
        raise RuntimeError(f"В обучении есть метки вне human/machine: {int(unknown.sum())}")
    return (frame["label"] == "machine").astype(int)


def _split_table(cfg: dict) -> pd.DataFrame:
    directory = resolve_path(cfg, cfg["paths"]["splits_dir"])
    parts = []
    for name in ("train", "validation", "test", "spectra_threshold"):
        path = directory / f"{name}.csv"
        part = pd.read_csv(path, dtype=str)
        part["split"] = name
        parts.append(part[["call_id", "label", "split"]].rename(columns={"label": "label_split"}))
    return pd.concat(parts, ignore_index=True)


def assert_no_forbidden(path: Path) -> None:
    header = pd.read_csv(path, nrows=0).columns
    leaked = {"external_id", "s3_key", "s3_bucket", "phone", "msisdn"} & set(header)
    if leaked:
        raise RuntimeError(f"В таблице есть запрещённые колонки: {sorted(leaked)}")
