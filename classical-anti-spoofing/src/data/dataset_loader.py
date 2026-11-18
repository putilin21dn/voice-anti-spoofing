from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.schema import FORBIDDEN_COLUMNS
from src.utils.config import resolve_path


SAFE_MANIFEST_COLUMNS = [
    "call_id",
    "timestamp",
    "outcome",
    "classification_result",
    "campaign_name",
]


def _read_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if columns is None:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    else:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, usecols=columns)
    overlap = FORBIDDEN_COLUMNS & set(frame.columns)
    if overlap:
        raise RuntimeError(f"В таблицу попали запрещённые колонки: {sorted(overlap)}")
    return frame


def _ids(path: Path) -> set[str]:
    frame = _read_csv(path, ["call_id"])
    return set(frame["call_id"].astype(str))


def _path_map(path: Path) -> dict[str, str]:
    frame = _read_csv(path)
    if "local_path" not in frame.columns:
        return {}
    return {str(row.call_id): str(row.local_path) for row in frame.itertuples(index=False) if row.call_id}


def load_dataset(cfg: dict) -> pd.DataFrame:
    samples_path = resolve_path(cfg, cfg["paths"]["samples_csv"])
    samples = _read_csv(samples_path)
    required = {"call_id", "sample", "local_path", "wav_ok", "client_channel", "source_channels", "sample_rate", "duration_ms"}
    missing = required - set(samples.columns)
    if missing:
        raise RuntimeError(f"В samples.csv нет колонок: {sorted(missing)}")
    samples = samples.drop_duplicates("call_id", keep="first").copy()

    meta_frames = []
    for key in ("human_manifest", "machine_manifest"):
        path = resolve_path(cfg, cfg["paths"][key])
        meta_frames.append(_read_csv(path, SAFE_MANIFEST_COLUMNS))
    meta = pd.concat(meta_frames, ignore_index=True).drop_duplicates("call_id", keep="first")
    meta = meta.drop(columns=[column for column in meta.columns if column != "call_id" and column in samples.columns], errors="ignore")

    frame = samples.merge(meta, on="call_id", how="left")
    test_ids = _ids(resolve_path(cfg, cfg["paths"]["test_csv"]))
    threshold_ids = _ids(resolve_path(cfg, cfg["paths"]["n100_csv"]))
    overlap = test_ids & threshold_ids
    if overlap:
        raise RuntimeError(f"Один call_id попал и в test, и в n100: {len(overlap)}")
    extra_paths = {}
    extra_paths.update(_path_map(resolve_path(cfg, cfg["paths"]["n100_csv"])))
    extra_paths.update(_path_map(resolve_path(cfg, cfg["paths"]["test_csv"])))

    def split_of(call_id: str) -> str:
        if call_id in test_ids:
            return "test"
        if call_id in threshold_ids:
            return "spectra_threshold"
        return "pool"

    def resolve_file(row: pd.Series) -> str:
        primary = row["local_path"]
        if primary and Path(primary).is_file():
            return primary
        alternate = extra_paths.get(str(row["call_id"]), "")
        if alternate and Path(alternate).is_file() and split_of(str(row["call_id"])) in {"test", "spectra_threshold"}:
            return alternate
        return primary

    frame["locked_split"] = frame["call_id"].map(split_of)
    frame["resolved_path"] = frame.apply(resolve_file, axis=1)
    frame["file_exists"] = frame["resolved_path"].map(lambda value: bool(value) and Path(value).is_file())
    frame["label"] = frame["sample"].where(frame["sample"].isin(["human", "machine"]), "unknown")
    frame["folder_name"] = frame["local_path"].map(lambda value: Path(value).parent.name if value else "")
    frame["label_folder_mismatch"] = (frame["folder_name"].isin(["human", "machine"])) & (frame["folder_name"] != frame["label"])
    leaked = FORBIDDEN_COLUMNS & set(frame.columns)
    if leaked:
        raise RuntimeError(f"Запрещённые колонки после сборки: {sorted(leaked)}")
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].fillna("")
    return frame.sort_values(["timestamp", "call_id"], kind="mergesort").reset_index(drop=True)
