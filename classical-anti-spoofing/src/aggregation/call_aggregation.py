from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.schema import FLAG_COLUMNS, METADATA_COLUMNS


AGGREGATIONS = ("mean", "median", "std", "min", "max", "p25", "p75")


def feature_columns(frame: pd.DataFrame) -> list[str]:
    reserved = set(METADATA_COLUMNS) | set(FLAG_COLUMNS)
    return [column for column in frame.columns if column not in reserved and not str(column).startswith("flag_")]


def aggregate_calls(segments: pd.DataFrame) -> pd.DataFrame:
    if segments.empty:
        return pd.DataFrame()
    columns = feature_columns(segments)
    rows: list[dict] = []
    for call_id, group in segments.groupby("call_id", sort=False):
        usable = group[pd.to_numeric(group["usable_for_features"], errors="coerce").fillna(0).astype(int) == 1]
        source = usable if len(usable) else group.iloc[0:0]
        row = {
            "call_id": call_id,
            "label": group["label"].iloc[0],
            "locked_split": group["locked_split"].iloc[0],
            "n_segments": int(len(group)),
            "n_usable_segments": int(len(usable)),
            "speech_duration_s": float(pd.to_numeric(group["duration"], errors="coerce").sum()),
            "recording_duration_s": float(group["recording_duration_s"].iloc[0]),
            "speech_ratio": float(group["speech_ratio"].iloc[0]),
            "silence_ratio": float(group["silence_ratio"].iloc[0]),
            "snr_db": float(pd.to_numeric(group["snr_db"], errors="coerce").iloc[0]),
            "snr_method": group["snr_method"].iloc[0],
            "sample_rate_hz": int(float(group["sample_rate_hz"].iloc[0])),
            "classification_result": group["classification_result"].iloc[0] if "classification_result" in group else "",
            "campaign_name": group["campaign_name"].iloc[0] if "campaign_name" in group else "",
        }
        for flag in FLAG_COLUMNS:
            if flag in group.columns:
                row[f"{flag}_rate"] = float(pd.to_numeric(group[flag], errors="coerce").fillna(0).mean())
        for column in columns:
            values = pd.to_numeric(source[column], errors="coerce").to_numpy(dtype=float) if len(source) else np.array([])
            finite = values[np.isfinite(values)] if values.size else np.array([])
            stats = _seven(finite)
            for name, value in stats.items():
                row[f"{column}__{name}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _seven(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {name: float("nan") for name in AGGREGATIONS}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)) if values.size > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
    }
