from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from src.features.schema import FLAG_COLUMNS, FORBIDDEN_COLUMNS, METADATA_COLUMNS
from src.utils.config import feature_fingerprint


class FeatureStore:
    def __init__(self, directory: Path, cfg: dict):
        self.directory = directory
        self.cfg = cfg
        self.segments_path = directory / "segments.csv"
        self.calls_path = directory / "calls.csv"
        self.status_path = directory / "call_status.csv"
        self.errors_path = directory / "errors.csv"
        self.done_path = directory / "completed_call_ids.txt"
        self.fingerprint_path = directory / "feature_fingerprint.txt"
        directory.mkdir(parents=True, exist_ok=True)
        self._guard_fingerprint()

    def completed(self) -> set[str]:
        self._drop_incomplete_segments()
        if not self.done_path.exists():
            return set()
        return {line.strip() for line in self.done_path.read_text(encoding="utf-8").splitlines() if line.strip()}

    def append_segments(self, rows: list[dict]) -> None:
        if not rows:
            return
        leaked = FORBIDDEN_COLUMNS & set(rows[0])
        if leaked:
            raise RuntimeError(f"Признаки содержат запрещённые поля: {sorted(leaked)}")
        frame = pd.DataFrame(rows)
        write_header = not self.segments_path.exists()
        frame.to_csv(self.segments_path, mode="a", header=write_header, index=False, encoding="utf-8")

    def mark_completed(self, call_id: str) -> None:
        with self.done_path.open("a", encoding="utf-8") as handle:
            handle.write(call_id + "\n")

    def append_status(self, row: dict) -> None:
        frame = pd.DataFrame([row])
        write_header = not self.status_path.exists()
        frame.to_csv(self.status_path, mode="a", header=write_header, index=False, encoding="utf-8")

    def append_error(self, row: dict) -> None:
        frame = pd.DataFrame([row])
        write_header = not self.errors_path.exists()
        frame.to_csv(self.errors_path, mode="a", header=write_header, index=False, encoding="utf-8")

    def write_calls(self, frame: pd.DataFrame) -> None:
        frame.to_csv(self.calls_path, index=False, encoding="utf-8")

    def save_config_copy(self) -> None:
        snapshot = {key: value for key, value in self.cfg.items() if not str(key).startswith("_")}
        with (self.directory / "config_snapshot.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(snapshot, handle, allow_unicode=True, sort_keys=False)

    def _guard_fingerprint(self) -> None:
        current = feature_fingerprint(self.cfg)
        if self.fingerprint_path.exists():
            saved = self.fingerprint_path.read_text(encoding="utf-8").strip()
            if saved != current:
                raise RuntimeError(
                    "Конфигурация признаков изменилась. Укажите другой каталог вывода или удалите незавершённый расчёт вручную."
                )
        else:
            self.fingerprint_path.write_text(current + "\n", encoding="utf-8")
            self.save_config_copy()
            columns = {"metadata": list(METADATA_COLUMNS), "flags": list(FLAG_COLUMNS)}
            (self.directory / "column_groups.json").write_text(
                json.dumps(columns, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    def _drop_incomplete_segments(self) -> None:
        if not self.segments_path.exists():
            return
        done = set()
        if self.done_path.exists():
            done = {line.strip() for line in self.done_path.read_text(encoding="utf-8").splitlines() if line.strip()}
        frame = pd.read_csv(self.segments_path, dtype={"call_id": str})
        kept = frame[frame["call_id"].isin(done)]
        if len(kept) != len(frame):
            kept.to_csv(self.segments_path, index=False, encoding="utf-8")
