from __future__ import annotations

import json
from pathlib import Path


def next_numbered_dir(parent: Path, prefix: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    number = 1
    while (parent / f"{prefix}_{number:03d}").exists():
        number += 1
    path = parent / f"{prefix}_{number:03d}"
    path.mkdir()
    return path


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value):
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Не сериализуется: {type(value)}")
