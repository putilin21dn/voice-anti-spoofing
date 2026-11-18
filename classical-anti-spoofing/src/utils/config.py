from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    root = project_root()
    config_path = Path(path) if path else root / "config" / "config.yaml"
    if not config_path.is_absolute():
        config_path = (root / config_path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_root"] = root
    cfg["_config_path"] = config_path
    return cfg


def _repository_layout():
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
            return module
    raise RuntimeError("Код вне репозитория voice-anti-spoofing не используется")


def resolve_path(cfg: dict[str, Any], value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    layout = _repository_layout()
    root = layout.repository_root(Path(cfg["_root"]))
    return layout.contained(root, Path(cfg["_root"]) / path)


def feature_fingerprint(cfg: dict[str, Any]) -> str:
    payload = {
        "audio": cfg["audio"],
        "vad": cfg["vad"],
        "quality": cfg["quality"],
        "features": cfg["features"],
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
