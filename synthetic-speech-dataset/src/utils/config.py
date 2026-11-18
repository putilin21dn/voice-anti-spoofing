from __future__ import annotations

from pathlib import Path

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = project_root() / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("config.yaml должен быть словарём")
    loaded["_root"] = str(project_root())
    loaded["_config_path"] = str(config_path)
    return loaded


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


def resolve_path(cfg: dict, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    layout = _repository_layout()
    root = layout.repository_root(Path(cfg["_root"]))
    return layout.contained(root, Path(cfg["_root"]) / path)
