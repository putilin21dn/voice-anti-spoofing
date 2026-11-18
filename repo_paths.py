"""Resolve component directories inside this repository only."""

from __future__ import annotations

import sys
from pathlib import Path


def repository_root(start: Path) -> Path:
    """Return the voice-anti-spoofing root that contains ``start``."""

    start = start.resolve()
    chain = [start] if start.is_dir() else []
    chain.extend(start.parents)
    for parent in chain:
        marker = parent / "pyproject.toml"
        if not marker.is_file():
            continue
        if 'name = "voice-anti-spoofing"' not in marker.read_text(encoding="utf-8"):
            continue
        if (parent / "classical-anti-spoofing").is_dir() and (parent / "synthetic-speech-dataset").is_dir():
            return parent
    raise RuntimeError(
        "Каталог voice-anti-spoofing не найден. Модули из папок вне этого репозитория не загружаются."
    )


def component_dir(start: Path, name: str) -> Path:
    """Return ``name`` under the repository root. A path outside the root is refused."""

    root = repository_root(start)
    path = (root / name).resolve()
    if path != root and root not in path.parents:
        raise RuntimeError(f"{name} оказался вне репозитория: {path}")
    if not path.is_dir():
        raise RuntimeError(f"В репозитории нет каталога {name}")
    return path


def contained(root: Path, path: Path) -> Path:
    path = path.resolve()
    root = root.resolve()
    if path != root and root not in path.parents:
        raise RuntimeError(f"Путь выходит за репозиторий voice-anti-spoofing: {path}")
    return path


def component_python(start: Path, name: str) -> Path:
    """Prefer the component's own venv. Otherwise use the current interpreter, never a path outside the repo."""

    folder = component_dir(start, name)
    local = folder / ".venv" / "Scripts" / "python.exe"
    if local.is_file():
        return contained(repository_root(start), local)
    return Path(sys.executable)
