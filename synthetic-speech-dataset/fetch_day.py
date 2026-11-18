"""Запускает скачивание дня так, чтобы src коллектора не пересекался с src этого репозитория."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _component(name: str) -> Path:
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
            return module.component_dir(here, name)
    raise RuntimeError("Код вне репозитория voice-anti-spoofing не используется")


COLLECTOR = _component("voice-dataset-collector")
sys.path = [item for item in sys.path if Path(item or ".").resolve() != ROOT]
sys.path.insert(0, str(COLLECTOR))

spec = importlib.util.spec_from_file_location("day_audio_main", ROOT / "src" / "collect" / "day_audio.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.main()
