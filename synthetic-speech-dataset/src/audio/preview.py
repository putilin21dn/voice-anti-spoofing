from __future__ import annotations

from pathlib import Path

from src.audio.loader import wav_bytes
from src.utils.config import resolve_path


def resolve_source(cfg: dict, source_file: str) -> Path:
    if not source_file or source_file.startswith(("/", "\\")) or ":" in source_file:
        raise PermissionError("Путь к аудио должен быть относительным")
    collector = resolve_path(cfg, cfg["paths"]["collector_root"])
    audio_root = resolve_path(cfg, cfg["paths"]["audio_root"])
    candidate = (collector / source_file).resolve()
    if candidate != audio_root and audio_root not in candidate.parents:
        raise PermissionError("Аудио вне разрешённого каталога")
    if candidate.suffix.lower() != ".wav" or not candidate.is_file():
        raise FileNotFoundError("WAV не найден")
    return candidate


def preview_bytes(cfg: dict, source_file: str, channel: str) -> bytes:
    path = resolve_source(cfg, source_file)
    if channel in {"mix", "all"}:
        return wav_bytes(path, None)
    if channel not in {"0", "1", "left", "right"}:
        raise ValueError("Канал: mix, 0 или 1")
    index = 0 if channel in {"0", "left"} else 1
    return wav_bytes(path, index)
