from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if data.size == 0:
        raise ValueError("Пустой WAV")
    return data, int(sample_rate)


def select_client_channel(
    data: np.ndarray,
    *,
    manifest_channel: str | int | None,
    config_channel: int,
    channel_override: int | None,
) -> tuple[np.ndarray, int, str]:
    """Берёт один канал. Каналы не смешиваются в mono."""

    channels = int(data.shape[1])
    if channels == 1:
        return np.ascontiguousarray(data[:, 0], dtype=np.float64), 0, "mono_client_extract"
    if channel_override is not None:
        index = int(channel_override)
        reason = "channel_override"
    elif manifest_channel not in (None, ""):
        index = int(manifest_channel)
        reason = "manifest_client_channel"
    else:
        index = int(config_channel)
        reason = "config_client_channel"
    if index < 0 or index >= channels:
        raise ValueError(f"Канал {index} отсутствует, в файле каналов: {channels}")
    return np.ascontiguousarray(data[:, index], dtype=np.float64), index, reason
