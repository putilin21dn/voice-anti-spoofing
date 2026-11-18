from __future__ import annotations

import io
import wave
from pathlib import Path


def probe_wav(path: Path) -> dict[str, str] | None:
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            frames = handle.getnframes()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            compression = handle.getcomptype()
    except (wave.Error, EOFError, OSError):
        return None
    if rate <= 0 or channels <= 0 or frames < 0 or width <= 0:
        return None
    codec = "pcm_s16le" if compression == "NONE" and width == 2 else ""
    return {
        "sample_rate": str(rate),
        "channel_count": str(channels),
        "duration_seconds": f"{frames / rate:.3f}",
        "codec": codec,
        "codec_source": "wav_header" if codec else "",
    }


def wav_bytes(path: Path, channel: int | None) -> bytes:
    """Возвращает WAV в памяти. Исходный файл не изменяется."""

    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        width = source.getsampwidth()
        rate = source.getframerate()
        frames = source.readframes(source.getnframes())
    if channel is None:
        selected = frames
        out_channels = channels
    else:
        if channel < 0 or channel >= channels:
            raise ValueError(f"В записи нет канала {channel}")
        if channels == 1:
            selected = frames
        else:
            frame_size = width * channels
            pieces = bytearray()
            offset = channel * width
            for position in range(0, len(frames) - frame_size + 1, frame_size):
                pieces.extend(frames[position + offset : position + offset + width])
            selected = bytes(pieces)
        out_channels = 1
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(out_channels)
        output.setsampwidth(width)
        output.setframerate(rate)
        output.writeframes(selected)
    return buffer.getvalue()
