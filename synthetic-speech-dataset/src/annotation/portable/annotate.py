"""Локальная разметка одной части. Зависимостей нет, наружу ничего не отправляет."""

from __future__ import annotations

import csv
import io
import json
import math
import re
import wave
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from uuid import UUID


ROOT = Path(__file__).resolve().parent
PAGE = ROOT / "index.html"
PRIORITY_PAGE = ROOT / "priority.html"
MANIFEST = ROOT / "manifest.csv"
CALLS = ROOT / "annotations.csv"
SEGMENTS = ROOT / "segments.csv"
AUDIO = ROOT / "audio"
CALL_ID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
ANNOTATOR_RE = re.compile(r"^[\w.\-]{1,32}$")
PRIMARY = {
    "human_live",
    "human_recorded",
    "synthetic_tts",
    "synthetic_voice_conversion",
    "synthetic_playback",
    "mixed",
    "unknown",
}
HUMAN = {"human_live", "human_recorded"}
SYNTHETIC = {"synthetic_tts", "synthetic_voice_conversion", "synthetic_playback"}
SECONDARY = {"", "ai_secretary", "voicemail", "anti_spam", "greeting_only"}
CONFIDENCE = {"", "high", "medium", "low"}
SOURCES = {"manual_listening", "verified_metadata", "known_generator", "unknown_source"}
STATUS = {"unreviewed", "in_progress", "reviewed", "needs_second_review", "disputed"}
CALL_FIELDS = [
    "call_id",
    "annotator_id",
    "primary_label",
    "secondary_label",
    "label_confidence",
    "label_source",
    "tts_generator",
    "voice_conversion_system",
    "speaker_id",
    "playback_method",
    "annotation_notes",
    "review_status",
    "updated_at",
]
SEGMENT_FIELDS = [
    "call_id",
    "segment_id",
    "start_time",
    "end_time",
    "speaker_id",
    "segment_label",
    "label_confidence",
    "label_source",
    "notes",
    "annotator_id",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [{key: value or "" for key, value in row.items() if key} for row in csv.DictReader(handle)]


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    temporary.replace(path)


def text(value: object, limit: int) -> str:
    raw = "" if value is None else str(value).strip()
    if "\n" in raw or "\r" in raw:
        raise ValueError("Поле содержит перевод строки")
    if len(raw) > limit:
        raise ValueError(f"Поле длиннее {limit} символов")
    return raw


def check_pair(primary: str, secondary: str) -> str | None:
    if secondary == "ai_secretary" and primary in HUMAN:
        return "AI-секретарь — синтетический голос. Для человека уберите ai_secretary."
    if secondary == "voicemail" and primary == "human_live":
        return "Записанная почта — human_recorded, а не human_live."
    return None


def validate(payload: dict, duration_s: float) -> dict:
    primary = text(payload.get("primary_label"), 64)
    secondary = text(payload.get("secondary_label"), 64)
    confidence = text(payload.get("label_confidence"), 16)
    source = text(payload.get("label_source"), 64)
    status = text(payload.get("review_status"), 32)
    annotator = text(payload.get("annotator_id"), 32)
    notes = "" if payload.get("annotation_notes") is None else str(payload.get("annotation_notes")).strip()
    if primary not in PRIMARY:
        raise ValueError("Неизвестная основная метка")
    if secondary not in SECONDARY or confidence not in CONFIDENCE or source not in SOURCES or status not in STATUS:
        raise ValueError("Дополнительная метка, уверенность, источник или статус не подходят")
    if not ANNOTATOR_RE.fullmatch(annotator):
        raise ValueError("Впишите имя разметчика: до 32 букв, цифр, точки, _ или -")
    if len(notes) > 4000:
        raise ValueError("Заметка длиннее 4000 символов")
    pair = check_pair(primary, secondary)
    if pair:
        raise ValueError(pair)
    if primary == "unknown":
        if source != "unknown_source" or confidence:
            raise ValueError("Для unknown источник unknown_source и пустая уверенность")
    elif source == "unknown_source" or confidence not in {"high", "medium", "low"}:
        raise ValueError("Подтверждённая метка требует источник и уверенность")
    tts = text(payload.get("tts_generator"), 200)
    conversion = text(payload.get("voice_conversion_system"), 200)
    speaker = text(payload.get("speaker_id"), 200)
    playback = text(payload.get("playback_method"), 200)
    if source == "known_generator" and not any((tts, conversion, speaker, playback)):
        raise ValueError("known_generator требует известный генератор, систему, говорящего или способ воспроизведения")
    if source == "known_generator" and primary == "synthetic_tts" and not tts:
        raise ValueError("known_generator для TTS требует имя генератора")
    if source == "known_generator" and primary == "synthetic_voice_conversion" and not conversion:
        raise ValueError("known_generator для voice conversion требует имя системы")
    if source == "known_generator" and primary == "synthetic_playback" and not playback:
        raise ValueError("known_generator для воспроизведения требует способ playback")
    if source == "known_generator" and primary in HUMAN and not speaker:
        raise ValueError("known_generator для человеческой речи требует speaker_id")
    raw_segments = payload.get("segments") or []
    if not isinstance(raw_segments, list) or len(raw_segments) > 50:
        raise ValueError("Сегменты должны быть списком не длиннее 50")
    segments = []
    families: set[str] = set()
    for index, raw in enumerate(raw_segments, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Сегмент {index} повреждён")
        try:
            start = float(raw.get("start_time"))
            end = float(raw.get("end_time"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Сегмент {index}: время должно быть числом") from exc
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start or end > duration_s + 0.05:
            raise ValueError(f"Сегмент {index} выходит за запись")
        label = text(raw.get("segment_label"), 64)
        seg_confidence = text(raw.get("label_confidence"), 16)
        seg_source = text(raw.get("label_source"), 64)
        if label not in PRIMARY or seg_confidence not in CONFIDENCE or seg_source not in SOURCES:
            raise ValueError(f"Сегмент {index}: метка, уверенность или источник не подходят")
        if label == "unknown" and seg_source != "unknown_source":
            raise ValueError(f"Сегмент {index}: для unknown источник unknown_source")
        if label != "unknown" and seg_source == "unknown_source":
            raise ValueError(f"Сегмент {index}: подтверждённая метка требует источник")
        if label in HUMAN:
            families.add("human")
        elif label in SYNTHETIC:
            families.add("synthetic")
        notes = "" if raw.get("notes") is None else str(raw.get("notes")).strip()
        if len(notes) > 1000:
            raise ValueError(f"Сегмент {index}: заметка длиннее 1000 символов")
        segment_id = text(raw.get("segment_id"), 64)
        if segment_id:
            UUID(segment_id)
        segments.append(
            {
                "segment_id": segment_id,
                "start_time": f"{start:.3f}",
                "end_time": f"{end:.3f}",
                "speaker_id": text(raw.get("speaker_id"), 200),
                "segment_label": label,
                "label_confidence": text(raw.get("label_confidence"), 16),
                "label_source": text(raw.get("label_source"), 64),
                "notes": notes,
            }
        )
    if families == {"human", "synthetic"} and primary not in {"mixed", "unknown"}:
        raise ValueError("В интервалах есть и человек, и синтетика. Звонку нужна метка mixed или unknown.")
    return {
        "primary_label": primary,
        "secondary_label": secondary,
        "label_confidence": confidence,
        "label_source": source,
        "tts_generator": tts,
        "voice_conversion_system": conversion,
        "speaker_id": speaker,
        "playback_method": playback,
        "annotation_notes": notes,
        "review_status": status,
        "annotator_id": annotator,
        "segments": segments,
    }


def wav_bytes(path: Path, channel: int | None) -> bytes:
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
            raise ValueError("Такого канала нет")
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


def blank(fact: dict[str, str], annotator: str) -> dict:
    row = {field: "" for field in CALL_FIELDS}
    row.update(fact)
    row["annotator_id"] = annotator
    row["primary_label"] = "unknown"
    row["label_source"] = "unknown_source"
    row["review_status"] = "unreviewed"
    row["segments"] = []
    return row


class App:
    def __init__(self) -> None:
        self.facts = {row["call_id"]: row for row in read_csv(MANIFEST) if CALL_ID_RE.fullmatch(row.get("call_id", ""))}

    def view(self, annotator: str) -> list[dict]:
        labels = {(row.get("call_id", ""), row.get("annotator_id", "")): row for row in read_csv(CALLS)}
        grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
        for row in read_csv(SEGMENTS):
            grouped.setdefault((row.get("call_id", ""), row.get("annotator_id", "")), []).append(row)
        items = []
        for call_id, fact in self.facts.items():
            chosen = labels.get((call_id, annotator)) or blank(fact, annotator)
            item = blank(fact, annotator)
            for field in CALL_FIELDS:
                if field in chosen:
                    item[field] = chosen.get(field, "")
            item["annotator_id"] = annotator
            item["duration_seconds"] = fact.get("duration_seconds", "")
            item["channel_count"] = fact.get("channel_count", "")
            item["client_channel"] = fact.get("client_channel", "0")
            item["campaign_name"] = fact.get("campaign_name", "")
            item["segments"] = grouped.get((call_id, annotator), [])
            items.append(item)
        return items

    def save(self, call_id: str, payload: dict) -> dict:
        fact = self.facts.get(call_id)
        if fact is None:
            raise ValueError("Этого звонка нет в вашей части")
        clean = validate(payload, float(fact.get("duration_seconds") or 0))
        annotator = clean["annotator_id"]
        row = {field: "" for field in CALL_FIELDS}
        row["call_id"] = call_id
        for key, value in clean.items():
            if key != "segments":
                row[key] = value
        row["updated_at"] = datetime.now(timezone.utc).isoformat()
        calls = [item for item in read_csv(CALLS) if not (item.get("call_id") == call_id and item.get("annotator_id") == annotator)]
        calls.append(row)
        calls.sort(key=lambda item: (item.get("call_id", ""), item.get("annotator_id", "")))
        write_csv(CALLS, CALL_FIELDS, calls)
        segments = [
            item
            for item in read_csv(SEGMENTS)
            if not (item.get("call_id") == call_id and item.get("annotator_id") == annotator)
        ]
        for segment in clean["segments"]:
            segments.append({"call_id": call_id, "annotator_id": annotator, **segment})
        segments.sort(key=lambda item: (item.get("call_id", ""), item.get("start_time", "")))
        write_csv(SEGMENTS, SEGMENT_FIELDS, segments)
        for item in self.view(annotator):
            if item["call_id"] == call_id:
                return item
        raise RuntimeError("Сохранённая запись не прочиталась")


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args) -> None:
            return

        def _send(self, status: int, body: bytes, content_type: str, extra: dict[str, str] | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: object) -> None:
            self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                page = PRIORITY_PAGE if _priority_pack() else PAGE
                self._send(200, page.read_bytes(), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/queue" and _priority_pack():
                self._queue(parse_qs(parsed.query))
                return
            if parsed.path == "/api/calls":
                annotator = (parse_qs(parsed.query).get("annotator") or [""])[0]
                rows = app.view(annotator)
                status = (parse_qs(parsed.query).get("review_status") or [""])[0]
                if status:
                    rows = [row for row in rows if row.get("review_status") == status]
                self._json(200, rows)
                return
            if parsed.path.startswith("/api/calls/"):
                call_id = parsed.path.removeprefix("/api/calls/")
                annotator = (parse_qs(parsed.query).get("annotator") or [""])[0]
                for row in app.view(annotator):
                    if row["call_id"] == call_id:
                        self._json(200, row)
                        return
                self._json(404, {"error": "Звонок не найден"})
                return
            if parsed.path.startswith("/api/audio/"):
                self._audio(parsed.path.removeprefix("/api/audio/"), parse_qs(parsed.query))
                return
            self._json(404, {"error": "Не найдено"})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            prefix = "/api/calls/"
            if not parsed.path.startswith(prefix):
                self._json(404, {"error": "Не найдено"})
                return
            call_id = parsed.path[len(prefix) :]
            length = int(self.headers.get("Content-Length") or "0")
            if length <= 0 or length > 1_000_000:
                self._json(400, {"error": "Пустое или слишком большое тело запроса"})
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                saved = app.save(call_id, payload)
            except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
                self._json(400, {"error": str(exc) or "Запрос не разобран"})
                return
            self._json(200, saved)

        def _queue(self, query: dict[str, list[str]]) -> None:
            annotator = (query.get("annotator") or [""])[0]
            if annotator and not ANNOTATOR_RE.fullmatch(annotator):
                self._json(400, {"error": "Некорректное имя разметчика"})
                return
            labels = {
                row.get("call_id", ""): row
                for row in read_csv(CALLS)
                if row.get("annotator_id", "") == annotator
            }
            items = []
            for call_id, fact in app.facts.items():
                label = labels.get(call_id, {})
                items.append(
                    {
                        "sort_index": fact.get("sort_index", ""),
                        "call_id": call_id,
                        "date": fact.get("date", ""),
                        "campaign": fact.get("campaign_name", ""),
                        "review_priority": fact.get("review_priority", ""),
                        "queues": fact.get("queues", ""),
                        "public_reasons": fact.get("public_reasons", ""),
                        "duration_seconds": fact.get("duration_seconds", ""),
                        "client_channel": fact.get("client_channel", "0"),
                        "speech_duration_s": fact.get("speech_duration_s", ""),
                        "n_segments": fact.get("n_segments", ""),
                        "n_windows": fact.get("n_windows", ""),
                        "padding_ratio": fact.get("padding_ratio", ""),
                        "spectra_status": fact.get("spectra_status", ""),
                        "classical_status": fact.get("classical_status", ""),
                        "review_status": label.get("review_status") or "unreviewed",
                        "primary_label": label.get("primary_label", ""),
                        "secondary_label": label.get("secondary_label", ""),
                        "label_confidence": label.get("label_confidence", ""),
                        "annotation_notes": label.get("annotation_notes", ""),
                        "updated_at": label.get("updated_at", ""),
                    }
                )
            items.sort(key=lambda row: int(row["sort_index"] or 0))
            self._json(200, {"calls": items, "total": len(items)})

        def _audio(self, call_id: str, query: dict[str, list[str]]) -> None:
            if not CALL_ID_RE.fullmatch(call_id) or call_id not in app.facts:
                self._json(404, {"error": "Звонок не найден"})
                return
            channel = (query.get("channel") or ["mix"])[0]
            index = None
            if channel in {"0", "1", "left", "right"}:
                index = 0 if channel in {"0", "left"} else 1
            elif channel not in {"mix", "all"}:
                self._json(400, {"error": "Канал: mix, 0 или 1"})
                return
            path = AUDIO / f"{call_id}.wav"
            try:
                payload = wav_bytes(path, index)
            except (OSError, wave.Error, ValueError):
                self._json(404, {"error": "Запись не открылась"})
                return
            start = 0
            end = len(payload) - 1
            status = 200
            range_header = self.headers.get("Range")
            if range_header and range_header.startswith("bytes="):
                piece = range_header.removeprefix("bytes=").split(",", 1)[0]
                left, _, right = piece.partition("-")
                if left:
                    start = int(left)
                if right:
                    end = int(right)
                end = min(end, len(payload) - 1)
                status = 206
            chunk = payload[start : end + 1] if payload else b""
            extra = {"Accept-Ranges": "bytes"}
            if status == 206:
                extra["Content-Range"] = f"bytes {start}-{end}/{len(payload)}"
            self._send(status, chunk, "audio/wav", extra)

    return Handler


def _priority_pack() -> bool:
    if not MANIFEST.is_file() or not PRIORITY_PAGE.is_file():
        return False
    with MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        header = handle.readline()
    return "review_priority" in header


def main() -> None:
    if not MANIFEST.is_file() or not (PRIORITY_PAGE.is_file() if _priority_pack() else PAGE.is_file()):
        raise SystemExit("Запускайте annotate.py из распакованной папки архива")
    port = 8771 if _priority_pack() else 8768
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(App()))
    print(f"Разметка вашей части: http://127.0.0.1:{port}")
    print("Слушает только этот компьютер. Остановка: Ctrl+C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Остановлено")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
