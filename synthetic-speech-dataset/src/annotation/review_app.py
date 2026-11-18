from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import csv

from src.annotation.schema import valid_annotator, valid_call_id
from src.annotation.share import assignment_path, load_assignment
from src.annotation.storage import AnnotationStore
from src.audio.preview import preview_bytes
from src.utils.config import resolve_path


LOGGER = logging.getLogger("synthetic-speech-dataset")
PAGE = Path(__file__).resolve().parent / "static" / "index.html"
PRIORITY_PAGE = Path(__file__).resolve().parent / "static" / "priority.html"
HIDDEN_BEFORE_LABEL = {
    "model_scores",
    "prior_classification",
    "prior_sample",
    "outcome",
    "goal_status",
    "room_suffix",
    "selection_reason",
    "training_role",
    "locked_split",
    "reason_codes",
}


class ReviewApp:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.store = AnnotationStore(
            resolve_path(cfg, cfg["paths"]["manifest_out"]),
            resolve_path(cfg, cfg["paths"]["calls_out"]),
            resolve_path(cfg, cfg["paths"]["segments_out"]),
        )


def handler_factory(app: ReviewApp):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args) -> None:
            LOGGER.info("%s %s", self.command, urlparse(self.path).path)

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: object) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def _error(self, status: int, message: str) -> None:
            self._json(status, {"error": message})

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                page = PRIORITY_PAGE if app.cfg.get("blind_priority") else PAGE
                self._send(200, page.read_bytes(), "text/html; charset=utf-8")
                return
            if app.cfg.get("blind_priority") and parsed.path == "/api/queue":
                self._queue(parse_qs(parsed.query))
                return
            if app.cfg.get("blind_priority") and parsed.path.startswith("/api/predictions/"):
                self._predictions(parsed.path.removeprefix("/api/predictions/"), parse_qs(parsed.query))
                return
            if parsed.path == "/api/health":
                self._json(200, {"ok": True, "calls": len(app.store.manifest())})
                return
            if parsed.path == "/api/shares":
                assigned = _assignment(app.cfg)
                self._json(200, {"shares": sorted(set(assigned.values()))})
                return
            if parsed.path == "/api/calls":
                self._calls(parse_qs(parsed.query))
                return
            if parsed.path.startswith("/api/calls/"):
                self._one(parsed.path.removeprefix("/api/calls/"), parse_qs(parsed.query))
                return
            if parsed.path.startswith("/api/audio/"):
                self._audio(parsed.path.removeprefix("/api/audio/"), parse_qs(parsed.query))
                return
            self._error(404, "Не найдено")

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            prefix = "/api/calls/"
            if not parsed.path.startswith(prefix):
                self._error(404, "Не найдено")
                return
            call_id = parsed.path[len(prefix) :]
            if not valid_call_id(call_id):
                self._error(400, "Некорректный call_id")
                return
            length = int(self.headers.get("Content-Length") or "0")
            if length <= 0 or length > 1_000_000:
                self._error(400, "Пустое или слишком большое тело запроса")
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError):
                self._error(400, "Тело запроса не JSON")
                return
            if not isinstance(payload, dict):
                self._error(400, "Ожидался объект JSON")
                return
            try:
                saved = app.store.save(call_id, payload)
            except ValueError as exc:
                self._error(400, str(exc))
                return
            if app.cfg.get("blind_priority"):
                saved = _without_model_hints(saved)
            self._json(200, saved)

        def _queue(self, query: dict[str, list[str]]) -> None:
            try:
                annotator = self._annotator(query)
            except ValueError as exc:
                self._error(400, str(exc))
                return
            try:
                queued = _read_queue(app.cfg)
            except ValueError as exc:
                self._error(500, str(exc))
                return
            saved = {
                row.get("call_id", ""): row
                for row in app.store.calls()
                if row.get("annotator_id", "") == annotator
            }
            assigned = _assignment(app.cfg)
            if assigned:
                queued = [row for row in queued if assigned.get(row["call_id"]) == "local"]
            items = []
            for row in queued:
                label = saved.get(row["call_id"], {})
                items.append(
                    {
                        "sort_index": row.get("sort_index", ""),
                        "call_id": row["call_id"],
                        "date": row.get("date", ""),
                        "campaign": row.get("campaign", ""),
                        "review_priority": row.get("review_priority", ""),
                        "queues": row.get("queues", ""),
                        "public_reasons": row.get("public_reasons", ""),
                        "duration_seconds": row.get("duration_seconds", ""),
                        "client_channel": row.get("client_channel", ""),
                        "speech_duration_s": row.get("spectra_speech_duration_s", ""),
                        "n_segments": row.get("spectra_n_segments", ""),
                        "n_windows": row.get("spectra_n_windows", ""),
                        "padding_ratio": row.get("padding_ratio", ""),
                        "spectra_status": row.get("spectra_status", ""),
                        "classical_status": row.get("classical_status", ""),
                        "review_status": label.get("review_status") or "unreviewed",
                        "primary_label": label.get("primary_label", ""),
                        "secondary_label": label.get("secondary_label", ""),
                        "label_confidence": label.get("label_confidence", ""),
                        "annotation_notes": label.get("annotation_notes", ""),
                        "updated_at": label.get("updated_at", ""),
                    }
                )
            self._json(200, {"calls": items, "total": len(items), "share": "local" if assigned else ""})

        def _predictions(self, call_id: str, query: dict[str, list[str]]) -> None:
            if not valid_call_id(call_id):
                self._error(400, "Некорректный call_id")
                return
            try:
                annotator = self._annotator(query)
            except ValueError as exc:
                self._error(400, str(exc))
                return
            saved = next(
                (
                    row
                    for row in app.store.calls()
                    if row.get("call_id") == call_id and row.get("annotator_id", "") == annotator
                ),
                None,
            )
            if saved is None or saved.get("review_status") not in {"reviewed", "needs_second_review", "disputed"}:
                self._error(403, "Сначала сохраните независимую метку")
                return
            try:
                table = _read_predictions(app.cfg)
            except ValueError as exc:
                self._error(500, str(exc))
                return
            row = table.get(call_id)
            if row is None:
                self._error(404, "Для звонка нет таблицы предсказаний")
                return
            self._json(200, {"call_id": call_id, "models": _prediction_view(row)})

        def _annotator(self, query: dict[str, list[str]]) -> str:
            value = (query.get("annotator") or [""])[0]
            if not valid_annotator(value):
                raise ValueError("Некорректный код разметчика")
            return value

        def _calls(self, query: dict[str, list[str]]) -> None:
            try:
                annotator = self._annotator(query)
            except ValueError as exc:
                self._error(400, str(exc))
                return
            status = (query.get("review_status") or [""])[0]
            share = (query.get("share") or [""])[0]
            rows = app.store.view(annotator)
            assigned = _assignment(app.cfg)
            if share:
                rows = [row for row in rows if assigned.get(row["call_id"]) == share]
            if status:
                rows = [row for row in rows if row.get("review_status") == status]
            if app.cfg.get("blind_priority"):
                rows = [_without_model_hints(row) for row in rows]
            else:
                scores = _model_scores(app.cfg)
                for row in rows:
                    row["model_scores"] = scores.get(row["call_id"], {})
            for row in rows:
                row["share"] = assigned.get(row["call_id"], "")
            self._json(200, rows)

        def _one(self, call_id: str, query: dict[str, list[str]]) -> None:
            if not valid_call_id(call_id):
                self._error(400, "Некорректный call_id")
                return
            try:
                annotator = self._annotator(query)
            except ValueError as exc:
                self._error(400, str(exc))
                return
            for row in app.store.view(annotator):
                if row["call_id"] == call_id:
                    if app.cfg.get("blind_priority"):
                        row = _without_model_hints(row)
                    else:
                        row["model_scores"] = _model_scores(app.cfg).get(call_id, {})
                    self._json(200, row)
                    return
            self._error(404, "Звонка нет в партии")

        def _audio(self, call_id: str, query: dict[str, list[str]]) -> None:
            if not valid_call_id(call_id):
                self._error(400, "Некорректный call_id")
                return
            fact = next((row for row in app.store.manifest() if row["call_id"] == call_id), None)
            if fact is None:
                self._error(404, "Звонка нет в партии")
                return
            channel = (query.get("channel") or ["mix"])[0]
            try:
                payload = preview_bytes(app.cfg, fact["source_file"], channel)
            except ValueError as exc:
                self._error(400, str(exc))
                return
            except FileNotFoundError:
                self._error(404, "WAV не найден")
                return
            except PermissionError:
                self._error(403, "Аудио вне разрешённого каталога")
                return
            self._audio_body(payload)

        def _audio_body(self, payload: bytes) -> None:
            start, end = 0, len(payload) - 1
            status = 200
            ranged = self.headers.get("Range")
            if ranged and ranged.startswith("bytes=") and payload:
                piece = ranged.split("=", 1)[1].split(",", 1)[0]
                left, _, right = piece.partition("-")
                try:
                    start = int(left) if left else 0
                    end = int(right) if right else len(payload) - 1
                except ValueError:
                    start, end = 0, -1
                if start < 0 or start >= len(payload) or end < start:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{len(payload)}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                end = min(end, len(payload) - 1)
                status = 206
            chunk = payload[start : end + 1] if payload else b""
            self.send_response(status)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(chunk)))
            self.send_header("Accept-Ranges", "bytes")
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(chunk)

    return Handler


def _assignment(cfg: dict) -> dict[str, str]:
    try:
        path = assignment_path(cfg)
    except ValueError:
        return {}
    return load_assignment(path)


def _read_scores(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            row["call_id"]: {key: value or "" for key, value in row.items() if key and key != "call_id"}
            for row in csv.DictReader(handle)
            if row.get("call_id")
        }


def _merge_long_scores(scores: dict[str, dict[str, str]], path: Path) -> None:
    if not path.is_file():
        return
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            call_id = row.get("call_id") or ""
            model = row.get("model") or ""
            if not call_id or not model:
                continue
            item = scores.setdefault(call_id, {})
            item[f"{model}_status"] = row.get("status") or ""
            item[f"{model}_class"] = row.get("predicted_class") or ""
            item[f"{model}_score"] = row.get("score") or ""


def _without_model_hints(row: dict) -> dict:
    return {key: value for key, value in row.items() if key not in HIDDEN_BEFORE_LABEL}


def _read_queue(cfg: dict) -> list[dict[str, str]]:
    raw = (cfg.get("paths") or {}).get("queue_csv") or ""
    if not raw:
        raise ValueError("В конфиге нет очереди прослушивания")
    path = resolve_path(cfg, raw)
    if not path.is_file():
        raise ValueError("Очередь прослушивания ещё не собрана")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [{key: value or "" for key, value in row.items() if key} for row in csv.DictReader(handle)]


def _read_predictions(cfg: dict) -> dict[str, dict[str, str]]:
    raw = (cfg.get("paths") or {}).get("predictions_csv") or ""
    if not raw:
        raise ValueError("В конфиге нет таблицы предсказаний")
    path = resolve_path(cfg, raw)
    if not path.is_file():
        raise ValueError("Таблица предсказаний ещё не собрана")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            row["call_id"]: {key: value or "" for key, value in row.items() if key}
            for row in csv.DictReader(handle)
            if row.get("call_id")
        }


def _prediction_view(row: dict[str, str]) -> list[dict[str, str]]:
    models = [
        ("Spectra-0", "spectra0", "synthetic, если score ниже порога"),
        ("AASIST3", "aasist3", "synthetic, если score ниже порога"),
        ("Logistic new", "logistic_new", "synthetic, если score не ниже порога"),
        ("SVM new", "svm_new", "synthetic, если score не ниже порога"),
        ("LightGBM new", "lightgbm_new", "synthetic, если score не ниже порога"),
        ("logistic_old_bundle", "logistic_old", "класс machine старой задачи, если score не ниже порога"),
        ("svm_old_bundle", "svm_old", "класс machine старой задачи, если score не ниже порога"),
        ("lightgbm_old_bundle", "lightgbm_old", "класс machine старой задачи, если score не ниже порога"),
    ]
    items = []
    for title, key, rule in models:
        items.append(
            {
                "model": title,
                "score": row.get(f"{key}_score", ""),
                "prediction": row.get(f"{key}_prediction", ""),
                "status": row.get(f"{key}_status", ""),
                "rule": rule,
            }
        )
    items.append({"model": "Классификация кампании «автоответчик»", "score": "", "prediction": row.get("amd_campaign", ""), "status": "", "rule": "отдельная задача автоответчика"})
    items.append({"model": "Живой фразовый детектор", "score": "", "prediction": row.get("amd_live", ""), "status": "", "rule": "отдельная задача автоответчика"})
    items.append({"model": "Исход answering_machine", "score": "", "prediction": row.get("amd_outcome", ""), "status": "", "rule": "исход кампании, не метка происхождения"})
    return items


def _model_scores(cfg: dict) -> dict[str, dict[str, str]]:
    raw = (cfg.get("paths") or {}).get("scores_csv") or ""
    if raw:
        finished = _read_scores(resolve_path(cfg, raw))
        if finished:
            return finished
    root = Path(cfg["_root"]) / "evaluation"
    scores: dict[str, dict[str, str]] = {}
    previous = root / "confirmed_001" / "20261113T155919" / "call_results"
    for name in ("spectra0.csv", "aasist3.csv", "logistic_regression.csv", "svm.csv", "lightgbm.csv"):
        _merge_long_scores(scores, previous / name)
    day_root = root / str(cfg.get("batch_id") or "day_20261113")
    if day_root.is_dir():
        partials = sorted(day_root.glob("*/logs/classical_scores.csv"))
        if partials:
            _merge_long_scores(scores, partials[-1])
    return scores


def make_server(cfg: dict, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    if host != "127.0.0.1":
        raise RuntimeError("Сервер слушает только 127.0.0.1")
    app = ReviewApp(cfg)
    return ThreadingHTTPServer((host, port), handler_factory(app))
