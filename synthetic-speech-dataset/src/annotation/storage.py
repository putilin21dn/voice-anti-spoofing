from __future__ import annotations

import csv
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.annotation.schema import (
    CALL_FIELDS,
    FACT_FIELDS,
    LABEL_FIELDS,
    SEGMENT_FIELDS,
    valid_annotator,
    valid_call_id,
    validate_annotation,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{key: (value or "") for key, value in row.items() if key} for row in csv.DictReader(handle)]


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    temporary.replace(path)


def blank_annotation(facts: dict[str, str]) -> dict[str, str]:
    row = {field: "" for field in CALL_FIELDS}
    for field in FACT_FIELDS:
        row[field] = facts.get(field, "")
    row["primary_label"] = "unknown"
    row["label_source"] = "unknown_source"
    row["review_status"] = "unreviewed"
    return row


class AnnotationStore:
    def __init__(self, manifest_path: Path, calls_path: Path, segments_path: Path) -> None:
        self.manifest_path = manifest_path
        self.calls_path = calls_path
        self.segments_path = segments_path
        self._lock = threading.Lock()

    def manifest(self) -> list[dict[str, str]]:
        rows = read_csv(self.manifest_path)
        return [row for row in rows if valid_call_id(row.get("call_id", ""))]

    def calls(self) -> list[dict[str, str]]:
        return read_csv(self.calls_path)

    def segments(self) -> list[dict[str, str]]:
        return read_csv(self.segments_path)

    def touched(self) -> bool:
        for row in self.calls():
            if row.get("annotator_id"):
                return True
            if row.get("primary_label") not in {"", "unknown"}:
                return True
            if row.get("review_status") not in {"", "unreviewed"}:
                return True
            if (row.get("annotation_notes") or "").strip():
                return True
        return any(row.get("call_id") for row in self.segments())

    def view(self, annotator_id: str = "") -> list[dict]:
        if not valid_annotator(annotator_id):
            raise ValueError("Некорректный код разметчика")
        facts = {row["call_id"]: row for row in self.manifest()}
        labels: dict[tuple[str, str], dict[str, str]] = {}
        for row in self.calls():
            labels[(row.get("call_id", ""), row.get("annotator_id", ""))] = row
        grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
        for row in self.segments():
            grouped.setdefault((row.get("call_id", ""), row.get("annotator_id", "")), []).append(row)
        items = []
        for call_id, fact in facts.items():
            chosen = labels.get((call_id, annotator_id))
            owner = annotator_id
            if chosen is None and annotator_id:
                chosen = labels.get((call_id, ""))
                owner = ""
            if chosen is None:
                others = [
                    row
                    for (cid, aid), row in labels.items()
                    if cid == call_id and aid != annotator_id
                ]
                if others:
                    chosen = sorted(others, key=lambda row: row.get("updated_at", ""))[-1]
                    owner = chosen.get("annotator_id", "")
            if chosen is None:
                chosen = blank_annotation(fact)
                owner = ""
            item = blank_annotation(fact)
            for field in LABEL_FIELDS:
                item[field] = chosen.get(field, "")
            item["annotator_id"] = annotator_id
            item["label_owner"] = owner
            if not item["primary_label"]:
                item["primary_label"] = "unknown"
                item["label_source"] = "unknown_source"
                item["review_status"] = "unreviewed"
            own_segments = grouped.get((call_id, owner), [])
            item["segments"] = [
                {field: segment.get(field, "") for field in SEGMENT_FIELDS}
                for segment in own_segments
            ]
            items.append(item)
        return items

    def save(self, call_id: str, payload: dict) -> dict:
        if not valid_call_id(call_id):
            raise ValueError("Некорректный call_id")
        with self._lock:
            facts = {row["call_id"]: row for row in self.manifest()}
            fact = facts.get(call_id)
            if fact is None:
                raise ValueError("Звонка нет в manifest этой партии")
            duration = float(fact.get("duration_seconds") or 0)
            clean = validate_annotation(payload, duration_s=duration)
            annotator = clean["annotator_id"]
            updated = blank_annotation(fact)
            updated.update({key: clean[key] for key in LABEL_FIELDS if key != "updated_at"})
            updated["updated_at"] = datetime.now(timezone.utc).isoformat()
            calls = [row for row in self.calls() if not (row.get("call_id") == call_id and row.get("annotator_id", "") == annotator)]
            calls.append(updated)
            calls.sort(key=lambda row: (row.get("call_id", ""), row.get("annotator_id", "")))
            write_csv(self.calls_path, CALL_FIELDS, calls)

            segments = [
                row
                for row in self.segments()
                if not (row.get("call_id") == call_id and row.get("annotator_id", "") == annotator)
            ]
            for segment in clean["segments"]:
                segment_id = segment["segment_id"] or str(uuid4())
                segments.append(
                    {
                        "call_id": call_id,
                        "segment_id": segment_id,
                        "start_time": segment["start_time"],
                        "end_time": segment["end_time"],
                        "speaker_id": segment["speaker_id"],
                        "segment_label": segment["segment_label"],
                        "label_confidence": segment["label_confidence"],
                        "label_source": segment["label_source"],
                        "notes": segment["notes"],
                        "annotator_id": annotator,
                    }
                )
            segments.sort(key=lambda row: (row.get("call_id", ""), row.get("annotator_id", ""), row.get("start_time", "")))
            write_csv(self.segments_path, SEGMENT_FIELDS, segments)
            saved = self.view(annotator)
        for item in saved:
            if item["call_id"] == call_id:
                return item
        raise RuntimeError("Сохранённая запись не прочиталась")
