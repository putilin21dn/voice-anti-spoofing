from __future__ import annotations

import csv
from pathlib import Path

from src.annotation.schema import valid_call_id
from src.audio.loader import probe_wav
from src.utils.config import resolve_path


MACHINE_MARKERS = ("автоответчик", "answering_machine", "anti_spam_bot")
REASON_ORDER = ("human_pool", "machine_answering", "machine_live_outcome", "label_conflict")


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def client_channel_index(external_id: str, channels: int) -> int:
    """Канал абонента. external_id используется только здесь и в manifest не пишется."""

    if channels <= 1:
        return 0
    if (external_id or "").startswith("excel-"):
        return 1
    return 0


def load_locked_splits(splits_dir: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    for name in ("train", "validation", "test", "spectra_threshold"):
        path = splits_dir / f"{name}.csv"
        if not path.exists():
            continue
        for row in read_table(path):
            call_id = row.get("call_id") or ""
            if valid_call_id(call_id):
                found[call_id] = name
    return found


def _conflict(prior_sample: str, outcome: str, classification: str) -> bool:
    if prior_sample != "human":
        return False
    text = f"{outcome}|{classification}"
    return any(marker in text for marker in MACHINE_MARKERS)


def _reason(prior_sample: str, outcome: str, classification: str, selection: dict) -> str:
    if _conflict(prior_sample, outcome, classification):
        return "label_conflict"
    if prior_sample == "human":
        return "human_pool"
    if prior_sample == "machine" and outcome == selection["machine_outcome"]:
        return "machine_answering"
    if prior_sample == "machine" and outcome in set(selection["live_outcomes"]):
        return "machine_live_outcome"
    return ""


def take_evenly(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if count <= 0 or not rows:
        return []
    if len(rows) <= count:
        return list(rows)
    chosen: list[dict[str, str]] = []
    used: set[int] = set()
    for index in range(count):
        pointer = round(index * (len(rows) - 1) / (count - 1)) if count > 1 else 0
        while pointer in used and pointer + 1 < len(rows):
            pointer += 1
        if pointer in used or pointer >= len(rows):
            continue
        used.add(pointer)
        chosen.append(rows[pointer])
    if len(chosen) < count:
        for pointer, row in enumerate(rows):
            if pointer in used:
                continue
            chosen.append(row)
            if len(chosen) == count:
                break
    return chosen


def build_batch(cfg: dict) -> tuple[list[dict[str, str]], dict]:
    selection = cfg["selection"]
    audio_root = resolve_path(cfg, cfg["paths"]["audio_root"])
    collector = resolve_path(cfg, cfg["paths"]["collector_root"])
    locked = load_locked_splits(resolve_path(cfg, cfg["paths"]["splits_dir"]))
    excluded = set(selection["exclude_locked_splits"])
    seen: set[str] = set()
    candidates: list[dict[str, str]] = []
    stats: dict = {
        "unreadable": [],
        "duplicate_call_ids": [],
        "pool_available": {reason: 0 for reason in REASON_ORDER},
        "pool_excluded_locked": {reason: 0 for reason in REASON_ORDER},
        "other_machine": 0,
    }

    for prior_sample, key in (("human", "human_manifest"), ("machine", "machine_manifest")):
        for source in read_table(resolve_path(cfg, cfg["paths"][key])):
            call_id = source.get("call_id") or ""
            if not valid_call_id(call_id):
                continue
            if call_id in seen:
                stats["duplicate_call_ids"].append(call_id)
                continue
            seen.add(call_id)
            outcome = source.get("outcome") or ""
            classification = source.get("classification_result") or ""
            reason = _reason(prior_sample, outcome, classification, selection)
            if not reason:
                if prior_sample == "machine":
                    stats["other_machine"] += 1
                continue
            audio = audio_root / prior_sample / f"{call_id}.wav"
            probed = probe_wav(audio)
            if probed is None:
                stats["unreadable"].append(call_id)
                continue
            split = locked.get(call_id, "")
            row = {
                "call_id": call_id,
                "source_file": audio.resolve().relative_to(collector).as_posix(),
                "data_source": "telephone_call",
                "campaign_id": "",
                "campaign_name": source.get("campaign_name") or "",
                "call_status": "",
                "goal_status": source.get("goal_status") or "",
                "outcome": outcome,
                "codec": probed["codec"],
                "codec_source": probed["codec_source"],
                "sample_rate": probed["sample_rate"],
                "duration_seconds": probed["duration_seconds"],
                "channel_count": probed["channel_count"],
                "annotated_channel": "client",
                "client_channel": str(client_channel_index(source.get("external_id") or "", int(probed["channel_count"]))),
                "prior_sample": prior_sample,
                "prior_classification": classification,
                "locked_split": split,
                "selection_reason": reason,
                "training_role": "locked_baseline" if split in excluded else "not_for_training",
                "batch_id": cfg["batch_id"],
                "timestamp": source.get("timestamp") or "",
            }
            stats["pool_available"][reason] += 1
            if split in excluded and not (reason == "label_conflict" and selection.get("always_include_conflicts")):
                stats["pool_excluded_locked"][reason] += 1
                continue
            candidates.append(row)

    selected: list[dict[str, str]] = []
    quotas = selection["quotas"]
    for reason in ("human_pool", "machine_answering", "machine_live_outcome"):
        pool = [row for row in candidates if row["selection_reason"] == reason and row["training_role"] != "locked_baseline"]
        pool.sort(key=lambda row: (row["timestamp"], row["call_id"]))
        selected.extend(take_evenly(pool, int(quotas[reason])))
    if selection.get("always_include_conflicts"):
        known = {row["call_id"] for row in selected}
        conflicts = [row for row in candidates if row["selection_reason"] == "label_conflict" and row["call_id"] not in known]
        conflicts.sort(key=lambda row: (row["timestamp"], row["call_id"]))
        selected.extend(conflicts)

    selected.sort(key=lambda row: (REASON_ORDER.index(row["selection_reason"]), row["timestamp"], row["call_id"]))
    for row in selected:
        row.pop("timestamp", None)
    stats["selected"] = {
        reason: sum(1 for row in selected if row["selection_reason"] == reason) for reason in REASON_ORDER
    }
    stats["rows"] = len(selected)
    return selected, stats
