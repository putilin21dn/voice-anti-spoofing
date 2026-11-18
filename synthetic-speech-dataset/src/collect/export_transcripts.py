"""Выгружает реплики дня из messages. Телефон, external_id и ключ S3 не читаются.

post — офлайн Tone ASR по записи. pre — живой ASR звонка. GigaAM в базе не хранится.
Номера в тексте заменяются до записи на диск.
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

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
sys.path.insert(0, str(COLLECTOR))

from config.settings import load_settings  # noqa: E402
from src.db.connection import connect  # noqa: E402


MANIFEST = ROOT / "data" / "manifests" / "day_20261113.csv"
OUT_DIR = ROOT / "data" / "transcripts"
TURNS_OUT = OUT_DIR / "day_20261113_turns.csv"
DIALOGUES_OUT = OUT_DIR / "day_20261113_dialogues.csv"
FORBIDDEN = {"phone", "phone_number", "external_id", "s3_key", "s3_bucket", "msisdn", "client_id"}
TURN_FIELDS = ["call_id", "source", "turn_index", "speaker", "offset_s", "text"]
DIALOGUE_FIELDS = [
    "call_id",
    "post_turns",
    "pre_turns",
    "post_client_turns",
    "post_dialogue",
    "post_client_text",
    "pre_dialogue",
    "pre_client_text",
]
# Десять и больше цифр подряд или через пробелы, скобки и дефисы — номер, не длительность.
_PHONE_RUN = re.compile(r"(?:\+?\d[\s\-().]{0,3}){10,}")


def redact(text: str) -> tuple[str, int]:
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        digits = re.sub(r"\D", "", match.group(0))
        if len(digits) < 10:
            return match.group(0)
        count += 1
        return "[номер]"

    return _PHONE_RUN.sub(replace, text or ""), count


def load_call_ids() -> list[str]:
    with MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row["call_id"] for row in rows if row.get("call_id")]
    if len(ids) != len(set(ids)):
        raise SystemExit("В manifest повторяется call_id")
    return ids


def fetch_messages(call_ids: list[str]) -> list[dict]:
    settings = load_settings()
    with connect(settings) as connection:
        cursor = connection.execute(
            """
            SELECT call_id::text AS call_id,
                   side,
                   type,
                   timestamp,
                   content
            FROM messages
            WHERE call_id = ANY(%(ids)s::uuid[])
              AND type IN ('post', 'pre')
            ORDER BY call_id, type, timestamp
            """,
            {"ids": call_ids},
        )
        columns = [item.name for item in cursor.description]
        if FORBIDDEN & set(columns):
            raise SystemExit("Запрос вернул запрещённую колонку")
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def speaker_of(side: str, content: str) -> str:
    if (content or "").strip().lower().startswith("служебный контекст"):
        return "service"
    if side == "agent":
        return "agent"
    if side == "client":
        return "client"
    return "unknown"


def build(call_ids: list[str], raw_rows: list[dict]) -> tuple[list[dict], list[dict], dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    redactions = 0
    for row in raw_rows:
        text, n = redact(row["content"] or "")
        redactions += n
        grouped[(row["call_id"], row["type"])].append(
            {
                "call_id": row["call_id"],
                "source": row["type"],
                "speaker": speaker_of(row["side"], row["content"] or ""),
                "timestamp": float(row["timestamp"]),
                "text": text,
            }
        )

    turns: list[dict] = []
    by_call: dict[str, dict[str, list[dict]]] = defaultdict(lambda: {"post": [], "pre": []})
    for (call_id, source), items in grouped.items():
        stamps = [item["timestamp"] for item in items]
        absolute = max(stamps) > 1e8
        origin = min(stamps) if absolute else 0.0
        for index, item in enumerate(items, start=1):
            offset = item["timestamp"] - origin if absolute else item["timestamp"]
            turn = {
                "call_id": call_id,
                "source": source,
                "turn_index": index,
                "speaker": item["speaker"],
                "offset_s": f"{offset:.3f}",
                "text": item["text"],
            }
            turns.append(turn)
            by_call[call_id][source].append(turn)

    def line(turn: dict) -> str:
        return f"[{turn['offset_s']} {turn['speaker']}] {turn['text']}"

    def client_lines(items: list[dict]) -> str:
        return "\n".join(turn["text"] for turn in items if turn["speaker"] == "client")

    dialogues = []
    for call_id in call_ids:
        post = by_call[call_id]["post"]
        pre = by_call[call_id]["pre"]
        dialogues.append(
            {
                "call_id": call_id,
                "post_turns": len(post),
                "pre_turns": len(pre),
                "post_client_turns": sum(turn["speaker"] == "client" for turn in post),
                "post_dialogue": "\n".join(line(turn) for turn in post),
                "post_client_text": client_lines(post),
                "pre_dialogue": "\n".join(line(turn) for turn in pre),
                "pre_client_text": client_lines(pre),
            }
        )

    stats = {
        "calls": len(call_ids),
        "raw_messages": len(raw_rows),
        "turns": len(turns),
        "with_post": sum(row["post_turns"] > 0 for row in dialogues),
        "with_pre": sum(row["pre_turns"] > 0 for row in dialogues),
        "with_post_client": sum(row["post_client_turns"] > 0 for row in dialogues),
        "post_only": sum(row["post_turns"] > 0 and row["pre_turns"] == 0 for row in dialogues),
        "pre_only": sum(row["pre_turns"] > 0 and row["post_turns"] == 0 for row in dialogues),
        "neither": sum(row["post_turns"] == 0 and row["pre_turns"] == 0 for row in dialogues),
        "redactions": redactions,
        "speakers": {},
    }
    speakers: dict[str, int] = defaultdict(int)
    for turn in turns:
        speakers[f"{turn['source']}:{turn['speaker']}"] += 1
    stats["speakers"] = dict(speakers)
    return turns, dialogues, stats


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    if FORBIDDEN & set(fields):
        raise SystemExit("В файле запрещённая колонка")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    call_ids = load_call_ids()
    raw_rows = fetch_messages(call_ids)
    turns, dialogues, stats = build(call_ids, raw_rows)
    write_csv(TURNS_OUT, TURN_FIELDS, turns)
    write_csv(DIALOGUES_OUT, DIALOGUE_FIELDS, dialogues)
    print(
        "calls={calls} raw_messages={raw_messages} turns={turns} "
        "with_post={with_post} with_pre={with_pre} with_post_client={with_post_client} "
        "post_only={post_only} pre_only={pre_only} neither={neither} redactions={redactions}".format(**stats)
    )
    print("speakers", " ".join(f"{key}={value}" for key, value in sorted(stats["speakers"].items())))


if __name__ == "__main__":
    main()
