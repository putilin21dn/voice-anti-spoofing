"""Проиграть JSON-ленту записанного звонка и записать JSONL shadow-событий."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from shadow_replay.replay import play_tape


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Проиграть записанный звонок так, будто он ещё идёт. "
            "Лента содержит кадры audio, classical, final и call_end. "
            "Текст реплики в неё не кладётся."
        )
    )
    parser.add_argument("tape", type=Path, help="JSON с полем events")
    parser.add_argument("--output", type=Path, help="JSONL. Без флага события печатаются в stdout")
    args = parser.parse_args(argv)
    tape = json.loads(args.tape.read_text(encoding="utf-8"))
    rows = play_tape(tape)
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    payload = "\n".join(lines) + ("\n" if lines else "")
    if args.output is None:
        sys.stdout.write(payload)
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
