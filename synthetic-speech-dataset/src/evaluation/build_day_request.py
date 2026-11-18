"""Собирает запрос на прогон из manifest дня. Телефон и ключ S3 в него не входят."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    manifest = Path(sys.argv[1])
    json_out = Path(sys.argv[2])
    csv_out = Path(sys.argv[3])
    calls = []
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            wav = Path(row["source_file"])
            if not wav.is_absolute():
                wav = ROOT / wav
            calls.append(
                {
                    "call_id": row["call_id"],
                    "wav_path": str(wav),
                    "client_channel": row.get("client_channel", "0"),
                    "duration_seconds": row.get("duration_seconds", ""),
                    "channel_count": row.get("channel_count", ""),
                    "prior_classification": row.get("prior_classification", ""),
                    "outcome": row.get("outcome", ""),
                    "campaign_name": row.get("campaign_name", ""),
                    "timestamp": row.get("timestamp", ""),
                }
            )
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps({"calls": calls}, ensure_ascii=False), encoding="utf-8")
    with csv_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["call_id", "wav_path", "client_channel"],
            lineterminator="\n",
        )
        writer.writeheader()
        for call in calls:
            writer.writerow({key: call[key] for key in writer.fieldnames})
    print(f"запрос: {len(calls)}", flush=True)


if __name__ == "__main__":
    main()
