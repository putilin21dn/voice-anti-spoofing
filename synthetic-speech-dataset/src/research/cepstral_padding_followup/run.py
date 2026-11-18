"""Build the 2026-11-13 cepstral and padding follow-up. No model is run."""

from __future__ import annotations

import argparse
import json

from src.research.cepstral_padding_followup.audio import run_audio
from src.research.cepstral_padding_followup.common import OUTPUT
from src.research.cepstral_padding_followup.plots import make_plots
from src.research.cepstral_padding_followup.report import write_reports
from src.research.cepstral_padding_followup.tables import build_tables


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", default="tables,audio,plots,report")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    steps = {step.strip() for step in args.steps.split(",") if step.strip()}
    status_path = OUTPUT / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    if "tables" in steps:
        status["tables"] = build_tables()
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print("tables done", flush=True)
    if "audio" in steps:
        status["audio"] = run_audio(resume=args.resume)
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print("audio done", flush=True)
    if "plots" in steps:
        make_plots()
        print("plots done", flush=True)
    if "report" in steps:
        write_reports(status)
        print("report done", flush=True)


if __name__ == "__main__":
    main()
