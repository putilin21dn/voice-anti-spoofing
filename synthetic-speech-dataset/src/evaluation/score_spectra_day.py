"""Spectra-0 по уже скачанной партии. AASIST3 в голос A не входит и здесь не считается."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.spectra_runner import _prepare, _spectra_call, load_modules  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402


def main() -> None:
    request = Path(sys.argv[1])
    output = Path(sys.argv[2])
    cfg = load_config("config/evaluation.yaml")
    spectra_root = resolve_path(cfg, cfg["paths"]["spectra_root"])
    webrtc_site = Path(torch.__file__).resolve().parent.parent
    inference, _aasist = load_modules(spectra_root, webrtc_site)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    calls = list(csv.DictReader(request.open(encoding="utf-8-sig", newline="")))
    print(f"Spectra-0: {len(calls)} звонков, устройство {device}", flush=True)
    model = inference.load_model(device)
    rows = []
    try:
        prepared = [_prepare(inference, call) for call in calls]
        total = len(prepared)
        for index, item in enumerate(prepared, start=1):
            windows: list[dict] = []
            rows.append(_spectra_call(inference, model, item, device, windows))
            if index == 1 or index % 25 == 0 or index == total:
                print(f"Spectra-0 {index}/{total}", flush=True)
    finally:
        del model
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit("Spectra-0 не вернула ни одного звонка")
    fields = list(rows[0].keys())
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"spectra: {output}", flush=True)


if __name__ == "__main__":
    main()
