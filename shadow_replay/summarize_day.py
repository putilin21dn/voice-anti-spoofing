"""Сводит голос A по уже посчитанным score дня. Идентификаторы звонков не печатает."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from shadow_replay.acoustic import high_confidence_synthetic, model_decision
from shadow_replay.ui_marks import interface_label, resulting_mark, text_split

ROOT = Path(__file__).resolve().parents[1]
DAY = ROOT / "synthetic-speech-dataset" / "evaluation" / "day_20261118"
MANIFEST = ROOT / "synthetic-speech-dataset" / "data" / "manifests" / "day_20261118.csv"


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    spectra = {row["call_id"]: row for row in _rows(DAY / "spectra0.csv")}
    classical: dict[str, dict[str, dict]] = {}
    for row in _rows(DAY / "classical_scores.csv"):
        classical.setdefault(row["call_id"], {})[row["model"]] = row
    manifest = {row["call_id"]: row for row in _rows(MANIFEST)}
    decisions = Counter()
    marks = Counter()
    synthetic_split = Counter()
    kept_on_synthetic = Counter()
    spectra_status = Counter()
    missing_models = Counter()
    output_rows = []
    for call_id, spectra_row in spectra.items():
        spectra_status[spectra_row.get("status") or ""] += 1
        specs = {
            "spectra0": (spectra_row.get("score"), spectra_row.get("status") or ""),
            "svm": (classical.get(call_id, {}).get("svm_new", {}).get("score"), classical.get(call_id, {}).get("svm_new", {}).get("status") or "missing"),
            "lightgbm": (
                classical.get(call_id, {}).get("lightgbm_new", {}).get("score"),
                classical.get(call_id, {}).get("lightgbm_new", {}).get("status") or "missing",
            ),
        }
        votes = []
        for model, (score, status) in specs.items():
            if status != "ok":
                missing_models[model] += 1
            vote = model_decision(model, None if score in {None, ""} else score, "ok" if status == "ok" else status)
            if vote:
                votes.append(vote)
        if len(votes) < 2:
            ensemble = "no_decision"
        else:
            ensemble = "synthetic" if sum(item == "synthetic" for item in votes) >= 2 else "human"
        decisions[ensemble] += 1
        stored = manifest.get(call_id, {}).get("prior_classification") or ""
        current = interface_label(stored)
        confident = high_confidence_synthetic(specs)
        mark = resulting_mark(stored, ensemble, high_confidence=confident)
        marks[mark] += 1
        if ensemble == "synthetic":
            split = text_split(stored)
            if split:
                synthetic_split[split] += 1
            elif confident:
                synthetic_split["высокая уверенность → автоответчик"] += 1
            else:
                synthetic_split["уверенность не высокая"] += 1
                kept_on_synthetic[current] += 1
        output_rows.append(
            {
                "call_id": call_id,
                "ensemble": ensemble,
                "interface_mark": current,
                "mark": mark,
                "text_split": text_split(stored),
                "high_confidence": "yes" if confident else "",
            }
        )
    destination = DAY / "ui_marks.csv"
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["call_id", "ensemble", "interface_mark", "mark", "text_split", "high_confidence"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"звонков со Spectra-0: {len(spectra)}")
    print(f"статус Spectra-0: {dict(spectra_status)}")
    print(f"голос A: {dict(decisions)}")
    print("отметки интерфейса после прогона:")
    for name, count in marks.most_common():
        print(f"  {count} {name}")
    print(f"synthetic и текст: {dict(synthetic_split)}")
    if kept_on_synthetic:
        print("synthetic без разделения текстом, отметка сохранена:")
        for name, count in kept_on_synthetic.most_common():
            print(f"  {count} {name}")
    print(f"модели без ok: {dict(missing_models)}")


if __name__ == "__main__":
    main()
