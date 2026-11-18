"""Печатает телефон звонков, которые голос назвал human. В файл номер не пишет."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]


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
PYTHON = Path(sys.executable)
VOTE = ROOT / "reports" / "callback_20261116" / "vote.csv"

SCRIPT = r"""
import sys
from config.settings import load_settings
from src.db.connection import connect
ids = [line.strip() for line in sys.stdin if line.strip()]
sql = '''
SELECT c.id::text, c.external_id
FROM calls AS c
WHERE c.id::text = ANY(%(ids)s)
'''
with connect(load_settings()) as connection:
    for call_id, external_id in connection.execute(sql, {"ids": ids}).fetchall():
        print(f"{call_id}\t{external_id or ''}")
"""


def phone_of(external_id: str) -> str:
    tail = (external_id or "").split("::", 1)[-1]
    number = tail.split("_", 1)[0]
    if number.isdigit() and len(number) >= 10:
        return number
    return ""


def main() -> None:
    frame = pd.read_csv(VOTE, encoding="utf-8-sig")
    humans = frame[frame["vote"].eq("human")].copy()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(COLLECTOR)
    env["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        [str(PYTHON), "-c", SCRIPT],
        input="\n".join(humans["call_id"]),
        text=True,
        capture_output=True,
        cwd=str(COLLECTOR),
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "запрос не выполнился").strip()
        raise SystemExit(message.splitlines()[-1][:300])
    found = {}
    for line in completed.stdout.splitlines():
        call_id, _, external_id = line.partition("\t")
        found[call_id] = phone_of(external_id)
    humans["phone"] = humans["call_id"].map(found)
    order = ["отказ_лпр", "перезвон", "асинхронная_связь", "неизвестно", "автоответчик"]
    humans["sort"] = humans["prior_classification"].map({name: index for index, name in enumerate(order)}).fillna(99)
    humans = humans.sort_values(["sort", "call_id"])
    for row in humans.itertuples(index=False):
        spectra = "Spectra одна synthetic" if row.spectra_pred == "synthetic" else "все human" if row.svm_pred == "human" and row.lgbm_pred == "human" else "не все human"
        print(f"{row.phone}\t{row.prior_classification}\t{row.outcome}\t{spectra}\t{row.call_id}")


if __name__ == "__main__":
    main()
