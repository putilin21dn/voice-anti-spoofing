"""Состав кампании B2X-16-11-2026-callback. Телефоны и ключи S3 не читаются."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

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

SCRIPT = r"""
from config.settings import load_settings
from src.db.connection import connect

sql = '''
SELECT
    c.direction::text AS direction,
    c.with_audio,
    c.is_finalized,
    c.duration >= 15 AS long_enough,
    c.outcome,
    r.classification_result,
    (c.collected_data ? 'answering_machine_text') AS live_amd,
    count(*) AS n,
    count(*) FILTER (WHERE c.duration >= 15) AS n_ge15
FROM calls AS c
JOIN call_campaign_results AS r ON r.call_id = c.id
JOIN call_campaigns AS camp ON camp.id = r.call_campaign_id
WHERE camp.analysis_type = 'b2x'
  AND camp.name = 'B2X-16-11-2026-callback'
GROUP BY 1, 2, 3, 4, 5, 6, 7
ORDER BY n DESC
'''
names = '''
SELECT camp.name, count(DISTINCT c.id) AS calls
FROM calls AS c
JOIN call_campaign_results AS r ON r.call_id = c.id
JOIN call_campaigns AS camp ON camp.id = r.call_campaign_id
WHERE camp.analysis_type = 'b2x'
  AND camp.name ILIKE '%callback%'
  AND c.timestamp >= TIMESTAMPTZ '2026-11-16 00:00:00+03'
  AND c.timestamp < TIMESTAMPTZ '2026-11-18 00:00:00+03'
GROUP BY camp.name
ORDER BY calls DESC
'''
with connect(load_settings()) as connection:
    print('NAMES')
    for name, n in connection.execute(names).fetchall():
        print(f'{n}\t{name}')
    print('ROWS')
    for row in connection.execute(sql).fetchall():
        print('\t'.join('' if item is None else str(item) for item in row))
"""


def main() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(COLLECTOR)
    completed = subprocess.run(
        [str(PYTHON), "-c", SCRIPT],
        text=True,
        capture_output=True,
        cwd=str(COLLECTOR),
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "census failed").strip()
        raise SystemExit(message.splitlines()[-1][:500])
    print(completed.stdout)


if __name__ == "__main__":
    main()
