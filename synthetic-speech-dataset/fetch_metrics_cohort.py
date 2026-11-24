"""Скачивает wav когорты метрик по call_id.

external_id живёт только в памяти на время запроса к S3. В лог он не попадает:
в ключе S3 есть номер телефона.
"""

from __future__ import annotations

import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _component(name: str) -> Path:
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


def _load_cohort_module():
    path = ROOT / "src" / "collect" / "metrics_cohort.py"
    spec = importlib.util.spec_from_file_location("metrics_cohort_list", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("Не удалось прочитать список когорты")
    spec.loader.exec_module(module)
    return module


def _lookup(connection, call_ids: list[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for start in range(0, len(call_ids), 200):
        chunk = call_ids[start : start + 200]
        rows = connection.execute(
            "SELECT id::text, external_id FROM calls WHERE id::text = ANY(%s)",
            (chunk,),
        ).fetchall()
        for call_id, external_id in rows:
            found[str(call_id)] = str(external_id or "")
    return found


def _download_one(storage, recording_key, external_id: str, destination: Path) -> str:
    from src.storage.s3_client import S3StorageError

    if not external_id:
        return "empty_external_id"
    try:
        storage.download(recording_key(external_id), destination)
    except S3StorageError as exc:
        return exc.kind
    except OSError:
        return "write_error"
    if not destination.is_file() or destination.stat().st_size <= 44:
        return "invalid_wav"
    return ""


def main() -> None:
    cohort = _load_cohort_module()
    rows = cohort.load_cohort(cohort.cohort_path(ROOT))
    missing = cohort.pending(ROOT, rows)
    print(f"когорта {len(rows)}, уже на диске {len(rows) - len(missing)}, скачать {len(missing)}", flush=True)
    if not missing:
        return
    collector = _component("voice-dataset-collector")
    sys.path.insert(0, str(collector))
    from config.settings import load_settings
    from src.db.connection import DatabaseError, connect
    from src.db.queries import recording_key
    from src.storage.s3_client import ReadOnlyS3

    settings = load_settings()
    try:
        with connect(settings) as connection:
            identities = _lookup(connection, [row["call_id"] for row in missing])
    except DatabaseError as exc:
        raise SystemExit(str(exc)) from exc
    storage = ReadOnlyS3(settings)
    failures: dict[str, int] = {}
    downloaded = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {}
        for row in missing:
            destination = cohort.wav_path(ROOT, row["source_file"])
            external_id = identities.get(row["call_id"])
            if external_id is None:
                failures["not_in_db"] = failures.get("not_in_db", 0) + 1
                print(f"ошибка {row['call_id']} not_in_db", flush=True)
                continue
            futures[
                pool.submit(_download_one, storage, recording_key, external_id, destination)
            ] = row["call_id"]
        done = 0
        for future in as_completed(futures):
            done += 1
            error = future.result()
            if error:
                failures[error] = failures.get(error, 0) + 1
                print(f"ошибка {futures[future]} {error}", flush=True)
            else:
                downloaded += 1
            if done % 50 == 0 or done == len(futures):
                print(f"скачано {downloaded}, ошибок {sum(failures.values())}, прошло {done}/{len(futures)}", flush=True)
    if failures:
        summary = ", ".join(f"{kind} {count}" for kind, count in sorted(failures.items()))
        raise SystemExit(f"не скачано: {summary}")
    print(f"готово, скачано {downloaded}", flush=True)


if __name__ == "__main__":
    main()
