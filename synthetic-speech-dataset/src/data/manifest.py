from __future__ import annotations

import json

from src.annotation.schema import CALL_FIELDS, FACT_FIELDS, SEGMENT_FIELDS
from src.annotation.storage import AnnotationStore, blank_annotation, write_csv
from src.data.selector import build_batch
from src.utils.config import resolve_path


def init_batch(cfg: dict, *, rebuild: bool = False) -> dict:
    manifest_path = resolve_path(cfg, cfg["paths"]["manifest_out"])
    calls_path = resolve_path(cfg, cfg["paths"]["calls_out"])
    segments_path = resolve_path(cfg, cfg["paths"]["segments_out"])
    store = AnnotationStore(manifest_path, calls_path, segments_path)
    if manifest_path.exists() and not rebuild:
        return {"created": False, "reason": "manifest уже есть", "rows": len(store.manifest())}
    if store.touched():
        raise RuntimeError("Разметка уже начата. Повторная сборка партии её не перезаписывает.")

    rows, stats = build_batch(cfg)
    if not rows:
        raise RuntimeError("Партия пустая. Проверьте пути к manifest и WAV.")
    write_csv(manifest_path, FACT_FIELDS, rows)
    write_csv(calls_path, CALL_FIELDS, [blank_annotation(row) for row in rows])
    write_csv(segments_path, SEGMENT_FIELDS, [])
    stats_path = manifest_path.with_name(manifest_path.stem + "_stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    stats["created"] = True
    stats["manifest"] = str(manifest_path)
    return stats
