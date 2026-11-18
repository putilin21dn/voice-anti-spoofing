from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from config.settings import Settings
from src.collector.call_selector import merge_rows, read_manifest, row_from_db, write_manifest
from src.db.connection import connect
from src.db.queries import fetch_campaign_calls


MOSCOW = timezone(timedelta(hours=3))

# Живой диалог B2X: человек дошёл до содержательного исхода.
# Это не биометрическая метка, а classification_result кампании.
HUMAN_CLASSIFICATIONS = frozenset(
    {
        "лид",
        "интерес_без_данных",
        "отказ_лпр",
        "не_лпр",
        "не_та_организация",
        "перезвон",
        "асинхронная_связь",
    }
)

# Автоответчик и явный автоматический собеседник.
MACHINE_CLASSIFICATIONS = frozenset({"автоответчик", "answering_machine", "anti_spam_bot"})
MACHINE_OUTCOMES = frozenset({"answering_machine", "anti_spam_bot"})


def month_bounds(month: str) -> tuple[datetime, datetime]:
    """Календарный месяц в Europe/Moscow, границы полуоткрытые."""

    year_text, month_text = month.split("-", 1)
    start = datetime(int(year_text), int(month_text), 1, tzinfo=MOSCOW)
    if start.month == 12:
        end = datetime(start.year + 1, 1, 1, tzinfo=MOSCOW)
    else:
        end = datetime(start.year, start.month + 1, 1, tzinfo=MOSCOW)
    return start, end


def assign_sample(
    *,
    outcome: str | None,
    classifications: list[str] | None,
    has_answering_machine_text: bool,
) -> str | None:
    """Машинная метка важнее человеческой, если у звонка есть обе."""

    labels = set(classifications or [])
    machine = (
        (outcome or "") in MACHINE_OUTCOMES
        or bool(labels & MACHINE_CLASSIFICATIONS)
        or has_answering_machine_text
    )
    if machine:
        return "machine"
    if labels & HUMAN_CLASSIFICATIONS:
        return "human"
    return None


def manual_machine_path(settings: Settings, analysis_type: str, month: str) -> Path:
    return settings.project_root / "data" / "review" / f"{analysis_type}_{month}_manual_machine.txt"


def manual_human_path(settings: Settings, analysis_type: str, month: str) -> Path:
    return settings.project_root / "data" / "review" / f"{analysis_type}_{month}_manual_human.txt"


def apply_manual_override(
    sample: str | None,
    call_id: str,
    manual_machine: set[str],
    manual_human: set[str],
) -> str | None:
    """Ручная проверка важнее автоматической метки. Робот важнее, если номер в обоих списках."""

    if call_id in manual_machine:
        return "machine"
    if call_id in manual_human:
        return "human"
    return sample


def load_manual_machine_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")}


def remember_manual_machine_ids(path: Path, call_ids: list[str]) -> set[str]:
    found = load_manual_machine_ids(path)
    found.update(call_ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sorted(found)) + "\n", encoding="utf-8")
    return found


def sample_paths(settings: Settings, analysis_type: str, month: str, sample: str) -> tuple[Path, Path]:
    slug = f"{analysis_type}_{month}_{sample}"
    manifest = settings.project_root / "data" / "manifests" / f"{slug}.csv"
    audio = settings.project_root / "data" / "audio" / analysis_type / month / sample
    return manifest, audio


def _matched_labels(record: dict, sample: str) -> str:
    labels = set(record.get("classifications") or [])
    if sample == "machine":
        matched = labels & MACHINE_CLASSIFICATIONS
        if (record.get("outcome") or "") in MACHINE_OUTCOMES:
            matched.add(str(record.get("outcome")))
        if record.get("has_answering_machine_text"):
            matched.add("answering_machine_text")
    else:
        matched = labels & HUMAN_CLASSIFICATIONS
    return "|".join(sorted(matched))


def export_samples(
    settings: Settings,
    *,
    analysis_type: str,
    month: str,
    dry_run: bool,
) -> dict[str, dict[str, int | Path]]:
    start, end = month_bounds(month)
    with connect(settings) as connection:
        records = fetch_campaign_calls(
            connection,
            settings,
            analysis_type=analysis_type,
            from_date=start,
            to_date=end,
        )
    manual_machine = load_manual_machine_ids(manual_machine_path(settings, analysis_type, month))
    manual_human = load_manual_machine_ids(manual_human_path(settings, analysis_type, month))
    grouped: dict[str, list[dict]] = {"human": [], "machine": []}
    for record in records:
        sample = apply_manual_override(
            assign_sample(
                outcome=record.get("outcome"),
                classifications=list(record.get("classifications") or []),
                has_answering_machine_text=bool(record.get("has_answering_machine_text")),
            ),
            str(record.get("call_id")),
            manual_machine,
            manual_human,
        )
        if sample is None:
            continue
        row = row_from_db(record, bucket=settings.s3_bucket)
        row["sample"] = sample
        row["analysis_type"] = analysis_type
        row["classification_result"] = _matched_labels(record, sample)
        names = record.get("campaign_names") or []
        row["campaign_name"] = "|".join(str(name) for name in names)
        grouped[sample].append(row)

    summary: dict[str, dict[str, int | Path]] = {}
    for sample, fresh in grouped.items():
        manifest, _audio = sample_paths(settings, analysis_type, month, sample)
        existing = read_manifest(manifest)
        if sample == "human" and manual_machine:
            existing = [row for row in existing if row["call_id"] not in manual_machine]
        if sample == "machine" and manual_human:
            existing = [row for row in existing if row["call_id"] not in manual_human]
        merged, added = merge_rows(existing, fresh)
        if not dry_run:
            write_manifest(manifest, merged)
        summary[sample] = {
            "selected": len(fresh),
            "added": 0 if dry_run else added,
            "manifest_rows": len(existing) if dry_run else len(merged),
            "manifest": manifest,
        }
    summary["seen"] = {"selected": len(records), "added": 0, "manifest_rows": 0, "manifest": Path()}
    return summary


def relocate_manual_machine(
    settings: Settings,
    *,
    analysis_type: str,
    month: str,
    call_ids: list[str],
) -> dict[str, list[str]]:
    """Переносит проверенные вручную звонки из human в machine."""

    human_manifest, human_audio = sample_paths(settings, analysis_type, month, "human")
    machine_manifest, machine_audio = sample_paths(settings, analysis_type, month, "machine")
    human_rows = read_manifest(human_manifest)
    machine_rows = read_manifest(machine_manifest)
    human_by_id = {row["call_id"]: row for row in human_rows}
    machine_ids = {row["call_id"] for row in machine_rows}
    moved: list[str] = []
    missing: list[str] = []
    machine_audio.mkdir(parents=True, exist_ok=True)

    for call_id in call_ids:
        row = human_by_id.get(call_id)
        if row is None:
            if call_id in machine_ids:
                moved.append(call_id)
            else:
                missing.append(call_id)
            continue
        source = Path(row["local_path"]) if row.get("local_path") else human_audio / f"{call_id}.wav"
        if not source.is_file():
            source = human_audio / f"{call_id}.wav"
        destination = machine_audio / f"{call_id}.wav"
        if source.is_file():
            if destination.exists() and source.resolve() != destination.resolve():
                destination.unlink()
            if source.resolve() != destination.resolve():
                source.replace(destination)
        elif not destination.is_file():
            missing.append(call_id)
            continue
        label = row.get("classification_result") or ""
        if "manual_review" not in label.split("|"):
            row["classification_result"] = f"{label}|manual_review" if label else "manual_review"
        row["sample"] = "machine"
        row["local_path"] = str(destination)
        if call_id not in machine_ids:
            machine_rows.append(row)
            machine_ids.add(call_id)
        human_by_id.pop(call_id, None)
        moved.append(call_id)

    order = [row["call_id"] for row in human_rows if row["call_id"] in human_by_id]
    write_manifest(human_manifest, [human_by_id[call_id] for call_id in order])
    write_manifest(machine_manifest, machine_rows)
    remember_manual_machine_ids(manual_machine_path(settings, analysis_type, month), moved)
    return {"moved": moved, "missing": missing}
