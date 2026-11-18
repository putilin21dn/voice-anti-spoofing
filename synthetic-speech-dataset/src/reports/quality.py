from __future__ import annotations

import json

from src.data.validation import validate_dataset
from src.utils.config import resolve_path


def _origin_line(check: dict) -> str:
    labels = check.get("labels") or {}
    confirmed = sum(count for label, count in labels.items() if label not in {"unknown", ""})
    if confirmed == 0:
        return "До прослушивания происхождение голоса — `unknown`."
    return (
        "Происхождение в этой партии — прослушивание канала абонента одним разметчиком. "
        "Поля генератора, voice conversion и playback пустые, поэтому это не метка known_generator."
    )


def render_quality(cfg: dict, check: dict) -> str:
    lines = [
        "# Проверка партии batch_001",
        "",
        "Партия собрана для ручной разметки. Она не является обучающей выборкой.",
        "Прежние метки human и machine сохранены в `prior_sample` и не перенесены в `primary_label`.",
        "",
        f"- звонков: {check['rows']}",
        f"- сегментов: {check['segments']}",
        f"- причины отбора: {check['reasons']}",
        f"- прежние split: {check['locked_splits']}",
        f"- роль для обучения: {check['training_roles']}",
        f"- метки происхождения: {check['labels']}",
        f"- дополнительный признак: {check['secondary']}",
        f"- уверенность: {check['confidence']}",
        f"- статус проверки: {check['review_status']}",
        f"- источник метки: {check['label_sources']}",
        f"- метка по группе отбора: {check['by_reason']}",
        f"- каналы записи: {check['channel_counts']}",
        f"- канал абонента: {check['client_channels']}",
        f"- sample rate: {check['sample_rates']}",
        f"- codec: {check['codecs']}",
        "",
        "## Ошибки структуры",
        "",
    ]
    if check["errors"]:
        lines.extend(f"- {item}" for item in check["errors"])
    else:
        lines.append("Структурных ошибок нет.")
    lines.extend(["", "## Что в данных отсутствует", ""])
    lines.extend(f"- {item}" for item in check["findings"])
    soft = check.get("soft_labels") or []
    if soft:
        lines.extend(["", "## Метки с невысокой уверенностью или незакрытым статусом", ""])
        for row in soft:
            secondary = row["secondary_label"] or "без доп. признака"
            lines.append(
                f"- `{row['call_id']}`: {row['primary_label']}, {secondary}, "
                f"уверенность {row['label_confidence'] or 'не указана'}, статус {row['review_status']}"
            )
    stats_path = resolve_path(cfg, cfg["paths"]["manifest_out"]).with_name("batch_001_stats.json")
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        lines.extend(
            [
                "",
                "## Сколько записей было доступно до квоты",
                "",
                f"- читаемые кандидаты по группам: {stats.get('pool_available')}",
                f"- исключены из-за старых test или n100: {stats.get('pool_excluded_locked')}",
                f"- machine без автоответчика и без живого исхода, в партию не брались: {stats.get('other_machine')}",
                f"- не открылись WAV: {len(stats.get('unreadable') or [])}",
                "",
            ]
        )
    lines.extend(
        [
            "",
            "## Что из этого можно размечать",
            "",
            "Все строки — телефонные звонки B2X, `data_source = telephone_call`.",
            "Метка ставится по каналу абонента. Канал нашего робота слушается отдельно и сам по себе звонок синтетическим не делает.",
            _origin_line(check),
            "Исход `answering_machine` и живые исходы кампании остаются контекстом, не доказательством синтеза.",
            "",
            "Звонки старых test и n100 в эту партию не отбирались, кроме конфликта метки, если он попадёт в исключённый split.",
            "Их нельзя использовать для обучения и подбора порога нового детектора.",
            "",
            "## Идентификаторы партии",
            "",
        ]
    )
    for call_id in check["manifest_ids"]:
        lines.append(f"- {call_id}")
    lines.append("")
    return "\n".join(lines)


def write_quality_report(cfg: dict) -> tuple[dict, str]:
    check = validate_dataset(cfg)
    text = render_quality(cfg, check)
    destination = resolve_path(cfg, cfg["paths"]["quality_report"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return check, str(destination)
