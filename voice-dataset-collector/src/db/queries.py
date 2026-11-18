from __future__ import annotations

from datetime import datetime
from typing import Any

from psycopg import sql
from psycopg.rows import dict_row

from config.settings import Settings


def recording_key(external_id: str) -> str:
    """Ключ финального WAV. Так его строят worker и backend.

    worker: calls/{room_name}/{room_name}.wav, а room_name пишется в calls.external_id.
    backend: calls/{call.external_id}/{call.external_id}.wav в отдаче записи,
    транскрибации и audio quality.
    """

    return f"calls/{external_id}/{external_id}.wav"


def build_calls_query(
    settings: Settings,
    *,
    from_date: datetime | None,
    to_date: datetime | None,
    goal_statuses: list[str],
    outcomes: list[str],
    end_reasons: list[str],
    min_duration_s: int | None,
    max_duration_s: int | None,
    require_audio: bool,
    require_finalized: bool,
    exclude_ids: list[str],
    limit: int,
    offset: int,
) -> tuple[sql.Composed, dict[str, Any]]:
    """Собирает выборку calls с привязанными параметрами.

    calls.duration хранится в секундах (worker передаёт int(monotonic delta) и пишет duration=%ds).
    duration_ms в результате — производное значение duration * 1000, не длительность WAV.
    Колонки status, s3_bucket и s3_key в таблице calls нет.
    """

    table = sql.SQL("{}.{}").format(sql.Identifier(settings.db_schema), sql.Identifier("calls"))
    conditions: list[sql.Composable] = []
    params: dict[str, Any] = {"limit": limit, "offset": offset}

    if from_date is not None:
        conditions.append(sql.SQL("c.created_at >= %(from_date)s"))
        params["from_date"] = from_date
    if to_date is not None:
        conditions.append(sql.SQL("c.created_at < %(to_date)s"))
        params["to_date"] = to_date
    if goal_statuses:
        conditions.append(sql.SQL("c.goal_status = ANY(%(goal_statuses)s)"))
        params["goal_statuses"] = goal_statuses
    if outcomes:
        conditions.append(sql.SQL("c.outcome = ANY(%(outcomes)s)"))
        params["outcomes"] = outcomes
    if end_reasons:
        conditions.append(sql.SQL("c.end_reason = ANY(%(end_reasons)s)"))
        params["end_reasons"] = end_reasons
    if min_duration_s is not None:
        conditions.append(sql.SQL("c.duration >= %(min_duration_s)s"))
        params["min_duration_s"] = min_duration_s
    if max_duration_s is not None:
        conditions.append(sql.SQL("c.duration <= %(max_duration_s)s"))
        params["max_duration_s"] = max_duration_s
    if require_audio:
        conditions.append(sql.SQL("c.with_audio IS TRUE"))
    if require_finalized:
        conditions.append(sql.SQL("c.is_finalized IS TRUE"))
    if exclude_ids:
        conditions.append(sql.SQL("c.id <> ALL(%(exclude_ids)s::uuid[])"))
        params["exclude_ids"] = exclude_ids

    where = sql.SQL(" AND ").join(conditions) if conditions else sql.SQL("TRUE")
    query = sql.SQL(
        """
        SELECT
            c.id::text AS call_id,
            c.created_at,
            c.timestamp,
            c.duration AS duration_s,
            (c.duration * 1000) AS duration_ms,
            c.external_id,
            c.with_audio,
            c.is_finalized,
            c.goal_status,
            c.outcome,
            c.end_reason,
            c.disconnected_by,
            c.direction::text AS direction
        FROM {table} AS c
        WHERE {where}
        ORDER BY c.created_at ASC, c.id ASC
        LIMIT %(limit)s
        OFFSET %(offset)s
        """
    ).format(table=table, where=where)
    return query, params


def fetch_campaign_calls(
    connection,
    settings: Settings,
    *,
    analysis_type: str,
    from_date: datetime,
    to_date: datetime,
) -> list[dict[str, Any]]:
    """Звонки кампании с классификациями. Телефон не выбирается.

    Один call_id — одна строка. Классификации всех результатов кампании
    собираются в массив, решение human/machine принимает вызывающий код.
    """

    calls = sql.SQL("{}.{}").format(sql.Identifier(settings.db_schema), sql.Identifier("calls"))
    results = sql.SQL("{}.{}").format(sql.Identifier(settings.db_schema), sql.Identifier("call_campaign_results"))
    campaigns = sql.SQL("{}.{}").format(sql.Identifier(settings.db_schema), sql.Identifier("call_campaigns"))
    query = sql.SQL(
        """
        SELECT
            c.id::text AS call_id,
            c.created_at,
            c.timestamp,
            c.duration AS duration_s,
            (c.duration * 1000) AS duration_ms,
            c.external_id,
            c.with_audio,
            c.is_finalized,
            c.goal_status,
            c.outcome,
            c.end_reason,
            c.disconnected_by,
            c.direction::text AS direction,
            coalesce(bool_or(c.collected_data ? 'answering_machine_text'), false) AS has_answering_machine_text,
            coalesce(
                array_agg(DISTINCT r.classification_result) FILTER (WHERE r.classification_result IS NOT NULL),
                ARRAY[]::text[]
            ) AS classifications,
            coalesce(
                array_agg(DISTINCT camp.name) FILTER (WHERE camp.name IS NOT NULL),
                ARRAY[]::text[]
            ) AS campaign_names
        FROM {calls} AS c
        JOIN {results} AS r ON r.call_id = c.id
        JOIN {campaigns} AS camp ON camp.id = r.call_campaign_id
        WHERE camp.analysis_type = %(analysis_type)s
          AND c.with_audio IS TRUE
          AND c.is_finalized IS TRUE
          AND c.timestamp >= %(from_date)s
          AND c.timestamp < %(to_date)s
        GROUP BY c.id
        ORDER BY min(c.timestamp) ASC, c.id ASC
        """
    ).format(calls=calls, results=results, campaigns=campaigns)
    params = {"analysis_type": analysis_type, "from_date": from_date, "to_date": to_date}
    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute(query, params)
        return list(cursor.fetchall())


def fetch_calls(connection, settings: Settings, **filters) -> list[dict[str, Any]]:
    query, params = build_calls_query(settings, **filters)
    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute(query, params)
        return list(cursor.fetchall())
