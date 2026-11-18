-- Основная выборка звонков для manifest.
-- Параметры передаются отдельно, значения в текст запроса не подставляются.
--
-- $1  from_date          timestamptz/timestamp, необязательно. Граница created_at, включительно.
-- $2  to_date            верхняя граница created_at, не включая этот момент.
-- $3  goal_statuses      text[]. В таблице calls нет колонки status.
--                        --status в CLI фильтрует goal_status:
--                        goal_achieved, goal_not_achieved, goal_callback, undetermined.
-- $4  outcomes           text[], необязательно. Свободная строка outcome.
-- $5  end_reasons        text[], необязательно. Причина завершения end_reason.
-- $6  min_duration_s     int. calls.duration хранится в секундах.
-- $7  max_duration_s     int. Секунды, включительно.
-- $8  exclude_ids        uuid[]. Уже выгруженные call_id, чтобы не дублировать manifest.
-- $9  limit              размер партии, по умолчанию 200.
-- $10 offset             сдвиг внутри одного и того же фильтра.
--
-- По умолчанию приложение добавляет:
--   with_audio IS TRUE   — worker выставил флаг после сохранения WAV;
--   is_finalized IS TRUE — звонок доведён до финальной публикации.
-- Отсутствие строк в call_metrics / call_audio_metrics звонок не исключает:
-- backend не требует эти метрики, чтобы запись существовала.
--
-- s3_bucket в БД не хранится. Приложение подставляет S3_BUCKET из окружения.
-- s3_key вычисляется как calls/{external_id}/{external_id}.wav.
-- duration_ms в CSV = duration * 1000 и повторяет секундную точность колонки duration.

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
FROM calls AS c
WHERE ($1::timestamp IS NULL OR c.created_at >= $1)
  AND ($2::timestamp IS NULL OR c.created_at < $2)
  AND ($3::text[] IS NULL OR c.goal_status = ANY($3))
  AND ($4::text[] IS NULL OR c.outcome = ANY($4))
  AND ($5::text[] IS NULL OR c.end_reason = ANY($5))
  AND ($6::int IS NULL OR c.duration >= $6)
  AND ($7::int IS NULL OR c.duration <= $7)
  AND c.with_audio IS TRUE
  AND c.is_finalized IS TRUE
  AND ($8::uuid[] IS NULL OR c.id <> ALL($8))
ORDER BY c.created_at ASC, c.id ASC
LIMIT $9
OFFSET $10;
