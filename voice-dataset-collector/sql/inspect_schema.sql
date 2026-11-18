-- Исследование схемы, на которую опирается collector.
-- Запросы только на чтение. Запускать от имени пользователя с SELECT.
--
-- По моделям backend (app/db/models/calls.py) рабочая таблица — calls
-- в схеме public, если DB_SCHEMA не переопределён.
-- Отдельных колонок status, s3_bucket и s3_key в модели нет.

SELECT table_schema, table_name
FROM information_schema.tables
WHERE table_schema = current_schema()
  AND table_name IN (
      'calls',
      'call_audio_metrics',
      'call_audio_events',
      'call_metrics',
      'call_turns',
      'messages'
  )
ORDER BY table_name;

SELECT column_name, data_type, udt_name, is_nullable
FROM information_schema.columns
WHERE table_schema = current_schema()
  AND table_name = 'calls'
ORDER BY ordinal_position;

-- Поля, которые collector читает для manifest.
-- duration — целые секунды. with_audio — флаг, что запись была сохранена.
-- external_id — room name; ключ объекта: calls/{external_id}/{external_id}.wav

SELECT
    id,
    created_at,
    timestamp,
    duration,
    external_id,
    with_audio,
    is_finalized,
    goal_status,
    outcome,
    end_reason,
    disconnected_by,
    direction
FROM calls
ORDER BY created_at DESC
LIMIT 20;
