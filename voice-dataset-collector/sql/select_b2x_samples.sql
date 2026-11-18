-- Две выборки B2X за календарный месяц Europe/Moscow.
-- Значения в запрос не подставляются: $1 analysis_type, $2 начало, $3 конец.
--
-- machine, если есть хотя бы одно:
--   calls.outcome IN ('answering_machine', 'anti_spam_bot')
--   classification_result IN ('автоответчик', 'answering_machine', 'anti_spam_bot')
--   calls.collected_data содержит answering_machine_text
--
-- human, если machine нет и classification_result один из:
--   лид, интерес_без_данных, отказ_лпр, не_лпр,
--   не_та_организация, перезвон, асинхронная_связь
--
-- Молчание, недозвон, «положил трубку» и «неизвестно» не входят ни в одну выборку.
-- Телефон не выбирается. external_id нужен только как часть ключа S3.

SELECT
    c.id::text AS call_id,
    c.created_at,
    c.timestamp,
    c.duration AS duration_s,
    c.external_id,
    c.outcome,
    camp.analysis_type,
    array_agg(DISTINCT r.classification_result) FILTER (WHERE r.classification_result IS NOT NULL) AS classifications,
    array_agg(DISTINCT camp.name) AS campaign_names,
    coalesce(bool_or(c.collected_data ? 'answering_machine_text'), false) AS has_answering_machine_text
FROM calls AS c
JOIN call_campaign_results AS r ON r.call_id = c.id
JOIN call_campaigns AS camp ON camp.id = r.call_campaign_id
WHERE camp.analysis_type = $1
  AND c.with_audio IS TRUE
  AND c.is_finalized IS TRUE
  AND c.timestamp >= $2
  AND c.timestamp < $3
GROUP BY c.id, camp.analysis_type
ORDER BY min(c.timestamp), c.id;
