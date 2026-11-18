# Проверка партии batch_001

Партия собрана для ручной разметки. Она не является обучающей выборкой.
Прежние метки human и machine сохранены в `prior_sample` и не перенесены в `primary_label`.

- звонков: 33
- сегментов: 0
- причины отбора: {'human_pool': 12, 'machine_answering': 12, 'machine_live_outcome': 8, 'label_conflict': 1}
- прежние split: {'train': 22, 'validation': 7, 'none': 3, 'test': 1}
- роль для обучения: {'not_for_training': 32, 'locked_baseline': 1}
- метки происхождения: {'synthetic_tts': 17, 'human_live': 13, 'synthetic_playback': 1, 'synthetic_voice_conversion': 2}
- дополнительный признак: {'ai_secretary': 5, 'none': 21, 'anti_spam': 2, 'voicemail': 5}
- уверенность: {'medium': 7, 'high': 25, 'low': 1}
- статус проверки: {'reviewed': 32, 'unreviewed': 1}
- источник метки: {'manual_listening': 33}
- метка по группе отбора: {'machine_answering': {'synthetic_tts': 9, 'synthetic_playback': 1, 'synthetic_voice_conversion': 2}, 'human_pool': {'human_live': 12}, 'machine_live_outcome': {'synthetic_tts': 8}, 'label_conflict': {'human_live': 1}}
- каналы записи: {'2': 33}
- канал абонента: {'0': 33}
- sample rate: {'8000': 33}
- codec: {'pcm_s16le': 33}

## Ошибки структуры

Структурных ошибок нет.

## Что в данных отсутствует

- campaign_id пуст у всех звонков: в выгрузке есть только campaign_name.
- call_status пуст у всех звонков: в таблице calls нет колонки status.
- speaker_id неизвестен.
- tts_generator абонента неизвестен. TTS нашего робота сюда не подставляется.
- voice_conversion_system неизвестна.
- playback_method неизвестен.
- Категория human_recorded в этой партии не подтверждена.
- Категория mixed в этой партии не подтверждена.
- Подтверждённые метки получены прослушиванием. Имени генератора в них нет.
- Метка сохранена, но статус проверки не reviewed: a113285a-037e-4aa9-bb15-28c5797cdee3.

## Метки с невысокой уверенностью или незакрытым статусом

- `03c03387-ccba-498d-adb0-2cb300d1062c`: synthetic_tts, ai_secretary, уверенность medium, статус reviewed
- `148f783c-45f6-47e5-ba1e-c5b07f602fc5`: synthetic_playback, anti_spam, уверенность medium, статус reviewed
- `610f1c8d-37b8-443a-b260-d79365a2568b`: synthetic_voice_conversion, ai_secretary, уверенность low, статус reviewed
- `63538d8d-cb8a-4085-bf74-3b0fefc3593a`: human_live, без доп. признака, уверенность medium, статус reviewed
- `9a208c8b-4bf1-48bf-ae9a-a248e0a8671a`: synthetic_tts, voicemail, уверенность medium, статус reviewed
- `a113285a-037e-4aa9-bb15-28c5797cdee3`: synthetic_tts, ai_secretary, уверенность high, статус unreviewed
- `aa410f9b-c042-4007-a599-51243be8de6a`: synthetic_tts, voicemail, уверенность medium, статус reviewed
- `ec7e6949-d970-43d8-8377-af5f0a063976`: synthetic_tts, voicemail, уверенность medium, статус reviewed
- `fbb05816-48f7-46ba-adbc-e9f96e011d76`: synthetic_tts, voicemail, уверенность medium, статус reviewed

## Сколько записей было доступно до квоты

- читаемые кандидаты по группам: {'human_pool': 1120, 'machine_answering': 278, 'machine_live_outcome': 35, 'label_conflict': 1}
- исключены из-за старых test или n100: {'human_pool': 108, 'machine_answering': 91, 'machine_live_outcome': 18, 'label_conflict': 0}
- machine без автоответчика и без живого исхода, в партию не брались: 62
- не открылись WAV: 0


## Что из этого можно размечать

Все строки — телефонные звонки B2X, `data_source = telephone_call`.
Метка ставится по каналу абонента. Канал нашего робота слушается отдельно и сам по себе звонок синтетическим не делает.
Происхождение в этой партии — прослушивание канала абонента одним разметчиком. Поля генератора, voice conversion и playback пустые, поэтому это не метка known_generator.
Исход `answering_machine` и живые исходы кампании остаются контекстом, не доказательством синтеза.

Звонки старых test и n100 в эту партию не отбирались, кроме конфликта метки, если он попадёт в исключённый split.
Их нельзя использовать для обучения и подбора порога нового детектора.

## Идентификаторы партии

- e77d23f1-4b04-4806-a61f-c2dc71af6c90
- fb3e9fb2-bf95-41ad-a9aa-5cfaa8db50fe
- bc4a2f3c-9745-49d5-8c57-f3b6155d7a94
- 0c2f9ea6-f19d-4c70-a423-591f2388635f
- 72853f81-5ae2-46c5-8bff-9aed82ae2322
- aa42b567-01a8-4748-9be7-2805e67dcc77
- 07a50ff8-9b29-41b4-ac8b-40652a952014
- 55dc62a9-e29c-451c-bd37-c0de4a7fd069
- 1f5385a9-f1a6-4a7d-a955-18ff6b36b972
- 63538d8d-cb8a-4085-bf74-3b0fefc3593a
- 6648a0cc-235b-4be1-9794-d600c2764cb0
- c29ef30e-6829-41b5-8664-e15b19c2c800
- fba7116e-fe01-42e0-9234-5192735197cc
- 610f1c8d-37b8-443a-b260-d79365a2568b
- c1311c50-59d3-4bc6-b2db-9b439a870010
- ec8b738c-3a7a-4cf4-8ce6-1338a3c50fba
- 148f783c-45f6-47e5-ba1e-c5b07f602fc5
- a113285a-037e-4aa9-bb15-28c5797cdee3
- fbb05816-48f7-46ba-adbc-e9f96e011d76
- aa410f9b-c042-4007-a599-51243be8de6a
- 03c03387-ccba-498d-adb0-2cb300d1062c
- bbdf70a6-3e54-4fb1-92ac-f04ea65651fe
- ec7e6949-d970-43d8-8377-af5f0a063976
- 9a208c8b-4bf1-48bf-ae9a-a248e0a8671a
- cb33adbb-dcca-429b-97c6-01ad065ccc37
- 48b6c2a3-55d3-4266-a4fd-a59a214e5e8d
- 8d7bfa96-7b2d-4e53-a683-c847980e489c
- 7e88bb1d-5990-4850-b31a-817eca1a88c5
- 1c76c0d9-225a-408b-9382-56a1d59ec1f5
- 67b3e6fa-e578-4b2d-b4bf-d2ec9577c736
- abdbdc2f-11de-4ed1-bcfc-f6c34b495f84
- 957e7a3b-bf1e-46fe-baa3-0ac9ba477026
- 1c9d5432-8bcc-4dc2-ac67-11ff015cf951
