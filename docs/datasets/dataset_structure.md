# Структура данных

Аудио и таблицы в Git не хранятся. Ниже — схема, которую задают конфиги и README источников. Файлы нужно положить локально по этим относительным путям.

## Идентификаторы

Рабочий идентификатор в аналитике — `call_id`.

`classical-anti-spoofing/src/features/schema.py` запрещает колонки `external_id`, `s3_key`, `s3_bucket`, `phone`, `msisdn`, `ani`, `dnis`. Тот же запрет повторён в исследовательских скриптах synthetic-speech-dataset. Манифест collector эти поля содержит, потому что по ним скачивается запись. Такой манифест в новый репозиторий не копировался.

Телефон в обычный лог collector, по его README, не пишется. Исключение — `callback_phones.py`, который специально печатает номер.

## Collector

Корень: `voice-dataset-collector/`.

| Путь | Назначение |
| --- | --- |
| `data/manifests/calls_manifest.csv` | Результат `export` |
| `data/manifests/b2x_2026-11_human.csv`, `b2x_2026-11_machine.csv` | Списки B2X. В заголовке есть `external_id`, `s3_key`, `s3_bucket` |
| `data/audio/` | Скачанные WAV |
| `data/samples/b2x/2026-11/samples.csv` | Уже извлечённый клиентский канал. Заголовок: `call_id`, `sample`, `client_channel`, `source_channels`, `sample_rate`, `sample_width`, `duration_ms`, `wav_ok`, `error`, `local_path` |
| `data/samples/b2x/2026-11/n100/samples.csv` | Набор, на котором выбирался порог Spectra-0 |
| `data/samples/b2x/2026-11/test20/samples.csv` | Зафиксированный test. В обучении и подборе порога не участвует |
| `data/review/` | Ручные списки и клипы разногласий моделей |

Метка `sample` здесь — `human` или `machine` кампании, не имя генератора. Пустой `unknown` в текущей выборке, по README classical, не участвует в обучении.

Ключ объекта, как он описан в README collector: `calls/{external_id}/{external_id}.wav`. Промежуточные `*_raw_*.mp4` collector не скачивает.

## Classical

Корень: `classical-anti-spoofing/`. Пути заданы в `config/config.yaml`.

Аудио: клиентский канал, 8 кГц, 16 бит, mono. Повышение до 16 кГц выключено (`allow_upsample: false`). Верхняя частота признаков не выше 4000 Гц.

Сплит по `call_id`, seed 42, доля validation 0.2. `test` — это test20. `spectra_threshold` — это n100. Train и validation берутся из остальных звонков.

`data/splits/split_metadata.json` скопирован и фиксирует правила: test не участвует в обучении и выборе порога; на n100 выбран порог Spectra-0 1.02; идентификатора говорящего и поля TTS-генератора нет.

Группы колонок признаков: `data/features/column_groups.json`. Сами `calls.csv` и `segments.csv` не переносились.

VAD в README classical: WebRTC VAD mode 2, кадр 20 мс, склейка паузы до 300 мс. Сегмент короче 500 мс сохраняется и не идёт в признаки.

## Spectra

Корень: `voice-anti-spoofing-spectra0/`.

| Путь | Назначение |
| --- | --- |
| `data/input/` | WAV для ручного прогона |
| `data/output/spectra0_results.csv` | Окна Spectra-0. Этот CSV не скопирован |
| `data/output/spectra0_test20_results.csv` | Прогон test20. Не скопирован |
| `models/spectra_0/model.py` | Архитектура, в Git |
| `models/spectra_0/model.safetensors` | Веса, только локально |
| `models/spectra_aasist3/` | То же для AASIST3 |

Окно: 64600 сэмплов при 16 кГц, pre-emphasis 0.97. Канал по умолчанию в README — первый. Каналы клиента и бота не смешиваются.

## Происхождение речи

Корень: `synthetic-speech-dataset/`.

Партия `batch_001` — проверка инструмента. В README у её строк `training_role = not_for_training`. Старые test и n100 в отбор этой партии не входят. До прослушивания `primary_label = unknown`.

Подтверждённые и дневные партии описываются конфигами `config/day_*.yaml`, `config/priority_20261111_20261112.yaml`, `config/callback_20261116.yaml`, `config/evaluation.yaml`. Сами CSV аннотаций и транскриптов не копировались.

Категории из `src/evaluation/dataset.py`:

- человек: `human_live`, `human_recorded`;
- синтетика: `synthetic_tts`, `synthetic_voice_conversion`, `synthetic_playback`;
- отдельно: `mixed`, `unknown`.

Один `call_id` должен попадать только в один split. Если появится `speaker_id`, один говорящий не должен быть одновременно в train и test. Это требование плана в `reports/modeling_plan.md`, а не уже построенный split по людям: поля говорящего в данных нет.

## Корневые каталоги `data/`

`data/raw`, `data/interim`, `data/processed`, `data/external` пустые. Текущие конфиги их не читают. Они оставлены как место для будущей локальной раскладки и закрыты `.gitignore`.
