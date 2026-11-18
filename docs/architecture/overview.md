# Архитектура

Репозиторий держит четыре самостоятельных компонента рядом. Общего пакета, который импортирует их все, нет. Production-код `genai-agents-worker` и `genai-agents-backend` сюда не входит и этим проектом не изменяется.

## Почему компоненты не вложены в `src/`

Ориентир из плана сборки предлагал `src/dataset_collector`, `src/classical_anti_spoofing`, `src/spectra0`, `src/synthetic_speech_dataset`. Фактический код так не устроен.

Каждый компонент запускается из своего корня и импортирует собственный пакет `src`:

- `from src.collector...` в collector;
- `from src.features...` и `from src.models...` в classical;
- скрипты Spectra лежат прямо в `src/run_inference.py`.

`synthetic-speech-dataset` ищет соседей через `repo_paths.component_dir`. Функция поднимается только до корня этого репозитория, где лежит `pyproject.toml` с именем `voice-anti-spoofing`, и отказывается брать каталог снаружи. Относительные пути конфигов проходят ту же проверку в `resolve_path`.

`classical-anti-spoofing/config/config.yaml` указывает на `../voice-dataset-collector/...` и `../voice-anti-spoofing-spectra0/...`.

Если переименовать каталоги или положить их на уровень глубже, эти пути сломаются. Поэтому в `voice-anti-spoofing` сохранены исходные имена и исходная глубина: родитель компонента — корень нового репозитория.

Алгоритмы, пороги и формулы при переносе не переписывались.

## Роли

| Компонент | Роль | Чего компонент не делает |
| --- | --- | --- |
| `voice-dataset-collector` | Read-only доступ к PostgreSQL и объектному хранилищу, манифесты и локальные WAV | Не импортирует worker, backend и Spectra |
| `classical-anti-spoofing` | Признаки телефонного канала 8 кГц и классические классификаторы | Не меняет Spectra-0 и не подключается к звонку |
| `voice-anti-spoofing-spectra0` | Локальный inference двух моделей Hugging Face | Не обучает модели и в пилоте не применяет порог |
| `synthetic-speech-dataset` | Разметка происхождения, офлайн-метрики и текстовые отчёты | Не заменяет AMD в живом звонке |

Связка только файловая. Один компонент читает CSV или WAV другого по относительному пути либо запускается его интерпретатором, чтобы разрешился нужный пакет `src`. Это записано в `origin_retrain.py`, `new_classical_scores.py` и `cpu_inference.py`.

## Две разные задачи

1. Старая: `human` против `machine` по метке кампании B2X. Это живой диалог против автоответчика. На ней обучены классические модели и выбран порог Spectra-0 1.02.
2. Новая: происхождение речи абонента. Метки задаёт `synthetic-speech-dataset/src/evaluation/dataset.py`: `human_live`, `human_recorded`, `synthetic_tts`, `synthetic_voice_conversion`, `synthetic_playback`, `mixed`, `unknown`.

`human_recorded` не переименовывается в синтетику. `mixed` и `unknown` не входят в бинарный знаменатель. Канал собственного робота в сравнение происхождения не входит: это TTS системы на другом канале.

## Где лежит решение Ensemble A

Определение голоса A и соседних комбинаций B, C, D и «2 из 2» написано в:

- `synthetic-speech-dataset/reports/voting_combinations_analysis.md`
- `synthetic-speech-dataset/reports/ensemble_a_error_analysis.md`
- `synthetic-speech-dataset/src/research/model_assisted_annotation/rules.py`
- `synthetic-speech-dataset/src/research/model_assisted_annotation/voting_combinations.py`

Голос A: Spectra-0, SVM new, LightGBM new. Synthetic, если synthetic сказали минимум две модели. Пороги берутся из `rules.THRESHOLDS` и `rules.WIDTHS`. При сборке репозитория эти константы не редактировались.

Отдельно существует более ранний baseline Spectra-0: зазор logit меньше 1.02. Это другой score и другой порог. Его источник — `classical-anti-spoofing/config/config.yaml` и `synthetic-speech-dataset/src/evaluation/metrics.py`. Оба определения сохранены рядом, без сведения к одному числу.

## Что сознательно отсутствует

- Копии worker и backend.
- Веса, аудио, большие таблицы и результаты inference в CSV.
- Объединённый модельный класс, в котором Spectra и LightGBM стали бы одним кодом.
- Вложенные `.git` исходных репозиториев.
