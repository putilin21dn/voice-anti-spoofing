# voice-anti-spoofing

Локальный инженерный репозиторий для разработки, исследования и оценки обнаружения синтетической речи и автоматических звонков. Это исследовательский код. Он не подключён к production и не меняет AMD, ASR, VAD звонка, call flow или логику завершения вызова.

## Первый запуск

Из корня репозитория, Python 3.11 или новее (для classical лучше 3.12):

```powershell
python scripts/check_setup.py
python scripts/bootstrap.py
python -m shadow_replay shadow_replay/examples/human_call.json
```

`bootstrap.py` создаёт три `.venv`, ставит `requirements.txt` и запускает тесты. База, аудио и веса для этого не нужны. Файл `voice-dataset-collector/.env` создаётся из примера пустым. `python scripts/run_tests.py` повторяет те же тесты.

Один тест synthetic ищет локальный `confirmed_001` и без него падает. Это отсутствие данных, не поломка установки. Collector, classical и `shadow_replay` от этого файла не зависят.

Spectra на записи ставится отдельно: torch и `model.safetensors` в Git не входят. SVM, LightGBM и logistic ансамбля A лежат в `synthetic-speech-dataset/reports/origin_retrain_20261113/models/`. Команды и границы: [docs/operations/local_setup.md](docs/operations/local_setup.md).

Компоненты собраны из четырёх существующих проектов и оставлены самостоятельными каталогами. Так сохраняются их импорты `src.*` и обращения к соседним каталогам (`../classical-anti-spoofing`, `../voice-dataset-collector`, `../voice-anti-spoofing-spectra0`). `genai-agents-worker` и `genai-agents-backend` в этот репозиторий не входят.

## Архитектура

```text
voice-anti-spoofing/
├── voice-dataset-collector/        чтение PostgreSQL и скачивание WAV
├── classical-anti-spoofing/        признаки 8 кГц и классические классификаторы
├── voice-anti-spoofing-spectra0/   локальный прогон Spectra-0 и Spectra-AASIST3
├── synthetic-speech-dataset/       разметка происхождения, оценка и отчёты
├── docs/                           обзор, протокол, данные, запуск
├── data/                           пустые каталоги для локальных данных
├── artifacts/                      пустые каталоги для локальных артефактов
└── pyproject.toml                  группы зависимостей без общей установки
```

Поток данных, как он описан в исходных README:

1. `voice-dataset-collector` читает звонки и может скачать WAV в свой `data/`.
2. `classical-anti-spoofing` читает уже извлечённый клиентский канал и CSV из collector, считает акустические признаки и обучает Logistic Regression, SVM и LightGBM.
3. `voice-anti-spoofing-spectra0` считает оконные оценки Spectra-0 и AASIST3. Порог решения в этом пилоте не применяется.
4. `synthetic-speech-dataset` размечает происхождение речи и сравнивает уже посчитанные решения. Часть скриптов запускается интерпретатором classical-anti-spoofing.

Подробнее: [docs/architecture/overview.md](docs/architecture/overview.md). Откуда взят каждый каталог: [docs/architecture/component_provenance.md](docs/architecture/component_provenance.md).

## Компоненты

| Каталог | Что делает | Как запускать |
| --- | --- | --- |
| `voice-dataset-collector` | Read-only выборка звонков из PostgreSQL и скачивание WAV из объектного хранилища | `python src/main.py check-db`, `check-s3`, `export` |
| `classical-anti-spoofing` | LFCC, MFCC, CQCC, F0, voice quality, spectral features; Logistic Regression, RBF SVM, LightGBM | `python src/main.py inspect-data`, `create-splits`, `extract-features`, `train`, `evaluate`, `compare-spectra0` |
| `voice-anti-spoofing-spectra0` | Загрузка и оконный inference Spectra-0 и Spectra-AASIST3 | `python src/check_environment.py`, `download_model.py`, `run_inference.py`, `run_aasist3.py` |
| `synthetic-speech-dataset` | Локальная разметка, контроль качества партии, офлайн-сравнение моделей | `python src/main.py init-batch`, `validate`, `serve`; `python -m unittest discover -s tests` |

Команды выполняются из каталога компонента внутри этого репозитория. Код ищет соседние компоненты только под корнем `voice-anti-spoofing` и отказывается выходить наружу.

## Модели и алгоритмы

Здесь лежат реализации, а не веса.

**Spectra-0.** Код архитектуры: `voice-anti-spoofing-spectra0/models/spectra_0/model.py`. Источник весов в карточке компонента: Hugging Face `lab260/spectra_0`. Скрипт приводит волну к 16 кГц, применяет pre-emphasis 0.97 и режет речь на окна 64600 сэмплов. Выход 0 — spoof, выход 1 — bonafide. Softmax в пилоте не считается калиброванной вероятностью, и сам пилот порог не ставит.

**Spectra-AASIST3.** Код: `voice-anti-spoofing-spectra0/models/spectra_aasist3/model.py`. Источник весов: `lab260/Spectra-AASIST3`. В `src/evaluation/metrics.py` порог звонка AASIST3 явно не зафиксирован. Отдельные константы для более позднего голосования лежат в `synthetic-speech-dataset/src/research/model_assisted_annotation/rules.py` и при переносе не менялись.

**Классические модели.** Logistic Regression, RBF SVM и LightGBM на агрегатах звонка. Варианты матрицы признаков, которые есть в коде и отчётах: полный набор, `no_energy`, `shape`, `voicing`, `mfcc`, `spectral`. Отбор признаков и порог идут по train и validation. Зафиксированный baseline Spectra-0 в `classical-anti-spoofing/config/config.yaml`: средний зазор `bonafide_logit - spoof_logit`, spoof если зазор меньше 1.02. Этот порог выбран на n100 и на test20 заново не подбирается.

**Ensemble A.** В отчётах `synthetic-speech-dataset/reports/voting_combinations_analysis.md` и `ensemble_a_error_analysis.md` голос A — это Spectra-0, SVM new и LightGBM new. Synthetic, если так сказали минимум две модели из трёх. Logistic Regression в это правило не входит. Числовые пороги и ширина пограничной зоны записаны в `rules.py`. `unknown` и `mixed` в бинарный знаменатель не входят.

Старые метки `human` / `machine` — это «живой диалог против автоответчика». Метки происхождения — `human_live`, `human_recorded`, `synthetic_tts`, `synthetic_voice_conversion`, `synthetic_playback`, `mixed`, `unknown`. Они не смешиваются.

Числа экспериментов в этом README не пересказываются. Они остаются в исходных Markdown-отчётах компонентов.

## Датасеты

Аудио, манифесты и таблицы признаков в Git не входят. Относительные пути в конфигах сохранены, поэтому локальные файлы ожидаются там же, где их искали исходные проекты:

- выборка B2X: `voice-dataset-collector/data/samples/b2x/2026-11/samples.csv` и соседние `n100`, `test20`;
- признаки и сплиты: `classical-anti-spoofing/data/features/`, `classical-anti-spoofing/data/splits/`;
- оконные оценки Spectra: `voice-anti-spoofing-spectra0/data/output/`;
- аннотации происхождения: `synthetic-speech-dataset/data/annotations/` и `data/manifests/`.

Схема и правила: [docs/datasets/dataset_structure.md](docs/datasets/dataset_structure.md). Где хранить файлы на диске: [docs/operations/data_storage.md](docs/operations/data_storage.md).

В манифестах collector есть `external_id`, `s3_key` и `s3_bucket`. Исходный `WHAT_TO_DOWNLOAD.txt` пишет, что в `external_id` и `s3_key` встречается телефон. Эти CSV не копировались.

## Тесты

Безопасные проверки — существующие unit tests. Они не должны скачивать данные и не должны обучать модели. Тест `RealSelectionTests` в synthetic-speech-dataset сам пропускается, если локальный манифест B2X отсутствует.

Из каталога компонента, в его окружении:

```powershell
cd voice-dataset-collector
python -m unittest discover -s tests

cd ..\classical-anti-spoofing
python -m unittest discover -s tests

cd ..\synthetic-speech-dataset
python -m unittest discover -s tests
```

`voice-anti-spoofing-spectra0` своих unit tests не содержит. `python src/check_environment.py` только печатает версии и доступность CUDA; веса для этого не нужны, но модуль импортирует PyTorch.

Общий `Makefile` вызывает те же три команды через `make test-collector`, `make test-classical`, `make test-synthetic` и `make test`.

## Скрипты

Новых скриптов обучения и inference этот перенос не добавляет. Точки входа остались внутри компонентов:

- collector: `voice-dataset-collector/src/main.py`
- classical: `classical-anti-spoofing/src/main.py`
- Spectra: `voice-anti-spoofing-spectra0/src/run_inference.py`, `run_aasist3.py`, `download_model.py`
- разметка и оценка: `synthetic-speech-dataset/src/main.py`
- дневная выгрузка через collector: `synthetic-speech-dataset/fetch_day.py`

Исследовательские скрипты лежат в `synthetic-speech-dataset/src/research/`. Часть из них рассчитана на интерпретатор `classical-anti-spoofing`, потому что импортирует его пакет `src`. Запускать полный прогон, обучение или inference ради проверки переноса не нужно.

`synthetic-speech-dataset/src/research/model_assisted_annotation/callback_phones.py` умеет извлекать телефон из `external_id` и печатать его. Это исходное поведение, оно не менялось. На реальной базе его лучше не запускать из этого репозитория.

## Локальные модели

Веса в репозиторий не входят. Каталоги и ожидаемые файлы:

| Что | Куда положить локально | Откуда брать |
| --- | --- | --- |
| Spectra-0 | `voice-anti-spoofing-spectra0/models/spectra_0/model.safetensors` | Hugging Face `lab260/spectra_0`. `model.py` уже лежит рядом |
| Spectra-AASIST3 | `voice-anti-spoofing-spectra0/models/spectra_aasist3/model.safetensors` | Hugging Face `lab260/Spectra-AASIST3`. `model.py` уже лежит рядом |
| Encoder | кэш Hugging Face | `facebook/wav2vec2-xls-r-300m`, догружается при первом запуске |
| Logistic / SVM / LightGBM | `classical-anti-spoofing/models/` | уже лежат в репозитории: `logistic`, `svm`, `lightgbm` и варианты `no_energy`, `shape`, `voicing`, `mfcc`, `spectral` |
| Модели происхождения 2026-11-13 | `synthetic-speech-dataset/reports/origin_retrain_20261113/models/` | в Git: `svm.joblib`, `lightgbm.joblib`, `logistic.joblib` |

Скачивание: из каталога Spectra выполнить `python src/download_model.py` только когда веса действительно нужны. При переносе модели не скачивались.

В карточке Spectra-0 одновременно встречаются Apache-2.0 и MIT. Отдельного LICENSE у этих четырёх проектов нет. Перед использованием модели сверьте лицензию в репозитории Hugging Face.

## Локальные датасеты

Положите файлы по относительным путям из `classical-anti-spoofing/config/config.yaml` и `synthetic-speech-dataset/config/config.yaml`, либо поправьте пути в своей локальной копии конфига. Исходные CSV и WAV оставлены в прежних каталогах на этой машине и в новый репозиторий не копировались.

Когорта метрик голоса A — 818 `call_id` в `synthetic-speech-dataset/data/cohorts/metrics_20261111_16.txt`. Сами wav качает `synthetic-speech-dataset/fetch_metrics_cohort.py` из базы и S3 в уже указанные относительные пути. Команда и границы для агента: skill `score-recorded-calls`.

Пустые `data/raw`, `data/interim`, `data/processed`, `data/external` — место для новой раскладки, если она понадобится. Текущие скрипты читают пути внутри компонентов, а не эти четыре каталога.

## Локальные артефакты

Компоненты пишут результаты в свои каталоги:

- `classical-anti-spoofing/reports/`, `logs/`, `models/`
- `voice-anti-spoofing-spectra0/data/output/`
- `synthetic-speech-dataset/reports/`, `evaluation/`, `logs/`

Корневые `artifacts/models`, `artifacts/checkpoints`, `artifacts/predictions`, `artifacts/experiments` зарезервированы и пока пустые. Они в Git не коммитятся.

Уже посчитанные текстовые отчёты перенесены: Markdown, небольшие JSON с метриками и YAML конфигов экспериментов. Вместе с ними в репозитории лежат 18 классических bundle `.joblib` в `classical-anti-spoofing/models/`. CSV предсказаний, рисунки и веса Spectra `.safetensors` не переносились.

## Что нельзя коммитить

Корневой `.gitignore` закрывает типы файлов и каталоги, а не только размер. Git LFS не настроен.

Не коммитить аудио, CSV и другие таблицы, массивы, веса, архивы, логи, кэши, виртуальные окружения, `.env` и локальные выгрузки. Подробный список: [docs/operations/data_storage.md](docs/operations/data_storage.md).

Исключение из правил возможно только отдельным решением и добавлением файла принудительно. Автоматически большие файлы в репозиторий не принимаются.

## Ограничения

- Единое виртуальное окружение не создавалось. У компонентов разные `requirements.txt`. Жёсткого конфликта закреплённых версий в этих файлах нет: ограничения заданы нижними границами. Spectra дополнительно требует PyTorch с CUDA-индексом, и этот пакет в `requirements.txt` не записан. Исследовательские скрипты synthetic-speech-dataset рассчитаны на интерпретатор classical-anti-spoofing.
- Python: classical указывает 3.12, Spectra — 3.11 или 3.12. `pyproject.toml` требует `>=3.11`.
- Истории Git четырёх источников не склеивались. У classical-anti-spoofing и synthetic-speech-dataset своего `.git` не было.
- В collector и Spectra в копию вошло рабочее дерево, включая незакоммиченные правки исходников. Сами исходные репозитории не изменялись.
- `voice-dataset-collector/config/settings.py` по-прежнему подставляет имя бакета по умолчанию, если `S3_BUCKET` пуст. Секреты и файл `.env` не копировались. В `.env.example` значение бакета очищено.
- CQCC считается spafe без подъёма частоты до 16 кГц. Это ограничение записано в исходном README classical.
- Идентификатора говорящего и имени TTS-генератора в разметке нет. Разрез по людям и по невиданным генераторам исходные отчёты не делают.
- На n100 выбирался порог Spectra-0 1.02, поэтому n100 не является независимым тестом этого порога. Независимый старый набор — test20.
- Классические модели обучены на задаче «живой диалог против автоответчика». Их score не является обученным детектором синтеза.
- Лицензионного файла в источниках нет. Корневой `LICENSE` поэтому не создавался.

Установка окружений: [docs/operations/local_setup.md](docs/operations/local_setup.md). Протокол метрик: [docs/evaluation/evaluation_protocol.md](docs/evaluation/evaluation_protocol.md).
