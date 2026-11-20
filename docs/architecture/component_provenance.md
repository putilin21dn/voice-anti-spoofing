# Происхождение компонентов

Истории репозиториев не объединялись. В этот проект скопировано рабочее дерево нужных файлов. Каталоги `.git` источников не копировались. Исходные деревья при сборке не изменялись.

Лицензионных файлов `LICENSE`, `NOTICE` и `COPYING` в четырёх источниках нет. Корневой `LICENSE` не придумывался. Для Spectra-0 карточка компонента просит отдельно проверить Apache-2.0 и MIT на Hugging Face.

## Источники

| Компонент в этом репозитории | Исходный каталог | Git | Состояние на момент копирования |
| --- | --- | --- | --- |
| `voice-dataset-collector/` | `voice-dataset-collector` | да, ветка `main`, HEAD `ecd8161` (`Save the B2X call lists so the same files can be downloaded again.`), remote `origin.cursor.com/dmitry-putilin/voice-dataset-collector` | В копию кода вошли незакоммиченные правки `src/collector/samples.py` и `tests/test_collector.py`: ручной список human и приоритет ручной метки. Незакоммиченные CSV, review-списки и `data/samples/.../test20/` не копировались |
| `classical-anti-spoofing/` | `classical-anti-spoofing` | нет | Скопировано текущее дерево без `.venv`, логов, весов и таблиц |
| `voice-anti-spoofing-spectra0/` | `voice-anti-spoofing-spectra0` | да, ветка `main`, HEAD `5926bef` (`Save the Spectra scores and the list of weights to download.`), remote `origin.cursor.com/dmitry-putilin/voice-anti-spoofing-spectra0` | В копию вошла незакоммиченная правка `src/run_inference.py`: аргумент `--output` и имя metadata рядом с выбранным CSV. Файлы `data/output/*.csv` и metadata этого прогона не копировались |
| `synthetic-speech-dataset/` | `synthetic-speech-dataset` | нет | Скопированы код, конфиги, HTML разметки и текстовые отчёты |

Целевой репозиторий на момент сборки: `https://github.com/putilin21dn/voice-anti-spoofing.git`, ветка `main`, единственный коммит `1ed7daa` (`Initial commit`). Новый коммит этой сборкой не создавался.

## Что изменено при переносе

Код алгоритмов, пороги, метки и отчёты не редактировались. Отличия от исходного дерева:

1. Не скопированы секреты, аудио, таблицы, веса, архивы, логи, кэши, виртуальные окружения и вложенные `.git`.
2. В `voice-dataset-collector/.env.example` значение `S3_BUCKET` очищено. В исходном примере стояло конкретное имя бакета. Код `config/settings.py` не менялся: при пустой переменной он по-прежнему подставляет своё значение по умолчанию.
3. Корневой `.env.example` собран заново и содержит только имена переменных и пустые или безопасные placeholder-значения.
4. Файлы `.gitignore` компонентов не оставлены. В источниках они снова разрешали часть CSV. Действует только корневой `.gitignore`, и Git LFS в `.gitattributes` нет.
5. Компоненты лежат в корне нового репозитория под прежними именами. См. `docs/decisions/0001-component-layout.md`.

## Что скопировано

- Все `.py` вне `.venv` и `__pycache__`: collector 17, classical 51, spectra0 6, synthetic 76.
- README и Markdown-отчёты, включая каталоги `reports/` и короткие `summary.md` / `data_quality_report.md` из `synthetic-speech-dataset/evaluation/`.
- `requirements.txt`, SQL, YAML конфигов, HTML страниц разметки, `WHAT_TO_DOWNLOAD.txt`.
- Небольшие JSON: метрики и конфиги экспериментов classical, `column_groups.json`, `split_metadata.json`, короткие meta и config JSON в отчётах synthetic. Файлы крупнее 80 КБ, `*request.json` и `*threads_*.json` не копировались.

## Что не скопировано

Категории, а не полный перечень путей:

- `.env` collector. В нём были непустые `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_SCHEMA`, `S3_HOST`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, `S3_BUCKET`, `S3_SECURE`. Значения здесь не приводятся.
- Аудио `.wav` во всех четырёх проектах, включая клипы ручного прослушивания и архивы `shares/*.zip`.
- CSV манифестов, сплитов, признаков, предсказаний и аннотаций. В заголовках манифестов collector есть `external_id`, `s3_key`, `s3_bucket`.
- Веса `.safetensors` Spectra-0 и AASIST3. Классические `.joblib` из `classical-anti-spoofing/models/` скопированы: 18 файлов, варианты full, `no_energy`, `shape`, `voicing`, `mfcc`, `spectral`. Копии тех же bundle внутри `reports/experiment_*/model.joblib` не копировались. Три joblib `origin_retrain_20261113/models/` добавлены в Git отдельно.
- Рисунки `.png` и `.svg`, логи, `.venv`, кэш Hugging Face `.cache`.
- Списки идентификаторов `completed_call_ids.txt`, `feature_fingerprint.txt` и тексты `data/review/*.txt`.

## Секреты и персональные данные

Обнаружено и не перенесено:

| Где | Что |
| --- | --- |
| `voice-dataset-collector/.env` | Непустые параметры базы и ключи объектного хранилища |
| `voice-dataset-collector/data/manifests/*.csv` | Колонки `external_id`, `s3_key`, `s3_bucket`. Исходная заметка говорит, что в ключе встречается телефон |
| `synthetic-speech-dataset/src/research/model_assisted_annotation/callback_phones.py` | Код, который печатает телефон, извлечённый из `external_id`. Самих номеров в файле нет. Файл скопирован как исходник, поведение не менялось |

`tests/test_identity.py` использует синтетическую цифровую строку, чтобы проверить отбрасывание цифр из имени комнаты. Это фикстура теста, не выгрузка абонента.

Поиск по текстовым отчётам не нашёл отдельных телефонных номеров вне UUID и этой тестовой фикстуры. Ложные срабатывания на длинных числах внутри UUID и метрик в отчёты не переносились как секреты.

## Другие каталоги рядом

В родительской папке есть и другие деревья. Они в проект не включались:

| Каталог | Почему не включён |
| --- | --- |
| `genai-agents-worker`, `genai-agents-backend` | Запрещены к копированию. Это внешний production-код |
| `karpov_course`, `math_for_ds`, `ml-engineer-roadmap-2026` | Учебные материалы, не часть anti-spoofing |
| `lab-generative-frankbybasta`, `lab_quality_gate_v2` | Отдельные лабораторные работы |
| `my_venv`, `.vscode`, корневые `lab*.py` | Локальное окружение, настройки редактора и разовые скрипты |

## Неоднозначные файлы, которые не переносились

Они могут пригодиться для воспроизведения, но это таблицы, рисунки или веса. Они остались только в исходных каталогах:

- `classical-anti-spoofing/data/splits/*.csv` и `data/features/calls.csv`, `segments.csv` — сплиты и матрицы признаков. `split_metadata.json` и `column_groups.json` скопированы.
- `classical-anti-spoofing/reports/**/predictions.csv`, `feature_importance.csv`, рисунки ROC и confusion matrix. Рядом лежат скопированные `summary.md` и `metrics.json`.
- `voice-anti-spoofing-spectra0/models/**/model.safetensors` и `data/output/*.csv`.
- Базовые classical bundle из `models/` скопированы. Три joblib прогона 2026-11-13 (`svm`, `lightgbm`, `logistic`) затем добавлены в Git по тому же пути.
- `synthetic-speech-dataset/reports/ensemble_a/*.csv`, `reports/voting_combinations/*.csv`, крупные `*_features.csv` и `per_call_predictions.csv`.
- Короткие `reports/cpu_inference/*threads_*.json`: в них есть покомпонентные `call_id` и score восьми звонков. Текстовый разбор — в `reports/cpu_inference_analysis.md`, он скопирован.
- `evaluation/**` кроме перечисленных Markdown: CSV, JSON запросов и логи прогонов.
