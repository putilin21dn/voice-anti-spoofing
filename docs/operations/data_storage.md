# Локальное хранение данных

Git хранит код, конфиги, схемы и текстовые отчёты. Записи звонков, таблицы, веса и результаты прогонов остаются на диске и в коммит не попадают.

## Куда класть файлы, чтобы старые пути заработали

Конфиги не переписывались. Они ищут соседей относительно своего компонента.

| Содержимое | Локальный путь внутри этого репозитория |
| --- | --- |
| Манифесты и WAV collector | `voice-dataset-collector/data/` |
| Клиентские семплы B2X | `voice-dataset-collector/data/samples/b2x/2026-11/` |
| Признаки classical | `classical-anti-spoofing/data/features/` |
| Сплиты | `classical-anti-spoofing/data/splits/` |
| Bundle classical | `classical-anti-spoofing/models/` |
| Веса Spectra-0 и AASIST3 | `voice-anti-spoofing-spectra0/models/spectra_0/`, `models/spectra_aasist3/` |
| CSV окон Spectra | `voice-anti-spoofing-spectra0/data/output/` |
| Аннотации и дневные манифесты | `synthetic-speech-dataset/data/` |
| Новые `.joblib` происхождения | `synthetic-speech-dataset/reports/origin_retrain_20261113/models/` |
| Прогоны оценки | `synthetic-speech-dataset/evaluation/` |

На этой машине аудио и таблицы по-прежнему можно держать локально по путям внутри `voice-anti-spoofing`. Код не читает исходники из каталогов рядом с репозиторием. Симлинки наружу не создавались.

Корневые `data/raw`, `data/interim`, `data/processed`, `data/external` и `artifacts/*` текущими скриптами не читаются. Это запас под новую раскладку.

## Что закрыто `.gitignore`

Правила одновременно по расширению и по каталогу. Ограничения только по размеру нет. Git LFS нет.

Типы: аудио и видео, CSV/TSV/Parquet/JSONL, NumPy и pickle, HDF5, Feather, Arrow, веса PyTorch, ONNX, SafeTensors, архивы, логи, картинки прогонов. Исключение — классические bundle `classical-anti-spoofing/models/**/*.joblib`. Остальные `.joblib` по-прежнему игнорируются.

Каталоги: локальные `data/` компонентов, `artifacts/`, `logs/`, `.venv`, `__pycache__`, `.cache`, `shares/`.

Секреты: `.env` и `.env.*`, кроме `.env.example`.

Исключения сделаны только для уже перенесённых мелких схем: `column_groups.json`, `config_snapshot.yaml`, `split_metadata.json`, `.gitkeep`, `model.py`.

Файл можно добавить вопреки правилу только явной командой `git add -f`. Это должно быть отдельным решением, а не следствием обычного `git add`.

## Что уже лежит в Git как отчёт

Markdown и небольшие JSON/YAML экспериментов. В том числе технические отчёты Ensemble A и classical. Рядом с ними в исходных проектах остаются CSV и PNG. Их локальные пути совпадают с относительными ссылками внутри Markdown, если положить файлы на те же места.

Примеры, которые отчёты называют по имени и которые в Git не вошли:

- `synthetic-speech-dataset/reports/ensemble_a/` — CSV и рисунки;
- `synthetic-speech-dataset/reports/voting_combinations/` — CSV метрик, исключённых звонков и переворотов голоса;
- `classical-anti-spoofing/reports/experiment_*/predictions.csv` и `*.png`;
- `voice-anti-spoofing-spectra0/data/output/*.csv`.

## Чего не класть даже локально в общий каталог без нужды

Реальные телефоны, `external_id`, `s3_key`, пароли, ключи доступа и выгрузки `.env`. Манифест collector нужен только машине, которая скачивает WAV, и не должен попадать в коммит.
