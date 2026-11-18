# Приоритетное прослушивание, 11–12 ноября 2026

Модели выбирают, какой звонок слушать. Метку происхождения ставит человек, не видя их решений до сохранения.

## Слушать

Сервер только на этом компьютере:

```powershell
cd synthetic-speech-dataset
.\..\my_venv\Scripts\python.exe -m src.main --config config\priority_20261111_20261112.yaml serve
```

Адрес: `http://127.0.0.1:8770`

Порядок списка: очередь A, затем C, затем B, затем контрольная D. Пробел включает канал абонента. «Сохранить и следующий» записывает метку и открывает следующий звонок, не показывая модели. «Сохранить и показать модели» открывает score после того, как метка уже лежит в файле.

Новые метки пишутся в `data/annotations/priority_20261111_20261112_calls.csv`. Аннотации `batch_001`, 13 и 16 ноября этот файл не затирает.

Вторая часть для другого человека: `shares/priority_20261111_20261112_guest.zip`. В архиве нет телефонов и оценок моделей. Уже прослушанные звонки в него не входят. После разметки файл `annotations.csv` из архива вливается так:

```powershell
.\..\my_venv\Scripts\python.exe -m src.main --config config\priority_20261111_20261112.yaml import-share --annotations путь\annotations.csv --segments путь\segments.csv
```

## Таблицы

- `per_call_predictions.csv` — score и решения. В интерфейс до сохранения метки не отдаётся.
- `review_queue_all.csv` — одна строка на звонок, все его очереди и причины.
- `review_queue_disagreements.csv`, `review_queue_borderline.csv`, `review_queue_technical.csv`, `review_queue_random_control.csv` — те же звонки по очередям. Складывать их размеры нельзя: звонок бывает в нескольких.
- `queue_summary.csv` — уникальные звонки и пересечения.
- `manual_annotation_template.csv` — пустая метка для человека.
- `sampling_frame.csv` — вероятность попасть в прослушивание.
- `methodology.md`, `sampling_plan.md`, `data_audit.md`, `summary.md`, `listening_results.md`, `analysis.md`, `common_misses.md`, `spectra_pack2.md`, `trio_overlap.md`, `length_slices.md`, `exp1_spectra_aggregation.md`, `exp2_vote.md`, `exp3_shared_errors.md`.

Идентификатор — `call_id`.
