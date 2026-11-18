# Кепстр и padding, день 2026-11-13

Проверка двух вопросов на уже посчитанном дне: межсегментная стабильность LFCC/CQCC и повтор хвоста в окнах Spectra-0. Модели заново не запускались и не обучались.

Читать сначала `summary.md`, затем `methodology.md`. CSV содержат только `call_id` как идентификатор звонка.

| Файл | Содержание |
| --- | --- |
| `summary.md` | Что видно на этом дне и чего из этого не следует |
| `methodology.md` | Откуда числа, как нарезаны группы, как устроен опыт с padding |
| `segment_level_cepstral_features.csv` | Средние кепстра каждого речевого сегмента и окна этого сегмента |
| `call_level_cepstral_summary.csv` | Межсегментное СКО, отдельный внутрисегментный MAD, margin и padding |
| `matched_group_analysis.csv` | Размеры групп, медианы, IQR, Cliff's delta и интервалы |
| `false_negative_segment_analysis.csv` | Пять пропусков по сегментам |
| `false_positive_padding_analysis.csv` | 32 FP, 151 TN и раскладка по терцилям |
| `padding_sensitivity.csv` | Реальный хвост, повтор и нули |
| `normalization_sensitivity.csv` | Без нормировки, пик и RMS |
| `visualizations/` | Margin, пять FN, padding и нормировка |

Повторный запуск из `synthetic-speech-dataset`:

```text
..\classical-anti-spoofing\.venv\Scripts\python.exe -m src.research.cepstral_padding_followup.run
```

`--resume` продолжает аудио, если прогон оборвался. Код не подключён к worker, backend и evaluation.
