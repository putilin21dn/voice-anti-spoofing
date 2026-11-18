# Пересечение ошибок, день 16.11.2026

Ретроспективный разбор 361 бинарного звонка. Пороги взяты с валидации 13 ноября и здесь не менялись. Модели не обучались, аудио заново не прогонялось. Логические OR и AND — просмотр уже размеченного дня, а не выбранная модель.

Идентификатор во всех таблицах — `call_id`. Предыдущие отчёты не изменялись.

## Как читать

- `summary.md` — ответы на вопросы исследования, списки звонков, ограничения.
- `methodology.md` — когорта, направление score, признаки и статистические правила.
- `per_call_predictions.csv` — 361 звонок × 8 моделей: score, порог, правило, метка, TP/FP/TN/FN.
- `pairwise_model_overlap.csv` — 28 пар, счётчики общих и раздельных ошибок.
- `synthetic_unique_detections.csv` — каждый synthetic-звонок в каждой паре: `both`, `only_a`, `only_b`, `neither`.
- `human_unique_false_positives.csv` — human-звонки, где хотя бы одна модель пары дала FP.
- `ensemble_or_analysis.csv`, `ensemble_and_analysis.csv` — четыре заранее заданные пары со Spectra-0 и сдвиг относительно одного Spectra-0.
- `new_vs_old_classical.csv` — новое и старое решение на каждом звонке, отдельно по трём семьям.
- `unique_error_features.csv` — ошибки Spectra-0 и нового LightGBM с хранимыми признаками.
- `metrics_with_confidence_intervals.csv` — интервалы Уилсона, bootstrap и точный Макнемар.
- `visualizations/` — общие обнаружения, общие FP, сдвиг OR/AND, признаки ошибок пары Spectra-0 и LightGBM.

Положительный класс — synthetic. В бинарной когорте 303 human и 58 synthetic.
