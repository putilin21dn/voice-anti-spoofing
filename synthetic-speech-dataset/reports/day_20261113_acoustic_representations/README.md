# Акустические представления, день 2026-11-13

Offline-исследование Log-Mel, Log-STFT, LFCC и CQCC на уже размеченных телефонных звонках. Это не новый прогон Spectra-0 и не обучение детектора.

Читать в таком порядке:

1. `conclusions.md` — выводы.
2. `methodology.md` — что именно посчитано и чего скрипт не делал.
3. `summary.md` — те же числа с таблицами групп, FP и FN.
4. CSV — сами значения. Идентификатор звонка везде `call_id`.

## Файлы

| Файл | Содержание |
| --- | --- |
| `conclusions.md` | Выводы по четырём представлениям, 32 FP и 5 FN |
| `feature_extraction_config.json` | Параметры преобразований, включая измеренную полосу spafe |
| `dataset_summary.csv` | Состав дня и контрольные числа Spectra-0 |
| `call_level_features.csv` | Признаки основного сигнала: канал абонента, pre-emphasis 0.97, без нормировки громкости |
| `segment_level_features.csv` | Те же признаки по речевым сегментам Spectra-0 |
| `log_normalization_comparison.csv` | Сырой сигнал, пиковая нормировка, pre-emphasis, логарифм мощности и лог-мел по отдельности |
| `error_analysis_fp.csv` | 32 ложных spoof Spectra-0 и положение их признаков внутри human и synthetic |
| `error_analysis_fn.csv` | 5 пропусков синтетики |
| `confounding_analysis.csv` | Эффект human/synthetic и связь с кампанией, длительностью, долей речи, громкостью |
| `padding_analysis.csv` | Повтор хвоста в окнах Spectra-0. В новые признаки повтор не входит |
| `protocol_selection.csv` | Кому и почему построены картинки |
| `summary_stats.json` | Машинные числа, по которым написан `summary.md` |
| `visualizations/` | Спектрограммы протокольной выборки и сводные профили |

`spectra_error_role` и `spectra0_margin_frozen` — метки уже закрытого прогона. Они выбирают FP и FN. Они не являются новыми признаками и не складываются с ними в одну метрику.

## Повторный запуск

```text
cd synthetic-speech-dataset
..\classical-anti-spoofing\.venv\Scripts\python.exe -m src.research.acoustic_representations.run
```

`--resume` продолжает `call_level_features.csv`, если прогон прервался. Код лежит в `src/research/acoustic_representations/` и в worker, backend и evaluation не подключён.
