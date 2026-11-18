# Протокол оценки

Этот файл описывает правила, которые уже записаны в коде и отчётах. Новых метрик сборка не считала. Пороги не пересматривались.

## Старая задача: живой диалог против автоответчика

Источник порога Spectra-0: `classical-anti-spoofing/config/config.yaml`.

- Score: средний по окнам зазор `bonafide_logit - spoof_logit`.
- Spoof, если зазор строго меньше 1.02. Равенство в `src/evaluation/metrics.py` остаётся bonafide.
- Порог выбран на n100 (`spectra_threshold`). На test20 и при оценке classical он не подбирается заново.
- n100 поэтому не является независимым тестом этого порога. Независимый зафиксированный набор старой задачи — test20.

Классические модели:

- Положительный класс старой задачи — `machine`, если score не ниже порога, сохранённого при обучении.
- Порог выбирается на validation. `evaluate` его не меняет.
- В матрицу не входят длительность записи, доля речи, SNR, число сегментов и частота дискретизации.
- Отбор колонок идёт только по train и validation, не больше 200 признаков (`max_features` в конфиге).
- `unknown` старой выборки в обучение не входит. В `split_metadata.json` строк `unknown` нет.

Цифры test20, validation и n100 лежат в `classical-anti-spoofing/reports/technical_report.md` и в `reports/experiment_*/metrics.json`. Краткое повторение тех же цифр есть в README компонента. Здесь они не дублируются, чтобы не разойтись с отчётом.

Варианты признаков (`no_energy`, `shape`, `voicing`, `mfcc`, `spectral`) лежат в своих каталогах `classical-anti-spoofing/models/` и своих `experiment_*` / `comparison_*`. Базовые `models/*.joblib` при варианте не перезаписываются. Эти 18 bundle сохранены в репозитории. Веса Spectra в Git не входят.

## Новая задача: происхождение речи

Метки и исключения: `synthetic-speech-dataset/src/evaluation/dataset.py`.

В бинарный знаменатель входят только подтверждённые:

- отрицательный класс: `human_live`, `human_recorded`;
- положительный класс: `synthetic_tts`, `synthetic_voice_conversion`, `synthetic_playback`.

`unknown` в знаменатель качества не входит. Он считается отдельной строкой. `mixed` тоже не входит в бинарный счёт: если границы ненадёжны, остаётся метка звонка, а не выдуманные интервалы.

`human_recorded` не переводится в synthetic. Исход кампании и код автоответчика сами по себе не доказывают синтетический голос. Это правило разметки в README synthetic-speech-dataset.

`src/evaluation/metrics.py` считает бинарные метрики только когда в знаменателе есть и человек, и подтверждённая синтетика. Для AASIST3 там прямо сказано, что порог звонка не зафиксирован и бинарные метрики не считаются. Оконные logits при этом можно сохранять.

Сегменты и целый звонок — разные таблицы. Окна одной записи не считаются независимыми примерами. План сравнения: `synthetic-speech-dataset/reports/modeling_plan.md`.

## Ensemble A и соседние правила

Источник формулировки: `reports/voting_combinations_analysis.md`. Источник чисел порога: `src/research/model_assisted_annotation/rules.py`.

| ID | Состав | Правило |
| --- | --- | --- |
| A | Spectra-0 + SVM new + LightGBM new | synthetic, если голосов synthetic минимум 2 из 3 |
| B | Spectra-0 + AASIST3 + LightGBM new | то же |
| C | Spectra-0 + AASIST3 + SVM new | то же |
| D | AASIST3 + SVM new + LightGBM new | то же |
| 2 из 2 | Spectra-0 + AASIST3 | synthetic, только если synthetic сказали обе |

Logistic Regression в эти правила не входит. Её порог в `rules.py` сохранён и в голосе не используется.

Направление в `rules.py`: для Spectra-0 и AASIST3 synthetic, если score ниже порога; для classical synthetic, если score не ниже порога. Ширина пограничной зоны — `rules.WIDTHS`, комментарий в файле: 0.10 IQR на 51 звонке валидации 13 ноября, на новых днях не пересчитывается.

Комментарий в том же файле: порог и направление не подбираются по 11–12 ноября.

Замена A в отчёте считается подтверждённой только если альтернатива 2 из 3 на обоих полных днях и на PACK1 и PACK2 каждого дня даёт не больше FN и не больше FP, чем A, и хотя бы на одном полном дне строго меньше ошибок FP+FN. Рост precision сам по себе замену не подтверждает. Правило 2 из 2 в этот отбор не входит.

Разбор пропусков A: `reports/ensemble_a_error_analysis.md`. CSV рядом с этим отчётом (`reports/ensemble_a/*.csv`) и рисунки в Git не входят.

Два порога Spectra-0 нельзя подменять один другим. 1.02 — порог зазора logit в classical и в `metrics.py`. Константа `rules.THRESHOLDS["spectra0"]` относится к score голосования. Оба значения оставлены как в источниках.

## Что не входит в знаменатель опубликованных комбинаций

По тексту `voting_combinations_analysis.md` для когорт 11–12 ноября и 16 ноября:

- из бинарного счёта выведены `unknown` и `mixed`;
- три звонка 16 ноября с разными основными метками на повторных строках исключены целиком, их список в отчёте совпадает с `reports/origin_retrain_20261113/test_disputes.csv` (сам CSV не копировался);
- подтипы synthetic не склеиваются в новые классы: для бинарного счёта все прежние synthetic-метки остаются положительным классом, все прежние human-метки — отрицательным.

Пакет читается из суффикса имени кампании PACK1 или PACK2.

## Где читать результаты

Текстовые отчёты на месте:

- classical: `classical-anti-spoofing/reports/technical_report.md`, `reports/experiment_*/summary.md`, `reports/comparison_*/summary.md`, `reports/shap_001/summary.md`, `reports/spectra0_baseline_001/summary.md`;
- происхождение и ансамбль: `synthetic-speech-dataset/reports/*.md` и вложенные `summary.md`, `methodology.md`;
- короткие заметки прогонов: `synthetic-speech-dataset/evaluation/**/summary.md` и `data_quality_report.md`.

Тяжёлые таблицы этих же прогонов остались в исходных каталогах на диске. Инструкция по локальному размещению: `docs/operations/data_storage.md`.
