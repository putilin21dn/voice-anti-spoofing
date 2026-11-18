# Сравнение комбинаций голосования для детекции синтетической речи

Офлайн-сравнение сохранённых решений. Нового inference, обучения и подбора порогов нет.

## 1. Цель эксперимента

Нужно сравнить правила, которые ставят звонку synthetic, когда за synthetic проголосовали минимум две модели из трёх. Текущая комбинация A — Spectra-0, SVM new и LightGBM new. Рядом считаются три замены одной из моделей на AASIST3 и, отдельно, правило «обе модели» для пары Spectra-0 + AASIST3.

| ID | Состав | Правило |
| --- | --- | --- |
| A | Spectra-0 + SVM + LightGBM | synthetic, если голосов synthetic минимум 2 из 3 |
| B | Spectra-0 + AASIST3 + LightGBM | synthetic, если голосов synthetic минимум 2 из 3 |
| C | Spectra-0 + AASIST3 + SVM | synthetic, если голосов synthetic минимум 2 из 3 |
| D | AASIST3 + SVM + LightGBM | synthetic, если голосов synthetic минимум 2 из 3 |
| 2 из 2 | Spectra-0 + AASIST3, оба голоса | synthetic, только если synthetic сказали обе модели |

Logistic regression в эти правила не входит. Её порог оставлен прежним и в расчёте голоса не используется. Подтипы synthetic не склеиваются в новые классы: для бинарного счёта все прежние synthetic-метки остаются положительным классом, все прежние human-метки — отрицательным.

Замена A считается подтверждённой только если альтернатива 2 из 3 на обоих полных днях и на PACK1 и PACK2 каждого дня даёт не больше FN и не больше FP, чем A, и хотя бы на одном полном дне строго меньше ошибок FP+FN. Рост precision сам по себе замену не подтверждает. Правило 2 из 2 в этот отбор не входит.

## 2. Источники данных и состав выборок

Обе когорты взяты из уже опубликованных разборов. 11–12 ноября — прослушанная приоритетная когорта. 16 ноября — размеченный день. Порог моделей выбирался на валидации 13 ноября и на этих звонках заново не считается.

| Файл | SHA-256, 16 символов | Байт |
| --- | --- | ---: |
| `reports/model_assisted_annotation_20261111_20261112/per_call_predictions.csv` | 6be6a458bd43a4b9 | 223489 |
| `data/annotations/priority_20261111_20261112_calls.csv` | eb9cba10de0b900c | 176479 |
| `reports/day_20261116_model_error_overlap/per_call_predictions.csv` | 59e50ae0e44a6695 | 391301 |
| `data/annotations/day_20261116_calls.csv` | 71ec621d16a74f89 | 139665 |
| `data/manifests/day_20261116.csv` | 785b1343f8805c25 | 123901 |
| `evaluation/day_20261116/20261116T152222/call_results/spectra0.csv` | ca507be492d08bce | 49975 |

11–12 ноября: в файле предсказаний 503 звонков, в файле меток 492. Повторов `call_id` в метках нет, расхождений основных меток нет. Из бинарного счёта выведены `unknown` (23) и `mixed` (12). Звонков без предсказания одной из четырёх моделей внутри бинарной когорты нет. В счёте 457 звонков: 134 synthetic и 323 human.

| Дата | synthetic_tts | synthetic_voice_conversion | synthetic_playback | human_live | human_recorded |
| --- | ---: | ---: | ---: | ---: | ---: |
| 11–12 | 134 | 0 | 0 | 323 | 0 |
| 16 | 58 | 0 | 0 | 303 | 0 |

Других бинарных подтипов в этих когортах нет. `unknown` и `mixed` в таблицу не входят.

16 ноября: строк аннотаций 392, уникальных звонков меньше из-за повторных строк. Одинаковая основная метка на двух строках встретилась у 5 звонков: оставлена первая строка, как в прежнем подсчёте. Три звонка с разными основными метками исключены целиком. Их `call_id` совпадают с `reports/origin_retrain_20261113/test_disputes.csv`.

| call_id | Основные метки |
| --- | --- |
| `14708dd0-11e2-4e07-9bfc-fae7808a4249` | human_live и unknown |
| `2d4ccfb6-4da6-4032-843a-0aeb45a52238` | synthetic_tts и unknown |
| `ebabebec-7daf-41bc-afd3-1c09a08f63c6` | synthetic_tts и unknown |

После снятия этих трёх звонков вне бинарного счёта остаются `unknown` (15) и `mixed` (5). В счёте 361 звонков: 58 synthetic и 303 human. Предсказания четырёх моделей есть у каждого из них. Список исключённых `call_id` лежит в `reports/voting_combinations/excluded_calls.csv`.

| Дата | Пакет | Звонков | Synthetic | Human |
| --- | --- | ---: | ---: | ---: |
| 11–12 | PACK1 | 267 | 115 | 152 |
| 11–12 | PACK2 | 190 | 19 | 171 |
| 16 | PACK1 | 134 | 51 | 83 |
| 16 | PACK2 | 227 | 7 | 220 |

Пакет читается из имени кампании: суффикс PACK1 или PACK2. Звонков бинарной когорты вне этих двух пакетов нет.

| Дата | Речь | Звонков | Synthetic | Human | Минимум, с | Максимум, с |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 11–12 | короткая | 153 | 29 | 124 | 0.76 | 9.68 |
| 11–12 | средняя | 152 | 48 | 104 | 9.72 | 17.04 |
| 11–12 | длинная | 152 | 57 | 95 | 17.06 | 232.38 |
| 16 | короткая | 119 | 7 | 112 | 1.70 | 9.58 |
| 16 | средняя | 116 | 30 | 86 | 9.74 | 16.98 |
| 16 | длинная | 126 | 21 | 105 | 17.12 | 455.24 |

Длительность речи — сумма речевых интервалов Spectra-0, не длительность файла. На 11–12 это колонка `spectra_speech_duration_s` в `per_call_predictions.csv`. На 16 ноября это `speech_duration_s` в `evaluation/day_20261116/20261116T152222/call_results/spectra0.csv`. Границы прежние и на 16 ноября заново не считались: короткая речь до 9.68 с включительно, средняя строго после 9.68 и до 17.04 с включительно, длинная строго длиннее 17.04 с. Неокруглённые терцили речи на бинарных звонках 11–12 ноября равны 9.680000 и 17.040000. В группировке используются опубликованные 9.68 и 17.04. Каждый звонок попадает ровно в одну группу, пропусков длительности нет.

Ровно на границе 9.68 с — 1 звонок, ровно на 17.04 с — 2 звонка. Граница 9.68 относится к короткой речи, граница 17.04 — к средней.

Кампания callback 16 ноября в эти когорты не входит: у неё нет той же полной бинарной разметки происхождения.

## 3. Использованные пороги и направления классификации

Порог применяется к уже сохранённому call-level score функцией `prediction` из `src/research/model_assisted_annotation/rules.py`. Для каждого звонка анализа пересчитанное решение совпало с сохранённой меткой. Score разных моделей не складываются и не читаются как вероятность.

| Модель | Что лежит в score | Порог | Synthetic, если | Минимум | Медиана | Максимум | Ровно на пороге |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| | | | **11–12 ноября** | | | | |
| Spectra-0 | margin | 0.22710928916931153 | synthetic, если score < порога | -13.398 | 2.504 | 15.469 | 0 |
| AASIST3 | средний bonafide logit | 0.11397590637207031 | synthetic, если score < порога | -3.072 | 4.066 | 12.670 | 0 |
| SVM | score | 0.0 | synthetic, если score ≥ порога | -2.007 | -0.730 | 1.514 | 0 |
| LightGBM | score | 0.3460832258410117 | synthetic, если score ≥ порога | 0.000 | 0.003 | 1.000 | 0 |
| | | | **16 ноября** | | | | |
| Spectra-0 | margin | 0.22710928916931153 | synthetic, если score < порога | -10.317 | 3.008 | 18.428 | 0 |
| AASIST3 | средний bonafide logit | 0.11397590637207031 | synthetic, если score < порога | -2.839 | 4.505 | 13.170 | 0 |
| SVM | score | 0.0 | synthetic, если score ≥ порога | -2.149 | -1.011 | 1.554 | 0 |
| LightGBM | score | 0.3460832258410117 | synthetic, если score ≥ порога | 0.000 | 0.001 | 1.000 | 0 |

У Spectra-0 и AASIST3 шкалы другие: это margin и средний bonafide logit. Число больше единицы или меньше нуля для них нормально и вероятностью не является. SVM new — `decision_function`, LightGBM new в сохранённом прогоне получен через `predict_proba`, но в голосовании участвует только бинарное сравнение с порогом 0.346. Logistic regression остаётся при пороге 0.7354826103642409 и в комбинации A–D не входит.

На пороге classical-модель говорит synthetic, потому что правило — «больше или равно». Spectra-0 и AASIST3 на пороге говорят human, потому что правило — «строго меньше». В этих когортах звонков ровно на пороге нет, так что граничный случай на исход не влияет.

## 4. Методика голосования

Сначала каждая модель переводится в human или synthetic. Пустое решение на human не заменяется. Дальше голос 2 из 3 считает число бинарных synthetic-решений и ставит synthetic при сумме 2 или 3. Сырой score в сумму не входит. Правило 2 из 2 ставит synthetic только при двух synthetic и считается отдельной колонкой `pair_2of2`. Оно не подаётся в функцию majority и не стоит в ряду A–D.

Перед записью отчёта скрипт проверяет:

- пороги и направление совпадают с rules.THRESHOLDS
- границы речи непересекающиеся: <=9.68, (9.68, 17.04], >17.04
- 2 из 3 — это сумма бинарных голосов >= 2; 2 из 2 — это пересечение и не смешано с 2 из 3
- на равном пороге classical — synthetic, Spectra-0 и AASIST3 — human; пустой status не становится human
- выборки 11–12 и 16 ноября не пересекаются
- размеры бинарных когорт совпали с предыдущими отчётами
- unknown, mixed и три расхождения меток исключены по прежней логике и не входят в знаменатель
- четыре модели стоят на одном наборе звонков, порог применён к бинарному решению, тройка совпала с прежним разбором
- базовая комбинация A и отдельные модели совпали с опубликованными TP/FP по дате, пакету и длительности
- в таблицах majority и 2 из 2 разведены, вход голоса бинарный

Дополнительно на каждой строке метрик проверяется: TP + FN равно числу synthetic, TN + FP равно числу human, сумма четырёх клеток равна размеру среза, specificity + FPR = 1. Смена голоса относительно A раскладывается на исправленные и новые FN и FP, и это разложение сходится с ΔFN и ΔFP.

## 5. Общие результаты по каждой дате

11–12 ноября, 457 звонков, 134 synthetic и 323 human. A: TP 109, FP 8, TN 315, FN 25, recall 0.813 (109/134), FPR 0.025 (8/323), precision 0.932 (109/117), ошибок 33.

| Правило | TP | FP | TN | FN | Recall | Specificity | FPR | Precision | Balanced accuracy | Ошибки |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A. Spectra-0 + SVM + LightGBM | 109 | 8 | 315 | 25 | 0.813 (109/134) | 0.975 (315/323) | 0.025 (8/323) | 0.932 (109/117) | 0.894 | 33 |
| B. Spectra-0 + AASIST3 + LightGBM | 93 | 9 | 314 | 41 | 0.694 (93/134) | 0.972 (314/323) | 0.028 (9/323) | 0.912 (93/102) | 0.833 | 50 |
| C. Spectra-0 + AASIST3 + SVM | 95 | 11 | 312 | 39 | 0.709 (95/134) | 0.966 (312/323) | 0.034 (11/323) | 0.896 (95/106) | 0.837 | 50 |
| D. AASIST3 + SVM + LightGBM | 105 | 10 | 313 | 29 | 0.784 (105/134) | 0.969 (313/323) | 0.031 (10/323) | 0.913 (105/115) | 0.876 | 39 |

Отдельное правило 2 из 2 на тех же 457 звонках:

| Правило | TP | FP | TN | FN | Recall | Specificity | FPR | Precision | Balanced accuracy | Ошибки |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pair_2of2. Spectra-0 + AASIST3, оба голоса | 75 | 9 | 314 | 59 | 0.560 (75/134) | 0.972 (314/323) | 0.028 (9/323) | 0.893 (75/84) | 0.766 | 68 |

16 ноября, 361 звонков, 58 synthetic и 303 human. A: TP 43, FP 3, TN 300, FN 15, recall 0.741 (43/58), FPR 0.010 (3/303), precision 0.935 (43/46), ошибок 18.

| Правило | TP | FP | TN | FN | Recall | Specificity | FPR | Precision | Balanced accuracy | Ошибки |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A. Spectra-0 + SVM + LightGBM | 43 | 3 | 300 | 15 | 0.741 (43/58) | 0.990 (300/303) | 0.010 (3/303) | 0.935 (43/46) | 0.866 | 18 |
| B. Spectra-0 + AASIST3 + LightGBM | 40 | 8 | 295 | 18 | 0.690 (40/58) | 0.974 (295/303) | 0.026 (8/303) | 0.833 (40/48) | 0.832 | 26 |
| C. Spectra-0 + AASIST3 + SVM | 39 | 8 | 295 | 19 | 0.672 (39/58) | 0.974 (295/303) | 0.026 (8/303) | 0.830 (39/47) | 0.823 | 27 |
| D. AASIST3 + SVM + LightGBM | 42 | 3 | 300 | 16 | 0.724 (42/58) | 0.990 (300/303) | 0.010 (3/303) | 0.933 (42/45) | 0.857 | 19 |

Отдельное правило 2 из 2 на тех же 361 звонках:

| Правило | TP | FP | TN | FN | Recall | Specificity | FPR | Precision | Balanced accuracy | Ошибки |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pair_2of2. Spectra-0 + AASIST3, оба голоса | 30 | 8 | 295 | 28 | 0.517 (30/58) | 0.974 (295/303) | 0.026 (8/303) | 0.789 (30/38) | 0.745 | 36 |

Отдельные модели на тех же звонках, без голосования:

### 11–12 ноября

| Правило | TP | FP | TN | FN | Recall | Specificity | FPR | Precision | Balanced accuracy | Ошибки |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Spectra-0 | 102 | 30 | 293 | 32 | 0.761 (102/134) | 0.907 (293/323) | 0.093 (30/323) | 0.773 (102/132) | 0.834 | 62 |
| AASIST3 | 75 | 14 | 309 | 59 | 0.560 (75/134) | 0.957 (309/323) | 0.043 (14/323) | 0.843 (75/89) | 0.758 | 73 |
| SVM | 105 | 15 | 308 | 29 | 0.784 (105/134) | 0.954 (308/323) | 0.046 (15/323) | 0.875 (105/120) | 0.869 | 44 |
| LightGBM | 104 | 13 | 310 | 30 | 0.776 (104/134) | 0.960 (310/323) | 0.040 (13/323) | 0.889 (104/117) | 0.868 | 43 |

### 16 ноября

| Правило | TP | FP | TN | FN | Recall | Specificity | FPR | Precision | Balanced accuracy | Ошибки |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Spectra-0 | 43 | 26 | 277 | 15 | 0.741 (43/58) | 0.914 (277/303) | 0.086 (26/303) | 0.623 (43/69) | 0.828 | 41 |
| AASIST3 | 30 | 14 | 289 | 28 | 0.517 (30/58) | 0.954 (289/303) | 0.046 (14/303) | 0.682 (30/44) | 0.736 | 42 |
| SVM | 38 | 7 | 296 | 20 | 0.655 (38/58) | 0.977 (296/303) | 0.023 (7/303) | 0.844 (38/45) | 0.816 | 27 |
| LightGBM | 40 | 5 | 298 | 18 | 0.690 (40/58) | 0.983 (298/303) | 0.017 (5/303) | 0.889 (40/45) | 0.837 | 23 |

![Recall](voting_combinations/figures/recall.png)

Подпись столбца — TP / число synthetic в когорте.

![Precision](voting_combinations/figures/precision.png)

Подпись столбца — TP / (TP + FP).

![FPR](voting_combinations/figures/fpr.png)

Подпись столбца — FP / число human. Ось начинается с нуля.

![FP и FN](voting_combinations/figures/fp_fn.png)

Числа над столбцами — абсолютные FP и FN. Вертикальная черта отделяет правило 2 из 2 от majority.

На 11–12 ноября A даёт 109 TP и 8 FP при 25 FN. B относительно A: FN +16, FP +1, ошибок +17. C относительно A: FN +14, FP +3, ошибок +17. D относительно A: FN +4, FP +2, ошибок +6. Правило 2 из 2 относительно A: FN +34, FP +1.

На 16 ноября A даёт 43 TP и 3 FP при 15 FN. B относительно A: FN +3, FP +5, ошибок +8. C относительно A: FN +4, FP +5, ошибок +9. D относительно A: FN +1, FP +0, ошибок +1. Правило 2 из 2 относительно A: FN +13, FP +5.

## 6. Результаты PACK1/PACK2

PACK1 держит почти всю синтетику: 115 из 134 на 11–12 и 51 из 58 на 16 ноября. PACK2 — это в основном human: 171 и 220 человек при 19 и 7 synthetic. Доля на семи synthetic меняется шагом около 14 процентных пунктов.

| Дата | Срез | n syn | n hum | Правило | TP | FP | TN | FN | Recall | FPR | Precision | Ошибки |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11–12 | PACK1 | 115 | 152 | A | 92 | 5 | 147 | 23 | 0.800 (92/115) | 0.033 (5/152) | 0.948 (92/97) | 28 |
| 11–12 | PACK1 | 115 | 152 | B | 75 | 4 | 148 | 40 | 0.652 (75/115) | 0.026 (4/152) | 0.949 (75/79) | 44 |
| 11–12 | PACK1 | 115 | 152 | C | 77 | 6 | 146 | 38 | 0.670 (77/115) | 0.039 (6/152) | 0.928 (77/83) | 44 |
| 11–12 | PACK1 | 115 | 152 | D | 88 | 7 | 145 | 27 | 0.765 (88/115) | 0.046 (7/152) | 0.926 (88/95) | 34 |
| 11–12 | PACK2 | 19 | 171 | A | 17 | 3 | 168 | 2 | 0.895 (17/19) | 0.018 (3/171) | 0.850 (17/20) | 5 |
| 11–12 | PACK2 | 19 | 171 | B | 18 | 5 | 166 | 1 | 0.947 (18/19) | 0.029 (5/171) | 0.783 (18/23) | 6 |
| 11–12 | PACK2 | 19 | 171 | C | 18 | 5 | 166 | 1 | 0.947 (18/19) | 0.029 (5/171) | 0.783 (18/23) | 6 |
| 11–12 | PACK2 | 19 | 171 | D | 17 | 3 | 168 | 2 | 0.895 (17/19) | 0.018 (3/171) | 0.850 (17/20) | 5 |
| 16 | PACK1 | 51 | 83 | A | 37 | 2 | 81 | 14 | 0.725 (37/51) | 0.024 (2/83) | 0.949 (37/39) | 16 |
| 16 | PACK1 | 51 | 83 | B | 33 | 3 | 80 | 18 | 0.647 (33/51) | 0.036 (3/83) | 0.917 (33/36) | 21 |
| 16 | PACK1 | 51 | 83 | C | 32 | 3 | 80 | 19 | 0.627 (32/51) | 0.036 (3/83) | 0.914 (32/35) | 22 |
| 16 | PACK1 | 51 | 83 | D | 36 | 2 | 81 | 15 | 0.706 (36/51) | 0.024 (2/83) | 0.947 (36/38) | 17 |
| 16 | PACK2 | 7 | 220 | A | 6 | 1 | 219 | 1 | 0.857 (6/7) | 0.005 (1/220) | 0.857 (6/7) | 2 |
| 16 | PACK2 | 7 | 220 | B | 7 | 5 | 215 | 0 | 1.000 (7/7) | 0.023 (5/220) | 0.583 (7/12) | 5 |
| 16 | PACK2 | 7 | 220 | C | 7 | 5 | 215 | 0 | 1.000 (7/7) | 0.023 (5/220) | 0.583 (7/12) | 5 |
| 16 | PACK2 | 7 | 220 | D | 6 | 1 | 219 | 1 | 0.857 (6/7) | 0.005 (1/220) | 0.857 (6/7) | 2 |

Правило 2 из 2 по пакетам, отдельно от таблицы выше:

| Дата | Срез | n syn | n hum | Правило | TP | FP | TN | FN | Recall | FPR | Precision | Ошибки |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11–12 | PACK1 | 115 | 152 | 2 из 2 | 60 | 4 | 148 | 55 | 0.522 (60/115) | 0.026 (4/152) | 0.938 (60/64) | 59 |
| 11–12 | PACK2 | 19 | 171 | 2 из 2 | 15 | 5 | 166 | 4 | 0.789 (15/19) | 0.029 (5/171) | 0.750 (15/20) | 9 |
| 16 | PACK1 | 51 | 83 | 2 из 2 | 24 | 3 | 80 | 27 | 0.471 (24/51) | 0.036 (3/83) | 0.889 (24/27) | 30 |
| 16 | PACK2 | 7 | 220 | 2 из 2 | 6 | 5 | 215 | 1 | 0.857 (6/7) | 0.023 (5/220) | 0.545 (6/11) | 6 |

![Пакеты](voting_combinations/figures/pack.png)

Число над столбцом recall — TP, над столбцом FPR — FP. Подпись группы показывает число synthetic и human, поэтому доля на PACK2 читается вместе с семью или девятнадцатью положительными звонками.

11–12, PACK1: A ловит 92 из 115 и даёт 5 FP на 152 human. B +17 FN и -1 FP, C +15 FN и +1 FP, D +4 FN и +2 FP. 11–12, PACK2: A ловит 17 из 19 и даёт 3 FP на 171 human. B -1 FN и +2 FP, C -1 FN и +2 FP, D +0 FN и +0 FP. 16, PACK1: A ловит 37 из 51 и даёт 2 FP на 83 human. B +4 FN и +1 FP, C +5 FN и +1 FP, D +1 FN и +0 FP. 16, PACK2: A ловит 6 из 7 и даёт 1 FP на 220 human. B -1 FN и +4 FP, C -1 FN и +4 FP, D +0 FN и +0 FP.

## 7. Результаты по длительности речи

Короткая: речь ≤ 9.68 с. Средняя: 9.68 < речь ≤ 17.04 с. Длинная: речь > 17.04 с. На 16 ноября короткая группа содержит 7 synthetic и 112 human. Precision и recall в ней стоят на единицах звонков.

| Дата | Срез | n syn | n hum | Правило | TP | FP | TN | FN | Recall | FPR | Precision | Ошибки |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11–12 | короткая | 29 | 124 | A | 24 | 3 | 121 | 5 | 0.828 (24/29) | 0.024 (3/124) | 0.889 (24/27) | 8 |
| 11–12 | короткая | 29 | 124 | B | 20 | 7 | 117 | 9 | 0.690 (20/29) | 0.056 (7/124) | 0.741 (20/27) | 16 |
| 11–12 | короткая | 29 | 124 | C | 21 | 7 | 117 | 8 | 0.724 (21/29) | 0.056 (7/124) | 0.750 (21/28) | 15 |
| 11–12 | короткая | 29 | 124 | D | 23 | 3 | 121 | 6 | 0.793 (23/29) | 0.024 (3/124) | 0.885 (23/26) | 9 |
| 11–12 | средняя | 48 | 104 | A | 35 | 1 | 103 | 13 | 0.729 (35/48) | 0.010 (1/104) | 0.972 (35/36) | 14 |
| 11–12 | средняя | 48 | 104 | B | 32 | 1 | 103 | 16 | 0.667 (32/48) | 0.010 (1/104) | 0.970 (32/33) | 17 |
| 11–12 | средняя | 48 | 104 | C | 32 | 3 | 101 | 16 | 0.667 (32/48) | 0.029 (3/104) | 0.914 (32/35) | 19 |
| 11–12 | средняя | 48 | 104 | D | 33 | 3 | 101 | 15 | 0.688 (33/48) | 0.029 (3/104) | 0.917 (33/36) | 18 |
| 11–12 | длинная | 57 | 95 | A | 50 | 4 | 91 | 7 | 0.877 (50/57) | 0.042 (4/95) | 0.926 (50/54) | 11 |
| 11–12 | длинная | 57 | 95 | B | 41 | 1 | 94 | 16 | 0.719 (41/57) | 0.011 (1/95) | 0.976 (41/42) | 17 |
| 11–12 | длинная | 57 | 95 | C | 42 | 1 | 94 | 15 | 0.737 (42/57) | 0.011 (1/95) | 0.977 (42/43) | 16 |
| 11–12 | длинная | 57 | 95 | D | 49 | 4 | 91 | 8 | 0.860 (49/57) | 0.042 (4/95) | 0.925 (49/53) | 12 |
| 16 | короткая | 7 | 112 | A | 4 | 1 | 111 | 3 | 0.571 (4/7) | 0.009 (1/112) | 0.800 (4/5) | 4 |
| 16 | короткая | 7 | 112 | B | 5 | 6 | 106 | 2 | 0.714 (5/7) | 0.054 (6/112) | 0.455 (5/11) | 8 |
| 16 | короткая | 7 | 112 | C | 5 | 6 | 106 | 2 | 0.714 (5/7) | 0.054 (6/112) | 0.455 (5/11) | 8 |
| 16 | короткая | 7 | 112 | D | 4 | 1 | 111 | 3 | 0.571 (4/7) | 0.009 (1/112) | 0.800 (4/5) | 4 |
| 16 | средняя | 30 | 86 | A | 23 | 1 | 85 | 7 | 0.767 (23/30) | 0.012 (1/86) | 0.958 (23/24) | 8 |
| 16 | средняя | 30 | 86 | B | 24 | 1 | 85 | 6 | 0.800 (24/30) | 0.012 (1/86) | 0.960 (24/25) | 7 |
| 16 | средняя | 30 | 86 | C | 23 | 1 | 85 | 7 | 0.767 (23/30) | 0.012 (1/86) | 0.958 (23/24) | 8 |
| 16 | средняя | 30 | 86 | D | 22 | 1 | 85 | 8 | 0.733 (22/30) | 0.012 (1/86) | 0.957 (22/23) | 9 |
| 16 | длинная | 21 | 105 | A | 16 | 1 | 104 | 5 | 0.762 (16/21) | 0.010 (1/105) | 0.941 (16/17) | 6 |
| 16 | длинная | 21 | 105 | B | 11 | 1 | 104 | 10 | 0.524 (11/21) | 0.010 (1/105) | 0.917 (11/12) | 11 |
| 16 | длинная | 21 | 105 | C | 11 | 1 | 104 | 10 | 0.524 (11/21) | 0.010 (1/105) | 0.917 (11/12) | 11 |
| 16 | длинная | 21 | 105 | D | 16 | 1 | 104 | 5 | 0.762 (16/21) | 0.010 (1/105) | 0.941 (16/17) | 6 |

Правило 2 из 2 по длительности речи:

| Дата | Срез | n syn | n hum | Правило | TP | FP | TN | FN | Recall | FPR | Precision | Ошибки |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11–12 | короткая | 29 | 124 | 2 из 2 | 16 | 7 | 117 | 13 | 0.552 (16/29) | 0.056 (7/124) | 0.696 (16/23) | 20 |
| 11–12 | средняя | 48 | 104 | 2 из 2 | 22 | 1 | 103 | 26 | 0.458 (22/48) | 0.010 (1/104) | 0.957 (22/23) | 27 |
| 11–12 | длинная | 57 | 95 | 2 из 2 | 37 | 1 | 94 | 20 | 0.649 (37/57) | 0.011 (1/95) | 0.974 (37/38) | 21 |
| 16 | короткая | 7 | 112 | 2 из 2 | 4 | 6 | 106 | 3 | 0.571 (4/7) | 0.054 (6/112) | 0.400 (4/10) | 9 |
| 16 | средняя | 30 | 86 | 2 из 2 | 15 | 1 | 85 | 15 | 0.500 (15/30) | 0.012 (1/86) | 0.938 (15/16) | 16 |
| 16 | длинная | 21 | 105 | 2 из 2 | 11 | 1 | 104 | 10 | 0.524 (11/21) | 0.010 (1/105) | 0.917 (11/12) | 11 |

![Длительность речи](voting_combinations/figures/duration.png)

11–12, короткая: 29 synthetic и 124 human, A даёт FN 5 и FP 3. Меньше всего ошибок FP+FN в этом срезе у A: FN 5, FP 3. 11–12, средняя: 48 synthetic и 104 human, A даёт FN 13 и FP 1. Меньше всего ошибок FP+FN в этом срезе у A: FN 13, FP 1. 11–12, длинная: 57 synthetic и 95 human, A даёт FN 7 и FP 4. Меньше всего ошибок FP+FN в этом срезе у A: FN 7, FP 4. 16, короткая: 7 synthetic и 112 human, A даёт FN 3 и FP 1. Меньше всего ошибок FP+FN в этом срезе у A: FN 3, FP 1. 16, средняя: 30 synthetic и 86 human, A даёт FN 7 и FP 1. Меньше всего ошибок FP+FN в этом срезе у B: FN 6, FP 1. 16, длинная: 21 synthetic и 105 human, A даёт FN 5 и FP 1. Меньше всего ошибок FP+FN в этом срезе у A: FN 5, FP 1.

## 8. Пересечение ошибок

Ошибка модели — бинарное решение. FN: звонок synthetic, модель сказала human. FP: звонок human, модель сказала synthetic. В паре «обе» значит, что ошиблись обе модели. «Только A» значит, что ошиблась первая и не ошиблась вторая.

| Дата | Ошибка | Модель A | Модель B | Обе | Только A | Только B | Ни одна | Jaccard |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 11–12 | FN | Spectra-0 | AASIST3 | 32 | 0 | 27 | 75 | 0.542 |
| 11–12 | FN | Spectra-0 | SVM | 8 | 24 | 21 | 81 | 0.151 |
| 11–12 | FN | Spectra-0 | LightGBM | 11 | 21 | 19 | 83 | 0.216 |
| 11–12 | FN | AASIST3 | SVM | 15 | 44 | 14 | 61 | 0.205 |
| 11–12 | FN | AASIST3 | LightGBM | 20 | 39 | 10 | 65 | 0.290 |
| 11–12 | FN | SVM | LightGBM | 20 | 9 | 10 | 95 | 0.513 |
| 11–12 | FP | Spectra-0 | AASIST3 | 9 | 21 | 5 | 288 | 0.257 |
| 11–12 | FP | Spectra-0 | SVM | 0 | 30 | 15 | 278 | 0.000 |
| 11–12 | FP | Spectra-0 | LightGBM | 1 | 29 | 12 | 281 | 0.024 |
| 11–12 | FP | AASIST3 | SVM | 2 | 12 | 13 | 296 | 0.074 |
| 11–12 | FP | AASIST3 | LightGBM | 1 | 13 | 12 | 297 | 0.038 |
| 11–12 | FP | SVM | LightGBM | 7 | 8 | 6 | 302 | 0.333 |
| 16 | FN | Spectra-0 | AASIST3 | 15 | 0 | 13 | 30 | 0.536 |
| 16 | FN | Spectra-0 | SVM | 6 | 9 | 14 | 29 | 0.207 |
| 16 | FN | Spectra-0 | LightGBM | 7 | 8 | 11 | 32 | 0.269 |
| 16 | FN | AASIST3 | SVM | 10 | 18 | 10 | 20 | 0.263 |
| 16 | FN | AASIST3 | LightGBM | 10 | 18 | 8 | 22 | 0.278 |
| 16 | FN | SVM | LightGBM | 14 | 6 | 4 | 34 | 0.583 |
| 16 | FP | Spectra-0 | AASIST3 | 8 | 18 | 6 | 271 | 0.250 |
| 16 | FP | Spectra-0 | SVM | 0 | 26 | 7 | 270 | 0.000 |
| 16 | FP | Spectra-0 | LightGBM | 0 | 26 | 5 | 272 | 0.000 |
| 16 | FP | AASIST3 | SVM | 0 | 14 | 7 | 282 | 0.000 |
| 16 | FP | AASIST3 | LightGBM | 0 | 14 | 5 | 284 | 0.000 |
| 16 | FP | SVM | LightGBM | 3 | 4 | 2 | 294 | 0.333 |

![Пересечение ошибок](voting_combinations/figures/error_overlap.png)

Диагональ матрицы — сколько ошибок у самой модели. Вне диагонали — сколько из них общие со второй моделью. Полная таблица и матрица также лежат в `reports/voting_combinations/pairwise_errors.csv` и `reports/voting_combinations/error_matrix.csv`.

На 11–12 ноября из 32 пропусков Spectra-0 SVM находит 24, LightGBM находит 21, AASIST3 находит 0. Из 30 ложных Spectra-0 SVM повторяет 0, LightGBM повторяет 1, AASIST3 повторяет 9. Synthetic, которые AASIST3 видит, а Spectra-0 нет: 0.

На 16 ноября из 15 пропусков Spectra-0 SVM находит 9, LightGBM находит 8, AASIST3 находит 0. Из 26 ложных Spectra-0 SVM повторяет 0, LightGBM повторяет 0, AASIST3 повторяет 8. Synthetic, которые AASIST3 видит, а Spectra-0 нет: 0.

Перекрытие пропусков Spectra-0 и AASIST3 высокое: на 11–12 Jaccard пропусков Spectra-0 и AASIST3 равен 0.542 (32 общих, 0 только Spectra-0, 27 только AASIST3); на 16 Jaccard пропусков Spectra-0 и AASIST3 равен 0.536 (15 общих, 0 только Spectra-0, 13 только AASIST3).

SVM и LightGBM закрывают часть пропусков Spectra-0 и почти не повторяют её ложные срабатывания. AASIST3 эту пару свойств не повторяет: пропуски Spectra-0 он в основном делит, а не закрывает, и часть ложных Spectra-0 подтверждает. Замена SVM или LightGBM на AASIST3 поэтому уменьшает разнообразие ошибок, на котором держится голос A.

### Synthetic, которые находит Spectra-0 и пропускает AASIST3

11–12 ноября: 27 звонков.

| call_id | Пакет | Речь, с | Длина | Метка | Spectra-0 | AASIST3 | SVM | LightGBM |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| `10666bbe-c36c-461b-8fd1-05d2fe5648af` | PACK1 | 12.78 | средняя | synthetic_tts | synthetic | human | synthetic | synthetic |
| `265eeedc-a234-4ecc-b436-c7550a954be9` | PACK1 | 10.74 | средняя | synthetic_tts | synthetic | human | human | human |
| `2fe261b0-4c8e-4050-a656-1be86cb7c0ad` | PACK1 | 16.76 | средняя | synthetic_tts | synthetic | human | synthetic | human |
| `30432a67-8696-4c6f-b8b9-7d06ec90315d` | PACK1 | 13.40 | средняя | synthetic_tts | synthetic | human | synthetic | synthetic |
| `312e2eca-469c-4065-9a7d-3e6eff53b271` | PACK1 | 12.52 | средняя | synthetic_tts | synthetic | human | human | human |
| `33f6d4aa-8f87-475d-838e-283ef0c2d721` | PACK1 | 13.10 | средняя | synthetic_tts | synthetic | human | synthetic | synthetic |
| `3f2d0b62-2c7f-4701-a91d-f12e404d4bc5` | PACK1 | 17.12 | длинная | synthetic_tts | synthetic | human | synthetic | synthetic |
| `4cd90209-e7c0-41c7-aa67-45d502d485f8` | PACK1 | 12.70 | средняя | synthetic_tts | synthetic | human | synthetic | synthetic |
| `50d74387-c6ad-4084-b66d-5ce66f428762` | PACK1 | 36.24 | длинная | synthetic_tts | synthetic | human | synthetic | synthetic |
| `58e01718-38f5-429d-afdb-a22dbab667ce` | PACK1 | 16.20 | средняя | synthetic_tts | synthetic | human | synthetic | synthetic |
| `5b516489-8166-428d-8988-6cf97aa0ac9a` | PACK1 | 11.02 | средняя | synthetic_tts | synthetic | human | human | human |
| `67631c21-d441-484e-9197-528b76e04d48` | PACK1 | 15.68 | средняя | synthetic_tts | synthetic | human | synthetic | synthetic |
| `7a80cec7-8b1c-46f1-a16a-a34647aac05e` | PACK1 | 7.76 | короткая | synthetic_tts | synthetic | human | synthetic | human |
| `806ce853-7d60-415a-bf42-489d8d2f052c` | PACK1 | 55.58 | длинная | synthetic_tts | synthetic | human | synthetic | synthetic |
| `82663895-4657-4791-a416-5cac11d88332` | PACK1 | 20.14 | длинная | synthetic_tts | synthetic | human | synthetic | human |
| `94f10a95-f479-4c9c-8712-f66b95b8871a` | PACK1 | 7.30 | короткая | synthetic_tts | synthetic | human | synthetic | synthetic |
| `97b5f5be-a7d4-4356-b716-bf0327f98bb9` | PACK1 | 13.76 | средняя | synthetic_tts | synthetic | human | human | synthetic |
| `989ccddb-808a-43ef-be10-9837afe0aa30` | PACK1 | 11.52 | средняя | synthetic_tts | synthetic | human | human | human |
| `a6ef2de6-fab5-479e-80f1-7f6e8b683d5d` | PACK1 | 11.88 | средняя | synthetic_tts | synthetic | human | synthetic | synthetic |
| `abd479d5-2012-4894-96fb-daf5ee080c6b` | PACK1 | 13.26 | средняя | synthetic_tts | synthetic | human | synthetic | synthetic |
| `c97da6ff-bf4b-4d14-a3a8-4d8e74fb55cd` | PACK1 | 9.34 | короткая | synthetic_tts | synthetic | human | human | human |
| `d74433eb-191e-413a-8dc2-c738f7f895d9` | PACK1 | 11.44 | средняя | synthetic_tts | synthetic | human | human | human |
| `df02136b-514c-4602-a790-0c3faef826fd` | PACK1 | 21.12 | длинная | synthetic_tts | synthetic | human | synthetic | synthetic |
| `e9256462-8676-49ed-818b-6c2e99f76ba2` | PACK1 | 13.76 | средняя | synthetic_tts | synthetic | human | synthetic | synthetic |
| `0826c11d-1b43-4043-b7b2-9cf051c65ae9` | PACK2 | 4.34 | короткая | synthetic_tts | synthetic | human | synthetic | synthetic |
| `6ff76ce0-c6e7-4e84-8104-f15237e32faf` | PACK2 | 9.30 | короткая | synthetic_tts | synthetic | human | synthetic | synthetic |
| `91d2767d-a5c6-42d8-aab8-426d4c27b895` | PACK2 | 4.24 | короткая | synthetic_tts | synthetic | human | synthetic | synthetic |

16 ноября: 13 звонков.

| call_id | Пакет | Речь, с | Длина | Метка | Spectra-0 | AASIST3 | SVM | LightGBM |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| `021d5c05-deaa-4630-99c6-c72c76d4f897` | PACK1 | 12.52 | средняя | synthetic_tts | synthetic | human | human | synthetic |
| `34e4efbe-d033-4b9a-948e-ec51aea4ddc8` | PACK1 | 11.52 | средняя | synthetic_tts | synthetic | human | synthetic | synthetic |
| `5218c50b-1d1b-44cc-bd1f-0d6a6d4d46bc` | PACK1 | 15.50 | средняя | synthetic_tts | synthetic | human | synthetic | synthetic |
| `646d0ec4-ac69-440b-bfc8-38c15d57a208` | PACK1 | 13.52 | средняя | synthetic_tts | synthetic | human | synthetic | synthetic |
| `886424dd-0798-4d1b-94e4-349de23cd322` | PACK1 | 11.22 | средняя | synthetic_tts | synthetic | human | human | human |
| `924fe24a-38f8-4c0a-9430-934d29287d56` | PACK1 | 12.82 | средняя | synthetic_tts | synthetic | human | synthetic | synthetic |
| `b6e98f1f-8c14-4dfb-8256-a051eb466341` | PACK1 | 13.74 | средняя | synthetic_tts | synthetic | human | synthetic | synthetic |
| `b9fc56b0-8189-4bb0-9d99-b3f5e494bfa1` | PACK1 | 13.26 | средняя | synthetic_tts | synthetic | human | synthetic | synthetic |
| `d25f82c3-3400-486e-ab8c-5bbb63ff30fe` | PACK1 | 13.14 | средняя | synthetic_tts | synthetic | human | synthetic | synthetic |
| `f0853a9a-580d-4de3-9af1-acce695bb825` | PACK1 | 14.02 | средняя | synthetic_tts | synthetic | human | human | human |
| `f240e62d-086a-433e-8a50-dc99870dc389` | PACK1 | 13.50 | средняя | synthetic_tts | synthetic | human | synthetic | synthetic |
| `f4e73c9b-bdec-4f4e-aa8b-36d2ac513617` | PACK1 | 12.00 | средняя | synthetic_tts | synthetic | human | human | human |
| `f3ad9c59-2ddc-4835-9b8e-868f1fb4321d` | PACK2 | 8.70 | короткая | synthetic_tts | synthetic | human | synthetic | synthetic |

### Synthetic, которые находит AASIST3 и пропускает Spectra-0

Таких звонков нет ни на одной дате.

### Human, которых модель назвала synthetic

11–12 ноября, Spectra-0 30, AASIST3 14, SVM 15, LightGBM 13. Ниже все human-звонки, где хотя бы одна из четырёх моделей сказала synthetic: 53.

| call_id | Пакет | Речь, с | Длина | Метка | Spectra-0 | AASIST3 | SVM | LightGBM |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| `09366939-f8dd-4387-907f-1efdafcb5fba` | PACK1 | 11.46 | средняя | human_live | human | human | human | synthetic |
| `18f09a05-292e-4f1f-b81c-feb29d91bd7b` | PACK1 | 9.66 | короткая | human_live | human | human | synthetic | human |
| `1b04c778-7254-4b92-8b9c-ee1b54d62ef2` | PACK1 | 118.24 | длинная | human_live | human | human | human | synthetic |
| `289bfde2-cfec-4344-badd-deecab7e480b` | PACK1 | 8.86 | короткая | human_live | synthetic | human | human | human |
| `29fb44cf-3980-47a8-861e-b183a7944cdb` | PACK1 | 7.42 | короткая | human_live | synthetic | synthetic | human | human |
| `2fc0606e-9f88-4bea-b9d7-912b67a04725` | PACK1 | 9.08 | короткая | human_live | human | human | human | synthetic |
| `33fb6077-f8a8-4045-8300-aab93bc25834` | PACK1 | 3.72 | короткая | human_live | synthetic | human | human | human |
| `3698e3cb-bee9-48a3-b03d-9e4914088060` | PACK1 | 5.64 | короткая | human_live | synthetic | human | human | human |
| `3740711b-bf70-4ad2-8ec6-e47512af8e14` | PACK1 | 11.56 | средняя | human_live | human | human | synthetic | synthetic |
| `392ca92a-bb64-46dc-9673-9e12a217fc10` | PACK1 | 5.42 | короткая | human_live | synthetic | human | human | human |
| `4acd26ad-94e3-40e1-b881-fa7a2b45b500` | PACK1 | 16.20 | средняя | human_live | human | human | synthetic | human |
| `5079f945-2390-40ed-a9b8-f674fe7ccd36` | PACK1 | 10.40 | средняя | human_live | human | synthetic | synthetic | human |
| `5281300b-4cd6-463f-8cd4-51e2da664ef4` | PACK1 | 16.60 | средняя | human_live | human | human | synthetic | human |
| `69cfb73f-6a74-4659-9a55-e1824b1989ad` | PACK1 | 27.26 | длинная | human_live | synthetic | synthetic | human | human |
| `6a0dbde8-ddd1-4845-bced-3684f87d82d7` | PACK1 | 8.16 | короткая | human_live | synthetic | human | human | human |
| `7f9115d1-7b39-4f3b-9972-360ff62c8486` | PACK1 | 30.18 | длинная | human_live | human | human | synthetic | synthetic |
| `8212bd75-f5c6-4847-ae90-1a8f40cd5460` | PACK1 | 8.68 | короткая | human_live | synthetic | human | human | human |
| `87ae4e7a-57f7-4509-bdc7-1bbbfa5bdd42` | PACK1 | 9.48 | короткая | human_live | synthetic | synthetic | human | human |
| `8cc015c4-6d0e-4ed7-94a9-44fe7e5513a4` | PACK1 | 96.72 | длинная | human_live | human | human | human | synthetic |
| `9ca19e9a-34f1-4a8c-ac26-8f0f48c33077` | PACK1 | 6.22 | короткая | human_live | synthetic | human | human | human |
| `a3cbab21-7c74-43f1-a638-ee6768e1508a` | PACK1 | 7.40 | короткая | human_live | synthetic | human | human | human |
| `acecb457-4110-4062-a469-fa858b7656b9` | PACK1 | 9.98 | средняя | human_live | human | synthetic | synthetic | human |
| `adba2fff-3ac1-4e08-9499-3f92add605c1` | PACK1 | 4.98 | короткая | human_live | synthetic | synthetic | human | synthetic |
| `bc94c174-4eaf-463f-91a8-3777ae16fd5f` | PACK1 | 70.30 | длинная | human_live | human | human | synthetic | human |
| `c1a8a5a1-ea0c-438f-b2c1-45c3a15724b7` | PACK1 | 30.90 | длинная | human_live | human | human | synthetic | synthetic |
| `d28190fa-02ad-474d-bc32-e4dcd15d08b6` | PACK1 | 2.86 | короткая | human_live | synthetic | human | human | human |
| `d862c896-4625-4197-834d-c22ad7201653` | PACK1 | 17.24 | длинная | human_live | human | human | synthetic | synthetic |
| `f8e190f6-f28c-48f7-9c9e-247511b0e5c7` | PACK1 | 5.94 | короткая | human_live | human | synthetic | human | human |
| `05675403-19e8-4957-927c-d79e66327d62` | PACK2 | 19.40 | длинная | human_live | human | human | synthetic | human |
| `099008ea-b3ff-4dd7-b556-522e8586dfbb` | PACK2 | 4.12 | короткая | human_live | human | synthetic | human | human |
| `24370cc2-9c23-424d-872a-26e968111af8` | PACK2 | 19.72 | длинная | human_live | human | human | human | synthetic |
| `24cd3f5f-b754-45dc-bda1-69b2b98d05e6` | PACK2 | 5.86 | короткая | human_live | synthetic | synthetic | human | human |
| `266dc1ed-2521-40b3-8de7-78717daf75cd` | PACK2 | 10.24 | средняя | human_live | human | human | synthetic | human |
| `2cc670e4-f79f-44ed-a28a-00e69a3c5ccb` | PACK2 | 59.64 | длинная | human_live | synthetic | human | human | human |
| `3112b2a4-0100-4349-a41d-c7ec4cdd474d` | PACK2 | 19.72 | длинная | human_live | synthetic | human | human | human |
| `3ef85021-2c86-44bd-af9f-d5e39c625493` | PACK2 | 4.66 | короткая | human_live | synthetic | human | human | human |
| `4534282c-4d9f-4949-8403-70d1b938defc` | PACK2 | 5.54 | короткая | human_live | synthetic | human | human | human |
| `4a67c0ca-b79f-4e5b-8200-5ab8d37446cb` | PACK2 | 12.42 | средняя | human_live | synthetic | human | human | human |
| `613b7c0a-17e8-41ac-aacb-72d6d5b2ce3f` | PACK2 | 13.80 | средняя | human_live | synthetic | synthetic | human | human |
| `61d05b46-d858-4d99-a3de-fdb02538c550` | PACK2 | 10.46 | средняя | human_live | human | synthetic | human | human |
| `782770c8-ba15-40f0-8b09-8c6c53374dae` | PACK2 | 7.66 | короткая | human_live | human | human | synthetic | synthetic |
| `7a3e4fc5-dd9a-43ad-8915-bcc043213235` | PACK2 | 4.88 | короткая | human_live | synthetic | human | human | human |
| `897d0a52-6e25-4bad-9e9f-b55dd2f5d6e1` | PACK2 | 16.58 | средняя | human_live | synthetic | human | human | human |
| `9f1179e9-fb5a-4573-9eb9-34f309bffbdc` | PACK2 | 5.50 | короткая | human_live | synthetic | human | human | human |
| `a0d6408b-29fa-498f-8c3e-ba58b4f6ef2d` | PACK2 | 8.70 | короткая | human_live | synthetic | synthetic | human | human |
| `a59194d7-3316-4e82-8720-14873d899e5e` | PACK2 | 17.04 | средняя | human_live | synthetic | human | human | human |
| `c7cacaf6-7da4-463a-aec7-428afc270818` | PACK2 | 7.36 | короткая | human_live | human | human | synthetic | synthetic |
| `cad0a404-33f7-46d7-b90c-9360a28b112a` | PACK2 | 26.88 | длинная | human_live | human | human | synthetic | synthetic |
| `cf8bcd45-88ed-4452-87bd-8d62189ba51c` | PACK2 | 20.76 | длинная | human_live | synthetic | human | human | human |
| `d6bfae28-a189-4094-a7dc-fb9b85b9396d` | PACK2 | 12.40 | средняя | human_live | synthetic | human | human | human |
| `ef86500b-6ad0-4f68-bb51-ebd95704403b` | PACK2 | 6.52 | короткая | human_live | synthetic | synthetic | human | human |
| `f1030e22-29b3-44ca-a8b5-aa27c2748774` | PACK2 | 6.06 | короткая | human_live | synthetic | synthetic | human | human |
| `f19d6c44-62bf-4e19-b142-61f43d2a5c5b` | PACK2 | 4.50 | короткая | human_live | synthetic | human | human | human |

16 ноября, Spectra-0 26, AASIST3 14, SVM 7, LightGBM 5. Ниже все human-звонки, где хотя бы одна из четырёх моделей сказала synthetic: 41.

| call_id | Пакет | Речь, с | Длина | Метка | Spectra-0 | AASIST3 | SVM | LightGBM |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| `0071b20d-e2cb-485a-96fb-16e3d3b6adf0` | PACK1 | 36.40 | длинная | human_live | human | human | human | synthetic |
| `17b83ac4-7422-43be-84ad-dcbc27c6a03b` | PACK1 | 2.88 | короткая | human_live | synthetic | human | human | human |
| `181b3375-dc18-407c-9ba3-5c0f92fa8404` | PACK1 | 3.12 | короткая | human_live | human | synthetic | human | human |
| `26af9434-3af8-4565-be99-e5840dd8f1e2` | PACK1 | 14.40 | средняя | human_live | synthetic | human | human | human |
| `51fd221e-737c-4aac-8836-aa127d60a0c1` | PACK1 | 3.96 | короткая | human_live | synthetic | synthetic | human | human |
| `55da55cb-da8a-48b0-a25a-7c85213402a6` | PACK1 | 8.64 | короткая | human_live | human | human | synthetic | human |
| `5ff00f96-5f6b-4e0b-96c9-97a0bbcc4995` | PACK1 | 12.82 | средняя | human_live | synthetic | synthetic | human | human |
| `6689c2c5-21bb-452b-926e-0290ab4de8c9` | PACK1 | 3.50 | короткая | human_live | synthetic | synthetic | human | human |
| `9f6deb76-d815-41ab-b1e4-d5062e997956` | PACK1 | 24.04 | длинная | human_live | synthetic | human | human | human |
| `bb926423-2cf5-4795-b996-4817428ecb9c` | PACK1 | 5.56 | короткая | human_live | human | human | synthetic | synthetic |
| `cf88924b-4dbb-4b19-9e9c-055ee3553b9e` | PACK1 | 67.46 | длинная | human_live | human | human | synthetic | synthetic |
| `d0bf96d8-c51c-4135-bb42-8afff2e1997e` | PACK1 | 4.20 | короткая | human_live | human | human | synthetic | human |
| `d544476b-21f6-4e54-b0cc-2f6cd578b45c` | PACK1 | 25.42 | длинная | human_live | synthetic | human | human | human |
| `e449d236-d615-4ba3-bcd6-5a832d9fc48e` | PACK1 | 3.34 | короткая | human_live | synthetic | human | human | human |
| `01125a1b-6f70-4a27-86c0-9660b92487ae` | PACK2 | 7.72 | короткая | human_live | synthetic | human | human | human |
| `069f2a16-2455-43c9-9c62-93fe7f5a8c66` | PACK2 | 3.48 | короткая | human_live | synthetic | human | human | human |
| `0bf51810-9158-4137-92f4-7754e7189f89` | PACK2 | 7.76 | короткая | human_live | synthetic | synthetic | human | human |
| `2350c447-356c-498d-97ff-18ee16b9edca` | PACK2 | 6.28 | короткая | human_live | synthetic | human | human | human |
| `2a724db6-a51b-4071-97d5-9a0f1fc63006` | PACK2 | 6.12 | короткая | human_live | human | synthetic | human | human |
| `2e9b17db-6d8c-4c40-aa44-05a3ffae7514` | PACK2 | 3.62 | короткая | human_live | human | synthetic | human | human |
| `4d1c4cbc-2602-47c9-ab81-aff73d477634` | PACK2 | 12.06 | средняя | human_live | human | human | synthetic | synthetic |
| `4d35d8f9-364b-44e9-8f84-2f4488ba2d59` | PACK2 | 5.48 | короткая | human_live | synthetic | human | human | human |
| `4fa77831-e8f4-44f4-8c5e-e760d28fc9fd` | PACK2 | 4.36 | короткая | human_live | synthetic | synthetic | human | human |
| `50063e1b-7dfe-4a14-a5f0-a7addbfcbf65` | PACK2 | 7.86 | короткая | human_live | synthetic | human | human | human |
| `61f175ed-3ca9-455e-b263-93e5c04db3ea` | PACK2 | 5.12 | короткая | human_live | synthetic | synthetic | human | human |
| `6c5165b0-d385-409a-8538-754455464927` | PACK2 | 10.90 | средняя | human_live | human | human | human | synthetic |
| `6c78ac3d-232c-4452-93c4-bd17ff2a9138` | PACK2 | 7.64 | короткая | human_live | synthetic | human | human | human |
| `6d78162a-14dd-4fe6-9ac4-41e599e7ddc4` | PACK2 | 8.54 | короткая | human_live | synthetic | human | human | human |
| `95f2b453-3770-4c93-a78c-847fe718f371` | PACK2 | 14.08 | средняя | human_live | synthetic | human | human | human |
| `9a970947-77b3-4d17-ba6d-693769f261ff` | PACK2 | 6.62 | короткая | human_live | human | synthetic | human | human |
| `9cfaf2ef-6303-4eb6-afd9-8bd1ae19adec` | PACK2 | 13.04 | средняя | human_live | synthetic | human | human | human |
| `9d1f58ac-bc1a-4ffd-90e3-81fcef7cf4e4` | PACK2 | 12.38 | средняя | human_live | human | human | synthetic | human |
| `a343399f-0873-4e12-91d5-161e89bd5d52` | PACK2 | 12.08 | средняя | human_live | synthetic | human | human | human |
| `b19f70e7-321a-4bb2-a078-04e1efd1e2dd` | PACK2 | 68.70 | длинная | human_live | synthetic | synthetic | human | human |
| `b1b34708-330b-4b3a-892a-0a55e81f6152` | PACK2 | 2.64 | короткая | human_live | human | synthetic | human | human |
| `b1d18800-977d-4aa6-9a88-325a70abca9e` | PACK2 | 3.88 | короткая | human_live | synthetic | human | human | human |
| `b6046250-79be-4160-978a-613d7aded7ba` | PACK2 | 12.10 | средняя | human_live | human | synthetic | human | human |
| `cd3c625c-7807-49f7-98cc-925752868e1d` | PACK2 | 5.34 | короткая | human_live | synthetic | human | human | human |
| `d668b695-e5ce-4603-830e-68176e00a882` | PACK2 | 5.48 | короткая | human_live | synthetic | human | human | human |
| `e4f8339e-9118-4b25-9bc1-a249551f4139` | PACK2 | 8.50 | короткая | human_live | human | human | synthetic | human |
| `ef756599-d127-4da0-b9de-393a8277a9a3` | PACK2 | 3.52 | короткая | human_live | synthetic | synthetic | human | human |

### Ошибки, общие для всех четырёх моделей

11–12 ноября: общие FN 7, общие FP 0.

Общие пропуски synthetic:

| call_id | Пакет | Речь, с | Длина | Метка | Spectra-0 | AASIST3 | SVM | LightGBM |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| `0502ed5b-e9c7-4d07-bb95-a08a9635242f` | PACK1 | 17.06 | длинная | synthetic_tts | human | human | human | human |
| `4205566c-00b2-4466-a890-cd77d1c23a00` | PACK1 | 25.04 | длинная | synthetic_tts | human | human | human | human |
| `5c66b44d-b4ef-4d9b-9f7a-675f6ee9dae5` | PACK1 | 14.44 | средняя | synthetic_tts | human | human | human | human |
| `6b161485-ea53-4975-a9ca-a8e780470751` | PACK1 | 9.24 | короткая | synthetic_tts | human | human | human | human |
| `849f517d-b76f-43c6-9866-22ae0b1ba415` | PACK1 | 5.54 | короткая | synthetic_tts | human | human | human | human |
| `c510304f-75f0-4e83-8793-c3605203bb18` | PACK1 | 17.62 | длинная | synthetic_tts | human | human | human | human |
| `f4d0fd3f-b5a9-4fd0-81bd-07d5425f3086` | PACK1 | 7.68 | короткая | synthetic_tts | human | human | human | human |

Общие ложные synthetic:

Нет human-звонка, которого все четыре модели назвали synthetic.

16 ноября: общие FN 6, общие FP 0.

Общие пропуски synthetic:

| call_id | Пакет | Речь, с | Длина | Метка | Spectra-0 | AASIST3 | SVM | LightGBM |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| `279953c2-c39a-40b9-a2d7-268c7a449cee` | PACK1 | 20.06 | длинная | synthetic_tts | human | human | human | human |
| `5a98b073-8e3f-40c4-af2b-e2cd8d7de045` | PACK1 | 27.68 | длинная | synthetic_tts | human | human | human | human |
| `5af092b8-bb25-4791-ad6a-d0dc68ec8ffb` | PACK1 | 27.18 | длинная | synthetic_tts | human | human | human | human |
| `5ce6abe9-78a3-4c1f-bc5b-c8116ad4fd80` | PACK1 | 28.76 | длинная | synthetic_tts | human | human | human | human |
| `758347c4-a53d-493e-94b4-c8f2dae41232` | PACK1 | 35.34 | длинная | synthetic_tts | human | human | human | human |
| `c76aa475-d6b3-4102-a695-ce2b61aabb09` | PACK1 | 9.58 | короткая | synthetic_tts | human | human | human | human |

Общие ложные synthetic:

Нет human-звонка, которого все четыре модели назвали synthetic.

Решения всех бинарных звонков, пакет, длительность речи и четыре голоса записаны в `reports/voting_combinations/call_decisions.csv`.

## 9. Сравнение с текущей комбинацией

Изменение считается как альтернатива минус A. Положительный recall — больше найденных synthetic. Положительные FPR и FP — больше ложных synthetic на человеческой речи. Положительный FN — больше пропусков. Процентные пункты — это разница долей, умноженная на 100. Один звонок на 11–12 — это 0.75 п.п. recall и 0.31 п.п. FPR. Один звонок 16 ноября — 1.72 п.п. recall и 0.33 п.п. FPR.

| Дата | Вариант | Δ Recall, п.п. | Δ Precision, п.п. | Δ FPR, п.п. | Δ FP | Δ FN | Δ ошибок |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 11–12 | B. Spectra-0 + AASIST3 + LightGBM | -11.94 | -1.99 | +0.31 | +1 | +16 | +17 |
| 11–12 | C. Spectra-0 + AASIST3 + SVM | -10.45 | -3.54 | +0.93 | +3 | +14 | +17 |
| 11–12 | D. AASIST3 + SVM + LightGBM | -2.99 | -1.86 | +0.62 | +2 | +4 | +6 |
| 16 | B. Spectra-0 + AASIST3 + LightGBM | -5.17 | -10.14 | +1.65 | +5 | +3 | +8 |
| 16 | C. Spectra-0 + AASIST3 + SVM | -6.90 | -10.50 | +1.65 | +5 | +4 | +9 |
| 16 | D. AASIST3 + SVM + LightGBM | -1.72 | -0.14 | +0.00 | +0 | +1 | +1 |

Правило 2 из 2 против A, той же арифметикой, но это не кандидат в ряд majority:

| Дата | Вариант | Δ Recall, п.п. | Δ Precision, п.п. | Δ FPR, п.п. | Δ FP | Δ FN | Δ ошибок |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 11–12 | 2 из 2. Spectra-0 + AASIST3, оба голоса | -25.37 | -3.88 | +0.31 | +1 | +34 | +35 |
| 16 | 2 из 2. Spectra-0 + AASIST3, оба голоса | -22.41 | -14.53 | +1.65 | +5 | +13 | +18 |

Смена итога относительно A разбирается по звонкам. Исправленный FN — synthetic, который A пропустил, а альтернатива взяла. Новый FN — наоборот. Для FP то же на человеческой речи.

| Дата | Вариант | Исправленный FN | Новый FN | Исправленный FP | Новый FP | Звонков с другим итогом |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 11–12 | B | 7 | 23 | 7 | 8 | 45 |
| 11–12 | C | 7 | 21 | 7 | 10 | 45 |
| 11–12 | D | 0 | 4 | 0 | 2 | 6 |
| 11–12 | 2 из 2 | 7 | 41 | 7 | 8 | 63 |
| 16 | B | 5 | 8 | 3 | 8 | 24 |
| 16 | C | 5 | 9 | 3 | 8 | 25 |
| 16 | D | 0 | 1 | 0 | 0 | 1 |
| 16 | 2 из 2 | 5 | 18 | 3 | 8 | 34 |

11–12 ноября, B:

| Что изменилось | call_id | Пакет | Речь, с | Метка | Spectra-0 | AASIST3 | SVM | LightGBM |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| исправленный FN | `1ad13b4a-2179-4c6a-a61b-186e01ec3d58` | PACK1 | 11.24 | synthetic_tts | synthetic | synthetic | human | human |
| исправленный FN | `378f5cd8-8118-4284-b886-824c7b1d7300` | PACK1 | 11.18 | synthetic_tts | synthetic | synthetic | human | human |
| исправленный FN | `515a85c3-f53c-48f9-85b7-040707fc40b1` | PACK1 | 5.40 | synthetic_tts | synthetic | synthetic | human | human |
| исправленный FN | `7d2de0e2-0b71-4d4a-a63d-357458b85d8f` | PACK1 | 12.62 | synthetic_tts | synthetic | synthetic | human | human |
| исправленный FN | `a7d2b48a-22b1-4d1a-a19a-5cf83e8492c2` | PACK1 | 11.36 | synthetic_tts | synthetic | synthetic | human | human |
| исправленный FN | `ca4e5833-7aaf-4805-8c44-a446c4f6f145` | PACK2 | 20.02 | synthetic_tts | synthetic | synthetic | human | human |
| исправленный FN | `fe7144ca-c49c-453d-9335-3b859fa1ac7f` | PACK1 | 11.04 | synthetic_tts | synthetic | synthetic | human | human |
| новый FN | `04e812f4-f546-4c67-a3e2-805271479970` | PACK1 | 9.36 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `279f97ea-1934-4cdc-a181-8d4f581987fd` | PACK1 | 4.16 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `2fe261b0-4c8e-4050-a656-1be86cb7c0ad` | PACK1 | 16.76 | synthetic_tts | synthetic | human | synthetic | human |
| новый FN | `30e91fa0-eb89-417a-b8b9-798aa10e2206` | PACK1 | 156.80 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `38ae525c-aa9c-407d-9c4c-db9a0ecaa3de` | PACK1 | 4.64 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `3a84e7c2-ccd1-4bd2-a738-983df255c7df` | PACK1 | 7.58 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `3cfe0ea9-8763-42e4-9a37-6e4cda1117a8` | PACK1 | 34.62 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `400f065f-e745-4510-b815-ce02f2ab2f52` | PACK1 | 13.90 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `50de2ea5-a6d9-4775-bd52-46d3d7cb3238` | PACK1 | 13.24 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `7a80cec7-8b1c-46f1-a16a-a34647aac05e` | PACK1 | 7.76 | synthetic_tts | synthetic | human | synthetic | human |
| новый FN | `7e6273fd-862f-4651-8a7f-5a855770e987` | PACK1 | 62.80 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `82663895-4657-4791-a416-5cac11d88332` | PACK1 | 20.14 | synthetic_tts | synthetic | human | synthetic | human |
| новый FN | `8dd38967-a03f-473e-ba70-07beb61adb84` | PACK1 | 13.78 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `c1f9f287-fa45-4073-9bf7-1882333cc5b1` | PACK1 | 12.82 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `c6696d60-dbd2-4f9d-b9e5-ea3c2aee88e4` | PACK1 | 10.86 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `dc14c9b3-90eb-4785-8599-2e2c23210b9f` | PACK1 | 39.38 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `e33a33f0-c112-4e7b-aec5-76fded49ad05` | PACK1 | 14.44 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `e7d96368-2789-4bb6-9e51-a5e85b9919cf` | PACK1 | 83.70 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `eaeb77d5-cf36-48e1-9fdc-f7cf8d0e0e53` | PACK1 | 37.68 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `f7ac24f8-0e0a-4041-b3e8-3220affaef07` | PACK1 | 232.38 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `f9c5a954-9c64-40c5-b69b-cee9f74293ad` | PACK1 | 17.82 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `f9e2f5b7-06d8-47a4-9db4-ee2bede482b7` | PACK1 | 13.86 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `fb1157ee-1541-4afb-9e89-a4281539bcc6` | PACK1 | 28.72 | synthetic_tts | human | human | synthetic | synthetic |
| исправленный FP | `3740711b-bf70-4ad2-8ec6-e47512af8e14` | PACK1 | 11.56 | human_live | human | human | synthetic | synthetic |
| исправленный FP | `782770c8-ba15-40f0-8b09-8c6c53374dae` | PACK2 | 7.66 | human_live | human | human | synthetic | synthetic |
| исправленный FP | `7f9115d1-7b39-4f3b-9972-360ff62c8486` | PACK1 | 30.18 | human_live | human | human | synthetic | synthetic |
| исправленный FP | `c1a8a5a1-ea0c-438f-b2c1-45c3a15724b7` | PACK1 | 30.90 | human_live | human | human | synthetic | synthetic |
| исправленный FP | `c7cacaf6-7da4-463a-aec7-428afc270818` | PACK2 | 7.36 | human_live | human | human | synthetic | synthetic |
| исправленный FP | `cad0a404-33f7-46d7-b90c-9360a28b112a` | PACK2 | 26.88 | human_live | human | human | synthetic | synthetic |
| исправленный FP | `d862c896-4625-4197-834d-c22ad7201653` | PACK1 | 17.24 | human_live | human | human | synthetic | synthetic |
| новый FP | `24cd3f5f-b754-45dc-bda1-69b2b98d05e6` | PACK2 | 5.86 | human_live | synthetic | synthetic | human | human |
| новый FP | `29fb44cf-3980-47a8-861e-b183a7944cdb` | PACK1 | 7.42 | human_live | synthetic | synthetic | human | human |
| новый FP | `613b7c0a-17e8-41ac-aacb-72d6d5b2ce3f` | PACK2 | 13.80 | human_live | synthetic | synthetic | human | human |
| новый FP | `69cfb73f-6a74-4659-9a55-e1824b1989ad` | PACK1 | 27.26 | human_live | synthetic | synthetic | human | human |
| новый FP | `87ae4e7a-57f7-4509-bdc7-1bbbfa5bdd42` | PACK1 | 9.48 | human_live | synthetic | synthetic | human | human |
| новый FP | `a0d6408b-29fa-498f-8c3e-ba58b4f6ef2d` | PACK2 | 8.70 | human_live | synthetic | synthetic | human | human |
| новый FP | `ef86500b-6ad0-4f68-bb51-ebd95704403b` | PACK2 | 6.52 | human_live | synthetic | synthetic | human | human |
| новый FP | `f1030e22-29b3-44ca-a8b5-aa27c2748774` | PACK2 | 6.06 | human_live | synthetic | synthetic | human | human |

11–12 ноября, C:

| Что изменилось | call_id | Пакет | Речь, с | Метка | Spectra-0 | AASIST3 | SVM | LightGBM |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| исправленный FN | `1ad13b4a-2179-4c6a-a61b-186e01ec3d58` | PACK1 | 11.24 | synthetic_tts | synthetic | synthetic | human | human |
| исправленный FN | `378f5cd8-8118-4284-b886-824c7b1d7300` | PACK1 | 11.18 | synthetic_tts | synthetic | synthetic | human | human |
| исправленный FN | `515a85c3-f53c-48f9-85b7-040707fc40b1` | PACK1 | 5.40 | synthetic_tts | synthetic | synthetic | human | human |
| исправленный FN | `7d2de0e2-0b71-4d4a-a63d-357458b85d8f` | PACK1 | 12.62 | synthetic_tts | synthetic | synthetic | human | human |
| исправленный FN | `a7d2b48a-22b1-4d1a-a19a-5cf83e8492c2` | PACK1 | 11.36 | synthetic_tts | synthetic | synthetic | human | human |
| исправленный FN | `ca4e5833-7aaf-4805-8c44-a446c4f6f145` | PACK2 | 20.02 | synthetic_tts | synthetic | synthetic | human | human |
| исправленный FN | `fe7144ca-c49c-453d-9335-3b859fa1ac7f` | PACK1 | 11.04 | synthetic_tts | synthetic | synthetic | human | human |
| новый FN | `04e812f4-f546-4c67-a3e2-805271479970` | PACK1 | 9.36 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `279f97ea-1934-4cdc-a181-8d4f581987fd` | PACK1 | 4.16 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `30e91fa0-eb89-417a-b8b9-798aa10e2206` | PACK1 | 156.80 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `38ae525c-aa9c-407d-9c4c-db9a0ecaa3de` | PACK1 | 4.64 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `3a84e7c2-ccd1-4bd2-a738-983df255c7df` | PACK1 | 7.58 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `3cfe0ea9-8763-42e4-9a37-6e4cda1117a8` | PACK1 | 34.62 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `400f065f-e745-4510-b815-ce02f2ab2f52` | PACK1 | 13.90 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `50de2ea5-a6d9-4775-bd52-46d3d7cb3238` | PACK1 | 13.24 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `7e6273fd-862f-4651-8a7f-5a855770e987` | PACK1 | 62.80 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `8dd38967-a03f-473e-ba70-07beb61adb84` | PACK1 | 13.78 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `97b5f5be-a7d4-4356-b716-bf0327f98bb9` | PACK1 | 13.76 | synthetic_tts | synthetic | human | human | synthetic |
| новый FN | `c1f9f287-fa45-4073-9bf7-1882333cc5b1` | PACK1 | 12.82 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `c6696d60-dbd2-4f9d-b9e5-ea3c2aee88e4` | PACK1 | 10.86 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `dc14c9b3-90eb-4785-8599-2e2c23210b9f` | PACK1 | 39.38 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `e33a33f0-c112-4e7b-aec5-76fded49ad05` | PACK1 | 14.44 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `e7d96368-2789-4bb6-9e51-a5e85b9919cf` | PACK1 | 83.70 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `eaeb77d5-cf36-48e1-9fdc-f7cf8d0e0e53` | PACK1 | 37.68 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `f7ac24f8-0e0a-4041-b3e8-3220affaef07` | PACK1 | 232.38 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `f9c5a954-9c64-40c5-b69b-cee9f74293ad` | PACK1 | 17.82 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `f9e2f5b7-06d8-47a4-9db4-ee2bede482b7` | PACK1 | 13.86 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `fb1157ee-1541-4afb-9e89-a4281539bcc6` | PACK1 | 28.72 | synthetic_tts | human | human | synthetic | synthetic |
| исправленный FP | `3740711b-bf70-4ad2-8ec6-e47512af8e14` | PACK1 | 11.56 | human_live | human | human | synthetic | synthetic |
| исправленный FP | `782770c8-ba15-40f0-8b09-8c6c53374dae` | PACK2 | 7.66 | human_live | human | human | synthetic | synthetic |
| исправленный FP | `7f9115d1-7b39-4f3b-9972-360ff62c8486` | PACK1 | 30.18 | human_live | human | human | synthetic | synthetic |
| исправленный FP | `c1a8a5a1-ea0c-438f-b2c1-45c3a15724b7` | PACK1 | 30.90 | human_live | human | human | synthetic | synthetic |
| исправленный FP | `c7cacaf6-7da4-463a-aec7-428afc270818` | PACK2 | 7.36 | human_live | human | human | synthetic | synthetic |
| исправленный FP | `cad0a404-33f7-46d7-b90c-9360a28b112a` | PACK2 | 26.88 | human_live | human | human | synthetic | synthetic |
| исправленный FP | `d862c896-4625-4197-834d-c22ad7201653` | PACK1 | 17.24 | human_live | human | human | synthetic | synthetic |
| новый FP | `24cd3f5f-b754-45dc-bda1-69b2b98d05e6` | PACK2 | 5.86 | human_live | synthetic | synthetic | human | human |
| новый FP | `29fb44cf-3980-47a8-861e-b183a7944cdb` | PACK1 | 7.42 | human_live | synthetic | synthetic | human | human |
| новый FP | `5079f945-2390-40ed-a9b8-f674fe7ccd36` | PACK1 | 10.40 | human_live | human | synthetic | synthetic | human |
| новый FP | `613b7c0a-17e8-41ac-aacb-72d6d5b2ce3f` | PACK2 | 13.80 | human_live | synthetic | synthetic | human | human |
| новый FP | `69cfb73f-6a74-4659-9a55-e1824b1989ad` | PACK1 | 27.26 | human_live | synthetic | synthetic | human | human |
| новый FP | `87ae4e7a-57f7-4509-bdc7-1bbbfa5bdd42` | PACK1 | 9.48 | human_live | synthetic | synthetic | human | human |
| новый FP | `a0d6408b-29fa-498f-8c3e-ba58b4f6ef2d` | PACK2 | 8.70 | human_live | synthetic | synthetic | human | human |
| новый FP | `acecb457-4110-4062-a469-fa858b7656b9` | PACK1 | 9.98 | human_live | human | synthetic | synthetic | human |
| новый FP | `ef86500b-6ad0-4f68-bb51-ebd95704403b` | PACK2 | 6.52 | human_live | synthetic | synthetic | human | human |
| новый FP | `f1030e22-29b3-44ca-a8b5-aa27c2748774` | PACK2 | 6.06 | human_live | synthetic | synthetic | human | human |

11–12 ноября, D:

| Что изменилось | call_id | Пакет | Речь, с | Метка | Spectra-0 | AASIST3 | SVM | LightGBM |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| новый FN | `2fe261b0-4c8e-4050-a656-1be86cb7c0ad` | PACK1 | 16.76 | synthetic_tts | synthetic | human | synthetic | human |
| новый FN | `7a80cec7-8b1c-46f1-a16a-a34647aac05e` | PACK1 | 7.76 | synthetic_tts | synthetic | human | synthetic | human |
| новый FN | `82663895-4657-4791-a416-5cac11d88332` | PACK1 | 20.14 | synthetic_tts | synthetic | human | synthetic | human |
| новый FN | `97b5f5be-a7d4-4356-b716-bf0327f98bb9` | PACK1 | 13.76 | synthetic_tts | synthetic | human | human | synthetic |
| новый FP | `5079f945-2390-40ed-a9b8-f674fe7ccd36` | PACK1 | 10.40 | human_live | human | synthetic | synthetic | human |
| новый FP | `acecb457-4110-4062-a469-fa858b7656b9` | PACK1 | 9.98 | human_live | human | synthetic | synthetic | human |

11–12 ноября, 2 из 2:

| Что изменилось | call_id | Пакет | Речь, с | Метка | Spectra-0 | AASIST3 | SVM | LightGBM |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| исправленный FN | `1ad13b4a-2179-4c6a-a61b-186e01ec3d58` | PACK1 | 11.24 | synthetic_tts | synthetic | synthetic | human | human |
| исправленный FN | `378f5cd8-8118-4284-b886-824c7b1d7300` | PACK1 | 11.18 | synthetic_tts | synthetic | synthetic | human | human |
| исправленный FN | `515a85c3-f53c-48f9-85b7-040707fc40b1` | PACK1 | 5.40 | synthetic_tts | synthetic | synthetic | human | human |
| исправленный FN | `7d2de0e2-0b71-4d4a-a63d-357458b85d8f` | PACK1 | 12.62 | synthetic_tts | synthetic | synthetic | human | human |
| исправленный FN | `a7d2b48a-22b1-4d1a-a19a-5cf83e8492c2` | PACK1 | 11.36 | synthetic_tts | synthetic | synthetic | human | human |
| исправленный FN | `ca4e5833-7aaf-4805-8c44-a446c4f6f145` | PACK2 | 20.02 | synthetic_tts | synthetic | synthetic | human | human |
| исправленный FN | `fe7144ca-c49c-453d-9335-3b859fa1ac7f` | PACK1 | 11.04 | synthetic_tts | synthetic | synthetic | human | human |
| новый FN | `04e812f4-f546-4c67-a3e2-805271479970` | PACK1 | 9.36 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `0826c11d-1b43-4043-b7b2-9cf051c65ae9` | PACK2 | 4.34 | synthetic_tts | synthetic | human | synthetic | synthetic |
| новый FN | `10666bbe-c36c-461b-8fd1-05d2fe5648af` | PACK1 | 12.78 | synthetic_tts | synthetic | human | synthetic | synthetic |
| новый FN | `279f97ea-1934-4cdc-a181-8d4f581987fd` | PACK1 | 4.16 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `2fe261b0-4c8e-4050-a656-1be86cb7c0ad` | PACK1 | 16.76 | synthetic_tts | synthetic | human | synthetic | human |
| новый FN | `30432a67-8696-4c6f-b8b9-7d06ec90315d` | PACK1 | 13.40 | synthetic_tts | synthetic | human | synthetic | synthetic |
| новый FN | `30e91fa0-eb89-417a-b8b9-798aa10e2206` | PACK1 | 156.80 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `33f6d4aa-8f87-475d-838e-283ef0c2d721` | PACK1 | 13.10 | synthetic_tts | synthetic | human | synthetic | synthetic |
| новый FN | `38ae525c-aa9c-407d-9c4c-db9a0ecaa3de` | PACK1 | 4.64 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `3a84e7c2-ccd1-4bd2-a738-983df255c7df` | PACK1 | 7.58 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `3cfe0ea9-8763-42e4-9a37-6e4cda1117a8` | PACK1 | 34.62 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `3f2d0b62-2c7f-4701-a91d-f12e404d4bc5` | PACK1 | 17.12 | synthetic_tts | synthetic | human | synthetic | synthetic |
| новый FN | `400f065f-e745-4510-b815-ce02f2ab2f52` | PACK1 | 13.90 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `4cd90209-e7c0-41c7-aa67-45d502d485f8` | PACK1 | 12.70 | synthetic_tts | synthetic | human | synthetic | synthetic |
| новый FN | `50d74387-c6ad-4084-b66d-5ce66f428762` | PACK1 | 36.24 | synthetic_tts | synthetic | human | synthetic | synthetic |
| новый FN | `50de2ea5-a6d9-4775-bd52-46d3d7cb3238` | PACK1 | 13.24 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `58e01718-38f5-429d-afdb-a22dbab667ce` | PACK1 | 16.20 | synthetic_tts | synthetic | human | synthetic | synthetic |
| новый FN | `67631c21-d441-484e-9197-528b76e04d48` | PACK1 | 15.68 | synthetic_tts | synthetic | human | synthetic | synthetic |
| новый FN | `6ff76ce0-c6e7-4e84-8104-f15237e32faf` | PACK2 | 9.30 | synthetic_tts | synthetic | human | synthetic | synthetic |
| новый FN | `7a80cec7-8b1c-46f1-a16a-a34647aac05e` | PACK1 | 7.76 | synthetic_tts | synthetic | human | synthetic | human |
| новый FN | `7e6273fd-862f-4651-8a7f-5a855770e987` | PACK1 | 62.80 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `806ce853-7d60-415a-bf42-489d8d2f052c` | PACK1 | 55.58 | synthetic_tts | synthetic | human | synthetic | synthetic |
| новый FN | `82663895-4657-4791-a416-5cac11d88332` | PACK1 | 20.14 | synthetic_tts | synthetic | human | synthetic | human |
| новый FN | `8dd38967-a03f-473e-ba70-07beb61adb84` | PACK1 | 13.78 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `91d2767d-a5c6-42d8-aab8-426d4c27b895` | PACK2 | 4.24 | synthetic_tts | synthetic | human | synthetic | synthetic |
| новый FN | `94f10a95-f479-4c9c-8712-f66b95b8871a` | PACK1 | 7.30 | synthetic_tts | synthetic | human | synthetic | synthetic |
| новый FN | `97b5f5be-a7d4-4356-b716-bf0327f98bb9` | PACK1 | 13.76 | synthetic_tts | synthetic | human | human | synthetic |
| новый FN | `a6ef2de6-fab5-479e-80f1-7f6e8b683d5d` | PACK1 | 11.88 | synthetic_tts | synthetic | human | synthetic | synthetic |
| новый FN | `abd479d5-2012-4894-96fb-daf5ee080c6b` | PACK1 | 13.26 | synthetic_tts | synthetic | human | synthetic | synthetic |
| новый FN | `c1f9f287-fa45-4073-9bf7-1882333cc5b1` | PACK1 | 12.82 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `c6696d60-dbd2-4f9d-b9e5-ea3c2aee88e4` | PACK1 | 10.86 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `dc14c9b3-90eb-4785-8599-2e2c23210b9f` | PACK1 | 39.38 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `df02136b-514c-4602-a790-0c3faef826fd` | PACK1 | 21.12 | synthetic_tts | synthetic | human | synthetic | synthetic |
| новый FN | `e33a33f0-c112-4e7b-aec5-76fded49ad05` | PACK1 | 14.44 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `e7d96368-2789-4bb6-9e51-a5e85b9919cf` | PACK1 | 83.70 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `e9256462-8676-49ed-818b-6c2e99f76ba2` | PACK1 | 13.76 | synthetic_tts | synthetic | human | synthetic | synthetic |
| новый FN | `eaeb77d5-cf36-48e1-9fdc-f7cf8d0e0e53` | PACK1 | 37.68 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `f7ac24f8-0e0a-4041-b3e8-3220affaef07` | PACK1 | 232.38 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `f9c5a954-9c64-40c5-b69b-cee9f74293ad` | PACK1 | 17.82 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `f9e2f5b7-06d8-47a4-9db4-ee2bede482b7` | PACK1 | 13.86 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `fb1157ee-1541-4afb-9e89-a4281539bcc6` | PACK1 | 28.72 | synthetic_tts | human | human | synthetic | synthetic |
| исправленный FP | `3740711b-bf70-4ad2-8ec6-e47512af8e14` | PACK1 | 11.56 | human_live | human | human | synthetic | synthetic |
| исправленный FP | `782770c8-ba15-40f0-8b09-8c6c53374dae` | PACK2 | 7.66 | human_live | human | human | synthetic | synthetic |
| исправленный FP | `7f9115d1-7b39-4f3b-9972-360ff62c8486` | PACK1 | 30.18 | human_live | human | human | synthetic | synthetic |
| исправленный FP | `c1a8a5a1-ea0c-438f-b2c1-45c3a15724b7` | PACK1 | 30.90 | human_live | human | human | synthetic | synthetic |
| исправленный FP | `c7cacaf6-7da4-463a-aec7-428afc270818` | PACK2 | 7.36 | human_live | human | human | synthetic | synthetic |
| исправленный FP | `cad0a404-33f7-46d7-b90c-9360a28b112a` | PACK2 | 26.88 | human_live | human | human | synthetic | synthetic |
| исправленный FP | `d862c896-4625-4197-834d-c22ad7201653` | PACK1 | 17.24 | human_live | human | human | synthetic | synthetic |
| новый FP | `24cd3f5f-b754-45dc-bda1-69b2b98d05e6` | PACK2 | 5.86 | human_live | synthetic | synthetic | human | human |
| новый FP | `29fb44cf-3980-47a8-861e-b183a7944cdb` | PACK1 | 7.42 | human_live | synthetic | synthetic | human | human |
| новый FP | `613b7c0a-17e8-41ac-aacb-72d6d5b2ce3f` | PACK2 | 13.80 | human_live | synthetic | synthetic | human | human |
| новый FP | `69cfb73f-6a74-4659-9a55-e1824b1989ad` | PACK1 | 27.26 | human_live | synthetic | synthetic | human | human |
| новый FP | `87ae4e7a-57f7-4509-bdc7-1bbbfa5bdd42` | PACK1 | 9.48 | human_live | synthetic | synthetic | human | human |
| новый FP | `a0d6408b-29fa-498f-8c3e-ba58b4f6ef2d` | PACK2 | 8.70 | human_live | synthetic | synthetic | human | human |
| новый FP | `ef86500b-6ad0-4f68-bb51-ebd95704403b` | PACK2 | 6.52 | human_live | synthetic | synthetic | human | human |
| новый FP | `f1030e22-29b3-44ca-a8b5-aa27c2748774` | PACK2 | 6.06 | human_live | synthetic | synthetic | human | human |

16 ноября, B:

| Что изменилось | call_id | Пакет | Речь, с | Метка | Spectra-0 | AASIST3 | SVM | LightGBM |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| исправленный FN | `1d78bf0d-164c-498a-8f9a-7f0319eab3a0` | PACK1 | 12.36 | synthetic_tts | synthetic | synthetic | human | human |
| исправленный FN | `358836c9-470e-488b-8e6b-e2486280858c` | PACK1 | 11.06 | synthetic_tts | synthetic | synthetic | human | human |
| исправленный FN | `9bcebd6d-65ec-4e31-ab28-e687791d0f08` | PACK1 | 14.56 | synthetic_tts | synthetic | synthetic | human | human |
| исправленный FN | `b417b694-4414-4fce-b424-160e80f000af` | PACK1 | 13.74 | synthetic_tts | synthetic | synthetic | human | human |
| исправленный FN | `c60e0b14-4155-41c9-b826-3bc3feb60149` | PACK2 | 7.88 | synthetic_tts | synthetic | synthetic | human | human |
| новый FN | `1029c895-257a-492c-b470-a6bfa06c2e03` | PACK1 | 13.02 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `34d21678-ee72-45f2-929e-03edf0420e12` | PACK1 | 12.24 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `3a8c6cf5-64eb-4dcd-8bc4-3040abc9f5e6` | PACK1 | 23.56 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `4fc1b005-7f63-48fd-8857-d10ce947556e` | PACK1 | 14.36 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `7ad89fd6-c06c-43f1-b6e2-04458ae10123` | PACK1 | 455.24 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `c77a1957-d1aa-46a5-b52a-af99b8a8e7d8` | PACK1 | 231.24 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `d7c183c3-572f-4192-b5cc-48fb6df17e6e` | PACK1 | 187.08 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `e79b0ef7-4459-4f02-94e8-d71322df2089` | PACK1 | 189.70 | synthetic_tts | human | human | synthetic | synthetic |
| исправленный FP | `4d1c4cbc-2602-47c9-ab81-aff73d477634` | PACK2 | 12.06 | human_live | human | human | synthetic | synthetic |
| исправленный FP | `bb926423-2cf5-4795-b996-4817428ecb9c` | PACK1 | 5.56 | human_live | human | human | synthetic | synthetic |
| исправленный FP | `cf88924b-4dbb-4b19-9e9c-055ee3553b9e` | PACK1 | 67.46 | human_live | human | human | synthetic | synthetic |
| новый FP | `0bf51810-9158-4137-92f4-7754e7189f89` | PACK2 | 7.76 | human_live | synthetic | synthetic | human | human |
| новый FP | `4fa77831-e8f4-44f4-8c5e-e760d28fc9fd` | PACK2 | 4.36 | human_live | synthetic | synthetic | human | human |
| новый FP | `51fd221e-737c-4aac-8836-aa127d60a0c1` | PACK1 | 3.96 | human_live | synthetic | synthetic | human | human |
| новый FP | `5ff00f96-5f6b-4e0b-96c9-97a0bbcc4995` | PACK1 | 12.82 | human_live | synthetic | synthetic | human | human |
| новый FP | `61f175ed-3ca9-455e-b263-93e5c04db3ea` | PACK2 | 5.12 | human_live | synthetic | synthetic | human | human |
| новый FP | `6689c2c5-21bb-452b-926e-0290ab4de8c9` | PACK1 | 3.50 | human_live | synthetic | synthetic | human | human |
| новый FP | `b19f70e7-321a-4bb2-a078-04e1efd1e2dd` | PACK2 | 68.70 | human_live | synthetic | synthetic | human | human |
| новый FP | `ef756599-d127-4da0-b9de-393a8277a9a3` | PACK2 | 3.52 | human_live | synthetic | synthetic | human | human |

16 ноября, C:

| Что изменилось | call_id | Пакет | Речь, с | Метка | Spectra-0 | AASIST3 | SVM | LightGBM |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| исправленный FN | `1d78bf0d-164c-498a-8f9a-7f0319eab3a0` | PACK1 | 12.36 | synthetic_tts | synthetic | synthetic | human | human |
| исправленный FN | `358836c9-470e-488b-8e6b-e2486280858c` | PACK1 | 11.06 | synthetic_tts | synthetic | synthetic | human | human |
| исправленный FN | `9bcebd6d-65ec-4e31-ab28-e687791d0f08` | PACK1 | 14.56 | synthetic_tts | synthetic | synthetic | human | human |
| исправленный FN | `b417b694-4414-4fce-b424-160e80f000af` | PACK1 | 13.74 | synthetic_tts | synthetic | synthetic | human | human |
| исправленный FN | `c60e0b14-4155-41c9-b826-3bc3feb60149` | PACK2 | 7.88 | synthetic_tts | synthetic | synthetic | human | human |
| новый FN | `021d5c05-deaa-4630-99c6-c72c76d4f897` | PACK1 | 12.52 | synthetic_tts | synthetic | human | human | synthetic |
| новый FN | `1029c895-257a-492c-b470-a6bfa06c2e03` | PACK1 | 13.02 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `34d21678-ee72-45f2-929e-03edf0420e12` | PACK1 | 12.24 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `3a8c6cf5-64eb-4dcd-8bc4-3040abc9f5e6` | PACK1 | 23.56 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `4fc1b005-7f63-48fd-8857-d10ce947556e` | PACK1 | 14.36 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `7ad89fd6-c06c-43f1-b6e2-04458ae10123` | PACK1 | 455.24 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `c77a1957-d1aa-46a5-b52a-af99b8a8e7d8` | PACK1 | 231.24 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `d7c183c3-572f-4192-b5cc-48fb6df17e6e` | PACK1 | 187.08 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `e79b0ef7-4459-4f02-94e8-d71322df2089` | PACK1 | 189.70 | synthetic_tts | human | human | synthetic | synthetic |
| исправленный FP | `4d1c4cbc-2602-47c9-ab81-aff73d477634` | PACK2 | 12.06 | human_live | human | human | synthetic | synthetic |
| исправленный FP | `bb926423-2cf5-4795-b996-4817428ecb9c` | PACK1 | 5.56 | human_live | human | human | synthetic | synthetic |
| исправленный FP | `cf88924b-4dbb-4b19-9e9c-055ee3553b9e` | PACK1 | 67.46 | human_live | human | human | synthetic | synthetic |
| новый FP | `0bf51810-9158-4137-92f4-7754e7189f89` | PACK2 | 7.76 | human_live | synthetic | synthetic | human | human |
| новый FP | `4fa77831-e8f4-44f4-8c5e-e760d28fc9fd` | PACK2 | 4.36 | human_live | synthetic | synthetic | human | human |
| новый FP | `51fd221e-737c-4aac-8836-aa127d60a0c1` | PACK1 | 3.96 | human_live | synthetic | synthetic | human | human |
| новый FP | `5ff00f96-5f6b-4e0b-96c9-97a0bbcc4995` | PACK1 | 12.82 | human_live | synthetic | synthetic | human | human |
| новый FP | `61f175ed-3ca9-455e-b263-93e5c04db3ea` | PACK2 | 5.12 | human_live | synthetic | synthetic | human | human |
| новый FP | `6689c2c5-21bb-452b-926e-0290ab4de8c9` | PACK1 | 3.50 | human_live | synthetic | synthetic | human | human |
| новый FP | `b19f70e7-321a-4bb2-a078-04e1efd1e2dd` | PACK2 | 68.70 | human_live | synthetic | synthetic | human | human |
| новый FP | `ef756599-d127-4da0-b9de-393a8277a9a3` | PACK2 | 3.52 | human_live | synthetic | synthetic | human | human |

16 ноября, D:

| Что изменилось | call_id | Пакет | Речь, с | Метка | Spectra-0 | AASIST3 | SVM | LightGBM |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| новый FN | `021d5c05-deaa-4630-99c6-c72c76d4f897` | PACK1 | 12.52 | synthetic_tts | synthetic | human | human | synthetic |

16 ноября, 2 из 2:

| Что изменилось | call_id | Пакет | Речь, с | Метка | Spectra-0 | AASIST3 | SVM | LightGBM |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| исправленный FN | `1d78bf0d-164c-498a-8f9a-7f0319eab3a0` | PACK1 | 12.36 | synthetic_tts | synthetic | synthetic | human | human |
| исправленный FN | `358836c9-470e-488b-8e6b-e2486280858c` | PACK1 | 11.06 | synthetic_tts | synthetic | synthetic | human | human |
| исправленный FN | `9bcebd6d-65ec-4e31-ab28-e687791d0f08` | PACK1 | 14.56 | synthetic_tts | synthetic | synthetic | human | human |
| исправленный FN | `b417b694-4414-4fce-b424-160e80f000af` | PACK1 | 13.74 | synthetic_tts | synthetic | synthetic | human | human |
| исправленный FN | `c60e0b14-4155-41c9-b826-3bc3feb60149` | PACK2 | 7.88 | synthetic_tts | synthetic | synthetic | human | human |
| новый FN | `021d5c05-deaa-4630-99c6-c72c76d4f897` | PACK1 | 12.52 | synthetic_tts | synthetic | human | human | synthetic |
| новый FN | `1029c895-257a-492c-b470-a6bfa06c2e03` | PACK1 | 13.02 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `34d21678-ee72-45f2-929e-03edf0420e12` | PACK1 | 12.24 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `34e4efbe-d033-4b9a-948e-ec51aea4ddc8` | PACK1 | 11.52 | synthetic_tts | synthetic | human | synthetic | synthetic |
| новый FN | `3a8c6cf5-64eb-4dcd-8bc4-3040abc9f5e6` | PACK1 | 23.56 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `4fc1b005-7f63-48fd-8857-d10ce947556e` | PACK1 | 14.36 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `5218c50b-1d1b-44cc-bd1f-0d6a6d4d46bc` | PACK1 | 15.50 | synthetic_tts | synthetic | human | synthetic | synthetic |
| новый FN | `646d0ec4-ac69-440b-bfc8-38c15d57a208` | PACK1 | 13.52 | synthetic_tts | synthetic | human | synthetic | synthetic |
| новый FN | `7ad89fd6-c06c-43f1-b6e2-04458ae10123` | PACK1 | 455.24 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `924fe24a-38f8-4c0a-9430-934d29287d56` | PACK1 | 12.82 | synthetic_tts | synthetic | human | synthetic | synthetic |
| новый FN | `b6e98f1f-8c14-4dfb-8256-a051eb466341` | PACK1 | 13.74 | synthetic_tts | synthetic | human | synthetic | synthetic |
| новый FN | `b9fc56b0-8189-4bb0-9d99-b3f5e494bfa1` | PACK1 | 13.26 | synthetic_tts | synthetic | human | synthetic | synthetic |
| новый FN | `c77a1957-d1aa-46a5-b52a-af99b8a8e7d8` | PACK1 | 231.24 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `d25f82c3-3400-486e-ab8c-5bbb63ff30fe` | PACK1 | 13.14 | synthetic_tts | synthetic | human | synthetic | synthetic |
| новый FN | `d7c183c3-572f-4192-b5cc-48fb6df17e6e` | PACK1 | 187.08 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `e79b0ef7-4459-4f02-94e8-d71322df2089` | PACK1 | 189.70 | synthetic_tts | human | human | synthetic | synthetic |
| новый FN | `f240e62d-086a-433e-8a50-dc99870dc389` | PACK1 | 13.50 | synthetic_tts | synthetic | human | synthetic | synthetic |
| новый FN | `f3ad9c59-2ddc-4835-9b8e-868f1fb4321d` | PACK2 | 8.70 | synthetic_tts | synthetic | human | synthetic | synthetic |
| исправленный FP | `4d1c4cbc-2602-47c9-ab81-aff73d477634` | PACK2 | 12.06 | human_live | human | human | synthetic | synthetic |
| исправленный FP | `bb926423-2cf5-4795-b996-4817428ecb9c` | PACK1 | 5.56 | human_live | human | human | synthetic | synthetic |
| исправленный FP | `cf88924b-4dbb-4b19-9e9c-055ee3553b9e` | PACK1 | 67.46 | human_live | human | human | synthetic | synthetic |
| новый FP | `0bf51810-9158-4137-92f4-7754e7189f89` | PACK2 | 7.76 | human_live | synthetic | synthetic | human | human |
| новый FP | `4fa77831-e8f4-44f4-8c5e-e760d28fc9fd` | PACK2 | 4.36 | human_live | synthetic | synthetic | human | human |
| новый FP | `51fd221e-737c-4aac-8836-aa127d60a0c1` | PACK1 | 3.96 | human_live | synthetic | synthetic | human | human |
| новый FP | `5ff00f96-5f6b-4e0b-96c9-97a0bbcc4995` | PACK1 | 12.82 | human_live | synthetic | synthetic | human | human |
| новый FP | `61f175ed-3ca9-455e-b263-93e5c04db3ea` | PACK2 | 5.12 | human_live | synthetic | synthetic | human | human |
| новый FP | `6689c2c5-21bb-452b-926e-0290ab4de8c9` | PACK1 | 3.50 | human_live | synthetic | synthetic | human | human |
| новый FP | `b19f70e7-321a-4bb2-a078-04e1efd1e2dd` | PACK2 | 68.70 | human_live | synthetic | synthetic | human | human |
| новый FP | `ef756599-d127-4da0-b9de-393a8277a9a3` | PACK2 | 3.52 | human_live | synthetic | synthetic | human | human |

## 10. Ограничения исследования

Обе когорты уже использовались в предыдущих отчётах. Пороги зафиксированы на валидации 13 ноября. Этот расчёт описывает те же звонки и не является новой независимой проверкой. Он не показывает, как правило поведёт себя на будущих звонках.

11–12 ноября — приоритетная прослушанная когорта, а не случайная выборка всех звонков этих дней. 16 ноября — один размеченный день. Доля synthetic разная: 134/457 и 58/361. PACK2 16 ноября содержит 7 synthetic, короткая речь этого дня — тоже 7. На таких группах один звонок сильно двигает recall.

Score моделей не калибровались и не сравнивались между собой. Голос видит только human/synthetic. Длительность речи взята из интервалов Spectra-0. Classical-длина речи в срезы не подставлялась: на 16 ноября эти два определения заметно расходятся, и прежние срезы использовали интервалы Spectra-0.

Метки ручные. Три спора 16 ноября исключены, а не разрешены большинством. `unknown` и `mixed` не добавлялись в human или synthetic. Ошибочные звонки из счёта не удалялись. Callback-кампания 16 ноября не смешивалась с пакетами.

## 11. Выводы и рекомендации

На 11–12 ноября наибольший recall среди правил 2 из 3 у A: 0.813 (109/134), FP 8. Наименьший FPR у A: 0.025 (8/323), при recall 0.813 (109/134).

На 16 ноября наибольший recall среди правил 2 из 3 у A: 0.741 (43/58), FP 3. Наименьший FPR совпадает у A и D: 0.010 (3/303). A recall 0.741 (43/58), FN 15; D recall 0.724 (42/58), FN 16.

На обоих полных днях наибольший recall у A. На 16 ноября наименьший FPR у A совпадает с другой комбинацией, но у A меньше пропусков synthetic. Это описание двух когорт, а не выбор правила на следующий день.

Отдельный вклад AASIST3 относительно Spectra-0: 11–12: AASIST3 добавляет 0 synthetic сверх Spectra-0 и пропускает 27 synthetic, которые Spectra-0 находит; 16: AASIST3 добавляет 0 synthetic сверх Spectra-0 и пропускает 13 synthetic, которые Spectra-0 находит.

Этого вклада мало для замены SVM или LightGBM. Те закрывают пропуски Spectra-0, которых AASIST3 не видит, и почти не подтверждают её ложные срабатывания. Голос A как раз использует это расхождение. Когда AASIST3 садится на место SVM или LightGBM, третий голос чаще повторяет Spectra-0, чем дополняет её.

По заранее заданному правилу ни B, ни C, ни D не держат одновременно FN и FP не хуже A на обоих днях и на PACK1 с PACK2. Менять текущую комбинацию Spectra-0 + SVM + LightGBM по этим двум когортам не стоит.

Правило 2 из 2 убирает часть ложных ценой пропусков, которые majority ещё берёт вторым голосом classical-модели. Относительно A это 11–12: FN +34, FP +1; 16: FN +13, FP +5. Заменой голоса 2 из 3 оно не является.

Эти выводы относятся к двум уже размеченным когортам. Они не доказывают, что соотношение сохранится на следующих звонках. Для окончательного выбора нужна независимая тестовая выборка, на которой порог и состав голоса заранее зафиксированы и не пересматриваются.

