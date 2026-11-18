# logistic

Порог и параметры выбраны на validation. Test в этом запуске не использовался.
Score не является калиброванной вероятностью.
Вариант: voicing.
Оставлены F0, доля озвученных кадров, автокорреляционная гармоничность, jitter, shimmer и HNR. Кепстр и спектр убраны до отбора.
Исключено колонок до отбора: 4592.

- train: 1002
- validation: 250
- признаков: 67
- параметры: {'C': 1.0}
- порог machine: 0.5

## Validation

- human: 202
- machine: 48
- recall_synthetic: 0.9791666666666666
- specificity_human: 0.9207920792079208
- precision_synthetic: 0.746031746031746
- f1_synthetic: 0.8468468468468469
- balanced_accuracy: 0.9499793729372937
- fpr: 0.07920792079207921
- fnr: 0.020833333333333332
- roc_auc: 0.9724628712871287
- pr_auc: 0.8500223578200617

## Permutation importance, первые 15

- f0_mean__p25: 0.12919
- f0_voiced_fraction__median: 0.10029
- f0_range__p25: 0.08936
- f0_iqr__median: 0.08657
- jitter_local__p25: 0.08408
- shimmer_local__min: 0.07081
- f0_iqr__max: 0.06722
- hnr_praat_db__mean: 0.06388
- hnr_praat_db__p25: 0.06190
- hnr_praat_db__std: 0.06147
- f0_range__median: 0.05802
- jitter_local__std: 0.05771
- acf_harmonicity_mean__min: 0.05417
- f0_median__std: 0.05097
- f0_mean__min: 0.04946

Коэффициент относится к стандартизованному признаку. Знак не доказывает причину синтетической речи.
