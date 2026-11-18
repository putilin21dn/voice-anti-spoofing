# lightgbm

Порог и параметры выбраны на validation. Test в этом запуске не использовался.
Score не является калиброванной вероятностью.
Вариант: voicing.
Оставлены F0, доля озвученных кадров, автокорреляционная гармоничность, jitter, shimmer и HNR. Кепстр и спектр убраны до отбора.
Исключено колонок до отбора: 4592.

- train: 1002
- validation: 250
- признаков: 67
- параметры: {'n_estimators': 200, 'num_leaves': 31, 'learning_rate': 0.05, 'min_child_samples': 20}
- порог machine: 0.013990800705877104

## Validation

- human: 202
- machine: 48
- recall_synthetic: 0.9583333333333334
- specificity_human: 0.905940594059406
- precision_synthetic: 0.7076923076923077
- f1_synthetic: 0.8141592920353983
- balanced_accuracy: 0.9321369636963697
- fpr: 0.09405940594059406
- fnr: 0.041666666666666664
- roc_auc: 0.9714315181518152
- pr_auc: 0.9062247537530566

## Permutation importance, первые 15

- f0_voiced_fraction__p25: 0.07760
- f0_range__max: 0.02180
- shimmer_local__p25: 0.01925
- f0_voiced_fraction__p75: 0.01894
- f0_voiced_fraction__median: 0.01696
- hnr_praat_db__min: 0.01665
- shimmer_local__p75: 0.01646
- f0_median__p25: 0.01566
- f0_median__min: 0.01279
- f0_mean__min: 0.01250
- f0_mean__std: 0.01240
- f0_median__std: 0.01190
- f0_range__std: 0.01031
- shimmer_local__min: 0.00932
- f0_voiced_fraction__std: 0.00873

Gain и permutation importance показывают вклад в эту модель, а не причину синтетической речи.
