# svm

Порог и параметры выбраны на validation. Test в этом запуске не использовался.
Score не является калиброванной вероятностью.
Вариант: voicing.
Оставлены F0, доля озвученных кадров, автокорреляционная гармоничность, jitter, shimmer и HNR. Кепстр и спектр убраны до отбора.
Исключено колонок до отбора: 4592.

- train: 1002
- validation: 250
- признаков: 67
- параметры: {'C': 0.3, 'gamma': 'scale'}
- порог machine: 0.0

## Validation

- human: 202
- machine: 48
- recall_synthetic: 0.9375
- specificity_human: 0.9405940594059405
- precision_synthetic: 0.7894736842105263
- f1_synthetic: 0.8571428571428571
- balanced_accuracy: 0.9390470297029703
- fpr: 0.0594059405940594
- fnr: 0.0625
- roc_auc: 0.9735973597359736
- pr_auc: 0.8757286838201164

## Permutation importance, первые 15

- f0_median__min: 0.02211
- f0_voiced_fraction__median: 0.02193
- f0_voiced_fraction__p25: 0.02143
- f0_mean__min: 0.02052
- f0_voiced_fraction__mean: 0.01935
- jitter_local__p25: 0.01854
- shimmer_local__min: 0.01656
- f0_voiced_fraction__p75: 0.01617
- f0_range__p25: 0.01409
- f0_voiced_fraction__max: 0.01349
- jitter_local__median: 0.01279
- f0_range__median: 0.00943
- f0_range__mean: 0.00893
- acf_harmonicity_mean__std: 0.00883
- f0_range__max: 0.00833

У RBF SVM нет одного коэффициента на признак. Permutation importance — ассоциация, не причина.
