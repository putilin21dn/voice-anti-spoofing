# logistic

Порог и параметры выбраны на validation. Test в этом запуске не использовался.
Score не является калиброванной вероятностью.
Вариант: no_energy.
До отбора убраны energy_rms и нулевой коэффициент LFCC, MFCC, CQCC: static, delta и delta-delta.
Исключено колонок до отбора: 252.

- train: 1002
- validation: 250
- признаков: 200
- параметры: {'C': 0.01}
- порог machine: 0.40594129203220825

## Validation

- human: 202
- machine: 48
- recall_synthetic: 0.9375
- specificity_human: 0.8762376237623762
- precision_synthetic: 0.6428571428571429
- f1_synthetic: 0.7627118644067796
- balanced_accuracy: 0.9068688118811881
- fpr: 0.12376237623762376
- fnr: 0.0625
- roc_auc: 0.9540016501650165
- pr_auc: 0.8672142833968928

## Permutation importance, первые 15

- f0_voiced_fraction__min: 0.01042
- mfcc_delta_01_min__mean: 0.00972
- lfcc_static_03_max__p25: 0.00972
- cqcc_delta2_05_min__p25: 0.00862
- cqcc_delta2_12_max__max: 0.00862
- f0_voiced_fraction__p25: 0.00734
- mfcc_static_11_max__min: 0.00703
- f0_voiced_fraction__mean: 0.00685
- mfcc_static_11_std__min: 0.00664
- acf_harmonicity_mean__p25: 0.00654
- lfcc_delta_10_std__p25: 0.00654
- acf_harmonicity_mean__mean: 0.00654
- mfcc_delta2_11_min__median: 0.00625
- acf_harmonicity_mean__min: 0.00495
- mfcc_static_17_max__p25: 0.00495

Коэффициент относится к стандартизованному признаку. Знак не доказывает причину синтетической речи.
