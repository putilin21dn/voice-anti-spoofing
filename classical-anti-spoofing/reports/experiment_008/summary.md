# lightgbm

Порог и параметры выбраны на validation. Test в этом запуске не использовался.
Score не является калиброванной вероятностью.
Вариант: no_energy.
До отбора убраны energy_rms и нулевой коэффициент LFCC, MFCC, CQCC: static, delta и delta-delta.
Исключено колонок до отбора: 252.

- train: 1002
- validation: 250
- признаков: 200
- параметры: {'n_estimators': 200, 'num_leaves': 15, 'learning_rate': 0.05, 'min_child_samples': 20}
- порог machine: 0.023523178525949477

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
- roc_auc: 0.9595709570957095
- pr_auc: 0.894979464488947

## Permutation importance, первые 15

- spectral_centroid_hz_mean__mean: 0.01607
- cqcc_delta2_12_max__max: 0.01467
- spectral_centroid_hz_mean__p75: 0.01407
- f0_voiced_fraction__p25: 0.01229
- cqcc_delta2_01_std__median: 0.01141
- spectral_flatness_mean__mean: 0.01042
- cqcc_delta2_01_std__mean: 0.00951
- mfcc_static_11_max__min: 0.00891
- mfcc_delta2_06_std__median: 0.00883
- cqcc_delta2_01_std__p25: 0.00833
- zero_crossing_rate_mean__p75: 0.00774
- mfcc_delta2_06_std__p75: 0.00763
- mfcc_static_17_max__p25: 0.00664
- zero_crossing_rate_mean__median: 0.00625
- cqcc_delta_01_std__mean: 0.00555

Gain и permutation importance показывают вклад в эту модель, а не причину синтетической речи.
