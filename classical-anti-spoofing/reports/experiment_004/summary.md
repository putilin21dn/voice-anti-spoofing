# lightgbm

Порог и параметры выбраны на validation. Test в этом запуске не использовался.
Score не является калиброванной вероятностью.

- train: 1002
- validation: 250
- признаков: 200
- параметры: {'n_estimators': 200, 'num_leaves': 31, 'learning_rate': 0.05, 'min_child_samples': 20}
- порог machine: 0.04007147659674559

## Validation

- human: 202
- machine: 48
- recall_synthetic: 0.9375
- specificity_human: 0.9257425742574258
- precision_synthetic: 0.75
- f1_synthetic: 0.8333333333333334
- balanced_accuracy: 0.9316212871287128
- fpr: 0.07425742574257425
- fnr: 0.0625
- roc_auc: 0.9755569306930693
- pr_auc: 0.9256466685959821

## Permutation importance, первые 15

- spectral_centroid_hz_mean__mean: 0.03224
- mfcc_delta2_00_std__p75: 0.02133
- lfcc_static_00_mean__median: 0.01925
- zero_crossing_rate_mean__p75: 0.01458
- zero_crossing_rate_mean__median: 0.01349
- mfcc_delta_00_min__max: 0.01300
- mfcc_delta_04_max__mean: 0.01250
- cqcc_delta2_01_std__median: 0.01091
- mfcc_static_17_max__p25: 0.01081
- f0_voiced_fraction__mean: 0.00932
- cqcc_delta_10_max__median: 0.00883
- zero_crossing_rate_mean__mean: 0.00883
- f0_voiced_fraction__p25: 0.00844
- f0_voiced_fraction__min: 0.00833
- spectral_centroid_hz_mean__max: 0.00833

Gain и permutation importance показывают вклад в эту модель, а не причину синтетической речи.
