# lightgbm

Порог и параметры выбраны на validation. Test в этом запуске не использовался.
Score не является калиброванной вероятностью.
Вариант: shape.
Оставлены коэффициенты LFCC, MFCC и CQCC начиная с 1 и спектральные меры. Энергия, F0 и качество голоса убраны до отбора.
Исключено колонок до отбора: 322.

- train: 1002
- validation: 250
- признаков: 200
- параметры: {'n_estimators': 200, 'num_leaves': 15, 'learning_rate': 0.05, 'min_child_samples': 20}
- порог machine: 0.08587803810710892

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
- roc_auc: 0.9619430693069306
- pr_auc: 0.902936161986271

## Permutation importance, первые 15

- spectral_centroid_hz_mean__mean: 0.02580
- spectral_centroid_hz_mean__p75: 0.02490
- spectral_flatness_mean__mean: 0.02083
- cqcc_delta2_12_max__max: 0.01815
- zero_crossing_rate_mean__p75: 0.01438
- mfcc_static_11_max__min: 0.01180
- lfcc_static_03_mean__median: 0.01151
- zero_crossing_rate_mean__max: 0.01141
- cqcc_delta2_12_max__median: 0.01141
- cqcc_delta2_12_min__median: 0.01141
- cqcc_delta2_12_max__p75: 0.01141
- cqcc_delta2_01_std__mean: 0.01101
- cqcc_delta2_10_min__median: 0.01091
- cqcc_delta_10_max__max: 0.01091
- mfcc_delta2_06_std__p25: 0.01042

Gain и permutation importance показывают вклад в эту модель, а не причину синтетической речи.
