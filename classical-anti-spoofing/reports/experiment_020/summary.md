# lightgbm

Порог и параметры выбраны на validation. Test в этом запуске не использовался.
Score не является калиброванной вероятностью.
Вариант: mfcc.
Оставлены только MFCC начиная с коэффициента 1. Энергия, LFCC, CQCC, спектр, F0 и качество голоса убраны до отбора.
Исключено колонок до отбора: 3066.

- train: 1002
- validation: 250
- признаков: 200
- параметры: {'n_estimators': 200, 'num_leaves': 31, 'learning_rate': 0.05, 'min_child_samples': 20}
- порог machine: 0.13358606780151605

## Validation

- human: 202
- machine: 48
- recall_synthetic: 0.8541666666666666
- specificity_human: 0.9306930693069307
- precision_synthetic: 0.7454545454545455
- f1_synthetic: 0.7961165048543689
- balanced_accuracy: 0.8924298679867987
- fpr: 0.06930693069306931
- fnr: 0.14583333333333334
- roc_auc: 0.9317244224422442
- pr_auc: 0.8226593746867097

## Permutation importance, первые 15

- mfcc_static_11_mean__mean: 0.04959
- mfcc_delta2_01_std__p25: 0.03700
- mfcc_static_01_mean__min: 0.02271
- mfcc_static_02_max__p25: 0.01677
- mfcc_delta_01_min__median: 0.01627
- mfcc_delta_17_max__p75: 0.01370
- mfcc_delta_06_max__p25: 0.00992
- mfcc_delta_19_max__median: 0.00833
- mfcc_delta2_04_std__median: 0.00823
- mfcc_delta_01_min__mean: 0.00734
- mfcc_delta2_01_std__min: 0.00635
- mfcc_static_10_std__p25: 0.00635
- mfcc_delta2_04_std__min: 0.00545
- mfcc_delta_07_std__p25: 0.00516
- mfcc_static_02_max__mean: 0.00505

Gain и permutation importance показывают вклад в эту модель, а не причину синтетической речи.
