# svm

Порог и параметры выбраны на validation. Test в этом запуске не использовался.
Score не является калиброванной вероятностью.
Вариант: mfcc.
Оставлены только MFCC начиная с коэффициента 1. Энергия, LFCC, CQCC, спектр, F0 и качество голоса убраны до отбора.
Исключено колонок до отбора: 3066.

- train: 1002
- validation: 250
- признаков: 200
- параметры: {'C': 1.0, 'gamma': 'scale'}
- порог machine: -0.12476249160605453

## Validation

- human: 202
- machine: 48
- recall_synthetic: 0.875
- specificity_human: 0.9108910891089109
- precision_synthetic: 0.7
- f1_synthetic: 0.7777777777777778
- balanced_accuracy: 0.8929455445544554
- fpr: 0.0891089108910891
- fnr: 0.125
- roc_auc: 0.9418316831683168
- pr_auc: 0.8531910637036274

## Permutation importance, первые 15

- mfcc_static_01_mean__min: 0.01526
- mfcc_static_11_mean__p25: 0.01060
- mfcc_static_02_max__mean: 0.00565
- mfcc_static_11_mean__median: 0.00495
- mfcc_static_11_mean__mean: 0.00495
- mfcc_static_09_max__p25: 0.00456
- mfcc_delta_11_std__p25: 0.00446
- mfcc_delta_01_min__mean: 0.00357
- mfcc_static_12_max__mean: 0.00297
- mfcc_delta_07_max__p25: 0.00248
- mfcc_delta2_14_max__p25: 0.00248
- mfcc_delta_14_min__p75: 0.00248
- mfcc_delta_14_min__median: 0.00248
- mfcc_delta_06_min__max: 0.00248
- mfcc_delta_06_max__p75: 0.00198

У RBF SVM нет одного коэффициента на признак. Permutation importance — ассоциация, не причина.
