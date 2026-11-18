# svm

Порог и параметры выбраны на validation. Test в этом запуске не использовался.
Score не является калиброванной вероятностью.

- train: 1002
- validation: 250
- признаков: 200
- параметры: {'C': 3.0, 'gamma': 'scale'}
- порог machine: -0.22144920850405767

## Validation

- human: 202
- machine: 48
- recall_synthetic: 0.9166666666666666
- specificity_human: 0.9455445544554455
- precision_synthetic: 0.8
- f1_synthetic: 0.8543689320388349
- balanced_accuracy: 0.931105610561056
- fpr: 0.054455445544554455
- fnr: 0.08333333333333333
- roc_auc: 0.9600866336633663
- pr_auc: 0.8628300622439233

## Permutation importance, первые 15

- mfcc_static_11_std__min: 0.01815
- mfcc_delta2_00_std__p75: 0.01656
- mfcc_static_07_std__min: 0.01597
- mfcc_static_17_max__p25: 0.01477
- lfcc_static_00_mean__max: 0.01438
- lfcc_static_00_mean__p25: 0.01438
- mfcc_static_11_std__p25: 0.01399
- cqcc_delta2_10_min__p75: 0.01399
- f0_voiced_fraction__std: 0.01378
- cqcc_delta_10_min__p75: 0.01349
- mfcc_delta2_00_std__median: 0.01339
- lfcc_static_18_max__p25: 0.01328
- cqcc_delta2_01_min__mean: 0.01289
- cqcc_delta2_01_std__p25: 0.01240
- mfcc_delta_06_std__min: 0.01240

У RBF SVM нет одного коэффициента на признак. Permutation importance — ассоциация, не причина.
