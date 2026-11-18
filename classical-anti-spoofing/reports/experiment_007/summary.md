# svm

Порог и параметры выбраны на validation. Test в этом запуске не использовался.
Score не является калиброванной вероятностью.
Вариант: no_energy.
До отбора убраны energy_rms и нулевой коэффициент LFCC, MFCC, CQCC: static, delta и delta-delta.
Исключено колонок до отбора: 252.

- train: 1002
- validation: 250
- признаков: 200
- параметры: {'C': 1.0, 'gamma': 'scale'}
- порог machine: -0.38624254602031194

## Validation

- human: 202
- machine: 48
- recall_synthetic: 0.9375
- specificity_human: 0.900990099009901
- precision_synthetic: 0.6923076923076923
- f1_synthetic: 0.7964601769911505
- balanced_accuracy: 0.9192450495049505
- fpr: 0.09900990099009901
- fnr: 0.0625
- roc_auc: 0.9647277227722771
- pr_auc: 0.8229116439324298

## Permutation importance, первые 15

- f0_voiced_fraction__median: 0.01487
- mfcc_delta_01_min__mean: 0.01388
- lfcc_static_03_mean__median: 0.01378
- f0_voiced_fraction__mean: 0.01180
- f0_voiced_fraction__p75: 0.01180
- cqcc_delta2_12_max__p75: 0.01141
- cqcc_delta2_12_max__max: 0.01130
- lfcc_static_03_mean__p25: 0.01120
- lfcc_delta_09_max__mean: 0.00932
- f0_voiced_fraction__p25: 0.00922
- f0_voiced_fraction__min: 0.00873
- mfcc_delta_06_std__median: 0.00873
- acf_harmonicity_mean__min: 0.00842
- acf_harmonicity_mean__p25: 0.00842
- cqcc_delta2_12_max__mean: 0.00833

У RBF SVM нет одного коэффициента на признак. Permutation importance — ассоциация, не причина.
