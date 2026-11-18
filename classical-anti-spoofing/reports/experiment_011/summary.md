# svm

Порог и параметры выбраны на validation. Test в этом запуске не использовался.
Score не является калиброванной вероятностью.
Вариант: shape.
Оставлены коэффициенты LFCC, MFCC и CQCC начиная с 1 и спектральные меры. Энергия, F0 и качество голоса убраны до отбора.
Исключено колонок до отбора: 322.

- train: 1002
- validation: 250
- признаков: 200
- параметры: {'C': 3.0, 'gamma': 'scale'}
- порог machine: -0.38855994652907133

## Validation

- human: 202
- machine: 48
- recall_synthetic: 0.9166666666666666
- specificity_human: 0.8960396039603961
- precision_synthetic: 0.676923076923077
- f1_synthetic: 0.7787610619469026
- balanced_accuracy: 0.9063531353135313
- fpr: 0.10396039603960396
- fnr: 0.08333333333333333
- roc_auc: 0.9555486798679869
- pr_auc: 0.8152838705667577

## Permutation importance, первые 15

- cqcc_delta_10_max__median: 0.02321
- cqcc_delta2_10_min__median: 0.02162
- cqcc_delta_09_max__p75: 0.02162
- cqcc_delta2_11_min__p75: 0.02112
- cqcc_delta_12_max__p75: 0.02013
- cqcc_delta_10_max__mean: 0.01974
- cqcc_delta_09_min__median: 0.01974
- cqcc_delta_10_max__p75: 0.01925
- mfcc_delta_15_max__p25: 0.01875
- cqcc_delta2_09_min__median: 0.01865
- cqcc_delta_10_max__max: 0.01815
- cqcc_delta2_12_min__median: 0.01755
- cqcc_delta2_09_min__p75: 0.01755
- cqcc_delta2_12_max__mean: 0.01716
- cqcc_delta_12_max__mean: 0.01656

У RBF SVM нет одного коэффициента на признак. Permutation importance — ассоциация, не причина.
