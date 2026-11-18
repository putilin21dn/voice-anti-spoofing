# logistic

Порог и параметры выбраны на validation. Test в этом запуске не использовался.
Score не является калиброванной вероятностью.

- train: 1002
- validation: 250
- признаков: 200
- параметры: {'C': 0.01}
- порог machine: 0.5237090176534015

## Validation

- human: 202
- machine: 48
- recall_synthetic: 0.9166666666666666
- specificity_human: 0.9356435643564357
- precision_synthetic: 0.7719298245614035
- f1_synthetic: 0.8380952380952381
- balanced_accuracy: 0.9261551155115512
- fpr: 0.06435643564356436
- fnr: 0.08333333333333333
- roc_auc: 0.9631806930693069
- pr_auc: 0.8962132352511566

## Permutation importance, первые 15

- lfcc_static_00_mean__max: 0.00813
- acf_harmonicity_mean__p25: 0.00495
- mfcc_delta_10_min__p75: 0.00396
- lfcc_delta2_00_max__p25: 0.00357
- mfcc_delta_06_max__min: 0.00347
- mfcc_delta2_11_max__median: 0.00347
- mfcc_delta_11_min__p75: 0.00297
- lfcc_delta2_00_max__mean: 0.00258
- mfcc_delta_06_max__median: 0.00248
- mfcc_delta_06_std__min: 0.00248
- mfcc_delta2_06_max__p25: 0.00248
- mfcc_delta2_06_std__min: 0.00248
- mfcc_delta2_06_min__max: 0.00248
- cqcc_delta2_08_max__mean: 0.00248
- lfcc_delta_00_std__p25: 0.00248

Коэффициент относится к стандартизованному признаку. Знак не доказывает причину синтетической речи.
