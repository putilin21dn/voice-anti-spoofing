# logistic

Порог и параметры выбраны на validation. Test в этом запуске не использовался.
Score не является калиброванной вероятностью.

- train: 1002
- validation: 250
- признаков: 200
- параметры: {'C': 0.01}
- порог machine: 0.49265447022723596

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
- roc_auc: 0.9631806930693069
- pr_auc: 0.8962132352511566

## Permutation importance, первые 15

- lfcc_static_00_mean__max: 0.00932
- acf_harmonicity_mean__p25: 0.00901
- cqcc_delta2_05_min__p25: 0.00714
- mfcc_delta_01_min__mean: 0.00675
- mfcc_delta2_00_max__max: 0.00654
- cqcc_delta_11_max__mean: 0.00604
- cqcc_delta2_12_max__max: 0.00565
- lfcc_static_03_mean__median: 0.00555
- lfcc_delta2_00_max__p25: 0.00516
- lfcc_static_03_mean__p25: 0.00505
- cqcc_delta2_11_max__mean: 0.00505
- cqcc_delta_11_max__median: 0.00505
- acf_harmonicity_mean__min: 0.00495
- lfcc_static_00_mean__p25: 0.00476
- mfcc_delta2_00_std__p75: 0.00456

Коэффициент относится к стандартизованному признаку. Знак не доказывает причину синтетической речи.
