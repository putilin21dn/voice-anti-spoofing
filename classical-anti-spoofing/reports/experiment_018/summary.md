# logistic

Порог и параметры выбраны на validation. Test в этом запуске не использовался.
Score не является калиброванной вероятностью.
Вариант: mfcc.
Оставлены только MFCC начиная с коэффициента 1. Энергия, LFCC, CQCC, спектр, F0 и качество голоса убраны до отбора.
Исключено колонок до отбора: 3066.

- train: 1002
- validation: 250
- признаков: 200
- параметры: {'C': 0.1}
- порог machine: 0.42650586668806345

## Validation

- human: 202
- machine: 48
- recall_synthetic: 0.875
- specificity_human: 0.8861386138613861
- precision_synthetic: 0.6461538461538462
- f1_synthetic: 0.7433628318584071
- balanced_accuracy: 0.880569306930693
- fpr: 0.11386138613861387
- fnr: 0.125
- roc_auc: 0.9273927392739274
- pr_auc: 0.7574463853370883

## Permutation importance, первые 15

- mfcc_static_11_mean__mean: 0.04701
- mfcc_static_01_mean__min: 0.03203
- mfcc_static_02_max__mean: 0.03133
- mfcc_delta_14_max__min: 0.02211
- mfcc_static_17_max__p25: 0.02071
- mfcc_delta_01_min__mean: 0.01943
- mfcc_delta2_18_min__median: 0.01904
- mfcc_delta_11_std__p25: 0.01832
- mfcc_delta2_17_max__p75: 0.01774
- mfcc_delta_15_max__p25: 0.01755
- mfcc_delta2_15_max__p25: 0.01745
- mfcc_delta2_17_min__p75: 0.01646
- mfcc_delta2_16_min__median: 0.01636
- mfcc_delta_08_max__p25: 0.01625
- mfcc_static_11_max__p75: 0.01615

Коэффициент относится к стандартизованному признаку. Знак не доказывает причину синтетической речи.
