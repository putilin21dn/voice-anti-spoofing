# logistic

Порог и параметры выбраны на validation. Test в этом запуске не использовался.
Score не является калиброванной вероятностью.
Вариант: shape.
Оставлены коэффициенты LFCC, MFCC и CQCC начиная с 1 и спектральные меры. Энергия, F0 и качество голоса убраны до отбора.
Исключено колонок до отбора: 322.

- train: 1002
- validation: 250
- признаков: 200
- параметры: {'C': 0.01}
- порог machine: 0.5497299188842178

## Validation

- human: 202
- machine: 48
- recall_synthetic: 0.8958333333333334
- specificity_human: 0.9158415841584159
- precision_synthetic: 0.7166666666666667
- f1_synthetic: 0.7962962962962963
- balanced_accuracy: 0.9058374587458746
- fpr: 0.08415841584158416
- fnr: 0.10416666666666667
- roc_auc: 0.9431724422442245
- pr_auc: 0.8402938888905951

## Permutation importance, первые 15

- zero_crossing_rate_mean__p75: 0.02867
- zero_crossing_rate_mean__mean: 0.02550
- spectral_centroid_hz_mean__p75: 0.02401
- zero_crossing_rate_mean__median: 0.02232
- zero_crossing_rate_mean__max: 0.01984
- mfcc_static_11_std__min: 0.01953
- cqcc_delta2_01_std__p75: 0.01914
- spectral_centroid_hz_mean__max: 0.01885
- cqcc_delta2_01_std__mean: 0.01815
- lfcc_static_03_mean__p25: 0.01815
- lfcc_static_03_mean__median: 0.01815
- lfcc_static_03_mean__mean: 0.01656
- spectral_centroid_hz_mean__median: 0.01656
- cqcc_delta_10_max__max: 0.01508
- cqcc_delta2_01_std__median: 0.01498

Коэффициент относится к стандартизованному признаку. Знак не доказывает причину синтетической речи.
