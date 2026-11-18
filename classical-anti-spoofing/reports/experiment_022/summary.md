# logistic

Порог и параметры выбраны на validation. Test в этом запуске не использовался.
Score не является калиброванной вероятностью.
Вариант: spectral.
Оставлены центроид, полоса, rolloff, flatness и zero-crossing rate. Кепстр, энергия, F0 и качество голоса убраны до отбора.
Исключено колонок до отбора: 4522.

- train: 1002
- validation: 250
- признаков: 139
- параметры: {'C': 0.1}
- порог machine: 0.4155940895403938

## Validation

- human: 202
- machine: 48
- recall_synthetic: 0.9375
- specificity_human: 0.8514851485148515
- precision_synthetic: 0.6
- f1_synthetic: 0.7317073170731707
- balanced_accuracy: 0.8944925742574257
- fpr: 0.1485148514851485
- fnr: 0.0625
- roc_auc: 0.9578176567656765
- pr_auc: 0.8607311002045422

## Permutation importance, первые 15

- zero_crossing_rate_mean__p75: 0.03274
- spectral_flatness_mean__p75: 0.01914
- spectral_centroid_hz_std__std: 0.01813
- spectral_flatness_mean__mean: 0.01458
- spectral_bandwidth_hz_min__min: 0.01357
- zero_crossing_rate_max__max: 0.01328
- zero_crossing_rate_max__p25: 0.01297
- spectral_flatness_max__p75: 0.01238
- spectral_centroid_hz_max__std: 0.01209
- zero_crossing_rate_mean__std: 0.01190
- spectral_flatness_mean__std: 0.01091
- spectral_flatness_min__min: 0.01011
- spectral_flatness_std__p25: 0.00990
- spectral_rolloff_hz_max__p25: 0.00972
- spectral_flatness_std__min: 0.00941

Коэффициент относится к стандартизованному признаку. Знак не доказывает причину синтетической речи.
