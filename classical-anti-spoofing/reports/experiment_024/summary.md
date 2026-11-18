# lightgbm

Порог и параметры выбраны на validation. Test в этом запуске не использовался.
Score не является калиброванной вероятностью.
Вариант: spectral.
Оставлены центроид, полоса, rolloff, flatness и zero-crossing rate. Кепстр, энергия, F0 и качество голоса убраны до отбора.
Исключено колонок до отбора: 4522.

- train: 1002
- validation: 250
- признаков: 139
- параметры: {'n_estimators': 200, 'num_leaves': 31, 'learning_rate': 0.05, 'min_child_samples': 20}
- порог machine: 0.04502100929193762

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
- roc_auc: 0.9554455445544555
- pr_auc: 0.8684892676974427

## Permutation importance, первые 15

- zero_crossing_rate_mean__p75: 0.04948
- spectral_bandwidth_hz_mean__min: 0.02201
- zero_crossing_rate_min__min: 0.02162
- spectral_flatness_mean__mean: 0.02013
- spectral_centroid_hz_mean__p75: 0.01825
- zero_crossing_rate_min__median: 0.01508
- spectral_bandwidth_hz_min__min: 0.01498
- spectral_bandwidth_hz_max__min: 0.01487
- spectral_flatness_mean__median: 0.01448
- spectral_bandwidth_hz_std__max: 0.01349
- spectral_bandwidth_hz_std__p75: 0.01279
- spectral_bandwidth_hz_mean__std: 0.01091
- spectral_flatness_max__std: 0.01042
- spectral_centroid_hz_min__mean: 0.01021
- zero_crossing_rate_mean__max: 0.00972

Gain и permutation importance показывают вклад в эту модель, а не причину синтетической речи.
