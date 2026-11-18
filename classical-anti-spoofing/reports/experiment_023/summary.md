# svm

Порог и параметры выбраны на validation. Test в этом запуске не использовался.
Score не является калиброванной вероятностью.
Вариант: spectral.
Оставлены центроид, полоса, rolloff, flatness и zero-crossing rate. Кепстр, энергия, F0 и качество голоса убраны до отбора.
Исключено колонок до отбора: 4522.

- train: 1002
- validation: 250
- признаков: 139
- параметры: {'C': 0.3, 'gamma': 'scale'}
- порог machine: -0.2166242832291352

## Validation

- human: 202
- machine: 48
- recall_synthetic: 0.9791666666666666
- specificity_human: 0.8366336633663366
- precision_synthetic: 0.5875
- f1_synthetic: 0.734375
- balanced_accuracy: 0.9079001650165016
- fpr: 0.16336633663366337
- fnr: 0.020833333333333332
- roc_auc: 0.9589521452145214
- pr_auc: 0.8598025969142732

## Permutation importance, первые 15

- zero_crossing_rate_min__min: 0.01279
- zero_crossing_rate_min__p25: 0.00922
- spectral_flatness_mean__std: 0.00844
- zero_crossing_rate_min__mean: 0.00724
- zero_crossing_rate_min__median: 0.00724
- zero_crossing_rate_mean__min: 0.00575
- spectral_rolloff_hz_min__p25: 0.00357
- spectral_rolloff_hz_std__mean: 0.00347
- zero_crossing_rate_mean__p25: 0.00328
- spectral_bandwidth_hz_max__max: 0.00307
- spectral_centroid_hz_min__p25: 0.00307
- spectral_flatness_mean__max: 0.00278
- spectral_bandwidth_hz_min__mean: 0.00268
- spectral_bandwidth_hz_std__min: 0.00248
- spectral_flatness_min__min: 0.00248

У RBF SVM нет одного коэффициента на признак. Permutation importance — ассоциация, не причина.
