# Сравнение со Spectra-0

Вариант классических моделей: spectral.
Порог Spectra-0 остаётся 1.02. Классические пороги взяты с validation.
Правила either/both заранее заданы и не выбирались по test.
Score не является калиброванной вероятностью.

## Spectra-0 отдельно

- n100: recall 0.707, specificity 0.868, balanced accuracy 0.788, ROC-AUC 0.826
- test20: recall 0.680, specificity 0.944, balanced accuracy 0.812, ROC-AUC 0.847

## Модели и заранее заданные объединения на test

- spectra0 spectra_only: recall 0.680, specificity 0.944, balanced accuracy 0.812
- logistic model_only: recall 0.960, specificity 0.944, balanced accuracy 0.952
- logistic either_machine: recall 0.960, specificity 0.889, balanced accuracy 0.924
- logistic both_machine: recall 0.680, specificity 1.000, balanced accuracy 0.840
- svm model_only: recall 0.880, specificity 0.889, balanced accuracy 0.884
- svm either_machine: recall 0.960, specificity 0.833, balanced accuracy 0.897
- svm both_machine: recall 0.600, specificity 1.000, balanced accuracy 0.800
- lightgbm model_only: recall 0.920, specificity 0.944, balanced accuracy 0.932
- lightgbm either_machine: recall 0.960, specificity 0.889, balanced accuracy 0.924
- lightgbm both_machine: recall 0.640, specificity 1.000, balanced accuracy 0.820

## Расхождения на test

- logistic: {'both_correct': 33, 'classical_only': 8, 'spectra_only': 1, 'both_wrong': 1}
  classical only: 6d812989-634c-4599-b03b-6d648b48cf05, b724cd06-933e-435d-967c-209d8e6653c8, 92708ab2-fdd5-4987-b1e8-5aba55f595f1, 5fb5f678-044c-4b72-be30-ddfba632059f, 11768ca6-42a7-4e75-aa07-ed924eb7925f, 2cc670e4-f79f-44ed-a28a-00e69a3c5ccb, f9c5a954-9c64-40c5-b69b-cee9f74293ad, 89a02384-e5ff-4beb-9761-0003f25dd571
  both wrong: dc14c9b3-90eb-4785-8599-2e2c23210b9f
- svm: {'both_correct': 30, 'classical_only': 8, 'spectra_only': 4, 'both_wrong': 1}
  classical only: 6d812989-634c-4599-b03b-6d648b48cf05, b724cd06-933e-435d-967c-209d8e6653c8, 92708ab2-fdd5-4987-b1e8-5aba55f595f1, 5fb5f678-044c-4b72-be30-ddfba632059f, 11768ca6-42a7-4e75-aa07-ed924eb7925f, 2cc670e4-f79f-44ed-a28a-00e69a3c5ccb, f9c5a954-9c64-40c5-b69b-cee9f74293ad, 89a02384-e5ff-4beb-9761-0003f25dd571
  both wrong: dc14c9b3-90eb-4785-8599-2e2c23210b9f
- lightgbm: {'both_correct': 32, 'classical_only': 8, 'spectra_only': 2, 'both_wrong': 1}
  classical only: 6d812989-634c-4599-b03b-6d648b48cf05, b724cd06-933e-435d-967c-209d8e6653c8, 92708ab2-fdd5-4987-b1e8-5aba55f595f1, 5fb5f678-044c-4b72-be30-ddfba632059f, 11768ca6-42a7-4e75-aa07-ed924eb7925f, 2cc670e4-f79f-44ed-a28a-00e69a3c5ccb, f9c5a954-9c64-40c5-b69b-cee9f74293ad, 89a02384-e5ff-4beb-9761-0003f25dd571
  both wrong: dc14c9b3-90eb-4785-8599-2e2c23210b9f

## Признаки, которые на n100 слабее повторяют зазор Spectra-0

Это описание n100, не независимый test. Корреляция не доказывает причину.
- spectral_flatness_mean__p75: corr machine -0.586, corr gap 0.210
- spectral_flatness_mean__mean: corr machine -0.575, corr gap 0.214
- spectral_centroid_hz_mean__std: corr machine -0.581, corr gap 0.237
- spectral_flatness_mean__max: corr machine -0.567, corr gap 0.230
- spectral_rolloff_hz_mean__std: corr machine -0.556, corr gap 0.217
- spectral_flatness_mean__median: corr machine -0.520, corr gap 0.173
- spectral_centroid_hz_mean__max: corr machine -0.633, corr gap 0.322
- spectral_rolloff_hz_mean__max: corr machine -0.622, corr gap 0.316
- zero_crossing_rate_min__median: corr machine -0.556, corr gap 0.257
- spectral_centroid_hz_mean__p75: corr machine -0.569, corr gap 0.277
- spectral_flatness_mean__std: corr machine -0.522, corr gap 0.220
- zero_crossing_rate_mean__std: corr machine -0.543, corr gap 0.252
