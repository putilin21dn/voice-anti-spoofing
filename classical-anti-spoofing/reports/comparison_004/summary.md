# Сравнение со Spectra-0

Вариант классических моделей: voicing.
Порог Spectra-0 остаётся 1.02. Классические пороги взяты с validation.
Правила either/both заранее заданы и не выбирались по test.
Score не является калиброванной вероятностью.

## Spectra-0 отдельно

- n100: recall 0.707, specificity 0.868, balanced accuracy 0.788, ROC-AUC 0.826
- test20: recall 0.680, specificity 0.944, balanced accuracy 0.812, ROC-AUC 0.847

## Модели и заранее заданные объединения на test

- spectra0 spectra_only: recall 0.680, specificity 0.944, balanced accuracy 0.812
- logistic model_only: recall 0.880, specificity 1.000, balanced accuracy 0.940
- logistic either_machine: recall 0.960, specificity 0.944, balanced accuracy 0.952
- logistic both_machine: recall 0.600, specificity 1.000, balanced accuracy 0.800
- svm model_only: recall 0.880, specificity 0.944, balanced accuracy 0.912
- svm either_machine: recall 1.000, specificity 0.889, balanced accuracy 0.944
- svm both_machine: recall 0.560, specificity 1.000, balanced accuracy 0.780
- lightgbm model_only: recall 0.920, specificity 0.833, balanced accuracy 0.877
- lightgbm either_machine: recall 1.000, specificity 0.778, balanced accuracy 0.889
- lightgbm both_machine: recall 0.600, specificity 1.000, balanced accuracy 0.800

## Расхождения на test

- logistic: {'both_correct': 32, 'classical_only': 8, 'spectra_only': 2, 'both_wrong': 1}
  classical only: 6d812989-634c-4599-b03b-6d648b48cf05, b724cd06-933e-435d-967c-209d8e6653c8, 92708ab2-fdd5-4987-b1e8-5aba55f595f1, 5fb5f678-044c-4b72-be30-ddfba632059f, 11768ca6-42a7-4e75-aa07-ed924eb7925f, 2cc670e4-f79f-44ed-a28a-00e69a3c5ccb, f9c5a954-9c64-40c5-b69b-cee9f74293ad, 89a02384-e5ff-4beb-9761-0003f25dd571
  both wrong: dc14c9b3-90eb-4785-8599-2e2c23210b9f
- svm: {'both_correct': 30, 'classical_only': 9, 'spectra_only': 4}
  classical only: 6d812989-634c-4599-b03b-6d648b48cf05, b724cd06-933e-435d-967c-209d8e6653c8, 92708ab2-fdd5-4987-b1e8-5aba55f595f1, 5fb5f678-044c-4b72-be30-ddfba632059f, dc14c9b3-90eb-4785-8599-2e2c23210b9f, 11768ca6-42a7-4e75-aa07-ed924eb7925f, 2cc670e4-f79f-44ed-a28a-00e69a3c5ccb, f9c5a954-9c64-40c5-b69b-cee9f74293ad, 89a02384-e5ff-4beb-9761-0003f25dd571
  both wrong: нет
- lightgbm: {'both_correct': 29, 'classical_only': 9, 'spectra_only': 5}
  classical only: 6d812989-634c-4599-b03b-6d648b48cf05, b724cd06-933e-435d-967c-209d8e6653c8, 92708ab2-fdd5-4987-b1e8-5aba55f595f1, 5fb5f678-044c-4b72-be30-ddfba632059f, dc14c9b3-90eb-4785-8599-2e2c23210b9f, 11768ca6-42a7-4e75-aa07-ed924eb7925f, 2cc670e4-f79f-44ed-a28a-00e69a3c5ccb, f9c5a954-9c64-40c5-b69b-cee9f74293ad, 89a02384-e5ff-4beb-9761-0003f25dd571
  both wrong: нет

## Признаки, которые на n100 слабее повторяют зазор Spectra-0

Это описание n100, не независимый test. Корреляция не доказывает причину.
- f0_voiced_fraction__min: corr machine 0.672, corr gap -0.290
- acf_harmonicity_mean__min: corr machine 0.633, corr gap -0.251
- f0_voiced_fraction__p25: corr machine 0.635, corr gap -0.272
- f0_range__p25: corr machine 0.572, corr gap -0.193
- f0_range__min: corr machine 0.546, corr gap -0.174
- acf_harmonicity_mean__p25: corr machine 0.576, corr gap -0.238
- f0_voiced_fraction__mean: corr machine 0.597, corr gap -0.267
- f0_range__mean: corr machine 0.509, corr gap -0.164
- f0_voiced_fraction__std: corr machine -0.539, corr gap 0.217
- f0_range__median: corr machine 0.508, corr gap -0.171
- f0_std__min: corr machine 0.508, corr gap -0.179
- f0_voiced_fraction__median: corr machine 0.558, corr gap -0.259
