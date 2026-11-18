# Сравнение со Spectra-0

Вариант классических моделей: shape.
Порог Spectra-0 остаётся 1.02. Классические пороги взяты с validation.
Правила either/both заранее заданы и не выбирались по test.
Score не является калиброванной вероятностью.

## Spectra-0 отдельно

- n100: recall 0.707, specificity 0.868, balanced accuracy 0.788, ROC-AUC 0.826
- test20: recall 0.680, specificity 0.944, balanced accuracy 0.812, ROC-AUC 0.847

## Модели и заранее заданные объединения на test

- spectra0 spectra_only: recall 0.680, specificity 0.944, balanced accuracy 0.812
- logistic model_only: recall 0.880, specificity 1.000, balanced accuracy 0.940
- logistic either_machine: recall 1.000, specificity 0.944, balanced accuracy 0.972
- logistic both_machine: recall 0.560, specificity 1.000, balanced accuracy 0.780
- svm model_only: recall 0.920, specificity 0.944, balanced accuracy 0.932
- svm either_machine: recall 1.000, specificity 0.889, balanced accuracy 0.944
- svm both_machine: recall 0.600, specificity 1.000, balanced accuracy 0.800
- lightgbm model_only: recall 0.960, specificity 1.000, balanced accuracy 0.980
- lightgbm either_machine: recall 1.000, specificity 0.944, balanced accuracy 0.972
- lightgbm both_machine: recall 0.640, specificity 1.000, balanced accuracy 0.820

## Расхождения на test

- logistic: {'both_correct': 31, 'classical_only': 9, 'spectra_only': 3}
  classical only: 6d812989-634c-4599-b03b-6d648b48cf05, b724cd06-933e-435d-967c-209d8e6653c8, 92708ab2-fdd5-4987-b1e8-5aba55f595f1, 5fb5f678-044c-4b72-be30-ddfba632059f, dc14c9b3-90eb-4785-8599-2e2c23210b9f, 11768ca6-42a7-4e75-aa07-ed924eb7925f, 2cc670e4-f79f-44ed-a28a-00e69a3c5ccb, f9c5a954-9c64-40c5-b69b-cee9f74293ad, 89a02384-e5ff-4beb-9761-0003f25dd571
  both wrong: нет
- svm: {'both_correct': 31, 'classical_only': 9, 'spectra_only': 3}
  classical only: 6d812989-634c-4599-b03b-6d648b48cf05, b724cd06-933e-435d-967c-209d8e6653c8, 92708ab2-fdd5-4987-b1e8-5aba55f595f1, 5fb5f678-044c-4b72-be30-ddfba632059f, dc14c9b3-90eb-4785-8599-2e2c23210b9f, 11768ca6-42a7-4e75-aa07-ed924eb7925f, 2cc670e4-f79f-44ed-a28a-00e69a3c5ccb, f9c5a954-9c64-40c5-b69b-cee9f74293ad, 89a02384-e5ff-4beb-9761-0003f25dd571
  both wrong: нет
- lightgbm: {'both_correct': 33, 'classical_only': 9, 'spectra_only': 1}
  classical only: 6d812989-634c-4599-b03b-6d648b48cf05, b724cd06-933e-435d-967c-209d8e6653c8, 92708ab2-fdd5-4987-b1e8-5aba55f595f1, 5fb5f678-044c-4b72-be30-ddfba632059f, dc14c9b3-90eb-4785-8599-2e2c23210b9f, 11768ca6-42a7-4e75-aa07-ed924eb7925f, 2cc670e4-f79f-44ed-a28a-00e69a3c5ccb, f9c5a954-9c64-40c5-b69b-cee9f74293ad, 89a02384-e5ff-4beb-9761-0003f25dd571
  both wrong: нет

## Признаки, которые на n100 слабее повторяют зазор Spectra-0

Это описание n100, не независимый test. Корреляция не доказывает причину.
- mfcc_delta_06_max__p25: corr machine 0.593, corr gap -0.141
- mfcc_delta_06_max__mean: corr machine 0.589, corr gap -0.147
- mfcc_delta_06_max__median: corr machine 0.566, corr gap -0.129
- lfcc_static_10_std__p25: corr machine 0.504, corr gap -0.022
- mfcc_delta2_06_max__p25: corr machine 0.552, corr gap -0.122
- mfcc_delta2_06_min__mean: corr machine -0.571, corr gap 0.161
- mfcc_delta2_06_min__p75: corr machine -0.548, corr gap 0.140
- mfcc_delta_04_max__p25: corr machine 0.583, corr gap -0.198
- mfcc_delta2_06_max__min: corr machine 0.522, corr gap -0.107
- mfcc_delta_06_max__min: corr machine 0.537, corr gap -0.133
- mfcc_static_17_max__p25: corr machine 0.479, corr gap -0.031
- mfcc_delta2_06_max__mean: corr machine 0.537, corr gap -0.135
