# Оценка классических моделей

Вариант: voicing.
Score моделей не калибровался. Пороги взяты из обучения на validation и здесь не менялись.
Порог на test не подбирался.

## logistic

- validation: recall 0.979, specificity 0.921, F1 0.847, balanced accuracy 0.950, ROC-AUC 0.972, calls 250
- test: recall 0.880, specificity 1.000, F1 0.936, balanced accuracy 0.940, ROC-AUC 0.971, calls 43
- spectra_threshold: recall 0.909, specificity 0.912, F1 0.914, balanced accuracy 0.911, ROC-AUC 0.961, calls 190

## svm

- validation: recall 0.938, specificity 0.941, F1 0.857, balanced accuracy 0.939, ROC-AUC 0.974, calls 250
- test: recall 0.880, specificity 0.944, F1 0.917, balanced accuracy 0.912, ROC-AUC 0.964, calls 43
- spectra_threshold: recall 0.889, specificity 0.945, F1 0.917, balanced accuracy 0.917, ROC-AUC 0.977, calls 190

## lightgbm

- validation: recall 0.958, specificity 0.906, F1 0.814, balanced accuracy 0.932, ROC-AUC 0.971, calls 250
- test: recall 0.920, specificity 0.833, F1 0.902, balanced accuracy 0.877, ROC-AUC 0.942, calls 43
- spectra_threshold: recall 0.919, specificity 0.923, F1 0.924, balanced accuracy 0.921, ROC-AUC 0.971, calls 190

## Test по длительности, числу сегментов и SNR

Границы корзин посчитаны на train.
- logistic speech_duration_s low: calls 11, balanced accuracy 1.0
- logistic speech_duration_s mid: calls 11, balanced accuracy 1.0
- logistic speech_duration_s high: calls 21, balanced accuracy 0.875
- logistic n_usable_segments mid: calls 23, balanced accuracy 0.9615384615384616
- logistic n_usable_segments high: calls 18, balanced accuracy 0.9
- logistic snr_db low: calls 11, balanced accuracy 0.875
- logistic snr_db mid: calls 8, balanced accuracy 1.0
- logistic snr_db high: calls 24, balanced accuracy 0.9444444444444444
- svm speech_duration_s low: calls 11, balanced accuracy 0.8166666666666667
- svm speech_duration_s mid: calls 11, balanced accuracy 1.0
- svm speech_duration_s high: calls 21, balanced accuracy 0.9166666666666667
- svm n_usable_segments mid: calls 23, balanced accuracy 0.8730769230769231
- svm n_usable_segments high: calls 18, balanced accuracy 0.95
- svm snr_db low: calls 11, balanced accuracy 0.875
- svm snr_db mid: calls 8, balanced accuracy 0.7333333333333334
- svm snr_db high: calls 24, balanced accuracy 0.9722222222222222
- lightgbm speech_duration_s low: calls 11, balanced accuracy 0.7166666666666667
- lightgbm speech_duration_s mid: calls 11, balanced accuracy 0.875
- lightgbm speech_duration_s high: calls 21, balanced accuracy 0.9583333333333333
- lightgbm n_usable_segments mid: calls 23, balanced accuracy 0.8115384615384615
- lightgbm n_usable_segments high: calls 18, balanced accuracy 0.95
- lightgbm snr_db low: calls 11, balanced accuracy 0.9285714285714286
- lightgbm snr_db mid: calls 8, balanced accuracy 0.6333333333333333
- lightgbm snr_db high: calls 24, balanced accuracy 0.9722222222222222
