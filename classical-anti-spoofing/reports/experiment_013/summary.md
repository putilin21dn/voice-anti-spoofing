# Оценка классических моделей

Вариант: shape.
Score моделей не калибровался. Пороги взяты из обучения на validation и здесь не менялись.
Порог на test не подбирался.

## logistic

- validation: recall 0.896, specificity 0.916, F1 0.796, balanced accuracy 0.906, ROC-AUC 0.943, calls 250
- test: recall 0.880, specificity 1.000, F1 0.936, balanced accuracy 0.940, ROC-AUC 0.984, calls 43
- spectra_threshold: recall 0.879, specificity 0.912, F1 0.897, balanced accuracy 0.895, ROC-AUC 0.966, calls 190

## svm

- validation: recall 0.917, specificity 0.896, F1 0.779, balanced accuracy 0.906, ROC-AUC 0.956, calls 250
- test: recall 0.920, specificity 0.944, F1 0.939, balanced accuracy 0.932, ROC-AUC 0.993, calls 43
- spectra_threshold: recall 0.909, specificity 0.934, F1 0.923, balanced accuracy 0.922, ROC-AUC 0.966, calls 190

## lightgbm

- validation: recall 0.938, specificity 0.926, F1 0.833, balanced accuracy 0.932, ROC-AUC 0.962, calls 250
- test: recall 0.960, specificity 1.000, F1 0.980, balanced accuracy 0.980, ROC-AUC 0.998, calls 43
- spectra_threshold: recall 0.899, specificity 0.934, F1 0.918, balanced accuracy 0.917, ROC-AUC 0.975, calls 190

## Test по длительности, числу сегментов и SNR

Границы корзин посчитаны на train.
- logistic speech_duration_s low: calls 11, balanced accuracy 0.9166666666666667
- logistic speech_duration_s mid: calls 11, balanced accuracy 1.0
- logistic speech_duration_s high: calls 21, balanced accuracy 0.9166666666666667
- logistic n_usable_segments mid: calls 23, balanced accuracy 0.9230769230769231
- logistic n_usable_segments high: calls 18, balanced accuracy 0.95
- logistic snr_db low: calls 11, balanced accuracy 0.875
- logistic snr_db mid: calls 8, balanced accuracy 0.8333333333333333
- logistic snr_db high: calls 24, balanced accuracy 0.9722222222222222
- svm speech_duration_s low: calls 11, balanced accuracy 0.9166666666666667
- svm speech_duration_s mid: calls 11, balanced accuracy 0.9285714285714286
- svm speech_duration_s high: calls 21, balanced accuracy 0.9444444444444444
- svm n_usable_segments mid: calls 23, balanced accuracy 0.9615384615384616
- svm n_usable_segments high: calls 18, balanced accuracy 0.8875
- svm snr_db low: calls 11, balanced accuracy 1.0
- svm snr_db mid: calls 8, balanced accuracy 0.8333333333333333
- svm snr_db high: calls 24, balanced accuracy 0.8888888888888888
- lightgbm speech_duration_s low: calls 11, balanced accuracy 0.9166666666666667
- lightgbm speech_duration_s mid: calls 11, balanced accuracy 1.0
- lightgbm speech_duration_s high: calls 21, balanced accuracy 1.0
- lightgbm n_usable_segments mid: calls 23, balanced accuracy 0.9615384615384616
- lightgbm n_usable_segments high: calls 18, balanced accuracy 1.0
- lightgbm snr_db low: calls 11, balanced accuracy 1.0
- lightgbm snr_db mid: calls 8, balanced accuracy 0.8333333333333333
- lightgbm snr_db high: calls 24, balanced accuracy 1.0
