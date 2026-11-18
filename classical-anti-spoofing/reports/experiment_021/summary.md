# Оценка классических моделей

Вариант: mfcc.
Score моделей не калибровался. Пороги взяты из обучения на validation и здесь не менялись.
Порог на test не подбирался.

## logistic

- validation: recall 0.875, specificity 0.886, F1 0.743, balanced accuracy 0.881, ROC-AUC 0.927, calls 250
- test: recall 0.880, specificity 0.889, F1 0.898, balanced accuracy 0.884, ROC-AUC 0.953, calls 43
- spectra_threshold: recall 0.879, specificity 0.901, F1 0.892, balanced accuracy 0.890, ROC-AUC 0.954, calls 190

## svm

- validation: recall 0.875, specificity 0.911, F1 0.778, balanced accuracy 0.893, ROC-AUC 0.942, calls 250
- test: recall 0.800, specificity 0.889, F1 0.851, balanced accuracy 0.844, ROC-AUC 0.951, calls 43
- spectra_threshold: recall 0.848, specificity 0.923, F1 0.884, balanced accuracy 0.886, ROC-AUC 0.954, calls 190

## lightgbm

- validation: recall 0.854, specificity 0.931, F1 0.796, balanced accuracy 0.892, ROC-AUC 0.932, calls 250
- test: recall 0.760, specificity 1.000, F1 0.864, balanced accuracy 0.880, ROC-AUC 0.971, calls 43
- spectra_threshold: recall 0.798, specificity 0.978, F1 0.878, balanced accuracy 0.888, ROC-AUC 0.954, calls 190

## Test по длительности, числу сегментов и SNR

Границы корзин посчитаны на train.
- logistic speech_duration_s low: calls 11, balanced accuracy 1.0
- logistic speech_duration_s mid: calls 11, balanced accuracy 0.7321428571428572
- logistic speech_duration_s high: calls 21, balanced accuracy 0.9027777777777777
- logistic n_usable_segments mid: calls 23, balanced accuracy 0.7846153846153847
- logistic n_usable_segments high: calls 18, balanced accuracy 1.0
- logistic snr_db low: calls 11, balanced accuracy 0.8035714285714286
- logistic snr_db mid: calls 8, balanced accuracy 0.7333333333333334
- logistic snr_db high: calls 24, balanced accuracy 0.9722222222222222
- svm speech_duration_s low: calls 11, balanced accuracy 0.9166666666666667
- svm speech_duration_s mid: calls 11, balanced accuracy 0.6607142857142857
- svm speech_duration_s high: calls 21, balanced accuracy 0.9027777777777777
- svm n_usable_segments mid: calls 23, balanced accuracy 0.7961538461538462
- svm n_usable_segments high: calls 18, balanced accuracy 0.8875
- svm snr_db low: calls 11, balanced accuracy 0.875
- svm snr_db mid: calls 8, balanced accuracy 0.5666666666666667
- svm snr_db high: calls 24, balanced accuracy 0.8611111111111112
- lightgbm speech_duration_s low: calls 11, balanced accuracy 0.9166666666666667
- lightgbm speech_duration_s mid: calls 11, balanced accuracy 0.7857142857142857
- lightgbm speech_duration_s high: calls 21, balanced accuracy 0.9166666666666667
- lightgbm n_usable_segments mid: calls 23, balanced accuracy 0.8846153846153846
- lightgbm n_usable_segments high: calls 18, balanced accuracy 0.85
- lightgbm snr_db low: calls 11, balanced accuracy 1.0
- lightgbm snr_db mid: calls 8, balanced accuracy 0.6666666666666666
- lightgbm snr_db high: calls 24, balanced accuracy 0.8888888888888888
