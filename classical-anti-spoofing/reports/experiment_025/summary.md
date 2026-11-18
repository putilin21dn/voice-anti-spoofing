# Оценка классических моделей

Вариант: spectral.
Score моделей не калибровался. Пороги взяты из обучения на validation и здесь не менялись.
Порог на test не подбирался.

## logistic

- validation: recall 0.938, specificity 0.851, F1 0.732, balanced accuracy 0.894, ROC-AUC 0.958, calls 250
- test: recall 0.960, specificity 0.944, F1 0.960, balanced accuracy 0.952, ROC-AUC 0.996, calls 43
- spectra_threshold: recall 0.949, specificity 0.846, F1 0.908, balanced accuracy 0.898, ROC-AUC 0.965, calls 190

## svm

- validation: recall 0.979, specificity 0.837, F1 0.734, balanced accuracy 0.908, ROC-AUC 0.959, calls 250
- test: recall 0.880, specificity 0.889, F1 0.898, balanced accuracy 0.884, ROC-AUC 0.969, calls 43
- spectra_threshold: recall 0.919, specificity 0.934, F1 0.929, balanced accuracy 0.927, ROC-AUC 0.965, calls 190

## lightgbm

- validation: recall 0.917, specificity 0.896, F1 0.779, balanced accuracy 0.906, ROC-AUC 0.955, calls 250
- test: recall 0.920, specificity 0.944, F1 0.939, balanced accuracy 0.932, ROC-AUC 0.982, calls 43
- spectra_threshold: recall 0.899, specificity 0.967, F1 0.932, balanced accuracy 0.933, ROC-AUC 0.976, calls 190

## Test по длительности, числу сегментов и SNR

Границы корзин посчитаны на train.
- logistic speech_duration_s low: calls 11, balanced accuracy 0.9
- logistic speech_duration_s mid: calls 11, balanced accuracy 1.0
- logistic speech_duration_s high: calls 21, balanced accuracy 0.9583333333333333
- logistic n_usable_segments mid: calls 23, balanced accuracy 0.95
- logistic n_usable_segments high: calls 18, balanced accuracy 0.95
- logistic snr_db low: calls 11, balanced accuracy 0.9285714285714286
- logistic snr_db mid: calls 8, balanced accuracy 1.0
- logistic snr_db high: calls 24, balanced accuracy 0.9722222222222222
- svm speech_duration_s low: calls 11, balanced accuracy 0.8166666666666667
- svm speech_duration_s mid: calls 11, balanced accuracy 0.9285714285714286
- svm speech_duration_s high: calls 21, balanced accuracy 0.9027777777777777
- svm n_usable_segments mid: calls 23, balanced accuracy 0.9115384615384616
- svm n_usable_segments high: calls 18, balanced accuracy 0.8375
- svm snr_db low: calls 11, balanced accuracy 0.9285714285714286
- svm snr_db mid: calls 8, balanced accuracy 0.7333333333333334
- svm snr_db high: calls 24, balanced accuracy 0.9444444444444444
- lightgbm speech_duration_s low: calls 11, balanced accuracy 0.8166666666666667
- lightgbm speech_duration_s mid: calls 11, balanced accuracy 1.0
- lightgbm speech_duration_s high: calls 21, balanced accuracy 0.9583333333333333
- lightgbm n_usable_segments mid: calls 23, balanced accuracy 0.9115384615384616
- lightgbm n_usable_segments high: calls 18, balanced accuracy 0.95
- lightgbm snr_db low: calls 11, balanced accuracy 0.9285714285714286
- lightgbm snr_db mid: calls 8, balanced accuracy 0.8333333333333333
- lightgbm snr_db high: calls 24, balanced accuracy 0.9722222222222222
