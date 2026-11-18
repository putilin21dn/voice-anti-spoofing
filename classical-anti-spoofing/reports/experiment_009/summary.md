# Оценка классических моделей

Вариант: no_energy.
Score моделей не калибровался. Пороги взяты из обучения на validation и здесь не менялись.
Порог на test не подбирался.

## logistic

- validation: recall 0.938, specificity 0.876, F1 0.763, balanced accuracy 0.907, ROC-AUC 0.954, calls 250
- test: recall 0.920, specificity 1.000, F1 0.958, balanced accuracy 0.960, ROC-AUC 0.978, calls 43
- spectra_threshold: recall 0.919, specificity 0.868, F1 0.901, balanced accuracy 0.894, ROC-AUC 0.966, calls 190

## svm

- validation: recall 0.938, specificity 0.901, F1 0.796, balanced accuracy 0.919, ROC-AUC 0.965, calls 250
- test: recall 0.920, specificity 0.944, F1 0.939, balanced accuracy 0.932, ROC-AUC 0.993, calls 43
- spectra_threshold: recall 0.919, specificity 0.912, F1 0.919, balanced accuracy 0.916, ROC-AUC 0.962, calls 190

## lightgbm

- validation: recall 0.938, specificity 0.876, F1 0.763, balanced accuracy 0.907, ROC-AUC 0.960, calls 250
- test: recall 0.960, specificity 0.944, F1 0.960, balanced accuracy 0.952, ROC-AUC 0.984, calls 43
- spectra_threshold: recall 0.919, specificity 0.890, F1 0.910, balanced accuracy 0.905, ROC-AUC 0.972, calls 190

## Test по длительности, числу сегментов и SNR

Границы корзин посчитаны на train.
- logistic speech_duration_s low: calls 11, balanced accuracy 0.9166666666666667
- logistic speech_duration_s mid: calls 11, balanced accuracy 1.0
- logistic speech_duration_s high: calls 21, balanced accuracy 0.9583333333333333
- logistic n_usable_segments mid: calls 23, balanced accuracy 0.9615384615384616
- logistic n_usable_segments high: calls 18, balanced accuracy 0.95
- logistic snr_db low: calls 11, balanced accuracy 1.0
- logistic snr_db mid: calls 8, balanced accuracy 0.8333333333333333
- logistic snr_db high: calls 24, balanced accuracy 0.9722222222222222
- svm speech_duration_s low: calls 11, balanced accuracy 0.9166666666666667
- svm speech_duration_s mid: calls 11, balanced accuracy 1.0
- svm speech_duration_s high: calls 21, balanced accuracy 0.9027777777777777
- svm n_usable_segments mid: calls 23, balanced accuracy 0.9615384615384616
- svm n_usable_segments high: calls 18, balanced accuracy 0.8875
- svm snr_db low: calls 11, balanced accuracy 1.0
- svm snr_db mid: calls 8, balanced accuracy 0.8333333333333333
- svm snr_db high: calls 24, balanced accuracy 0.8888888888888888
- lightgbm speech_duration_s low: calls 11, balanced accuracy 0.8166666666666667
- lightgbm speech_duration_s mid: calls 11, balanced accuracy 1.0
- lightgbm speech_duration_s high: calls 21, balanced accuracy 1.0
- lightgbm n_usable_segments mid: calls 23, balanced accuracy 0.9115384615384616
- lightgbm n_usable_segments high: calls 18, balanced accuracy 1.0
- lightgbm snr_db low: calls 11, balanced accuracy 1.0
- lightgbm snr_db mid: calls 8, balanced accuracy 0.7333333333333334
- lightgbm snr_db high: calls 24, balanced accuracy 1.0
