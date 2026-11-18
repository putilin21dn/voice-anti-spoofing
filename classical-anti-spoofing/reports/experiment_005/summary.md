# Оценка классических моделей

Score моделей не калибровался. Пороги взяты из обучения на validation и здесь не менялись.
Порог на test не подбирался.

## logistic

- validation: recall 0.938, specificity 0.926, F1 0.833, balanced accuracy 0.932, ROC-AUC 0.963, calls 250
- test: recall 0.960, specificity 1.000, F1 0.980, balanced accuracy 0.980, ROC-AUC 0.989, calls 43
- spectra_threshold: recall 0.919, specificity 0.912, F1 0.919, balanced accuracy 0.916, ROC-AUC 0.977, calls 190

## svm

- validation: recall 0.917, specificity 0.946, F1 0.854, balanced accuracy 0.931, ROC-AUC 0.960, calls 250
- test: recall 0.960, specificity 1.000, F1 0.980, balanced accuracy 0.980, ROC-AUC 0.993, calls 43
- spectra_threshold: recall 0.909, specificity 0.956, F1 0.933, balanced accuracy 0.933, ROC-AUC 0.973, calls 190

## lightgbm

- validation: recall 0.938, specificity 0.926, F1 0.833, balanced accuracy 0.932, ROC-AUC 0.976, calls 250
- test: recall 0.960, specificity 1.000, F1 0.980, balanced accuracy 0.980, ROC-AUC 0.989, calls 43
- spectra_threshold: recall 0.929, specificity 0.945, F1 0.939, balanced accuracy 0.937, ROC-AUC 0.981, calls 190

## Test по длительности, числу сегментов и SNR

Границы корзин посчитаны на train. Корзина `n_usable_segments low` на test содержит 2 звонка, метрика для неё не считалась. Три модели дали одинаковые решения на всём test, поэтому срезы совпадают.

- speech_duration_s low: 11 звонков (human 5, machine 6), balanced accuracy 0.917
- speech_duration_s mid: 11 звонков (human 4, machine 7), balanced accuracy 1.000
- speech_duration_s high: 21 звонок (human 9, machine 12), balanced accuracy 1.000
- n_usable_segments mid: 23 звонка (human 10, machine 13), balanced accuracy 0.962
- n_usable_segments high: 18 звонков (human 8, machine 10), balanced accuracy 1.000
- snr_db low: 11 звонков (human 7, machine 4), balanced accuracy 1.000
- snr_db mid: 8 звонков (human 5, machine 3), balanced accuracy 0.833
- snr_db high: 24 звонка (human 6, machine 18), balanced accuracy 1.000
