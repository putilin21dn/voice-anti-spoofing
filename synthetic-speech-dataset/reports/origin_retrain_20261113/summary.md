# Переобучение на 13 ноября, тест — размеченная часть 16 ноября

Положительный класс — синтетический голос: synthetic_tts, synthetic_voice_conversion, synthetic_playback. human_live и human_recorded — человек. unknown и mixed в счёт не входят.
Порог и состав признаков выбирались только на валидации 13 ноября. Метки 16 ноября в этот выбор не входили.
Старые bundle classical-anti-spoofing и веса Spectra-0 / AASIST3 не изменялись. Новые файлы лежат в этом каталоге.
Score — ранжирующая оценка, не вероятность.

- бинарных звонков 13 ноября: 251
- train / validation: 200 / 51, доля валидации 0.2, seed 42
- с речью, попали в обучение: 251
- признаков после отбора на train: 200

## Валидация, по ней выбирался порог

- logistic (retrained_classical): порог 0.7355, recall 1.000, specificity 0.973, balanced accuracy 0.986.
- svm (retrained_classical): порог 0.0000, recall 1.000, specificity 0.919, balanced accuracy 0.959.
- lightgbm (retrained_classical): порог 0.3461, recall 1.000, specificity 0.946, balanced accuracy 0.973.
- spectra0 (threshold_only): порог 0.2271, recall 1.000, specificity 0.946, balanced accuracy 0.973. Старый порог 1.02: recall 1.000, specificity 0.919.
- aasist3 (threshold_only): порог 0.1140, recall 0.929, specificity 1.000, balanced accuracy 0.964.

## Тест 16 ноября, промежуточный срез

На момент подсчёта было 158 бинарных звонков, все из местной половины. Новые модели посчитаны на 146 из них: у остальных ещё не было признаков. Старые модели взяты из уже готовой таблицы оценок, поэтому у них n=158. Вторая половина и ещё не прослушанные звонки сюда не входят. Это не закрытый день.
Человек 122, синтетика 36.

- logistic (retrained_classical): n 146, recall 0.528, specificity 0.991, precision 0.950, TP 19 FP 1 TN 109 FN 17.
- svm (retrained_classical): n 146, recall 0.639, specificity 0.964, precision 0.852, TP 23 FP 4 TN 106 FN 13.
- lightgbm (retrained_classical): n 146, recall 0.694, specificity 0.982, precision 0.926, TP 25 FP 2 TN 108 FN 11.
- spectra0 (threshold_only): n 146, recall 0.750, specificity 0.909, precision 0.730, TP 27 FP 10 TN 100 FN 9.
- spectra0_frozen_1.02 (unchanged): n 146, recall 0.778, specificity 0.873, precision 0.667, TP 28 FP 14 TN 96 FN 8.
- aasist3 (threshold_only): n 146, recall 0.500, specificity 0.964, precision 0.818, TP 18 FP 4 TN 106 FN 18.
- logistic_old_bundle (frozen_answering_machine_model): n 158, recall 0.667, specificity 0.959, precision 0.828, TP 24 FP 5 TN 117 FN 12.
- svm_old_bundle (frozen_answering_machine_model): n 158, recall 0.611, specificity 0.984, precision 0.917, TP 22 FP 2 TN 120 FN 14.
- lightgbm_old_bundle (frozen_answering_machine_model): n 158, recall 0.639, specificity 0.967, precision 0.852, TP 23 FP 4 TN 118 FN 13.

Валидационные числа показывают, какой порог выиграла сетка. Они не являются проверкой на новом дне.
Старые logistic, SVM и LightGBM в тесте — это прежняя задача «живой диалог или автоответчик», их класс machine не равен синтетике.
