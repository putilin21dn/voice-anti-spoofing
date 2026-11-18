# Проверка таблицы звонков

Ассоциации посчитаны только на train. Длительность и SNR не входят в признаки модели. Высокая корреляция не означает, что признак вызван синтезом.

- звонков с признаками: 1496
- дубли call_id: 0
- расхождение метки и split: 0
- без split: 10
- нечитаемые файлы: 10
- unknown: 0
- без пригодных сегментов: 11
- статусы: {'ok': 1485, 'error': 10, 'insufficient_speech': 1}
- split: {'train': 1002, 'validation': 251, 'spectra_threshold': 190, 'test': 43, 'nan': 10}
- чужие locked_split: {'train': [], 'validation': [], 'test': [], 'spectra_threshold': []}

## Признаки, сильнее всего связанные с machine на train

- mfcc_static_00_mean__mean: corr=0.497, с длительностью речи -0.058
- lfcc_static_00_mean__mean: corr=0.495, с длительностью речи -0.054
- mfcc_delta2_00_max__mean: corr=0.477, с длительностью речи -0.021
- mfcc_delta_06_max__mean: corr=0.459, с длительностью речи 0.036
- cqcc_static_00_mean__mean: corr=0.459, с длительностью речи -0.050
- f0_voiced_fraction__mean: corr=0.449, с длительностью речи -0.082
- mfcc_delta_00_min__mean: corr=-0.445, с длительностью речи -0.004
- mfcc_delta2_00_std__mean: corr=0.427, с длительностью речи -0.093
- mfcc_delta_06_std__mean: corr=0.426, с длительностью речи -0.050
- mfcc_static_11_max__mean: corr=0.421, с длительностью речи 0.037
- cqcc_delta2_12_max__mean: corr=0.405, с длительностью речи 0.079
- mfcc_delta2_06_std__mean: corr=0.404, с длительностью речи -0.046
- cqcc_delta_10_max__mean: corr=0.404, с длительностью речи 0.039
- cqcc_delta2_12_min__mean: corr=-0.402, с длительностью речи -0.071
- cqcc_delta2_01_std__mean: corr=0.399, с длительностью речи -0.088

## Среди них сильнее связаны с длительностью, чем с меткой

Таких среди первых 30 нет.

## Распределения на train

Доля machine внутри межквартильного размаха human. Границы взяты только с train.

- speech_duration_s: human median 10.900, machine median 9.000, machine внутри human IQR 0.601
- recording_duration_s: human median 28.823, machine median 20.041, machine внутри human IQR 0.363
- speech_ratio: human median 0.396, machine median 0.563, machine внутри human IQR 0.264
- snr_db: human median 49.382, machine median 54.047, machine внутри human IQR 0.316
- n_usable_segments: human median 4.000, machine median 3.000, machine внутри human IQR 0.611
- spectral_centroid_hz_mean__mean: human median 973.107, machine median 807.915, machine внутри human IQR 0.352
- spectral_flatness_mean__mean: human median 0.058, machine median 0.028, machine внутри human IQR 0.326
- f0_voiced_fraction__mean: human median 0.460, machine median 0.716, machine внутри human IQR 0.161
- mfcc_static_00_mean__mean: human median -265.035, machine median -224.037, machine внутри human IQR 0.233
- mfcc_delta_06_max__mean: human median 5.407, machine median 6.805, machine внутри human IQR 0.212
- acf_harmonicity_mean__mean: human median 0.407, machine median 0.496, machine внутри human IQR 0.352

## Выбросы на train

Граница Тьюки посчитана по всему train, без разреза по метке.

- speech_duration_s: 85 звонков, human 68, machine 17
- recording_duration_s: 91 звонков, human 69, machine 22
- speech_ratio: 0 звонков, human 0, machine 0
- snr_db: 3 звонков, human 2, machine 1
- n_usable_segments: 80 звонков, human 61, machine 19
- spectral_centroid_hz_mean__mean: 3 звонков, human 3, machine 0
- spectral_flatness_mean__mean: 24 звонков, human 24, machine 0
- f0_voiced_fraction__mean: 0 звонков, human 0, machine 0
- mfcc_static_00_mean__mean: 3 звонков, human 0, machine 3
- mfcc_delta_06_max__mean: 29 звонков, human 5, machine 24
- acf_harmonicity_mean__mean: 2 звонков, human 1, machine 1

## Пропуски по группам на train

- lfcc: колонок 1680, доля NaN 0.000
- mfcc: колонок 1680, доля NaN 0.000
- cqcc: колонок 1092, доля NaN 0.000
- f0: колонок 42, доля NaN 0.002
- voice_quality: колонок 28, доля NaN 0.000
- spectral: колонок 140, доля NaN 0.000

## Исходы кампании

Колонки с телефоном не читались. Исход в модель не входит.

- human outcome=: 629
- human outcome=b2x_async: 16
- human outcome=b2x_callback: 54
- human outcome=b2x_lead: 30
- human outcome=b2x_not_lpr: 22
- human outcome=b2x_wrong_org: 61
- human outcome=connection_issue: 8
- human outcome=refused: 295
- human outcome=silence: 5
- machine outcome=: 41
- machine outcome=answering_machine: 278
- machine outcome=b2x_callback: 11
- machine outcome=b2x_lead: 9
- machine outcome=b2x_not_lpr: 6
- machine outcome=connection_issue: 8
- machine outcome=refused: 8
- machine outcome=silence: 13
- human classification_result=асинхронная_связь: 22
- human classification_result=лид: 30
- human classification_result=не_лпр: 40
- human classification_result=не_та_организация: 77
- human classification_result=отказ_лпр: 717
- human classification_result=перезвон: 233
- machine classification_result=answering_machine|answering_machine_text|автоответчик: 244
- machine classification_result=answering_machine|автоответчик: 34
- machine classification_result=автоответчик: 71
- machine classification_result=перезвон|manual_review: 11
