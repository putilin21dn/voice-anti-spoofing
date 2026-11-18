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
