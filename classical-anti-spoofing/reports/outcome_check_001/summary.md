# Метка sample и поле outcome

Пороги моделей не менялись. Outcome и classification_result в обучение не входят.
consistent: machine с answering_machine или human с живым исходом.
conflict: machine с живым исходом или human с answering_machine.
empty и ambiguous (silence, connection_issue) посчитаны отдельно.

## Сколько звонков в каждой корзине

- spectra_threshold: {'consistent': 113, 'empty': 62, 'conflict': 10, 'ambiguous': 5}
- test: {'consistent': 23, 'empty': 11, 'conflict': 9}
- validation: {'empty': 122, 'consistent': 122, 'ambiguous': 5, 'conflict': 2}

## Ошибки замороженных моделей

- spectra_threshold logistic ambiguous: 0 из 5, доля 0.000
- spectra_threshold logistic conflict: 0 из 10, доля 0.000
- spectra_threshold logistic consistent: 12 из 113, доля 0.106
- spectra_threshold logistic empty: 4 из 62, доля 0.065
- spectra_threshold svm ambiguous: 0 из 5, доля 0.000
- spectra_threshold svm conflict: 0 из 10, доля 0.000
- spectra_threshold svm consistent: 9 из 113, доля 0.080
- spectra_threshold svm empty: 4 из 62, доля 0.065
- spectra_threshold lightgbm ambiguous: 0 из 5, доля 0.000
- spectra_threshold lightgbm conflict: 0 из 10, доля 0.000
- spectra_threshold lightgbm consistent: 8 из 113, доля 0.071
- spectra_threshold lightgbm empty: 4 из 62, доля 0.065
- spectra_threshold spectra0 ambiguous: 1 из 5, доля 0.200
- spectra_threshold spectra0 conflict: 1 из 10, доля 0.100
- spectra_threshold spectra0 consistent: 25 из 113, доля 0.221
- spectra_threshold spectra0 empty: 14 из 62, доля 0.226
- test logistic conflict: 0 из 9, доля 0.000
- test logistic consistent: 1 из 23, доля 0.043
- test logistic empty: 0 из 11, доля 0.000
- test svm conflict: 0 из 9, доля 0.000
- test svm consistent: 1 из 23, доля 0.043
- test svm empty: 0 из 11, доля 0.000
- test lightgbm conflict: 0 из 9, доля 0.000
- test lightgbm consistent: 1 из 23, доля 0.043
- test lightgbm empty: 0 из 11, доля 0.000
- test spectra0 conflict: 0 из 9, доля 0.000
- test spectra0 consistent: 5 из 23, доля 0.217
- test spectra0 empty: 4 из 11, доля 0.364
- validation logistic ambiguous: 1 из 5, доля 0.200
- validation logistic consistent: 9 из 122, доля 0.074
- validation logistic empty: 8 из 121, доля 0.066
- validation svm ambiguous: 1 из 5, доля 0.200
- validation svm consistent: 10 из 122, доля 0.082
- validation svm empty: 4 из 121, доля 0.033
- validation lightgbm ambiguous: 1 из 5, доля 0.200
- validation lightgbm consistent: 10 из 122, доля 0.082
- validation lightgbm empty: 7 из 121, доля 0.058

## Перекрытие score

Доля machine, чей score попал в межквартильный размах human. Меньше — классы дальше друг от друга.

- spectra_threshold logistic: human median 0.044, machine median 0.931, перекрытие 0.020
- spectra_threshold svm: human median -1.385, machine median 0.818, перекрытие 0.030
- spectra_threshold lightgbm: human median 0.000, machine median 0.996, перекрытие 0.020
- spectra_threshold spectra0: human median 3.374, machine median -1.027, перекрытие 0.172
- test logistic: human median 0.119, machine median 0.926, перекрытие 0.000
- test svm: human median -1.302, machine median 0.876, перекрытие 0.000
- test lightgbm: human median 0.000, machine median 0.987, перекрытие 0.040
- test spectra0: human median 3.492, machine median -2.090, перекрытие 0.240
- validation logistic: human median 0.065, machine median 0.896, перекрытие 0.021
- validation svm: human median -1.342, machine median 0.812, перекрытие 0.021
- validation lightgbm: human median 0.000, machine median 0.989, перекрытие 0.021

На test конфликтные исходы не дали ни одной ошибки ни классическим моделям, ни Spectra-0: 0 из 9. Единственная ошибка трёх классических моделей, `a6eb780a-85ae-4c5b-8108-d9f2134e5d07`, имеет outcome `answering_machine`. Длинный machine `dc14c9b3-90eb-4785-8599-2e2c23210b9f`, который Spectra-0 помечает human, имеет пустой outcome и classification_result `не_та_организация|manual_review`.
