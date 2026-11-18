# Classical Anti-Spoofing

Локальный эксперимент: акустические признаки и классические классификаторы для отделения человеческой речи от автоматической. Модуль не подключается к worker, backend и не меняет Spectra-0.

Аудио остаётся на этой машине. Номера телефонов в таблицы и логи не пишутся. Исходные WAV не изменяются.

## Что уже можно запускать

Признаки, проверка сигнала, аудит звонков, обучение Logistic Regression, SVM и LightGBM, оценка на validation / test / n100 и сравнение со Spectra-0. Порог Spectra-0 не меняется. Пороги классических моделей выбираются на validation и в `evaluate` не пересматриваются.

## Окружение (Windows PowerShell)

Нужен Python 3.12.

```powershell
cd classical-anti-spoofing
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Если активация заблокирована:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Секреты не нужны. `.env` можно не создавать.

## Данные

Источник задан в `config/config.yaml` и указывает на уже извлечённый клиентский канал:

`voice-dataset-collector/data/samples/b2x/2026-11/samples.csv`

Файлы не копируются. Метка берётся только из колонки `sample`: `human`, `machine`. Пустого `unknown` в текущей выборке нет, такие строки сохраняются в manifest и не участвуют в обучении.

`human` и `machine` — метки кампании B2X, а не имя TTS-генератора. Идентификатора говорящего в разметке нет, поэтому разрез по людям и по невиданным генераторам сделать нельзя.

Колонки `external_id` и `s3_key` не читаются: в них встречается телефон.

Каналы в mono не смешиваются. Готовые семплы уже одноканальные, 8 кГц, 16 бит. Повышение частоты до 16 кГц отключено. Верхняя частота признаков не выше Найквиста, то есть 4000 Гц.

## Разделение звонков

Один `call_id` попадает только в один split. `test` — это готовый `test20`, его нельзя использовать для обучения и порога. `spectra_threshold` — это `n100`, на нём уже выбирался порог Spectra-0. Train и validation берутся только из остальных звонков, seed `42`, доля validation `0.2`, стратификация по метке.

```powershell
python src/main.py inspect-data
python src/main.py create-splits
```

## Признаки

VAD — отдельная копия WebRTC VAD: mode 2, кадр 20 мс, склейка паузы до 300 мс. Это то же правило сегментов, что у Spectra-0, но сегменты короче 500 мс сохраняются и помечаются как непригодные. Тишина не идёт в голосовые признаки. Короткие куски не повторяются, чтобы дотянуть длину.

| Группа | Что считается | Зачем |
| --- | --- | --- |
| LFCC | 20 кепстральных коэффициентов по линейной шкале, mean/std/min/max, дельта и дельта-дельта | Различия огибающей спектра, которые часто используют против синтеза |
| MFCC | 20 коэффициентов по мел-шкале до 4000 Гц, те же статистики | Дополнительное описание тембра |
| CQCC | spafe 0.3.3: CQT, логарифм мощности, resample этого спектра, DCT | Отдельная кепстральная группа. Волна не повышается до 16 кГц. `resampling_ratio: 1.0` не повторяет геометрическое пересэмплирование оригинального MATLAB CQCC; это ограничение записано явно |
| F0 | mean, median, std, range, IQR и доля озвученных кадров | Высота тона. Нули и неозвученные кадры в среднее не входят |
| Voice quality | jitter, shimmer и HNR Praat, плюс автокорреляционная гармоничность | Praat на 8 кГц помечен `telephone_8khz_limited`. Автокорреляция в диапазоне 60–400 Гц считается отдельно и не называется HNR. Сбой или короткий фрагмент даёт NaN |
| Spectral | centroid, bandwidth, rolloff, flatness, zero-crossing rate | Форма спектра и шумность внутри телефонной полосы |

Основная единица будущего обучения — звонок. Сегментные признаки агрегируются mean, median, std, min, max, p25 и p75 только по пригодным сегментам, чтобы длинный звонок не получил больший вес числом строк.

Повторный запуск пропускает `call_id`, которые уже есть в `data/features/completed_call_ids.txt`. Ошибка файла туда не записывается, поэтому следующий запуск пробует его снова. Смена параметров признаков требует другой каталог: старый расчёт сам не затирается.

```powershell
python src/main.py extract-features --splits pool --per-class 8
python src/main.py validate-features
python src/main.py extract-features
```

Полный прогон без `--per-class` продолжит уже посчитанные звонки.

## Spectra-0

Baseline не меняется: средний по всем окнам зазор `bonafide_logit - spoof_logit`, spoof если зазор меньше 1.02. Это не калиброванная вероятность.

```powershell
python src/main.py compare-spectra0
```

На `n100` порог уже выбирался, поэтому эти метрики не являются независимым тестом. Независимый набор — `test20`.

## Обучение и сравнение

Длительность записи, доля речи, SNR, число сегментов и частота дискретизации в матрицу модели не входят: они описывают структуру записи. Отбор признаков и подбор параметров идут только по train и validation. `test20` и `n100` в обучение не попадают. `n100` всё равно нельзя считать независимым тестом Spectra-0: порог 1.02 выбран на нём.

```powershell
python src/main.py audit-calls
python src/main.py audit-outcomes
python src/main.py train --model logistic
python src/main.py train --model svm
python src/main.py train --model lightgbm
python src/main.py evaluate
python src/main.py compare-spectra0
```

`audit-calls` пишет распределения Human/Machine, пропуски, выбросы и исходы кампании в новый `reports/data_audit_NNN`. Последний такой отчёт — `data_audit_003`. Каждый запуск обучения пишет новый каталог `reports/experiment_NNN` или `reports/comparison_NNN` и не затирает предыдущий. Модель сохраняется в `models/{logistic,svm,lightgbm}.joblib`. Score — ранжирующая оценка, не калиброванная вероятность. Для SVM это `decision_function`, для Logistic Regression и LightGBM — `predict_proba` положительного класса.

Текущие цифры и оговорки: `reports/technical_report.md`. Кратко по `test20` (18 human, 25 machine), пороги заморожены на validation:

| Модель | Recall machine | Specificity human | Balanced accuracy | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Spectra-0, порог 1.02 | 0.680 | 0.944 | 0.812 | 0.847 |
| Logistic Regression | 0.960 | 1.000 | 0.980 | 0.989 |
| RBF SVM | 0.960 | 1.000 | 0.980 | 0.993 |
| LightGBM | 0.960 | 1.000 | 0.980 | 0.989 |

На validation balanced accuracy около 0.93. На `n100` у классических моделей она 0.92–0.94, у Spectra-0 0.79. Test содержит 43 звонка: три модели ошиблись на одном и том же коротком machine-звонке, который Spectra-0 распознал. Порог после просмотра test не менялся. В маршрутизацию production эти модели не ставились.

Отдельный вариант `no_energy` убирает до отбора RMS и нулевой коэффициент LFCC, MFCC и CQCC, затем заново выбирает 200 признаков на train. Базовые файлы `models/*.joblib` при этом не перезаписываются: вариант лежит в `models/no_energy/`.

```powershell
python src/main.py train --model logistic --variant no_energy
python src/main.py train --model svm --variant no_energy
python src/main.py train --model lightgbm --variant no_energy
python src/main.py evaluate --variant no_energy
python src/main.py compare-spectra0 --variant no_energy
```

На test20 без энергии balanced accuracy стала 0.960 / 0.932 / 0.952. Девять звонков, которые полный набор ловил, а Spectra-0 нет, остались пойманными. Отчёт: `reports/experiment_009` и `reports/comparison_002`.

Два следующих варианта режут матрицу ещё уже. `shape` оставляет кепстр с коэффициента 1 и спектральные меры. `voicing` оставляет F0, долю озвученных кадров, гармоничность, jitter, shimmer и HNR. Оба пишутся в свои каталоги `models/shape` и `models/voicing`.

```powershell
python src/main.py train --model logistic --variant shape
python src/main.py evaluate --variant shape
python src/main.py compare-spectra0 --variant shape
python src/main.py train --model logistic --variant voicing
python src/main.py evaluate --variant voicing
python src/main.py compare-spectra0 --variant voicing
```

На test20 balanced accuracy формы спектра: 0.940 / 0.932 / 0.980. Озвученности: 0.940 / 0.912 / 0.877. LightGBM только по форме спектра повторяет полный набор на этих 43 звонках. Обе группы ловят те же пропуски Spectra-0. Отчёты: `experiment_013`, `comparison_003`, `experiment_017`, `comparison_004`.

`mfcc` оставляет только MFCC с коэффициента 1. `spectral` оставляет центроид, полосу, rolloff, flatness и zero-crossing rate. На n100 balanced accuracy MFCC 0.890 / 0.886 / 0.888, спектральных мер 0.898 / 0.927 / 0.933. Spectra-0 там 0.788. Отчёты: `experiment_021`, `comparison_005`, `experiment_025`, `comparison_006`.
