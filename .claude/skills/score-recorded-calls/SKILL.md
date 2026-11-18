---
name: score-recorded-calls
description: >-
  Scores recorded calls with ensemble A and keeps B2X interface chips. Use when
  the user asks for a day run, offline scoring, прогон, отметки, synthetic,
  автоответчик, антиспам-бот, or shadow replay of recordings.
---

# Прогон записанных звонков

Сначала `python scripts/check_setup.py`. Если нет venv, Spectra-0 или пары `svm.joblib` и `lightgbm.joblib` ансамбля A, прогон не начинать: сказать, какого файла нет. Веса не скачивать, пока пользователь сам не попросил.

## Голос A

Три модели: Spectra-0, SVM new, LightGBM new. Synthetic, если так сказали минимум две валидные. Меньше двух голосов — `no_decision`, это не human.

- Spectra synthetic, если margin строго ниже `0.22710928916931153`
- SVM synthetic, если score не ниже `0`
- LightGBM synthetic, если score не ниже `0.3460832258410117`

Пороги и `BORDER_WIDTHS` в `shadow_replay/constants.py` не подбирать заново. `predicted_class` старых bundle не является решением ансамбля.

## Отметка интерфейса

Правило в `shadow_replay/ui_marks.py`.

- Не synthetic: оставить чип интерфейса из `prior_classification`.
- Synthetic и текст уже назвал `anti_spam_bot`: «Антиспам-бот».
- Synthetic и текст уже назвал `автоответчик`: «Автоответчик».
- Synthetic, текст их не разделил, и `high_confidence_synthetic` истинна: «Автоответчик».
- Иначе чип не менять.

Высокая уверенность — все три модели сказали synthetic и каждая дальше пограничной полосы. Два голоса из трёх и голос внутри полосы чип не переписывают.

В базу и в интерфейс результат не писать. Hangup не вызывать. Телефоны, `external_id`, `s3_key`, `s3_bucket`, `room_suffix` не печатать.

Сводка уже посчитанного дня: из корня `python -m shadow_replay.summarize_day`. Она пишет только локальный CSV и печатает количества.
