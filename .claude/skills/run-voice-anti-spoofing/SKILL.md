---
name: run-voice-anti-spoofing
description: >-
  Installs and runs the voice-anti-spoofing repo after a clone: check_setup,
  bootstrap, unit tests, and the shadow_replay example. Use when the user asks
  to clone, install, bootstrap, set up, run tests, запустить, развернуть, or
  проверить, что проект стартует.
---

# Запуск voice-anti-spoofing

Команды из корня репозитория. Код и пути не выходят за этот корень.

## Что запускается сразу

1. `python scripts/check_setup.py`
2. Если venv нет: `python scripts/bootstrap.py`
3. Повтор тестов: `python scripts/run_tests.py`
4. Пример без моделей: `python -m shadow_replay shadow_replay/examples/human_call.json`

Нужен Python 3.11+, для classical лучше 3.12. `bootstrap.py` ставит requirements только в `voice-dataset-collector`, `classical-anti-spoofing` и `synthetic-speech-dataset`.

Один тест synthetic ищет локальный `confirmed_001` и без него падает. Collector, classical и shadow_replay от этого файла не зависят.

## Что без локальных файлов не запускать

- Чтение базы и скачивание WAV: нужен заполненный `voice-dataset-collector/.env`. Значения не печатать и не коммитить.
- Inference Spectra: отдельный venv, torch и `model.safetensors`. Команды в `docs/operations/local_setup.md`.
- Ансамбль A: `synthetic-speech-dataset/reports/origin_retrain_20261113/models/svm.joblib` и `lightgbm.joblib` лежат в Git. Logistic того же прогона лежит рядом и в голосе A не используется.

Для прогона записей читай skill `score-recorded-calls`.

## Границы

- Не импортировать код из каталогов рядом с репозиторием и не из AppData.
- Не коммитить аудио, CSV, safetensors, лишние joblib, `.env`, логи.
- Не менять production AMD, hangup и базу.
- 18 bundle в `classical-anti-spoofing/models/` и три joblib прогона 2026-11-13 уже в Git. Другие `.joblib` не добавлять.
