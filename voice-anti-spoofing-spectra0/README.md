# Spectra-0 standalone pilot

Изолированный эксперимент, не подключённый к backend.

## 1. Окружение (Windows PowerShell)

Рекомендуется Python 3.11 или 3.12. Создай venv:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Если PowerShell блокирует активацию:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 2. PyTorch с CUDA для RTX 5060 Ti

Устанавливаем CUDA 13.2 wheel для Windows:

```powershell
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu132
python -m pip install -r requirements.txt
```

## 3. Проверка GPU

```powershell
python src/check_environment.py
```

Нужен результат `CUDA available: True` и имя RTX 5060 Ti. Если CUDA False, остановись и пришли вывод команды — не продолжай загрузку модели.

## 4. Загрузка Spectra-0

```powershell
python src/download_model.py
```

Скрипт скачивает только `model.py` и `model.safetensors` из `lab260/spectra_0`. При первом инференсе Hugging Face автоматически скачает дополнительный encoder `facebook/wav2vec2-xls-r-300m`. Нужен интернет и несколько GB свободного места.

## 5. Подготовить тестовые записи

Положи WAV-файлы в `data/input/`.

По умолчанию выбирается первый канал (обычно левый/клиентский). Для stereo-звонков заранее проверь порядок каналов. Второй канал:
```powershell
python src/run_inference.py --input data/input --channel 1
```

Не смешивай голос клиента и бота. Для первого smoke test предпочтительны mono WAV.

## 6. Запуск

```powershell
python src/run_inference.py --input data/input --channel 0 --device auto
```

Принудительно CPU:
```powershell
python src/run_inference.py --input data/input --channel 0 --device cpu
```

## 7. Результаты

В `data/output/` появятся:
- `spectra0_results.csv` — результаты по каждому окну аудио;
- `run_metadata.json` — параметры запуска.

Модель принимает float32 waveform, обычно 16 kHz; выход index 0 — spoof, index 1 — bonafide. Скрипт ресемплирует аудио в 16 kHz, применяет pre-emphasis 0.97 и режет на неперекрывающиеся окна по 64 600 сэмплов (около 4.04 секунды). Короткий хвост повторяется до длины окна по примеру model card.

`spoof_softmax` и `bonafide_softmax` — нормализованные выходы модели, **не откалиброванные вероятности**. Порог решения не применяется. `higher_logit_class` — только argmax для ориентировки, не production-классификация.

## Ограничения

Это технический smoke test, не оценка качества модели. Для оценки нужны размеченные настоящие люди, человеческие автоответчики, синтетическая речь и AI-секретари, записанные через телефонный канал. Делить выборку нужно по звонкам/спикерам/генераторам, а не случайно по окнам. Окна одного звонка нельзя считать независимыми примерами.

## Ссылки и лицензия

Модель: https://huggingface.co/lab260/spectra_0

В карточке вход указан как float32 waveform, обычно 16 kHz; logits: 0=spoof, 1=bonafide. Страница текущего репозитория показывает Apache-2.0, тогда как текст карточки содержит MIT — перед коммерческим использованием нужно проверить актуальную лицензию непосредственно в репозитории и зафиксировать её.

Все аудио остаются локально; не загружай записи с персональными данными в публичные сервисы.
