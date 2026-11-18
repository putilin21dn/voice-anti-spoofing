# Локальная установка

Общее окружение при сборке не создавалось и пакеты не устанавливались. Ставьте зависимости компонента только когда собираетесь его запускать.

Рекомендуемые интерпретаторы взяты из README источников: Python 3.12 для classical, Python 3.11 или 3.12 для Spectra. Collector и synthetic отдельную версию в README не фиксируют. `pyproject.toml` допускает `>=3.11`.

## Collector

```powershell
cd voice-dataset-collector
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Дальше заполните `.env` локально. Проверка соединения: `python src/main.py check-db` и `python src/main.py check-s3`. Обе команды, по README, только читают.

Нужны `psycopg[binary]`, `boto3`, `python-dotenv`. Те же ограничения записаны в группе `dataset-collector` корневого `pyproject.toml`.

## Classical

```powershell
cd classical-anti-spoofing
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Секреты не нужны. Список пакетов: numpy, scipy, pandas, soundfile, librosa, PyYAML, tqdm, webrtcvad-wheels, spafe, praat-parselmouth, scikit-learn, lightgbm, matplotlib, shap. Версии в `requirements.txt` — нижние границы, они не повышались.

Перед `extract-features` и `train` нужны локальные WAV и CSV. См. `docs/operations/data_storage.md`.

## Spectra

Отдельного `.venv` в исходном каталоге не было. README ставит PyTorch с индекса CUDA 13.2, затем остальное:

```powershell
cd voice-anti-spoofing-spectra0
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu132
python -m pip install -r requirements.txt
python src/check_environment.py
```

`requirements.txt` не содержит torch. В нём transformers, huggingface_hub, safetensors, soundfile, numpy, scipy, pandas, tqdm.

`download_model.py` скачивает `model.safetensors`. Запускайте его только когда веса нужны. Первый inference может дополнительно скачать `facebook/wav2vec2-xls-r-300m`.

## Synthetic speech dataset

Собственный `requirements.txt` содержит только `PyYAML`. Этого хватает не для всех скриптов. Разметка и `src/main.py` дополнительно используют модули, которые в классическом окружении уже есть. Скрипты признаков и переобучения прямо требуют интерпретатор classical-anti-spoofing, чтобы пакетом `src` был его код.

Для unit tests и локальной разметки достаточно окружения, где установлен PyYAML и зависимости, которые импортируют тесты. Проверка без данных:

```powershell
cd synthetic-speech-dataset
python -m unittest discover -s tests
```

`python src/main.py serve` слушает только `127.0.0.1:8765` и аудио наружу не отправляет.

## Почему не одно окружение на всех

Нижние границы версий в четырёх `requirements.txt` друг другу не противоречат: ни один файл не закрепляет верхнюю границу, которая отвергает нижнюю границу другого. Общая установка всё равно плохая идея по двум причинам из самих README:

- Spectra ставит PyTorch отдельной командой с CUDA-индексом. Обычный `pip install -r` этот шаг не выполняет.
- Исследовательский код synthetic импортирует `src` classical и должен видеть именно тот интерпретатор. Второе дерево `src` в том же процессе эти импорты перепутает.

Группы `[project.optional-dependencies]` в `pyproject.toml` повторяют файлы требований и не включают torch. Команда установки всего сразу в этот проект не добавлялась.

## Тесты, которые можно запускать без данных и весов

- `voice-dataset-collector`: `python -m unittest discover -s tests`
- `classical-anti-spoofing`: `python -m unittest discover -s tests`
- `synthetic-speech-dataset`: `python -m unittest discover -s tests`

Не запускайте `train`, `extract-features` на полном наборе, `run_inference.py` и `download_model.py`, если цель — только проверить сборку.
