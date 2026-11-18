# tests

Общего набора тестов нет. У каждого компонента свой каталог, и он запускается из корня компонента, чтобы пакет `src` разрешился правильно.

```powershell
cd voice-dataset-collector
python -m unittest discover -s tests

cd ..\classical-anti-spoofing
python -m unittest discover -s tests

cd ..\synthetic-speech-dataset
python -m unittest discover -s tests
```

У Spectra unit tests нет. Тесты не скачивают модели и не читают полный датасет. `RealSelectionTests` пропускается, если локальный манифест B2X не найден.
