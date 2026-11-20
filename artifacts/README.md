# artifacts

Каталоги `models`, `checkpoints`, `predictions` и `experiments` пустые и закрыты от Git.

Уже обученные bundle и веса Spectra кладутся в каталоги компонентов, потому что конфиги указывают туда:

- `classical-anti-spoofing/models/`
- `voice-anti-spoofing-spectra0/models/`
- `synthetic-speech-dataset/reports/origin_retrain_20261113/models/`

Текстовые отчёты экспериментов уже лежат в `reports/` компонентов и в Git входят. Предсказания и рисунки — нет. В Git из весов входят 18 bundle `classical-anti-spoofing/models/` и три joblib прогона 2026-11-13.
