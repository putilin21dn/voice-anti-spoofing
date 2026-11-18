# configs

Общего конфига, который подменяет компоненты, нет. Иначе рядом жили бы две копии одних и тех же порогов.

| Компонент | Конфиг |
| --- | --- |
| classical | `classical-anti-spoofing/config/config.yaml` |
| synthetic | `synthetic-speech-dataset/config/` |
| collector | переменные в `.env`, образец `voice-dataset-collector/.env.example` |
| spectra | аргументы CLI в `src/run_inference.py` и `src/run_aasist3.py` |

Порог Spectra-0 1.02 и константы `rules.py` при сборке не менялись.
