# Официальные метрики происхождения: день 2026-11-13

Слушание закрыто. Метки: `data/annotations/day_20261113_calls.csv`, 279 из 279. Скоры взяты из уже законченного прогона `evaluation/day_20261113/20261113T175850` (`data/manifests/day_20261113_scores.csv`). Модели заново не считались, порог не двигался, обучение не запускалось.

Промежуточный файл `reports/day_20261113_interim_analysis.md` — срез на 105 метках. Для выводов нужен этот файл.

Задача: бинарно отличить синтетический голос абонента от несинтетического. Подтипы — пометки для разбора ошибок.

## Правила счёта

- Spectra-0: spoof, если средний margin < 1.02. Margin = bonafide_logit − spoof_logit. Равенство 1.02 — bonafide. Порог на этом дне не подбирался.
- AASIST3: опубликованного порога звонка нет. В таблицах средний bonafide logit. Класс звонка не назначался.
- Logistic / SVM / LightGBM остаются на старой задаче «живой диалог против автоответчика». Их human/machine не читаются как «человек / синтетика». Пороги bundle: logistic ≥ 0.49265447022723596, SVM ≥ −0.22144920850405767, LightGBM ≥ 0.04007147659674559. Положительный класс — machine.
- В бинарный знаменатель входят `human_live` и `human_recorded` против `synthetic_tts`, `synthetic_voice_conversion`, `synthetic_playback`.
- `unknown` в знаменатель не входит. `mixed` в бинарные метрики не входит.
- `synthetic_voice_conversion` пустая. Нулями не заполняется.
- Метка относится к каналу абонента. Канал агента — наш TTS и звонок синтетическим не делает.
- Телефонов, external_id и ключей S3 в отчёте нет. Идентификатор — `call_id`.

## Состав

279 исходящих B2X, Москва, 2026-11-13, WAV не короче 15 с, стерео 8 кГц pcm_s16le, канал абонента 0. PACK1: 169. PACK2: 110.

| primary | n | в бинарном счёте |
| --- | ---: | --- |
| human_live | 181 | не синтетика |
| human_recorded | 2 | не синтетика |
| synthetic_tts | 67 | синтетика |
| synthetic_playback | 1 | синтетика |
| mixed | 7 | нет |
| unknown | 21 | нет |

`synthetic_voice_conversion` пусто.

Знаменатель: 183 несинтетических + 68 синтетических = 251. Вне счёта: 28.

Источник: `manual_listening` 258, `unknown_source` 21 (все 21 — это `unknown`). Уверенность: medium 245, high 8, low 5, пусто 21. Статус: reviewed 253, needs_second_review 26. Из них 21 — `unknown`, и ещё 5 — `human_live` с уже поставленным происхождением. Эти 5 в знаменателе остаются. Непросмотренных и пустых primary нет. Сегментов нет: одна метка на звонок.

Вторичные пометки только у `synthetic_tts`: без пометки 35, ai_secretary 20, voicemail 7, anti_spam 5. Единственная заметка: `806814b8-2bdc-4418-b6f1-a9fa6a05b94a`, human_live — до 15 секунды музыка ожидания, дальше живой человек.

| сторона | n | длительность, медиана / среднее, с | исходы |
| --- | ---: | --- | --- |
| human_live | 181 | 28.8 / 35.6 | пусто 103, refused 60, b2x_wrong_org 11, b2x_lead 5, b2x_not_lpr 1, b2x_callback 1 |
| human_recorded | 2 | 16.4 и 57.0 | оба answering_machine |
| synthetic_tts | 67 | 76.8 / 103.8 | answering_machine 29, b2x_callback 16, refused 7, connection_issue 7, пусто 3, b2x_not_lpr 3, silence 1, b2x_wrong_org 1 |
| synthetic_playback | 1 | 25.0 | пусто |
| mixed | 7 | медиана 39.4 | пусто 5, refused 1, b2x_not_lpr 1 |
| unknown | 21 | медиана 30.1, один 441.5 | пусто 10, silence 9, refused 1, b2x_lead 1 |

PACK1: human_live 86, synthetic_tts 57, unknown 16, mixed 7, human_recorded 2, playback 1. PACK2: human_live 95, synthetic_tts 10, unknown 5. Синтетика сосредоточена в PACK1 и длиннее живых разговоров.

Скоры: Spectra-0 и AASIST3 ok на 278, `no_speech` на 1 (`e3c1b701`, unknown). Классические ok на 276; `insufficient_speech` на `6fcc8af6` и `c084b44b`, оба unknown; тот же `no_speech`. В бинарном знаменателе пустых скоров нет.

## Spectra-0

Положительный класс — синтетика, предсказание положительного — spoof.

|  | предсказано spoof | предсказано bonafide |
| --- | ---: | ---: |
| синтетика, 68 | TP 63 | FN 5 |
| не синтетика, 183 | FP 32 | TN 151 |

- recall 63/68 = 0.926
- specificity 151/183 = 0.825
- precision 63/95 = 0.663
- f1 0.773
- accuracy 214/251 = 0.853
- balanced accuracy 0.876

| корзина margin | не синтетика, 183 | синтетика, 68 |
| --- | ---: | ---: |
| < −2 | 3 | 50 |
| [−2, 0) | 9 | 5 |
| [0, 0.5) | 7 | 4 |
| [0.5, 1.02) | 13 | 4 |
| [1.02, 1.5) | 10 | 1 |
| [1.5, 3) | 45 | 2 |
| ≥ 3 | 96 | 2 |

Медиана margin: human_live 3.248 (среднее 3.928, min −2.791, max 14.037), synthetic_tts −5.795 (среднее −4.688, min −12.824, max 8.438). Основная масса разъехалась. Ошибки — хвосты. Пять пропусков синтетики лежат выше 1.46, один из них на 8.44. Три живых ложных spoof лежат ниже −2. Сдвиг порога 1.02 оба хвоста сразу не убирает.

По вторичным пометкам TTS: ai_secretary 20/20 spoof, без пометки 33/35, voicemail 6/7, anti_spam 3/5.

### Пропуски синтетики, Spectra-0 bonafide

| call_id | метка | вторичная | уверенность | margin | logistic | svm | lightgbm | AASIST logit | сек | исход |
| --- | --- | --- | --- | ---: | --- | --- | --- | ---: | ---: | --- |
| a037652f-505c-4f81-a49d-a2ed36434761 | synthetic_tts | — | medium | 8.438 | human | machine | machine | 12.205 | 41.2 | b2x_wrong_org, prior не_та_организация |
| 7eb5d6cd-8612-484b-a770-eb3113faf791 | synthetic_tts | voicemail | medium | 4.198 | machine | machine | machine | 7.694 | 15.7 | answering_machine |
| 60f772ee-765d-48b3-9099-2f9a50cc0e25 | synthetic_tts | anti_spam | high | 2.211 | machine | machine | machine | 4.184 | 392.6 | answering_machine |
| cf7c2727-4fd9-4032-aefd-71871f58c582 | synthetic_tts | anti_spam | medium | 1.906 | machine | machine | machine | 3.247 | 157.3 | prior автоответчик |
| 8b34a750-b5df-4a6d-be75-7fbc207a9546 | synthetic_tts | — | medium | 1.464 | human | human | human | 1.625 | 335.1 | connection_issue |

`a037652f` — новый и самый далёкий пропуск: margin 8.44 и AASIST logit 12.2, оба смотрят в сторону bonafide. `60f772ee` и `cf7c2727` — длинная реклама «легко запомнить». `7eb5d6cd` — короткий voicemail АТС. `8b34a750` не поймал никто: Spectra bonafide, все три классические human.

`60f772ee` раньше был в confirmed_001 единственным роботом, которого Spectra-0 назвала bonafide. Слушатель поставил `synthetic_tts` / anti_spam / high.

### Ложные spoof на несинтетике

32 звонка. Три живых ниже −2, дальше хвост до порога. `c28eca4d` — единственный здесь `human_recorded`.

| call_id | метка | уверенность | review | margin | logistic | svm | lightgbm | AASIST logit | сек | исход / prior |
| --- | --- | --- | --- | ---: | --- | --- | --- | ---: | ---: | --- |
| fc0ba21c-245c-4388-b0ce-5da9ff149fae | human_live | medium | reviewed | −2.791 | human | human | human | 1.006 | 27.3 | отказ_лпр |
| 8dbd9d14-84ef-456d-922d-64a95f861196 | human_live | medium | reviewed | −2.045 | human | human | human | 5.088 | 17.0 | неизвестно |
| ed9640a9-b85e-43d4-8220-76726f51697f | human_live | medium | reviewed | −2.038 | human | human | human | −1.494 | 27.3 | отказ_лпр |
| 1e663645-f3e0-4f8e-9dc5-c5af53a8f9aa | human_live | medium | reviewed | −1.841 | human | human | human | 5.742 | 36.6 | положил_трубку |
| 0d3b7a75-2ffc-4241-9ecd-f7ce1858f162 | human_live | medium | reviewed | −1.617 | human | human | human | 0.475 | 20.2 | неизвестно |
| 514aaa2f-88ec-4e2f-9287-5f2369811517 | human_live | medium | reviewed | −1.039 | human | human | human | 4.816 | 23.1 | отказ_лпр |
| a9726ff1-8de1-4b7d-93af-8b03835a7552 | human_live | medium | reviewed | −1.026 | human | human | human | −1.363 | 31.2 | отказ_лпр |
| 0702ad57-c3f1-43ca-89cf-66e03891aca1 | human_live | medium | reviewed | −0.609 | human | human | human | −1.456 | 17.3 | отказ_лпр |
| 152969ae-97d4-40bb-a96a-57ccbebd64bd | human_live | medium | reviewed | −0.414 | human | human | human | −2.621 | 15.9 | положил_трубку |
| 9b1b065c-78d9-4c7c-b93f-0f283855b094 | human_live | medium | needs_second_review | −0.142 | human | human | machine | −0.301 | 32.6 | положил_трубку |
| 2bfab363-324f-4c98-96fd-0a2824fb5ad4 | human_live | medium | reviewed | −0.086 | human | human | human | 5.998 | 15.2 | неизвестно |
| 94c6a13a-3098-4066-a121-5b6e8b4361c6 | human_live | medium | reviewed | −0.029 | human | human | human | 6.094 | 22.3 | отказ_лпр |
| 071cbec7-7e21-441a-b44d-1a80f10a9459 | human_live | low | needs_second_review | 0.048 | human | human | human | −0.240 | 16.4 | неизвестно |
| 1d10c111-2251-4334-880e-b591c4f9f150 | human_live | medium | reviewed | 0.077 | human | human | human | 1.998 | 42.9 | refused / отказ_лпр |
| 6e8b7dad-5883-42b2-bf51-88a8cbec6ae6 | human_live | medium | reviewed | 0.119 | human | human | human | 2.702 | 78.2 | refused / отказ_лпр |
| 86ac857a-d6ad-4ed0-845a-a2de09a84c38 | human_live | low | needs_second_review | 0.294 | human | human | human | 1.021 | 18.0 | неизвестно |
| 04ae08b6-8f92-42d8-8a30-6263d416ee59 | human_live | medium | reviewed | 0.363 | human | human | human | 5.022 | 24.4 | отказ_лпр |
| fa8ec8a7-38f6-40c2-a10e-44b97c62263e | human_live | medium | reviewed | 0.422 | human | human | human | 2.014 | 59.4 | b2x_wrong_org |
| 7f2e9e45-3e02-4124-9c5d-1939619daf0b | human_live | medium | reviewed | 0.492 | human | human | human | −0.937 | 37.0 | неизвестно |
| c0e7084f-5815-494f-bd37-386ba13b901b | human_live | medium | reviewed | 0.511 | human | human | human | 1.534 | 17.2 | неизвестно |
| a84f8c4e-dca2-4797-8877-ca2c07bf05ea | human_live | medium | reviewed | 0.536 | human | human | human | 1.065 | 40.7 | refused / отказ_лпр |
| f0226490-648e-4cea-bed7-e6df89bced48 | human_live | medium | reviewed | 0.586 | human | human | human | 1.921 | 46.4 | отказ_лпр |
| b2602358-15c2-4259-b807-fab146323ba8 | human_live | medium | reviewed | 0.599 | human | human | human | 3.507 | 63.3 | refused / отказ_лпр |
| c2d7695f-8c54-41c6-bcc4-98670f4d6ff1 | human_live | medium | reviewed | 0.712 | human | human | human | 1.636 | 41.5 | refused / отказ_лпр |
| abfe6c6f-cd61-4316-9435-06d50e03bca5 | human_live | medium | reviewed | 0.762 | human | human | human | 2.893 | 18.7 | отказ_лпр |
| 170036b1-0db5-464a-b1d8-3b7ac0d64560 | human_live | medium | reviewed | 0.812 | human | human | human | 2.850 | 63.4 | refused / отказ_лпр |
| c2f8191e-43c0-419a-9db0-e605fd16c023 | human_live | medium | reviewed | 0.837 | human | human | human | 0.369 | 20.3 | refused / асинхронная_связь |
| 12fdecde-bc19-47dd-8e69-bcde321588da | human_live | medium | reviewed | 0.849 | human | human | human | 2.116 | 17.2 | отказ_лпр |
| 826e5466-ed26-4526-a8f6-23834247092f | human_live | medium | reviewed | 0.875 | human | human | human | 4.871 | 16.2 | неизвестно |
| 21b99534-4c2a-4db7-9f78-7cc251f7efe1 | human_live | medium | reviewed | 0.915 | human | human | human | 4.368 | 30.4 | отказ_лпр |
| c28eca4d-e82f-4ede-8e57-3582360c7a26 | human_recorded | low | reviewed | 0.975 | machine | machine | machine | −0.166 | 57.0 | answering_machine |
| 935f7bd5-750f-4277-b006-d6470e205e11 | human_live | medium | reviewed | 0.988 | human | human | human | 5.428 | 33.3 | refused / отказ_лпр |

На 30 из 31 живого ложного spoof все три классические модели сказали human. `6e8b7dad`, `f0226490`, `c2d7695f` — те же три human_live из confirmed_001, которых Spectra-0 уже называла spoof.

`human_recorded` всего два, отдельный вывод по подтипу не делается. Второй, `27ee4ba5-9d02-4514-9ee8-0fc225982616`: high, reviewed, Spectra bonafide 4.957, AASIST 7.458, все три классические machine, 16.4 с. Для бинарной задачи это верное bonafide. Для старой задачи machine здесь согласуется с автоответчиком.

`synthetic_playback` один: `8f662623-d653-4289-8318-b75737beece0`, low, reviewed, 25.0 с. Spectra spoof 0.818, он входит в 63 TP. AASIST logit 7.230. Все три классические human.

## Классические модели

Это совпадение их старого класса machine со стороной происхождения. Recall детектора синтеза из этих чисел не читается.

| модель | machine на синтетике, 68 | machine на несинтетике, 183 |
| --- | ---: | ---: |
| logistic | 52 | 13 |
| svm | 54 | 12 |
| lightgbm | 53 | 13 |

Оба `human_recorded` входят в эти 13/12/13: все три модели сказали machine. На `human_live` все три сказали machine у восьми звонков, и Spectra-0 на всех восьми bonafide: `324000a7` margin 7.853, `54ba0359` 6.330, `78b338e3` 2.509, `99159456` 8.227, `9a3a5654` 1.306, `a4f48b90` 6.097, `a8ba522b` 12.711, `db7caa0c` 9.535. Это отказы и «не та организация», не синтетика.

Все три классические сказали human на 13 синтетических. У 12 из них Spectra-0 spoof. Тринадцатый — пропуск `8b34a750`.

| call_id | метка | вторичная | Spectra | margin | AASIST logit | сек | исход |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 4b69ac8d-7a85-46df-8266-617138b4044c | synthetic_tts | ai_secretary | spoof | −9.122 | −1.539 | 18.4 | answering_machine |
| 0ebec9e1-aa91-4fe4-a3c6-eb1bec154e69 | synthetic_tts | — | spoof | −7.302 | −2.090 | 57.6 | b2x_callback |
| b0931366-6e3c-423b-9d5c-82f2d329c376 | synthetic_tts | — | spoof | −5.796 | −1.432 | 71.0 | connection_issue |
| acd5b754-215f-4acf-b5d8-b6b62b734dcc | synthetic_tts | — | spoof | −5.695 | −2.658 | 49.5 | b2x_callback |
| 897929c0-b6e4-4a09-bf38-03ababeabb3e | synthetic_tts | anti_spam | spoof | −4.087 | −0.527 | 47.1 | b2x_callback |
| a145ac8a-8d18-48b3-b2bc-20d3f795a73c | synthetic_tts | — | spoof | −3.980 | −1.507 | 96.1 | refused |
| a6102189-a3d8-41a4-86b5-9ff57d3c1080 | synthetic_tts | — | spoof | −3.929 | −1.975 | 99.6 | refused |
| f5559138-e5cd-4f8a-a9c5-e545b92f4639 | synthetic_tts | — | spoof | −2.986 | −1.960 | 135.8 | answering_machine |
| baa46fb6-2bee-4563-8259-189269e43fed | synthetic_tts | — | spoof | −1.170 | −3.090 | 31.5 | пусто |
| 44285cc8-f37a-400c-9388-6c46a2701256 | synthetic_tts | — | spoof | −0.484 | 1.407 | 89.5 | b2x_callback |
| 9b37b82b-a3a8-4238-b25f-66d136877ea8 | synthetic_tts | — | spoof | 0.265 | −1.683 | 232.6 | refused |
| 8f662623-d653-4289-8318-b75737beece0 | synthetic_playback | — | spoof | 0.818 | 7.230 | 25.0 | пусто |
| 8b34a750-b5df-4a6d-be75-7fbc207a9546 | synthetic_tts | — | bonafide | 1.464 | 1.625 | 335.1 | connection_issue |

## AASIST3

Порога нет. Средний bonafide logit.

| группа | n | mean | median | min | p25 | p75 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| human_live | 181 | 4.899 | 4.800 | −2.621 | 2.850 | 6.619 | 12.196 |
| human_recorded | 2 | 3.646 | 3.646 | −0.166 | — | — | 7.458 |
| synthetic_tts | 67 | −0.581 | −1.600 | −3.090 | −2.013 | −0.022 | 12.205 |
| synthetic_playback | 1 | 7.230 | — | — | — | — | — |
| mixed | 7 | 5.919 | 4.898 | 4.072 | 4.453 | 6.154 | 11.250 |
| unknown | 20 | 0.426 | 0.223 | −3.583 | −0.776 | 1.700 | 4.687 |

Медианы live и tts разъехались (4.80 против −1.60). Хвосты пересекаются: максимум TTS 12.205 — это пропуск Spectra `a037652f`. Все пять пропусков Spectra имеют положительный logit. Playback тоже 7.230 при Spectra spoof. Порог по этим числам не ставится.

## unknown и mixed

`mixed`, 7 звонков, все Spectra bonafide, margin от 1.640 до 7.754, AASIST logit от 4.072 до 11.250. В бинарный счёт не входят.

`unknown`, 21 звонок. У всех needs_second_review, источник `unknown_source`, уверенность пустая. Spectra на 20 со скором: bonafide 6, spoof 14. Медиана margin 0.028. Плюс `e3c1b701` без речи, скора нет.

| call_id | Spectra | margin | logistic | svm | lightgbm | AASIST | сек | исход |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | --- |
| 14f06437-0709-4312-bbd3-cc773fe9d260 | bonafide | 1.356 | machine | human | machine | 2.012 | 30.0 | silence |
| 19654c1a-c65f-4475-b777-f77bcf871a94 | spoof | −0.019 | human | human | human | 0.739 | 38.9 | пусто |
| 1d677d01-fe89-4065-9603-e1a3c50bf91f | spoof | 0.371 | human | human | human | 1.578 | 35.6 | refused |
| 21e4881b-b5cd-49a8-9787-43d1cd3894e3 | spoof | −0.707 | human | human | human | −3.583 | 30.0 | silence |
| 46deb85a-2c31-4b29-bc5e-9f0d5349400e | bonafide | 1.231 | human | human | human | −0.746 | 46.3 | пусто |
| 57524297-59b4-42c4-a756-5f991dce072d | spoof | −0.315 | human | human | human | −3.430 | 30.1 | silence |
| 584f45e0-3464-4dae-93a5-43375a640a55 | bonafide | 6.598 | human | human | human | 3.969 | 43.2 | пусто |
| 6fcc8af6-4ca9-4488-9a66-2c4e1ccdf1db | spoof | −1.941 | insufficient_speech | insufficient_speech | insufficient_speech | −1.210 | 30.0 | silence |
| 88c51fb8-1005-4830-b05a-047959051bc5 | bonafide | 1.268 | machine | machine | machine | 2.831 | 30.1 | silence |
| 9571e7b5-7d77-4f4c-95fc-c95a632f548f | bonafide | 2.476 | machine | machine | machine | 3.581 | 30.1 | silence |
| 97a6020d-73aa-4df4-af6c-2fb8281ab0b0 | spoof | −0.038 | human | human | human | −0.222 | 77.7 | silence |
| a62c4d21-203f-4734-9893-ce08262768f8 | spoof | 0.153 | machine | machine | machine | 1.596 | 31.5 | silence |
| c084b44b-407e-4baa-a366-668aad6476a6 | spoof | 0.093 | insufficient_speech | insufficient_speech | insufficient_speech | −0.532 | 15.6 | пусто |
| dbb2a173-b616-491d-9dbd-f7891b02f1b4 | spoof | −2.255 | human | human | human | −0.866 | 44.4 | пусто |
| de084b60-4f32-4aa5-89dc-2a29faf338c9 | spoof | −1.403 | human | human | human | −0.334 | 441.5 | b2x_lead |
| e314e1c8-3f26-4340-8e4d-000970bc131a | spoof | −1.522 | human | human | human | −0.489 | 26.6 | пусто |
| e3c1b701-7e06-4a95-bc58-9c9f1e3ae671 | no_speech | — | no_speech | no_speech | no_speech | — | 24.9 | пусто |
| e8dc2b09-26ff-40a1-aa6c-ea86ec11815e | bonafide | 3.664 | human | machine | machine | 4.687 | 30.1 | silence |
| eb1d9f07-0367-4acb-9f4e-857206effc22 | spoof | −0.941 | human | human | human | −3.195 | 18.2 | пусто |
| f1a0fd07-682e-46ce-938e-377fc5d3a536 | spoof | −0.358 | machine | machine | machine | 0.668 | 22.8 | пусто |
| fa06e86a-67f6-43a3-a15e-4c18fde7c682 | spoof | 0.075 | human | human | human | 1.471 | 44.6 | пусто |

`de084b60` в confirmed_001 был роботом без подтипа. В дневной разметке слушатель оставил `unknown`.

## Что из этого следует

1. На всём прослушанном дне Spectra-0 при замороженном 1.02 отделяет стороны: recall 0.926, specificity 0.825, precision 0.663, f1 0.773 на 251 звонке с ясной стороной. Медианы margin 3.25 и −5.80.
2. Precision ниже, чем на промежуточных 105, потому что дослушанная часть — в основном живые, и 32 из них попали ниже 1.02. Specificity почти не сдвинулась. Дополнительные TTS Spectra в основном ловит: 62 из 67 spoof.
3. Пять пропусков — не шум у порога. Ближайший к 1.02 имеет margin 1.46, дальний 8.44. Три живых ложных spoof ниже −2. Порог по этому дню не переносится.
4. Классические модели отвечают на другую задачу. На синтетике machine у них 52–54 из 68. Все три говорят human на 13 синтетических, и на 12 из этих 13 Spectra как раз spoof. На живых отказах они иногда говорят machine при Spectra bonafide.
5. Длительность и кампания смешаны с меткой. Медиана TTS 77 с против 29 с у живых. 57 из 67 TTS лежат в PACK1. Долю «детектор синтеза» и долю «длинный автоответчик PACK1» этот один день до конца не разделяет.
6. AASIST3 в среднем смотрит туда же, но класса звонка у него нет. На пяти пропусках Spectra его logit положительный, на единственном playback тоже.
7. `human_recorded` = 2, `synthetic_playback` = 1, `synthetic_voice_conversion` пусто. По этим подтипам отдельной метрики нет.
8. 21 unknown и 7 mixed остаются вне знаменателя.
