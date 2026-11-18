# Промежуточный разбор: детекция синтетического голоса, день 2026-11-13

Срез меток: 2026-11-15 21:43 UTC, файл `data/annotations/day_20261113_calls.csv`. На момент среза сохранено 105 звонков из 279. Слушатель продолжает разметку, числа сдвинутся. Это не финальный отчёт и не повод менять пороги.

Задача, которую надо анализировать: бинарно отличить синтетический голос абонента от несинтетического. Подтипы внутри стороны — пометки для разбора ошибок, не отдельные метрики.

## Что можно и чего нельзя делать с этими данными

- Порог Spectra-0 заморожен: spoof, если средний по окнам margin < 1.02. Margin = bonafide_logit − spoof_logit. Равенство 1.02 — bonafide. Порог на этом наборе не подбирать и не предлагать как результат.
- Класс AASIST3 на звонок не объявлен: опубликованного порога нет. Число в таблице — средний bonafide logit по окнам. Порог для него не выдумывать.
- Logistic / SVM / LightGBM обучены на старой задаче «живой диалог против автоответчика». Их human/machine нельзя читать как «человек / синтетика».
- `unknown` не входит в знаменатель. `mixed` не входит в бинарные метрики.
- Пустые категории не заполнять нулями. `synthetic_voice_conversion` в этом срезе нет.
- Метки слушает один человек по каналу абонента. Канал агента — наш TTS и сам по себе звонок синтетическим не делает.
- Телефоны, external_id и ключи S3 в данных нет и добавлять их не нужно. Идентификатор — `call_id`.

## Откуда звонки

Москва, календарный день 2026-11-13, исходящие B2X с записью, длительность WAV не короче 15 секунд. В наборе 279 стерео, 8 кГц, pcm_s16le, канал абонента — 0. Кампании дня: `B2X-13-11-2026-PACK1` и `PACK2`. Прогон моделей уже закончен, папка `evaluation/day_20261113/20261113T175850`, плоские скоры `data/manifests/day_20261113_scores.csv` (279 строк).

На всём дне, ещё до меток: Spectra-0 bonafide 169, spoof 109, `no_speech` 1. Logistic human 202, machine 74, пустой класс у 3 (`insufficient_speech` 2, `no_speech` 1).

Размеченные 105 — не случайная подвыборка. Это то, что успели прослушать в интерфейсе. Почти всё из PACK1.

## Правила метки

Метка — происхождение голоса абонента, не сценарий и не исход звонка.

| Метка | Смысл в бинарной задаче |
| --- | --- |
| human_live | не синтетика |
| human_recorded | не синтетика. Записанный голос человека, включая voicemail человека |
| synthetic_tts | синтетика |
| synthetic_voice_conversion | синтетика. В срезе 0 |
| synthetic_playback | синтетика. Синтетика через динамик или трубку остаётся синтетикой |
| mixed | в бинарный счёт не входит. Только если в одном звонке есть и человеческая, и синтетическая семья |
| unknown | в знаменатель не входит. Механизм неясен, в Human или Synthetic не дожимается |

Два записанных голоса одного происхождения — одна метка звонка, не `mixed`. Исходы `answering_machine`, `b2x_lead`, `refused`, `b2x_callback` и старые human/machine сами по себе синтетику не доказывают. Неестественный звук сам по себе тоже не синтетика.

Источник метки в срезе: `manual_listening` 93, `unknown_source` 12 (это как раз строки с primary `unknown` и пустой уверенностью). Уверенность: medium 83, high 6, low 4, пусто 12. Статус: reviewed 91, needs_second_review 14. Непросмотренных среди сохранённых нет. Заметка одна: `806814b8-2bdc-4418-b6f1-a9fa6a05b94a`, human_live — до 15 секунды музыка ожидания, дальше живой человек.

## Состав среза

| primary | n | сторона |
| --- | ---: | --- |
| human_live | 50 | не синтетика |
| human_recorded | 2 | не синтетика |
| synthetic_tts | 33 | синтетика |
| synthetic_playback | 1 | синтетика |
| unknown | 12 | вне знаменателя |
| mixed | 7 | вне бинарных метрик |

Вторичные пометки есть только у `synthetic_tts`: ai_secretary 7, anti_spam 3, voicemail 2, без пометки 21.

Длительность и исход сильно различаются по сторонам. Это смешение, его нельзя игнорировать.

| сторона | n | длительность, медиана / среднее, с | заметные исходы |
| --- | ---: | --- | --- |
| не синтетика (live + recorded) | 52 | live: медиана 23.6, среднее 30.5. recorded: 16.4 и 57.0 | live: пустой исход 35, refused 12, b2x_wrong_org 3. Оба recorded: answering_machine |
| синтетика | 34 | tts: медиана 97.1, среднее 129.6, max 392.6. playback: 25.0 | tts: answering_machine 13, b2x_callback 7, connection_issue 5, refused 4, пусто 3, b2x_not_lpr 1 |
| unknown | 12 | медиана 30.1, один выброс 441.5 | silence 8, пусто 3, b2x_lead 1 |
| mixed | 7 | медиана 39.4 | пусто 5, refused 1, b2x_not_lpr 1 |

Синтетические звонки в срезе в несколько раз длиннее живых. Живые чаще короткие отказы. Часть разделимости скора может идти от сценария «длинный автоответчик», а не только от факта синтеза.

## Пороги, как они заморожены

| Модель | Что за число | Решение |
| --- | --- | --- |
| Spectra-0 | средний margin по окнам речи со status=ok, включая добитые хвосты | spoof, если < 1.02 |
| Logistic | predict_proba класса machine | machine, если ≥ 0.49265447022723596 |
| SVM | decision_function, не вероятность | machine, если ≥ −0.22144920850405767 |
| LightGBM | predict_proba класса machine | machine, если ≥ 0.04007147659674559 |
| AASIST3 | средний bonafide logit | класса звонка нет, `threshold_applied` = false |

Окна Spectra-0 и AASIST3: WebRTC VAD mode 2, 8 кГц, 20 мс, склейка паузы до 300 мс, сегменты короче 500 мс выбрасываются, ресемпл 16 кГц, неперекрывающиеся окна 64600 сэмплов (4.0375 с), короткий хвост повторяется внутри того же речевого сегмента. У Spectra pre-emphasis 0.97 внутри каждого окна. У AASIST3 pre-emphasis на всей волне 16 кГц до нарезки. Порог 1.02 применяется к среднему звонка, не к каждому окну.

## Бинарный счёт Spectra-0

Положительный класс — синтетика, предсказание положительного — spoof. Знаменатель: 52 несинтетических + 34 синтетических. `unknown` и `mixed` не входят.

|  | предсказано spoof | предсказано bonafide |
| --- | ---: | ---: |
| синтетика, 34 | TP 30 | FN 4 |
| не синтетика, 52 | FP 9 | TN 43 |

- recall 30/34 = 0.882
- specificity 43/52 = 0.827
- precision 30/39 = 0.769
- f1 0.822
- accuracy 73/86 = 0.849

Распределение margin.

| корзина margin | не синтетика, 52 | синтетика, 34 |
| --- | ---: | ---: |
| < −2 | 2 | 25 |
| [−2, 0) | 1 | 2 |
| [0, 0.5) | 2 | 1 |
| [0.5, 1.02) | 4 | 2 |
| [1.02, 1.5) | 4 | 1 |
| [1.5, 3) | 13 | 2 |
| ≥ 3 | 26 | 1 |

Основная масса уже разъехалась: медиана margin на human_live 3.115 (среднее 4.027, min −2.791, max 14.037), на synthetic_tts −5.695 (среднее −4.529, min −12.824, max 4.198). Ошибки — хвосты, не сдвиг всей группы. Два FP на живых людях лежат глубоко ниже нуля, четыре FN — выше 1.02 с запасом. Подбор порога на этом срезе эти хвосты не уберёт оба сразу.

### Пропуски синтетики, Spectra-0 bonafide

| call_id | метка | вторичная | уверенность | margin | logistic | svm | lightgbm | AASIST logit | сек | исход |
| --- | --- | --- | --- | ---: | --- | --- | --- | ---: | ---: | --- |
| 7eb5d6cd-8612-484b-a770-eb3113faf791 | synthetic_tts | voicemail | medium | 4.198 | machine | machine | machine | 7.694 | 15.7 | answering_machine |
| 60f772ee-765d-48b3-9099-2f9a50cc0e25 | synthetic_tts | anti_spam | high | 2.211 | machine | machine | machine | 4.184 | 392.6 | answering_machine |
| cf7c2727-4fd9-4032-aefd-71871f58c582 | synthetic_tts | anti_spam | medium | 1.906 | machine | machine | machine | 3.247 | 157.3 | answering_machine, prior автоответчик |
| 8b34a750-b5df-4a6d-be75-7fbc207a9546 | synthetic_tts | — | medium | 1.464 | human | human | human | 1.625 | 335.1 | connection_issue |

Три из четырёх — длинные, и на них классические модели сказали machine. Это сценарий автоответчика, который старые модели как раз умеют, а Spectra-0 здесь принимает за bonafide. `8b34a750` не поймал никто: Spectra bonafide, все три классические human, AASIST logit тоже положительный.

`60f772ee` раньше уже был в наборе confirmed_001 как подтверждённый робот и единственный из тех 12 роботов, кого Spectra-0 назвала bonafide (margin 2.211). Слушатель теперь поставил `synthetic_tts` / anti_spam / high.

### Ложные spoof на несинтетике

| call_id | метка | уверенность | review | margin | logistic | svm | lightgbm | AASIST logit | сек | исход / prior |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- | ---: | --- |
| fc0ba21c-245c-4388-b0ce-5da9ff149fae | human_live | medium | reviewed | −2.791 | human | human | human | 1.006 | 27.3 | отказ_лпр |
| ed9640a9-b85e-43d4-8220-76726f51697f | human_live | medium | reviewed | −2.038 | human | human | human | −1.494 | 27.3 | отказ_лпр |
| 0d3b7a75-2ffc-4241-9ecd-f7ce1858f162 | human_live | medium | reviewed | −1.617 | human | human | human | 0.475 | 20.2 | неизвестно |
| 071cbec7-7e21-441a-b44d-1a80f10a9459 | human_live | low | needs_second_review | 0.048 | human | human | human | −0.240 | 16.4 | неизвестно |
| fa8ec8a7-38f6-40c2-a10e-44b97c62263e | human_live | medium | reviewed | 0.422 | human | human | human | 2.014 | 59.4 | b2x_wrong_org |
| b2602358-15c2-4259-b807-fab146323ba8 | human_live | medium | reviewed | 0.599 | human | human | human | 3.507 | 63.3 | refused / отказ_лпр |
| 170036b1-0db5-464a-b1d8-3b7ac0d64560 | human_live | medium | reviewed | 0.812 | human | human | human | 2.850 | 63.4 | refused / отказ_лпр |
| 12fdecde-bc19-47dd-8e69-bcde321588da | human_live | medium | reviewed | 0.849 | human | human | human | 2.116 | 17.2 | отказ_лпр |
| c28eca4d-e82f-4ede-8e57-3582360c7a26 | human_recorded | low | reviewed | 0.975 | machine | machine | machine | −0.166 | 57.0 | answering_machine |

На восьми `human_live` все три классические модели сказали human. Два отказа ЛПР (`fc0ba21c`, `ed9640a9`) Spectra-0 считает spoof уверенно, не на грани порога. Пять других живых лежат между 0 и 1.02. `c28eca4d` — записанный человеческий автоответчик, низкая уверенность, margin 0.975, почти порог; классические модели здесь machine, что для старой задачи согласуется с автоответчиком.

Второй `human_recorded`, `27ee4ba5-9d02-4514-9ee8-0fc225982616`: высокая уверенность, reviewed, два разных записанных голоса, механизм слушатель свёл к человеческой записи. Spectra-0 bonafide 4.957, AASIST logit 7.458, все три классические machine, исход answering_machine, 16.4 с. Для бинарной задачи это верное bonafide. n=2, отдельно про recorded вывод не делается.

Единственный `synthetic_playback`, `8f662623-d653-4289-8318-b75737beece0`: низкая уверенность, reviewed, 25.0 с, prior «не та организация». Слушатель слышал записанный голос или TTS и поставил playback. Spectra-0 spoof, margin 0.818. AASIST logit 7.230, все четыре окна AASIST bonafide. Logistic 0.381 human, SVM −0.657 human, LightGBM 0.00013 human. Классическое human здесь значит «не похоже на автоответчик» и синтетике не противоречит. Окна Spectra-0 на этом звонке: короткое добитое −2.38, полное +6.24, добитое −7.98, добитое +7.39. Среднее 0.818 утащили два коротких добитых spoof-окна. Порог к окнам не применяется.

## Классические модели на той же бинарной разметке

Считать их recall по синтетике бессмысленно как качество детектора синтеза. Ниже — насколько их старый класс machine совпал со стороной.

| модель | machine на синтетике, 34 | machine на несинтетике, 52 |
| --- | --- | --- |
| logistic | 23 | 4 |
| svm | 23 | 5 |
| lightgbm | 23 | 5 |

На `synthetic_tts` у всех трёх одинаково: machine 23, human 10. Playback все три назвали human. Совместные промахи, где все три сказали human при синтетической метке: 11 звонков. У 10 из них Spectra-0 как раз spoof.

| call_id | метка | вторичная | Spectra | margin | AASIST logit | сек | исход |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 4b69ac8d-7a85-46df-8266-617138b4044c | synthetic_tts | ai_secretary | spoof | −9.122 | −1.539 | 18.4 | answering_machine |
| 897929c0-b6e4-4a09-bf38-03ababeabb3e | synthetic_tts | anti_spam | spoof | −4.087 | −0.527 | 47.1 | b2x_callback |
| 8b34a750-b5df-4a6d-be75-7fbc207a9546 | synthetic_tts | — | bonafide | 1.464 | 1.625 | 335.1 | connection_issue |
| 8f662623-d653-4289-8318-b75737beece0 | synthetic_playback | — | spoof | 0.818 | 7.230 | 25.0 | пусто |
| 9b37b82b-a3a8-4238-b25f-66d136877ea8 | synthetic_tts | — | spoof | 0.265 | −1.683 | 232.6 | refused |
| a145ac8a-8d18-48b3-b2bc-20d3f795a73c | synthetic_tts | — | spoof | −3.980 | −1.507 | 96.1 | refused |
| a6102189-a3d8-41a4-86b5-9ff57d3c1080 | synthetic_tts | — | spoof | −3.929 | −1.975 | 99.6 | refused |
| acd5b754-215f-4acf-b5d8-b6b62b734dcc | synthetic_tts | — | spoof | −5.695 | −2.658 | 49.5 | b2x_callback |
| b0931366-6e3c-423b-9d5c-82f2d329c376 | synthetic_tts | — | spoof | −5.796 | −1.432 | 71.0 | connection_issue |
| baa46fb6-2bee-4563-8259-189269e43fed | synthetic_tts | — | spoof | −1.170 | −3.090 | 31.5 | пусто |
| f5559138-e5cd-4f8a-a9c5-e545b92f4639 | synthetic_tts | — | spoof | −2.986 | −1.960 | 135.8 | answering_machine |

Паттерн по всему срезу, не только по знаменателю: Spectra spoof и одновременно все три классические human — 24 звонка (synthetic_tts 9, human_live 8, unknown 6, synthetic_playback 1). Для живого диалога это ожидаемое расхождение задач. Для синтетики без сценария автоответчика классические модели молчат, Spectra-0 часто нет.

Обратное на несинтетике, где хотя бы одна классическая сказала machine:

| call_id | метка | кто сказал machine | Spectra | margin | исход |
| --- | --- | --- | --- | ---: | --- |
| 27ee4ba5-9d02-4514-9ee8-0fc225982616 | human_recorded | все три | bonafide | 4.957 | answering_machine |
| c28eca4d-e82f-4ede-8e57-3582360c7a26 | human_recorded | все три | spoof | 0.975 | answering_machine |
| 324000a7-d133-43c0-ba25-23b92f745550 | human_live | все три | bonafide | 7.853 | отказ_лпр |
| db7caa0c-be09-4934-8630-11424c02e57b | human_live | все три | bonafide | 9.535 | отказ_лпр |
| 70af9cfa-20b6-4dd9-89b0-879779c9365e | human_live | svm, lightgbm | bonafide | 1.353 | положил_трубку, needs_second_review |

## AASIST3

Порога нет. Средний bonafide logit:

| группа | n | mean | median | min | p25 | p75 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| human_live | 50 | 4.793 | 4.022 | −1.494 | 2.591 | 7.086 | 12.196 |
| human_recorded | 2 | 3.646 | 3.646 | −0.166 | — | — | 7.458 |
| synthetic_tts | 33 | −0.544 | −1.546 | −3.090 | −1.960 | 0.114 | 7.694 |
| synthetic_playback | 1 | 7.230 | — | — | — | — | — |
| mixed | 7 | 5.919 | 4.898 | 4.072 | 4.453 | 6.154 | 11.250 |
| unknown | 12 | 0.671 | 0.704 | −3.583 | −0.373 | 2.217 | 4.687 |

Группы live и tts по медиане разъехались (4.02 против −1.55), но хвосты пересекаются. Все четыре пропуска Spectra-0 имеют положительный AASIST logit (7.69, 4.18, 3.25, 1.63). Единственный playback имеет logit 7.23 при Spectra spoof. Любой порог, поставленный на этом срезе, будет подгонкой. Его нет в протоколе.

## unknown и mixed

`mixed`, 7 звонков, все Spectra bonafide, margin от 1.64 до 7.75, AASIST logit от 4.07 до 11.25. В бинарный счёт не входят, потому что в звонке заявлены обе семьи.

`unknown`, 12 звонков. Уверенность пустая, источник `unknown_source`, 11 из 12 в needs_second_review. Spectra: bonafide 4, spoof 8. Margin сжат около нуля: медиана −0.029, от −1.522 до 3.664. Исход silence у 8. Отдельно длинный `de084b60-4f32-4aa5-89dc-2a29faf338c9`: 441.5 с, исход b2x_lead, needs_second_review, Spectra spoof −1.403, все три классические human. Раньше в confirmed_001 этот звонок был отмечен как робот без подтипа; в дневной разметке слушатель поставил `unknown`, не подтип синтеза.

| call_id | review | Spectra | margin | logistic | svm | lightgbm | AASIST | сек | исход |
| --- | --- | --- | ---: | --- | --- | --- | ---: | ---: | --- |
| 14f06437-0709-4312-bbd3-cc773fe9d260 | needs_second_review | bonafide | 1.356 | machine | human | machine | 2.012 | 30.0 | silence |
| 19654c1a-c65f-4475-b777-f77bcf871a94 | needs_second_review | spoof | −0.019 | human | human | human | 0.739 | 38.9 | пусто |
| 21e4881b-b5cd-49a8-9787-43d1cd3894e3 | needs_second_review | spoof | −0.707 | human | human | human | −3.583 | 30.0 | silence |
| 57524297-59b4-42c4-a756-5f991dce072d | needs_second_review | spoof | −0.315 | human | human | human | −3.430 | 30.1 | silence |
| 88c51fb8-1005-4830-b05a-047959051bc5 | needs_second_review | bonafide | 1.268 | machine | machine | machine | 2.831 | 30.1 | silence |
| 9571e7b5-7d77-4f4c-95fc-c95a632f548f | needs_second_review | bonafide | 2.476 | machine | machine | machine | 3.581 | 30.1 | silence |
| 97a6020d-73aa-4df4-af6c-2fb8281ab0b0 | needs_second_review | spoof | −0.038 | human | human | human | −0.222 | 77.7 | silence |
| a62c4d21-203f-4734-9893-ce08262768f8 | needs_second_review | spoof | 0.153 | machine | machine | machine | 1.596 | 31.5 | silence |
| de084b60-4f32-4aa5-89dc-2a29faf338c9 | needs_second_review | spoof | −1.403 | human | human | human | −0.334 | 441.5 | b2x_lead |
| e314e1c8-3f26-4340-8e4d-000970bc131a | needs_second_review | spoof | −1.522 | human | human | human | −0.489 | 26.6 | пусто |
| e8dc2b09-26ff-40a1-aa6c-ea86ec11815e | needs_second_review | bonafide | 3.664 | human | machine | machine | 4.687 | 30.1 | silence |
| f1a0fd07-682e-46ce-938e-377fc5d3a536 | needs_second_review | spoof | −0.358 | machine | machine | machine | 0.668 | 22.8 | пусто |

Плюс один `human_live` на втором прослушивании, не unknown: `70af9cfa-20b6-4dd9-89b0-879779c9365e`, medium, Spectra bonafide 1.353. И `071cbec7` из таблицы FP.

## Что из этого уже следует

1. На прослушанной части Spectra-0 уже отделяет синтетику от несинтетики: recall 0.882, specificity 0.827 при замороженном 1.02. Масса margin разъехалась (медианы 3.12 и −5.70). Это срез 86 звонков с ясной стороной, не оценка всего дня.
2. Ошибки двух видов. Четыре синтетики Spectra принимает за bonafide с margin 1.46–4.20; три из них — длинные автоответчики, и старые модели говорят machine. Девять несинтетических она называет spoof; два живых отказа ЛПР с margin −2.79 и −2.04 выглядят как настоящие ложные spoof, не как дрожание порога.
3. Классические модели отвечают на другую задачу. На синтетике они говорят machine только в 23 из 34 и промахиваются втроём на одних и тех же 11 звонках, где Spectra чаще spoof. На живом диалоге они почти всегда human, что для старой задачи нормально и синтетику не проверяет.
4. Длительность и исход смешаны с меткой. Синтетика в срезе намного длиннее и чаще answering_machine. Пока разметка не дойдёт до короткой синтетики и длинных живых разговоров, долю «детектор синтеза» и долю «детектор длинного автоответчика» не разделить.
5. AASIST3 в среднем смотрит в ту же сторону, что Spectra, но без порога классом не является. Его logit положительный как раз на пропущенных Spectra синтетиках и на единственном playback.
6. `human_recorded` = 2, `synthetic_playback` = 1, `synthetic_voice_conversion` = 0. По этим подтипам ничего устойчивого нет.
7. 12 unknown и 7 mixed честно вынуты из знаменателя. Unknown в основном короткие silence около нулевого margin. Дожимать их в класс нельзя.

## Чего в анализе просить не стоит

- Новый порог Spectra-0 или AASIST3, подобранный по этим 105 меткам.
- Сводить classical machine к «синтетика найдена».
- Класть unknown и mixed в accuracy.
- Считать 0.88 recall оценкой продакшена: размечена примерно треть дня, выборка неслучайная, сценарий смещён.
- Обучать или дообучать что-либо на этом срезе. Старые тестовые звонки, на которых выбирался порог 1.02, сюда не подмешаны.
