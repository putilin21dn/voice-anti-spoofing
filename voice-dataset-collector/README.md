# Voice Dataset Collector

Независимая утилита для набора датасета человеческой и синтетической речи. Она читает звонки из PostgreSQL и скачивает исходные WAV из S3. Код `genai-agents-worker`, `genai-agents-backend` и `voice-anti-spoofing-spectra0` она не импортирует и не изменяет.

Записи могут содержать персональные данные. Файлы остаются на этой машине: утилита не отправляет аудио во внешние API и не распознаёт речь.

## Откуда берутся данные

Таблица `calls` (схема `public`, если не задан другой `DB_SCHEMA`):

| Поле manifest | Источник |
| --- | --- |
| `call_id` | `calls.id` |
| `created_at` | `calls.created_at` |
| `timestamp` | `calls.timestamp`, время начала звонка |
| `duration_s` | `calls.duration`, целые секунды |
| `duration_ms` | `duration * 1000`. Это не длительность файла, а секунды из БД, умноженные на 1000 |
| `external_id` | `calls.external_id`. Для звонков worker это имя комнаты |
| `audio_available` | `calls.with_audio`. Worker ставит `true` после сохранения WAV |
| `is_finalized` | `calls.is_finalized` |
| `goal_status` | `calls.goal_status` |
| `outcome` | `calls.outcome` |
| `end_reason` | `calls.end_reason` |
| `disconnected_by` | `calls.disconnected_by` |
| `direction` | `calls.direction` |
| `s3_bucket` | не колонка БД. Берётся из `S3_BUCKET` |
| `s3_key` | не колонка БД. Вычисляется как `calls/{external_id}/{external_id}.wav` |

Колонки `status` в `calls` нет. Параметр `--status` фильтрует `goal_status`: `goal_achieved`, `goal_not_achieved`, `goal_callback`, `undetermined`.

Ключ проверен по рабочему коду, а не подобран по имени:

- worker сохраняет финал в `calls/{room_name}/{room_name}.wav`, а `room_name` уходит в `external_id`;
- backend читает тот же ключ при отдаче записи, транскрибации и проверке качества аудио.

Промежуточные объекты `calls/{room}/{room}_raw_1.mp4` и `_raw_2.mp4` worker удаляет после сборки WAV. Collector их не скачивает.

По умолчанию в выборку попадают звонки с `with_audio = true` и `is_finalized = true`. Отсутствие строк в `call_metrics` или `call_audio_metrics` звонок не исключает: backend не требует эти метрики, чтобы запись существовала.

Телефон в manifest и в обычный лог не попадает.

## Установка

Команды для PowerShell из каталога `voice-dataset-collector`:

```powershell
cd voice-dataset-collector
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Дальше заполните `.env` сами. Файл не коммитится.

Соответствие имён backend и collector:

| Backend / worker | Collector |
| --- | --- |
| `POSTGRES_HOST` | `DB_HOST` |
| `POSTGRES_PORT` | `DB_PORT` |
| `POSTGRES_DB` | `DB_NAME` |
| `POSTGRES_USER` | `DB_USER` |
| `POSTGRES_PASSWORD` | `DB_PASSWORD` |
| `S3_HOST`, `S3_PORT`, `S3_SECURE` | те же имена |
| `S3_BUCKET` | `S3_BUCKET` (в рабочих проектах по умолчанию `genai`) |
| `S3_REGION` | `S3_REGION` (по умолчанию `eu-central-1`) |
| `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY` | те же имена |

`S3_ENDPOINT_URL` необязателен. Если он пуст, адрес собирается так же, как в клиентах worker и backend: для MinIO это `http://host:port` или `https://host:port`, для host с `amazonaws.com` свой endpoint не задаётся. Пустые ключи не отправляются в boto3, поэтому остаётся стандартная цепочка, включая IAM-роль. `S3_SESSION_TOKEN` нужен только для временных ключей; рабочие проекты его не используют.

Для MinIO клиент включает path-style addressing. Так же worker включает `force_path_style` при выгрузке записи.

## Проверка PostgreSQL

```powershell
python src/main.py check-db
```

Ожидаемый вывод содержит имя базы, пользователя, схему и `calls=yes`. Пароль не печатается. Сессия включается с `default_transaction_read_only`. Клиент отключает GSSAPI (`gssencmode=disable`): иначе сервер закрывает соединение до ответа.

Если схема отличается от моделей, выполните вручную `sql/inspect_schema.sql`. Основной запрос с комментариями лежит в `sql/select_calls.sql`.

## Проверка S3

```powershell
python src/main.py check-s3
```

Команда вызывает `HeadBucket`. Это чтение. Bucket не создаётся, объекты не изменяются. Если политика разрешает `GetObject`, но запрещает `HeadBucket`, проверка завершится ошибкой доступа, а скачивание конкретного ключа всё ещё может работать.

## Первые 200 звонков

```powershell
python src/main.py export --limit 200
```

По умолчанию:

- только `with_audio = true` и `is_finalized = true`;
- сортировка по `created_at`, затем по `id`;
- `call_id`, которые уже есть в manifest, не добавляются повторно.

Результат: `data\manifests\calls_manifest.csv`.

Полезные фильтры:

```powershell
python src/main.py export --limit 200 --from-date 2026-01-01 --to-date 2026-01-31
python src/main.py export --limit 200 --status goal_achieved
python src/main.py export --limit 200 --min-duration-s 5 --max-duration-s 600
python src/main.py export --limit 200 --end-reason user_hangup
python src/main.py export --limit 200 --dry-run
```

`--from-date` включает указанный день, `--to-date` тоже включает день целиком. Граница сравнивается с `created_at`. `--min-duration-s` и `--max-duration-s` сравниваются с `calls.duration` в секундах.

Чтобы увидеть и незавершённые звонки или звонки без флага записи:

```powershell
python src/main.py export --limit 200 --include-unfinalized --include-without-audio
```

Повтор той же команды берёт следующие ещё не выгруженные звонки. Чтобы заново пролистать одну и ту же выборку, отключите пропуск и задайте сдвиг:

```powershell
python src/main.py export --limit 200 --offset 0 --no-skip-exported
python src/main.py export --limit 200 --offset 200 --no-skip-exported
```

`--dry-run` выполняет запрос и печатает отчёт, CSV не изменяет.

## Скачивание аудио

Сначала проверьте одну-три записи:

```powershell
python src/main.py download --limit 3
```

После проверки скачивайте партию:

```powershell
python src/main.py download --limit 200
```

Команда идёт по manifest сверху вниз, скачивает ещё не полученные файлы и останавливается на `--limit`. Уже проверенный файл не перезаписывается: он должен лежать по `local_path`, его размер должен совпадать с `file_size_bytes`, а `wav_ok` не должен быть `false`. Перед новым скачиванием выполняется `HeadObject`. Байты пишутся во временный файл и заменяют назначение только после успешной загрузки.

Локальное имя — `{call_id}.wav` в `data\audio\`. Каналы, частота и формат не меняются. В manifest дописываются размер, число каналов, sample rate, длительность по заголовку WAV и `download_status`.

Следующая партия — снова `export`, затем `download`. Уже скачанные строки пропускаются:

```powershell
python src/main.py export --limit 200
python src/main.py download --limit 200
```

Дополнительные флаги download:

```powershell
python src/main.py download --limit 200 --offset 200
python src/main.py download --limit 200 --from-date 2026-01-01 --to-date 2026-01-31 --status undetermined
python src/main.py download --limit 200 --dry-run
python src/main.py download --limit 200 --retry-missing
```

`--offset` пропускает N строк, которые ещё нужно скачивать. `--dry-run` делает `HeadObject`, но не пишет файлы и не обновляет manifest. `--retry-missing` заново проверяет строки со статусами `missing` и `invalid_wav`.

Ошибка доступа или сети останавливает партию, чтобы не повторять заведомо сломанное подключение на каждой строке.

## Выборки B2X

`samples` берёт только кампании `call_campaigns.analysis_type = b2x` и звонки с `calls.timestamp` внутри календарного месяца по Москве.

Человеческая выборка — `classification_result` из живого диалога: `лид`, `интерес_без_данных`, `отказ_лпр`, `не_лпр`, `не_та_организация`, `перезвон`, `асинхронная_связь`.

Машинная выборка — `автоответчик`, `answering_machine`, `anti_spam_bot` или текст автоответчика в `collected_data`. Если у звонка есть обе группы меток, он попадает только в машинную. Молчание, недозвон и сброс на приветствии не входят ни туда, ни туда.

```powershell
python src/main.py samples --analysis-type b2x --month 2026-11
```

Manifest: `data\manifests\b2x_2026-11_human.csv` и `data\manifests\b2x_2026-11_machine.csv`. WAV: `data\audio\b2x\2026-11\human` и `data\audio\b2x\2026-11\machine`.

Из этих стереозаписей можно собрать семпл только абонента. Канал 0 — абонент, канал 1 — робот. У `external_id`, который начинается с `excel-`, каналы наоборот, как в backend. Частота и глубина отсчётов не меняются.

```powershell
python src/main.py extract-samples --analysis-type b2x --month 2026-11
```

Результат: `data\samples\b2x\2026-11\human`, `data\samples\b2x\2026-11\machine` и `samples.csv`. В этом manifest нет `external_id`.

Короткая подвыборка по 100 звонков из каждой папки, равномерно по времени:

```powershell
python src/main.py extract-samples --subset 100 --analysis-type b2x --month 2026-11
```

Она лежит в `data\samples\b2x\2026-11\n100`.

## Где лежат результаты

| Что | Путь |
| --- | --- |
| Manifest | `data\manifests\calls_manifest.csv` |
| WAV | `data\audio\{call_id}.wav` |
| Лог | `data\logs\collector.log` |
| Запросы | `sql\inspect_schema.sql`, `sql\select_calls.sql` |

`data\audio\`, `data\logs\`, CSV manifest и `.env` указаны в `.gitignore`.

## Статусы и ошибки

| `download_status` | Значение |
| --- | --- |
| `pending` | строка выгружена из БД, файл ещё не скачивался |
| `downloaded` | объект скачан, WAV открылся |
| `already_present` | локальный файл уже был целым, повторно не скачивался |
| `missing` | `HeadObject` или скачивание вернуло отсутствие ключа |
| `auth_error` | доступ запрещён, ключ недействителен или токен истёк |
| `network_error` | endpoint недоступен или оборвалось соединение |
| `download_error` | другая ошибка S3 или записи на диск |
| `invalid_wav` | байты сохранены без изменений, но модуль `wave` не открыл файл |
| `skipped_no_key` | пустой `external_id`, ключ построить нельзя |

`s3_object_status`: пусто до проверки, затем `exists` или `missing`.

`audio_available` повторяет флаг `with_audio` из БД. После успешного скачивания в этой строке стоит `true`. Это не заменяет проверку S3: фактическое наличие объекта смотрите в `s3_object_status`.

`wav_duration_ms` измерен по заголовку скачанного файла. `duration_ms` по-прежнему равен секундам из `calls.duration`, умноженным на 1000.

В конце партии печатается отчёт: сколько звонков найдено, сколько строк отобрано, сколько объектов есть в S3, сколько файлов скачано, сколько уже было на диске, сколько ключей отсутствует, сколько ошибок подключения и доступа, сколько WAV открылось. На шаге `export` объекты S3 не проверяются, и отчёт прямо об этом говорит.

Живое скачивание даже 1–3 звонков из этого окружения не запускалось: в новом проекте нет заполненного `.env`.
