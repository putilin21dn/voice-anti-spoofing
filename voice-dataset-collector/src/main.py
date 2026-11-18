from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import Settings, load_settings
from src.collector.audio_downloader import download_audio
from src.collector.extract_client import copy_subset, extract_client_samples
from src.collector.call_selector import export_calls, parse_day, split_csv_arg
from src.collector.samples import export_samples, sample_paths
from src.db.connection import DatabaseError, check_connection
from src.storage.s3_client import ReadOnlyS3, S3StorageError
from src.utils.logging_config import setup_logging


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Выгрузка call_id из PostgreSQL и оригинальных WAV из S3."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check-db", help="Проверить подключение к PostgreSQL и наличие таблицы calls")
    sub.add_parser("check-s3", help="Проверить чтение bucket через HeadBucket")

    export = sub.add_parser("export", help="Записать партию звонков в data/manifests/calls_manifest.csv")
    download = sub.add_parser("download", help="Скачать WAV по строкам manifest")
    for target in (export, download):
        target.add_argument("--limit", type=int, default=200)
        target.add_argument("--offset", type=int, default=0)
        target.add_argument("--from-date", default="")
        target.add_argument("--to-date", default="")
        target.add_argument(
            "--status",
            action="append",
            default=[],
            help="Фильтр calls.goal_status. Колонки status в таблице нет.",
        )
        target.add_argument("--dry-run", action="store_true")

    export.add_argument("--outcome", action="append", default=[])
    export.add_argument("--end-reason", action="append", default=[])
    export.add_argument("--min-duration-s", type=int, default=None)
    export.add_argument("--max-duration-s", type=int, default=None)
    export.add_argument("--include-without-audio", action="store_true")
    export.add_argument("--include-unfinalized", action="store_true")
    export.add_argument(
        "--no-skip-exported",
        action="store_true",
        help="Не исключать call_id, которые уже есть в manifest. Вместе с --offset листает одну и ту же выборку.",
    )
    download.add_argument(
        "--retry-missing",
        action="store_true",
        help="Повторно проверить объекты, ранее отмеченные как missing.",
    )
    samples = sub.add_parser(
        "samples",
        help="Две выборки кампании: human и machine. По умолчанию B2X за ноябрь 2026.",
    )
    samples.add_argument("--analysis-type", default="b2x")
    samples.add_argument("--month", default="2026-11", help="Календарный месяц Europe/Moscow, ГГГГ-ММ")
    samples.add_argument("--dry-run", action="store_true")
    samples.add_argument("--no-download", action="store_true")
    extract = sub.add_parser(
        "extract-samples",
        help="Собрать моно-семпл абонента из стереозаписей human и machine.",
    )
    extract.add_argument("--analysis-type", default="b2x")
    extract.add_argument("--month", default="2026-11")
    extract.add_argument("--subset", type=int, default=0, help="Сколько семплов взять из каждой выборки")
    return parser


def _print_report(title: str, rows: list[tuple[str, int | str]]) -> None:
    print(title)
    for label, value in rows:
        print(f"  {label}: {value}")


def _run_export(settings: Settings, args: argparse.Namespace) -> int:
    logger = setup_logging(settings.log_dir)
    from_date = parse_day(args.from_date, end=False) if args.from_date else None
    to_date = parse_day(args.to_date, end=True) if args.to_date else None
    try:
        stats = export_calls(
            settings,
            from_date=from_date,
            to_date=to_date,
            goal_statuses=split_csv_arg(args.status),
            outcomes=split_csv_arg(args.outcome),
            end_reasons=split_csv_arg(args.end_reason),
            min_duration_s=args.min_duration_s,
            max_duration_s=args.max_duration_s,
            require_audio=not args.include_without_audio,
            require_finalized=not args.include_unfinalized,
            skip_exported=not args.no_skip_exported,
            limit=args.limit,
            offset=args.offset,
            dry_run=args.dry_run,
        )
    except DatabaseError as exc:
        logger.error("Подключение к PostgreSQL: %s", exc)
        _print_report(
            "Партия не выполнена",
            [
                ("Найдено звонков", 0),
                ("Отобрано записей", 0),
                ("Ошибки подключения", 1),
            ],
        )
        return 1

    logger.info(
        "export found=%s added=%s dry_run=%s %s",
        stats["found"],
        stats["added"],
        args.dry_run,
        settings.public_summary(),
    )
    _print_report(
        "Партия export завершена",
        [
            ("Найдено звонков", stats["found"]),
            ("Отобрано записей", stats["selected"]),
            ("Новых строк в manifest", stats["added"]),
            ("Уже были в manifest", stats["already_in_manifest"]),
            ("Строк в manifest", stats["manifest_rows"]),
            ("Объекты S3", "не проверялись"),
            ("Скачано файлов", 0),
            ("Уже были скачаны", 0),
            ("Объекты отсутствуют", 0),
            ("Ошибки подключения", 0),
            ("Не удалось скачать", 0),
            ("WAV прошли проверку", 0),
        ],
    )
    return 0


def _run_samples(settings: Settings, args: argparse.Namespace) -> int:
    logger = setup_logging(settings.log_dir)
    try:
        summary = export_samples(
            settings,
            analysis_type=args.analysis_type,
            month=args.month,
            dry_run=args.dry_run,
        )
    except DatabaseError as exc:
        logger.error("Выборки: %s", exc)
        print(f"PostgreSQL: {exc}")
        return 1
    seen = summary.pop("seen")
    print(f"Кампания {args.analysis_type}, месяц {args.month}")
    print(f"  Звонков с аудио: {seen['selected']}")
    exit_code = 0
    for sample in ("human", "machine"):
        item = summary[sample]
        manifest, audio = sample_paths(settings, args.analysis_type, args.month, sample)
        print(f"  {sample}: {item['selected']} звонков, файл {manifest.name}")
        logger.info("%s selected=%s manifest=%s", sample, item["selected"], manifest.name)
        if args.dry_run or args.no_download or not item["selected"]:
            continue
        stats = download_audio(
            settings,
            limit=int(item["selected"]),
            offset=0,
            goal_statuses=[],
            from_date="",
            to_date="",
            dry_run=False,
            retry_missing=False,
            manifest_path=manifest,
            audio_dir=audio,
        )
        print(
            f"    скачано {stats['downloaded']}, уже были {stats['already_present']}, "
            f"нет в S3 {stats['missing']}, WAV ок {stats['wav_ok']}, "
            f"ошибки доступа {stats['auth_errors']}, ошибки сети {stats['connection_errors']}"
        )
        if stats["auth_errors"] or stats["connection_errors"]:
            exit_code = 1
    return exit_code


def _run_download(settings: Settings, args: argparse.Namespace) -> int:
    logger = setup_logging(settings.log_dir)
    from_date = args.from_date
    to_date = parse_day(args.to_date, end=True).isoformat() if args.to_date else ""
    if args.from_date and len(args.from_date) == 10:
        from_date = parse_day(args.from_date, end=False).isoformat()
    try:
        stats = download_audio(
            settings,
            limit=args.limit,
            offset=args.offset,
            goal_statuses=split_csv_arg(args.status),
            from_date=from_date,
            to_date=to_date,
            dry_run=args.dry_run,
            retry_missing=args.retry_missing,
        )
    except (DatabaseError, S3StorageError) as exc:
        logger.error("Скачивание остановлено: %s", exc)
        return 1

    logger.info(
        "download selected=%s downloaded=%s already=%s missing=%s wav_ok=%s errors=%s",
        stats["selected"],
        stats["downloaded"],
        stats["already_present"],
        stats["missing"],
        stats["wav_ok"],
        stats["download_errors"],
    )
    _print_report(
        "Партия download завершена",
        [
            ("Строк в manifest", stats["manifest_rows"]),
            ("Отобрано записей", stats["selected"]),
            ("Объекты существуют в S3", stats["s3_exists"]),
            ("Скачано файлов", stats["downloaded"]),
            ("Уже были скачаны", stats["already_present"]),
            ("Объекты отсутствуют", stats["missing"]),
            ("Ошибки подключения", stats["connection_errors"]),
            ("Ошибки доступа", stats["auth_errors"]),
            ("Не удалось скачать", stats["download_errors"]),
            ("WAV прошли проверку", stats["wav_ok"]),
        ],
    )
    return 1 if stats["auth_errors"] or stats["connection_errors"] else 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        settings = load_settings()
    except ValueError as exc:
        print(f"Некорректная конфигурация: {exc}")
        return 2

    if args.command == "check-db":
        logger = setup_logging(settings.log_dir)
        try:
            info = check_connection(settings)
        except DatabaseError as exc:
            logger.error("Проверка PostgreSQL: %s", exc)
            print(f"PostgreSQL: {exc}")
            return 1
        logger.info("PostgreSQL доступен, schema=%s calls=%s", info["schema_name"], info["calls_table"])
        print(
            f"PostgreSQL: database={info['database_name']} user={info['user_name']} "
            f"schema={info['schema_name']} calls={info['calls_table']}"
        )
        return 0 if info["calls_table"] == "yes" else 1

    if args.command == "check-s3":
        logger = setup_logging(settings.log_dir)
        try:
            ReadOnlyS3(settings).check_bucket()
        except S3StorageError as exc:
            logger.error("Проверка S3: %s", exc)
            print(f"S3: {exc}")
            return 1
        logger.info("S3 bucket доступен для чтения: %s", settings.s3_bucket)
        print(f"S3: bucket {settings.s3_bucket} доступен для чтения")
        return 0

    if args.command == "export":
        if args.limit < 1:
            print("--limit должен быть больше 0")
            return 2
        return _run_export(settings, args)
    if args.command == "download":
        if args.limit < 1:
            print("--limit должен быть больше 0")
            return 2
        return _run_download(settings, args)
    if args.command == "samples":
        return _run_samples(settings, args)
    if args.command == "extract-samples" and args.subset:
        stats = copy_subset(
            settings,
            analysis_type=args.analysis_type,
            month=args.month,
            count=args.subset,
        )
        _print_report(
            f"Подвыборка по {args.subset}",
            [("human", stats["human"]), ("machine", stats["machine"])],
        )
        return 0
    if args.command == "extract-samples":
        logger = setup_logging(settings.log_dir)
        stats = extract_client_samples(settings, analysis_type=args.analysis_type, month=args.month)
        logger.info(
            "extract-samples written=%s already=%s failed=%s human=%s machine=%s",
            stats["written"],
            stats["already_present"],
            stats["failed"],
            stats["human"],
            stats["machine"],
        )
        _print_report(
            "Семплы абонента готовы",
            [
                ("Записано новых", stats["written"]),
                ("Уже были", stats["already_present"]),
                ("Не удалось прочитать", stats["failed"]),
                ("human", stats["human"]),
                ("machine", stats["machine"]),
            ],
        )
        return 1 if stats["failed"] else 0
    print(f"Неизвестная команда: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
