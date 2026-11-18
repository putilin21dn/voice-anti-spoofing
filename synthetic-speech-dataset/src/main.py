from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.annotation.review_app import make_server
from src.annotation.share import export_share, import_share
from src.data.manifest import init_batch
from src.evaluation.pipeline import inspect_dataset, run_day_scores, run_evaluation, validate_readiness
from src.reports.quality import write_quality_report
from src.utils.config import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Локальная выборка и разметка Synthetic Speech Detection")
    parser.add_argument("--config", default="config/config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init-batch", help="Собрать первую партию из уже скачанных WAV")
    init.add_argument("--rebuild", action="store_true", help="Пересобрать партию, если разметка ещё не начата")
    sub.add_parser("validate", help="Проверить manifest, WAV и пустые метки")
    sub.add_parser("serve", help="Открыть локальный инструмент разметки")
    export = sub.add_parser("export-share", help="Собрать zip с половиной звонков для второго разметчика")
    export.add_argument("--share", default="guest", choices=["local", "guest"])
    imported = sub.add_parser("import-share", help="Влить annotations.csv второго разметчика, не затирая свои метки")
    imported.add_argument("--annotations", required=True)
    imported.add_argument("--segments", default="")
    imported.add_argument("--share", default="guest", choices=["local", "guest"])
    sub.add_parser("inspect-dataset", help="Показать состав закрываемой выборки, без моделей")
    sub.add_parser("validate-readiness", help="Проверить пачку, WAV, канал и артефакты моделей")
    evaluation = sub.add_parser("run-evaluation", help="Сравнить пять готовых моделей")
    evaluation.add_argument("--batch", default="confirmed_001")
    evaluation.add_argument("--dry-run", action="store_true")
    evaluation.add_argument("--call-ids", default="")
    evaluation.add_argument("--run-name", default="")
    evaluation.add_argument("--reuse-run", default="", help="Каталог прошлого прогона, чьи уже посчитанные звонки не считаются заново")
    day_scores = sub.add_parser("run-day-scores", help="Посчитать пять моделей по партии дня без меток")
    day_scores.add_argument("--day-config", default="config/day_20261113.yaml")
    day_scores.add_argument("--reuse-run", default="confirmed_001/20261113T155919")
    report = sub.add_parser("generate-report", help="Показать уже собранный отчёт прогона")
    report.add_argument("--batch", default="confirmed_001")
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.command == "init-batch":
        result = init_batch(cfg, rebuild=args.rebuild)
        check, report = write_quality_report(cfg)
        print(f"звонков: {check['rows']}")
        print(f"отбор: {check['reasons']}")
        print(f"отчёт: {report}")
        if not result.get("created"):
            print(result["reason"])
        if check["errors"]:
            print(f"ошибок: {len(check['errors'])}")
            raise SystemExit(1)
        return

    if args.command == "validate":
        check, report = write_quality_report(cfg)
        print(f"звонков: {check['rows']}")
        print(f"меток шаблона: {check['labels']}")
        print(f"отчёт: {report}")
        if check["errors"]:
            for item in check["errors"]:
                print(item)
            raise SystemExit(1)
        return

    if args.command == "export-share":
        export_share(cfg, args.share)
        return

    if args.command == "import-share":
        segments = Path(args.segments) if args.segments else None
        result = import_share(cfg, Path(args.annotations), segments, args.share)
        print(f"принято: {result['accepted']}")
        print(f"заменено своих же строк этого разметчика: {result['replaced']}")
        print(f"отклонено: {result['rejected']}")
        return

    if args.command == "serve":
        log_dir = resolve_path(cfg, cfg["paths"]["logs_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(message)s",
            handlers=[logging.FileHandler(log_dir / "review.log", encoding="utf-8")],
        )
        host = str(cfg.get("host") or "127.0.0.1")
        port = int(cfg.get("port") or 8765)
        server = make_server(cfg, host, port)
        print(f"Разметка: http://{host}:{port}")
        print("Слушает только этот компьютер. Остановка: Ctrl+C")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Остановлено")
        finally:
            server.server_close()
        return

    if args.command == "inspect-dataset":
        summary = inspect_dataset()
        print(f"пачка: {summary['batch_id']} статус: {summary['status']}")
        print(f"звонков: {summary['calls']}")
        print(f"в категориальных таблицах: {summary['in_category_metrics']}")
        print(f"роботы без подтипа: {summary['unclassified_robots']}")
        print(f"категории: {summary['categories']}")
        return

    if args.command == "validate-readiness":
        _readiness_command()
        return

    if args.command == "run-evaluation":
        if args.batch != "confirmed_001":
            raise SystemExit("Сейчас подготовлена только закрытая пачка confirmed_001")
        call_ids = [item for item in args.call_ids.split(",") if item]
        run_dir = run_evaluation(
            dry_run=args.dry_run,
            call_ids=call_ids,
            run_name=args.run_name,
            reuse_from=args.reuse_run,
        )
        print(f"прогон: {run_dir}")
        return

    if args.command == "run-day-scores":
        run_dir = run_day_scores(reuse_from=args.reuse_run, config_path=args.day_config)
        print(f"прогон: {run_dir}")
        return

    if args.command == "generate-report":
        report_path = _latest_summary(args.batch)
        print(report_path.read_text(encoding="utf-8"))
        print(f"файл: {report_path}")


def _readiness_command() -> None:
    payload, errors = validate_readiness()
    print(f"статус: {payload['status'].get('status')}")
    print(f"звонков: {len(payload['calls'])}")
    models = (payload["check"].get("models") or {})
    for name, info in models.items():
        print(f"{name}: признаков {info['n_features']}, порог {info['threshold']}")
    if errors:
        for item in errors:
            print(item)
        raise SystemExit(1)
    print("готовность: можно считать")


def _latest_summary(batch: str) -> Path:
    root = Path(__file__).resolve().parents[1] / "evaluation" / batch
    found = [path / "summary" / "summary.md" for path in root.iterdir() if (path / "summary" / "summary.md").is_file()] if root.exists() else []
    if not found:
        raise SystemExit(f"Отчёт для {batch} ещё не собран")
    return max(found, key=lambda path: path.stat().st_mtime)


if __name__ == "__main__":
    main()
