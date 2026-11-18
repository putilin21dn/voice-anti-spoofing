"""Создаёт venv компонентов и ставит их requirements.

Spectra и CUDA в эту команду не входят: веса и torch ставятся отдельно,
когда нужен inference. Секреты не читаются и не печатаются.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMPONENTS = (
    "voice-dataset-collector",
    "classical-anti-spoofing",
    "synthetic-speech-dataset",
)


def interpreter(component: Path) -> Path:
    folder = "Scripts" if sys.platform == "win32" else "bin"
    name = "python.exe" if sys.platform == "win32" else "python"
    return component / ".venv" / folder / name


def create(component: Path) -> None:
    target = component / ".venv"
    if interpreter(component).is_file():
        print(f"venv уже есть: {component.name}")
        return
    print(f"venv: {component.name}")
    venv.create(target, with_pip=True)


def install(component: Path) -> None:
    requirements = component / "requirements.txt"
    python = interpreter(component)
    if not python.is_file():
        raise SystemExit(f"Нет интерпретатора: {python}")
    print(f"зависимости: {component.name}")
    subprocess.run(
        [str(python), "-m", "pip", "install", "--upgrade", "pip"],
        check=True,
    )
    subprocess.run(
        [str(python), "-m", "pip", "install", "-r", str(requirements)],
        check=True,
    )


def main() -> None:
    if sys.version_info < (3, 11):
        raise SystemExit("Нужен Python 3.11 или новее. Для classical лучше 3.12.")
    parser = argparse.ArgumentParser(description="Окружения компонентов без скачивания данных и весов")
    parser.add_argument("--no-test", action="store_true", help="Только venv и pip")
    args = parser.parse_args()
    for name in COMPONENTS:
        component = ROOT / name
        if not (component / "requirements.txt").is_file():
            raise SystemExit(f"Нет requirements.txt: {name}")
        create(component)
        install(component)
    env_example = ROOT / "voice-dataset-collector" / ".env.example"
    env_file = ROOT / "voice-dataset-collector" / ".env"
    if env_example.is_file() and not env_file.is_file():
        env_file.write_text(env_example.read_text(encoding="utf-8"), encoding="utf-8")
        print("создан voice-dataset-collector/.env из примера. Значения пустые, заполните их локально.")
    if args.no_test:
        return
    subprocess.run([sys.executable, str(ROOT / "scripts" / "run_tests.py")], check=True)


if __name__ == "__main__":
    main()
