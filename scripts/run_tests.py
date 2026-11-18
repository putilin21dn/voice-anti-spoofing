"""Тесты, которым не нужны база, аудио и веса."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def interpreter(component: str) -> Path:
    folder = "Scripts" if sys.platform == "win32" else "bin"
    name = "python.exe" if sys.platform == "win32" else "python"
    return ROOT / component / ".venv" / folder / name


def run(python: Path, args: list[str], cwd: Path) -> int:
    if not python.is_file():
        print(f"нет интерпретатора: {python}")
        return 1
    print(f"тест: {cwd.name} {' '.join(args)}")
    completed = subprocess.run([str(python), *args], cwd=cwd)
    return completed.returncode


def main() -> None:
    codes = []
    suites = (
        ("voice-dataset-collector", ["-m", "unittest", "discover", "-s", "tests"]),
        ("classical-anti-spoofing", ["-m", "unittest", "discover", "-s", "tests"]),
        ("synthetic-speech-dataset", ["-m", "unittest", "discover", "-s", "tests"]),
    )
    for name, args in suites:
        codes.append(run(interpreter(name), args, ROOT / name))
    codes.append(
        run(
            Path(sys.executable),
            ["-m", "unittest", "discover", "-s", "shadow_replay/tests", "-t", "."],
            ROOT,
        )
    )
    if any(codes):
        raise SystemExit(
            "Часть тестов не прошла. Без локального confirmed_001 один тест synthetic "
            "падает на отсутствующем манифесте. Остальные наборы должны быть зелёными."
        )
    print("тесты прошли")


if __name__ == "__main__":
    main()
