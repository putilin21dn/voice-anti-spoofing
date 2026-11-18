"""Печатает, чего не хватает для запуска. Значения секретов не показывает."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def interpreter(component: str) -> Path:
    folder = "Scripts" if sys.platform == "win32" else "bin"
    name = "python.exe" if sys.platform == "win32" else "python"
    return ROOT / component / ".venv" / folder / name


def line(ready: bool, text: str) -> None:
    print(("готово" if ready else "нет   ") + "  " + text)


def main() -> None:
    print(f"Python {sys.version.split()[0]}")
    line(sys.version_info >= (3, 11), "Python 3.11+")
    for name in (
        "voice-dataset-collector",
        "classical-anti-spoofing",
        "synthetic-speech-dataset",
    ):
        line(interpreter(name).is_file(), f"venv {name}")
    spectra_python = interpreter("voice-anti-spoofing-spectra0")
    line(spectra_python.is_file(), "venv voice-anti-spoofing-spectra0 (нужен только для inference)")
    env_file = ROOT / "voice-dataset-collector" / ".env"
    line(env_file.is_file(), "voice-dataset-collector/.env (для чтения базы; значения здесь не печатаются)")
    joblibs = list((ROOT / "classical-anti-spoofing" / "models").rglob("*.joblib"))
    line(len(joblibs) >= 18, f"классические bundle в git: {len(joblibs)}")
    spectra = ROOT / "voice-anti-spoofing-spectra0" / "models" / "spectra_0" / "model.safetensors"
    aasist = ROOT / "voice-anti-spoofing-spectra0" / "models" / "spectra_aasist3" / "model.safetensors"
    line(spectra.is_file(), "локальный вес Spectra-0")
    line(aasist.is_file(), "локальный вес Spectra-AASIST3")
    origin = ROOT / "synthetic-speech-dataset" / "reports" / "origin_retrain_20261113" / "models"
    line((origin / "svm.joblib").is_file() and (origin / "lightgbm.joblib").is_file(), "локальные svm.joblib и lightgbm.joblib ансамбля A")
    example = ROOT / "shadow_replay" / "examples" / "human_call.json"
    line(example.is_file(), "пример ленты shadow_replay")
    print("Без базы и весов запускаются тесты и python -m shadow_replay на примере.")


if __name__ == "__main__":
    main()
