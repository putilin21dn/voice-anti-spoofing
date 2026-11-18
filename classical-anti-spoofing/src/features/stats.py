from __future__ import annotations

import numpy as np


STAT_NAMES = ("mean", "std", "min", "max")


def coefficient_names(prefix: str, n_coeffs: int, kinds: tuple[str, ...]) -> list[str]:
    names: list[str] = []
    for kind in kinds:
        for index in range(n_coeffs):
            for stat in STAT_NAMES:
                names.append(f"{prefix}_{kind}_{index:02d}_{stat}")
    return names


def scalar_names(prefix: str, stats: tuple[str, ...] = STAT_NAMES) -> list[str]:
    return [f"{prefix}_{stat}" for stat in stats]


def _finite(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def summarize_vector(prefix: str, values: np.ndarray) -> dict[str, float]:
    finite = _finite(values)
    if finite.size == 0:
        return {name: float("nan") for name in scalar_names(prefix)}
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_std": float(np.std(finite)),
        f"{prefix}_min": float(np.min(finite)),
        f"{prefix}_max": float(np.max(finite)),
    }


def summarize_matrix(prefix: str, kind: str, matrix: np.ndarray | None, n_coeffs: int) -> dict[str, float]:
    names = coefficient_names(prefix, n_coeffs, (kind,))
    if matrix is None or matrix.size == 0 or matrix.shape[0] == 0:
        return {name: float("nan") for name in names}
    out: dict[str, float] = {}
    for index in range(n_coeffs):
        finite = _finite(matrix[:, index])
        stem = f"{prefix}_{kind}_{index:02d}"
        if finite.size == 0:
            for stat in STAT_NAMES:
                out[f"{stem}_{stat}"] = float("nan")
            continue
        out[f"{stem}_mean"] = float(np.mean(finite))
        out[f"{stem}_std"] = float(np.std(finite))
        out[f"{stem}_min"] = float(np.min(finite))
        out[f"{stem}_max"] = float(np.max(finite))
    return out


def delta_matrix(features: np.ndarray, width: int) -> np.ndarray | None:
    """Регрессионная дельта. На короткой траектории возвращает None, а не дополненный ноль."""

    if features.ndim != 2 or features.shape[0] < (2 * width + 1) or width < 1:
        return None
    denominator = 2.0 * sum(n * n for n in range(1, width + 1))
    padded = np.pad(features, ((width, width), (0, 0)), mode="edge")
    total = np.zeros_like(features, dtype=np.float64)
    length = features.shape[0]
    for step in range(1, width + 1):
        total += step * (padded[width + step : width + step + length] - padded[width - step : width - step + length])
    return total / denominator
