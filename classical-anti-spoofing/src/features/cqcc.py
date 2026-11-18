from __future__ import annotations

import numpy as np

from src.audio.preprocessing import resolve_fmax
from src.features.lfcc import _pack
from src.features.stats import coefficient_names


def cqcc_available() -> tuple[bool, str]:
    try:
        from spafe.features.cqcc import cqcc  # noqa: F401
        from spafe.utils.preprocessing import SlidingWindow  # noqa: F401
    except Exception as exc:
        return False, f"{exc.__class__.__name__}: {exc}"
    return True, "spafe.features.cqcc"


def _kinds(include_deltas: bool) -> tuple[str, ...]:
    if include_deltas:
        return ("static", "delta", "delta2")
    return ("static",)


def cqcc_names(cfg: dict) -> list[str]:
    group = cfg["features"]["cqcc"]
    return coefficient_names("cqcc", int(group["num_ceps"]), _kinds(bool(group["include_deltas"])))


def compute_cqcc(audio: np.ndarray, sample_rate: int, cfg: dict) -> dict[str, float]:
    """CQCC через spafe. Аудио не ресемплируется вверх.

    scipy.signal.resample внутри spafe применяется к логарифму CQT, а не к волне.
    high_freq ограничивается Найквистом: spafe сам отклоняет частоту выше fs/2.
    """

    from spafe.features.cqcc import cqcc
    from spafe.utils.preprocessing import SlidingWindow

    group = cfg["features"]["cqcc"]
    names = cqcc_names(cfg)
    empty = {name: float("nan") for name in names}
    win = float(group["win_len_s"])
    if audio.size < int(sample_rate * win):
        return empty
    fmax = resolve_fmax(sample_rate, float(group["fmax"]))
    window = SlidingWindow(win, float(group["win_hop_s"]), "hamming")
    cepstra = cqcc(
        np.ascontiguousarray(audio, dtype=np.float64),
        fs=int(sample_rate),
        num_ceps=int(group["num_ceps"]),
        pre_emph=float(group["preemphasis"]) != 0,
        pre_emph_coeff=float(group["preemphasis"]),
        window=window,
        nfft=int(group["nfft"]),
        low_freq=float(group["fmin"]),
        high_freq=fmax,
        number_of_octaves=int(group["number_of_octaves"]),
        number_of_bins_per_octave=int(group["bins_per_octave"]),
        resampling_ratio=float(group["resampling_ratio"]),
        normalize=None,
    )
    matrix = np.asarray(cepstra, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < int(group["num_ceps"]):
        return empty
    return _pack(
        "cqcc",
        matrix[:, : int(group["num_ceps"])],
        names,
        bool(group["include_deltas"]),
        int(group["delta_width"]),
    )
