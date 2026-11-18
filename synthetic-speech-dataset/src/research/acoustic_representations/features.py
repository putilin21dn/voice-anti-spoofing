"""Log-mel, log-STFT, LFCC and two CQCC variants on one 8 kHz segment.

The segment passed to the primary extractors is already pre-emphasized once.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct

from src.research.acoustic_representations.config import (
    CQCC_BINS,
    CQCC_BINS_PER_OCTAVE,
    CQCC_CEPS,
    CQCC_FMIN_HZ,
    CQCC_HOP,
    CQCC_OCTAVES,
    CQCC_PAD_MODE,
    CQCC_RES_TYPE,
    CQCC_SCALE,
    CQCC_SPARSITY,
    CQCC_TUNING,
    CQCC_WINDOW,
    F0_FMAX_HZ,
    F0_FMIN_HZ,
    F0_FRAME_LENGTH,
    F0_HOP,
    FMAX_HZ,
    FMIN_HZ,
    FLOOR_POWER,
    HOP_LENGTH,
    LFCC_CEPS,
    LFCC_FILTERS,
    LOG_FLOOR,
    MEL_HTK,
    MEL_NORM,
    N_FFT,
    N_MELS,
    N_STFT_BANDS,
    ROLLOFF,
    SAMPLE_RATE,
    SPAFE_F0_HZ,
    SPAFE_HOP_S,
    SPAFE_NFFT,
    SPAFE_Q_RATE,
    SPAFE_RESAMPLING_RATIO,
    SPAFE_WIN_S,
    SPAFE_WINDOW,
    WIN_LENGTH,
)

_MEL_WEIGHTS: np.ndarray | None = None
_LFCC_WEIGHTS: np.ndarray | None = None
_HAMMING: np.ndarray | None = None
_FREQS: np.ndarray | None = None
_SPAFE_BAND: dict[str, float] | None = None


def frequencies() -> np.ndarray:
    global _FREQS
    if _FREQS is None:
        _FREQS = np.fft.rfftfreq(N_FFT, d=1.0 / SAMPLE_RATE)
    return _FREQS


def hamming() -> np.ndarray:
    global _HAMMING
    if _HAMMING is None:
        _HAMMING = np.hamming(WIN_LENGTH).astype(np.float64)
    return _HAMMING


def mel_weights() -> np.ndarray:
    global _MEL_WEIGHTS
    if _MEL_WEIGHTS is None:
        import librosa

        _MEL_WEIGHTS = np.asarray(
            librosa.filters.mel(
                sr=SAMPLE_RATE,
                n_fft=N_FFT,
                n_mels=N_MELS,
                fmin=FMIN_HZ,
                fmax=FMAX_HZ,
                htk=MEL_HTK,
                norm=MEL_NORM,
            ),
            dtype=np.float64,
        )
    return _MEL_WEIGHTS


def linear_filterbank() -> np.ndarray:
    """Triangular linear filterbank. Same construction as the classical LFCC code."""

    global _LFCC_WEIGHTS
    if _LFCC_WEIGHTS is None:
        n_freqs = N_FFT // 2 + 1
        freqs = np.linspace(0.0, SAMPLE_RATE / 2.0, n_freqs)
        edges = np.linspace(FMIN_HZ, FMAX_HZ, LFCC_FILTERS + 2)
        weights = np.zeros((LFCC_FILTERS, n_freqs), dtype=np.float64)
        for index in range(LFCC_FILTERS):
            left, center, right = edges[index : index + 3]
            rising = (freqs - left) / max(center - left, 1e-12)
            falling = (right - freqs) / max(right - center, 1e-12)
            weights[index] = np.maximum(0.0, np.minimum(rising, falling))
        _LFCC_WEIGHTS = weights
    return _LFCC_WEIGHTS


def frame_count(n_samples: int) -> int:
    if n_samples < WIN_LENGTH:
        return 0
    return 1 + (n_samples - WIN_LENGTH) // HOP_LENGTH


def stft_power(audio: np.ndarray) -> np.ndarray:
    """Power spectrogram, shape (frames, n_fft/2+1). No edge padding and no tail repeat."""

    count = frame_count(audio.size)
    if count == 0:
        return np.empty((0, N_FFT // 2 + 1), dtype=np.float64)
    offsets = HOP_LENGTH * np.arange(count, dtype=np.int64)
    index = offsets[:, None] + np.arange(WIN_LENGTH, dtype=np.int64)[None, :]
    frames = audio[index] * hamming()
    return np.abs(np.fft.rfft(frames, n=N_FFT)) ** 2


def log_safe(power: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(power, LOG_FLOOR))


def mel_power(power: np.ndarray) -> np.ndarray:
    if power.size == 0:
        return np.empty((0, N_MELS), dtype=np.float64)
    return power @ mel_weights().T


def lfcc_matrix(power: np.ndarray) -> np.ndarray:
    if power.size == 0:
        return np.empty((0, LFCC_CEPS), dtype=np.float64)
    bands = np.maximum(power @ linear_filterbank().T, LOG_FLOOR)
    return dct(np.log(bands), type=2, norm="ortho", axis=1)[:, :LFCC_CEPS]


def _cqt_frequencies() -> np.ndarray:
    import librosa

    return librosa.cqt_frequencies(
        n_bins=CQCC_BINS, fmin=CQCC_FMIN_HZ, bins_per_octave=CQCC_BINS_PER_OCTAVE
    )


def cqcc_librosa(audio: np.ndarray) -> np.ndarray:
    import librosa

    if audio.size < WIN_LENGTH:
        return np.empty((0, CQCC_CEPS), dtype=np.float64)
    transform = librosa.cqt(
        np.ascontiguousarray(audio, dtype=np.float32),
        sr=SAMPLE_RATE,
        hop_length=CQCC_HOP,
        fmin=CQCC_FMIN_HZ,
        n_bins=CQCC_BINS,
        bins_per_octave=CQCC_BINS_PER_OCTAVE,
        tuning=CQCC_TUNING,
        sparsity=CQCC_SPARSITY,
        window=CQCC_WINDOW,
        scale=CQCC_SCALE,
        pad_mode=CQCC_PAD_MODE,
        res_type=CQCC_RES_TYPE,
    )
    power = np.abs(np.asarray(transform, dtype=np.complex128).T) ** 2
    logged = log_safe(power)
    width = min(CQCC_CEPS, logged.shape[1])
    coeffs = dct(logged, type=2, norm="ortho", axis=1)[:, :width]
    if width < CQCC_CEPS:
        coeffs = np.pad(coeffs, ((0, 0), (0, CQCC_CEPS - width)))
    return coeffs


def spafe_effective_band() -> dict[str, float]:
    """Frequencies spafe keeps when the constant-Q window must fit in nfft."""

    global _SPAFE_BAND
    if _SPAFE_BAND is None:
        tmp = np.array(
            [
                SPAFE_F0_HZ * 2 ** ((octave * CQCC_BINS_PER_OCTAVE + step) / CQCC_BINS_PER_OCTAVE)
                for octave in range(CQCC_OCTAVES)
                for step in range(CQCC_BINS_PER_OCTAVE)
            ]
        )
        kept = tmp[(CQCC_FMIN_HZ <= tmp) & (tmp <= FMAX_HZ)]
        quality = SPAFE_Q_RATE / (2 ** (1.0 / CQCC_BINS_PER_OCTAVE) - 1.0)
        windows = np.ceil(quality * SAMPLE_RATE / kept).astype(np.int64)
        fit = windows <= SPAFE_NFFT
        usable = kept[fit]
        _SPAFE_BAND = {
            "declared_bins_in_range": float(kept.size),
            "effective_bins": float(usable.size),
            "dropped_bins": float((~fit).sum()),
            "effective_fmin_hz": float(usable[0]) if usable.size else np.nan,
            "effective_fmax_hz": float(usable[-1]) if usable.size else np.nan,
            "q": float(quality),
        }
    return _SPAFE_BAND


def cqcc_spafe(audio: np.ndarray) -> np.ndarray:
    from spafe.features.cqcc import cqcc
    from spafe.utils.preprocessing import SlidingWindow

    if audio.size < int(SAMPLE_RATE * SPAFE_WIN_S):
        return np.empty((0, CQCC_CEPS), dtype=np.float64)
    window = SlidingWindow(SPAFE_WIN_S, SPAFE_HOP_S, SPAFE_WINDOW)
    cepstra = cqcc(
        np.ascontiguousarray(audio, dtype=np.float64),
        fs=SAMPLE_RATE,
        num_ceps=CQCC_CEPS,
        pre_emph=False,
        pre_emph_coeff=0.97,
        window=window,
        nfft=SPAFE_NFFT,
        low_freq=CQCC_FMIN_HZ,
        high_freq=FMAX_HZ,
        number_of_octaves=CQCC_OCTAVES,
        number_of_bins_per_octave=CQCC_BINS_PER_OCTAVE,
        resampling_ratio=SPAFE_RESAMPLING_RATIO,
        normalize=None,
        f0=SPAFE_F0_HZ,
        q_rate=SPAFE_Q_RATE,
    )
    matrix = np.asarray(cepstra, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return np.empty((0, CQCC_CEPS), dtype=np.float64)
    if matrix.shape[1] < CQCC_CEPS:
        matrix = np.pad(matrix, ((0, 0), (0, CQCC_CEPS - matrix.shape[1])))
    return matrix[:, :CQCC_CEPS]


def f0_track(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """pyin on the raw segment. Returns f0 and voiced flag. Empty if the segment is short."""

    import librosa

    if audio.size < F0_FRAME_LENGTH:
        return np.empty(0), np.empty(0, dtype=bool)
    f0, voiced, _probability = librosa.pyin(
        np.ascontiguousarray(audio, dtype=np.float32),
        fmin=F0_FMIN_HZ,
        fmax=F0_FMAX_HZ,
        sr=SAMPLE_RATE,
        frame_length=F0_FRAME_LENGTH,
        hop_length=F0_HOP,
    )
    return np.asarray(f0, dtype=np.float64), np.asarray(voiced, dtype=bool)


def spectral_shape(power: np.ndarray) -> dict[str, float]:
    """Shape statistics of a linear power spectrogram. Empty input yields NaN."""

    empty = {
        "centroid_hz_mean": np.nan,
        "centroid_hz_std": np.nan,
        "rolloff85_hz_mean": np.nan,
        "flatness_mean": np.nan,
        "flux_l1shape_mean": np.nan,
        "log_energy_mean": np.nan,
        "floor_fraction": np.nan,
    }
    if power.size == 0:
        return empty
    freqs = frequencies()
    total = power.sum(axis=1)
    safe_total = np.maximum(total, LOG_FLOOR)
    centroid = (power * freqs[None, :]).sum(axis=1) / safe_total
    cumulative = np.cumsum(power, axis=1)
    threshold = ROLLOFF * total
    crossed = cumulative >= threshold[:, None]
    index = np.argmax(crossed, axis=1)
    missing = ~crossed.any(axis=1)
    rolloff = freqs[index].astype(np.float64)
    rolloff[missing] = np.nan
    geometric = np.exp(np.mean(np.log(np.maximum(power, LOG_FLOOR)), axis=1))
    arithmetic = np.maximum(power.mean(axis=1), LOG_FLOOR)
    flatness = geometric / arithmetic
    normalized = power / safe_total[:, None]
    if power.shape[0] >= 2:
        flux = float(np.mean(np.linalg.norm(np.diff(normalized, axis=0), axis=1)))
    else:
        flux = np.nan
    finite_centroid = centroid[np.isfinite(centroid)]
    finite_rolloff = rolloff[np.isfinite(rolloff)]
    return {
        "centroid_hz_mean": float(np.mean(finite_centroid)) if finite_centroid.size else np.nan,
        "centroid_hz_std": float(np.std(finite_centroid, ddof=1)) if finite_centroid.size > 1 else np.nan,
        "rolloff85_hz_mean": float(np.mean(finite_rolloff)) if finite_rolloff.size else np.nan,
        "flatness_mean": float(np.mean(flatness)),
        "flux_l1shape_mean": flux,
        "log_energy_mean": float(np.mean(log_safe(power))),
        "floor_fraction": float(np.mean(total < FLOOR_POWER)),
    }


def band_log_means(power: np.ndarray) -> np.ndarray:
    """Mean log-power in equal-Hz bands, shape (n_bands,). NaN if there are no frames."""

    if power.size == 0:
        return np.full(N_STFT_BANDS, np.nan)
    logged = log_safe(power)
    freqs = frequencies()
    edges = np.linspace(FMIN_HZ, FMAX_HZ, N_STFT_BANDS + 1)
    means = np.empty(N_STFT_BANDS, dtype=np.float64)
    for index in range(N_STFT_BANDS):
        if index == N_STFT_BANDS - 1:
            mask = (freqs >= edges[index]) & (freqs <= edges[index + 1])
        else:
            mask = (freqs >= edges[index]) & (freqs < edges[index + 1])
        means[index] = float(np.mean(logged[:, mask])) if mask.any() else np.nan
    return means


def log_of_band_means(power: np.ndarray) -> np.ndarray:
    """Log of the mean linear power in the same bands. Differs from the mean of the log."""

    if power.size == 0:
        return np.full(N_STFT_BANDS, np.nan)
    freqs = frequencies()
    edges = np.linspace(FMIN_HZ, FMAX_HZ, N_STFT_BANDS + 1)
    means = np.empty(N_STFT_BANDS, dtype=np.float64)
    for index in range(N_STFT_BANDS):
        if index == N_STFT_BANDS - 1:
            mask = (freqs >= edges[index]) & (freqs <= edges[index + 1])
        else:
            mask = (freqs >= edges[index]) & (freqs < edges[index + 1])
        means[index] = float(np.log(max(float(np.mean(power[:, mask])), LOG_FLOOR))) if mask.any() else np.nan
    return means


def mel_tilt(log_mel: np.ndarray) -> float:
    if log_mel.size == 0:
        return np.nan
    low = np.mean(log_mel[:, :8], axis=1)
    high = np.mean(log_mel[:, -8:], axis=1)
    return float(np.mean(low - high))


def level_temporal_std(log_mel: np.ndarray) -> float:
    if log_mel.shape[0] < 2:
        return np.nan
    level = np.mean(log_mel, axis=1)
    return float(np.std(level, ddof=1))


def matrix_flux(power: np.ndarray) -> float:
    if power.shape[0] < 2:
        return np.nan
    total = np.maximum(power.sum(axis=1, keepdims=True), LOG_FLOOR)
    normalized = power / total
    return float(np.mean(np.linalg.norm(np.diff(normalized, axis=0), axis=1)))


def temporal_mad(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 2:
        return np.full(matrix.shape[1], np.nan)
    return np.mean(np.abs(np.diff(matrix, axis=0)), axis=0)


def cqt_top_hz() -> float:
    return float(_cqt_frequencies()[-1])
