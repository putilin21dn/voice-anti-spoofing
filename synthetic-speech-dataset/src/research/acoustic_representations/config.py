"""Fixed parameters for the 2026-11-13 acoustic-representation study.

The primary waveform is the subscriber channel after one pre-emphasis of 0.97.
Amplitude normalization is a separate comparison, not part of the primary features.
"""

from __future__ import annotations

SAMPLE_RATE = 8000
CLIENT_CHANNEL = 0
PREEMPHASIS = 0.97
PEAK_NORM_TARGET = 0.99

WINDOW = "hamming_symmetric"
WIN_LENGTH = 160  # 20 ms at 8 kHz
HOP_LENGTH = 80  # 10 ms at 8 kHz
N_FFT = 256
STFT_CENTER = False
FMIN_HZ = 0.0
FMAX_HZ = 4000.0
LOG_TYPE = "natural"
LOG_FLOOR = 1e-10
N_MELS = 26
MEL_NORM = "slaney"
MEL_HTK = False
N_STFT_BANDS = 16
ROLLOFF = 0.85

LFCC_FILTERS = 20
LFCC_CEPS = 20
LFCC_DCT = "scipy_dct_type2_ortho"

CQCC_FMIN_HZ = 62.5
CQCC_BINS_PER_OCTAVE = 24
CQCC_OCTAVES = 6
CQCC_BINS = CQCC_OCTAVES * CQCC_BINS_PER_OCTAVE  # 144, top bin stays under 4 kHz
CQCC_CEPS = 13
CQCC_HOP = HOP_LENGTH
CQCC_WINDOW = "hann"
CQCC_TUNING = 0.0
CQCC_SPARSITY = 0.0
CQCC_SCALE = True
CQCC_PAD_MODE = "constant"
CQCC_RES_TYPE = "soxr_hq"

SPAFE_NFFT = 512
SPAFE_WIN_S = 0.032
SPAFE_HOP_S = 0.01
SPAFE_WINDOW = "hamming"
SPAFE_F0_HZ = 120.0
SPAFE_Q_RATE = 1.0
SPAFE_RESAMPLING_RATIO = 1.0
SPAFE_PREEMPH_INSIDE = False  # the segment is already pre-emphasized once

F0_FMIN_HZ = 60.0
F0_FMAX_HZ = 400.0
F0_FRAME_LENGTH = 1024
F0_HOP = HOP_LENGTH

SPECTRA_THRESHOLD = 1.02
SPECTRA_MODEL_WINDOW_S = 64600 / 16000  # 4.0375 s, frozen window used only for padding stats
CQT_LONG_WINDOW_SAMPLES = 4433  # longest librosa wavelet at these CQT settings, measured

MIN_NONSPEECH_S = 0.2
FLOOR_POWER = 1e-8
CLIP_LEVEL = 32767 / 32768

CANDIDATE_MIN_CLIFF = 0.33
CANDIDATE_MIN_MID_CLIFF = 0.20
CANDIDATE_MAX_ABS_SPEARMAN = 0.30

HUMAN_LABELS = ("human_live", "human_recorded")
SYNTH_LABELS = ("synthetic_tts", "synthetic_voice_conversion", "synthetic_playback")
FORBIDDEN_COLUMNS = ("phone", "phone_number", "external_id", "s3_key", "s3_bucket", "msisdn")

FEATURE_NOTES: dict[str, str] = {
    "logmel_mean": "Средний натуральный логарифм мел-энергии по всем полосам и кадрам. Чувствителен к громкости.",
    "logmel_std": "СКО лог-мел-энергии по всем полосам и кадрам. Смешивает разброс по частоте и по времени.",
    "logmel_iqr": "Межквартильный размах всей лог-мел-матрицы.",
    "logmel_temporal_std_mean": "Среднее по 26 полосам от СКО лог-мел-энергии во времени. Низкое значение — ровный уровень в полосах.",
    "logmel_level_temporal_std": "СКО по времени среднего лог-мел-уровня кадра. Описывает, плавает ли громкость речи.",
    "logmel_low_high_tilt": "Средняя лог-энергия нижних 8 мел-полос минус верхние 8. Положительный наклон — больше низов.",
    "mel_flux_l1shape_mean": "Среднее L2-расстояние между соседними L1-нормированными мел-спектрами. Форма, не громкость.",
    "stft_centroid_hz_mean": "Средний спектральный центроид по линейной мощности STFT, Гц. Центр тяжести спектра.",
    "stft_rolloff85_hz_mean": "Средняя частота, ниже которой лежит 85% линейной мощности STFT.",
    "stft_flatness_mean": "Средняя спектральная плоскостность линейной мощности: геом. среднее / арифм. среднее.",
    "stft_flux_l1shape_mean": "Среднее L2-расстояние между соседними L1-нормированными спектрами мощности STFT.",
    "logstft_mean": "Средний натуральный логарифм мощности STFT. Чувствителен к громкости.",
    "logstft_temporal_std_mean": "Среднее по 16 равным полосам от СКО лог-мощности во времени.",
    "lfcc_c00_mean": "Средний LFCC c0. После DCT лог-энергии линейных полос это в основном общий уровень.",
    "lfcc_c01_mean": "Средний LFCC c1. Грубый спектральный наклон на линейной шкале частот.",
    "lfcc_temporal_mad_mean": "Среднее по коэффициентам от среднего |Δ| между соседними кадрами LFCC.",
    "lfcc_between_segment_std_mean": "Среднее по коэффициентам от СКО сегментных средних LFCC. Различие кусков одного звонка.",
    "cqcc_c00_mean": "Средний CQCC c0 librosa. Уровень лог-энергии constant-Q, не доказательство синтеза.",
    "cqcc_c01_mean": "Средний CQCC c1 librosa. Наклон constant-Q спектра.",
    "cqcc_temporal_mad_mean": "Средняя покадровая изменчивость CQCC librosa.",
    "cqcc_between_segment_std_mean": "Разброс сегментных средних CQCC librosa.",
    "spafe_cqcc_c00_mean": "Средний c0 диагностического spafe CQCC. Полоса фактически уже заявленной.",
    "spafe_cqcc_c01_mean": "Средний c1 диагностического spafe CQCC.",
    "spafe_cqcc_temporal_mad_mean": "Покадровая изменчивость spafe CQCC.",
    "spafe_cqcc_between_segment_std_mean": "Разброс сегментных средних spafe CQCC.",
    "explanatory_f0_median_hz": "Медиана F0 по voiced-кадрам сырого сигнала, Гц. Не входит в четыре представления.",
    "explanatory_f0_iqr_hz": "Межквартильный размах F0. Малый размах — стабильная высота тона на этом звонке.",
    "explanatory_f0_voiced_fraction": "Доля кадров pyin, помеченных как voiced, внутри речевых сегментов.",
    "raw_speech_rms": "RMS сырого канала абонента на речевых интервалах, до pre-emphasis и до нормализации.",
    "raw_speech_peak": "Пик модуля сырого речевого сигнала. Близость к 1 означает клиппинг int16.",
    "raw_clip_fraction": "Доля речевых отсчётов на полной шкале int16.",
    "speech_to_nonspeech_db": "20 log10(RMS речи / RMS вне речи). Вне речи может быть тишина, шум или музыка, не проверенный шум.",
    "spectra_padding_ratio": "Доля повторённого хвоста в окнах Spectra-0. В признаки этого исследования повтор не входит.",
    "speech_ratio": "Доля речевых интервалов Spectra-0 в длительности звонка.",
    "duration_s": "Длительность WAV, секунды.",
}


def config_document() -> dict:
    return {
        "study": "day_20261113_acoustic_representations",
        "primary_waveform": {
            "channel": "subscriber channel 0",
            "sample_rate_hz": SAMPLE_RATE,
            "preemphasis": PREEMPHASIS,
            "preemphasis_scope": "each speech segment separately, no state carried across gaps",
            "amplitude_normalization": "none",
            "resample": "none",
        },
        "speech_intervals": {
            "source": "evaluation/day_20261113/20261113T175850/window_results/spectra0.csv",
            "rule": "per call_id and segment_index, start=min(start_time), end=max(end_time)",
            "vad_rerun": False,
            "model_inference_rerun": False,
            "minimum_segment_inherited_s": 0.5,
            "time_rounding": "window CSV stores seconds rounded to 3 decimals",
        },
        "stft": {
            "window": WINDOW,
            "win_length": WIN_LENGTH,
            "hop_length": HOP_LENGTH,
            "n_fft": N_FFT,
            "center": STFT_CENTER,
            "fmin_hz": FMIN_HZ,
            "fmax_hz": FMAX_HZ,
            "power": "abs(rfft)^2, unnormalized periodogram",
            "log": LOG_TYPE,
            "log_floor": LOG_FLOOR,
            "applies_to": ["log_mel", "log_stft", "lfcc"],
        },
        "log_mel": {
            "n_mels": N_MELS,
            "norm": MEL_NORM,
            "htk": MEL_HTK,
            "fmin_hz": FMIN_HZ,
            "fmax_hz": FMAX_HZ,
            "log": f"{LOG_TYPE}(max(mel_power, {LOG_FLOOR}))",
        },
        "log_stft": {
            "n_bands_for_summary": N_STFT_BANDS,
            "band_width_hz": FMAX_HZ / N_STFT_BANDS,
            "log": f"{LOG_TYPE}(max(power, {LOG_FLOOR}))",
            "centroid_rolloff_flatness": "computed on linear power of the same STFT",
            "rolloff": ROLLOFF,
        },
        "lfcc": {
            "n_filters": LFCC_FILTERS,
            "n_ceps": LFCC_CEPS,
            "fmin_hz": FMIN_HZ,
            "fmax_hz": FMAX_HZ,
            "filter_shape": "triangular, unnormalized, same construction as classical-anti-spoofing",
            "dct": LFCC_DCT,
            "log_floor": LOG_FLOOR,
            "second_preemphasis": False,
        },
        "cqcc_librosa": {
            "role": "primary CQCC",
            "fmin_hz": CQCC_FMIN_HZ,
            "n_bins": CQCC_BINS,
            "bins_per_octave": CQCC_BINS_PER_OCTAVE,
            "hop_length": CQCC_HOP,
            "window": CQCC_WINDOW,
            "tuning": CQCC_TUNING,
            "sparsity": CQCC_SPARSITY,
            "scale": CQCC_SCALE,
            "pad_mode": CQCC_PAD_MODE,
            "center": True,
            "res_type": CQCC_RES_TYPE,
            "internal_resample": "librosa may downsample lower octaves by 2 from 8000 Hz; it does not create content above 4000 Hz",
            "longest_wavelet_samples": CQT_LONG_WINDOW_SAMPLES,
            "n_ceps": CQCC_CEPS,
            "dct": LFCC_DCT,
            "log_floor": LOG_FLOOR,
            "power": "abs(cqt)^2",
        },
        "cqcc_spafe": {
            "role": "diagnostic, not the primary CQCC",
            "library": "spafe 0.3.3",
            "declared_fmin_hz": CQCC_FMIN_HZ,
            "declared_fmax_hz": FMAX_HZ,
            "number_of_octaves": CQCC_OCTAVES,
            "bins_per_octave": CQCC_BINS_PER_OCTAVE,
            "f0_hz": SPAFE_F0_HZ,
            "q_rate": SPAFE_Q_RATE,
            "nfft": SPAFE_NFFT,
            "win_len_s": SPAFE_WIN_S,
            "win_hop_s": SPAFE_HOP_S,
            "window": SPAFE_WINDOW,
            "resampling_ratio": SPAFE_RESAMPLING_RATIO,
            "pre_emph_inside_spafe": SPAFE_PREEMPH_INSIDE,
            "normalize": None,
            "n_ceps": CQCC_CEPS,
            "effective_band_note": "bins whose constant-Q window exceeds nfft=512 are dropped; effective band is about 539-3953 Hz",
        },
        "comparison_variants": {
            "raw": "subscriber channel, no pre-emphasis, no amplitude normalization",
            "peak_norm": f"peak scaled to {PEAK_NORM_TARGET} on the raw segment, no pre-emphasis, clipping is not repaired",
            "preemph": "primary",
            "spectral_log": "natural log of STFT power versus the same power left linear; primary waveform",
            "log_mel": "natural log after the mel filterbank versus the mel energies themselves; primary waveform",
        },
        "f0": {
            "role": "explanatory only, not one of the four representations",
            "waveform": "raw segment, before pre-emphasis",
            "method": "librosa.pyin",
            "fmin_hz": F0_FMIN_HZ,
            "fmax_hz": F0_FMAX_HZ,
            "frame_length": F0_FRAME_LENGTH,
            "hop_length": F0_HOP,
        },
        "spectra0_control": {
            "threshold": SPECTRA_THRESHOLD,
            "unchanged": True,
            "inference_rerun": False,
            "padding_statistics_source": "existing window_results/spectra0.csv",
            "repeated_audio_in_new_features": False,
        },
        "candidate_rule": {
            "min_abs_cliffs_delta": CANDIDATE_MIN_CLIFF,
            "min_abs_mid_tertile_cliffs_delta": CANDIDATE_MIN_MID_CLIFF,
            "max_abs_spearman_with_duration_speech_ratio_rms": CANDIDATE_MAX_ABS_SPEARMAN,
            "requires_same_sign_in_pack1_and_pack2": True,
            "requires_same_sign_in_all_duration_tertiles": True,
            "meaning": "a feature worth a later independent experiment, not evidence that a detector improved",
        },
    }
