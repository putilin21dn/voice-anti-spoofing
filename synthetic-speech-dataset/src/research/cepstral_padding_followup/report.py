"""Russian reports. Numbers come from the CSV files of this follow-up."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.acoustic_representations.signal import read_csv
from src.research.cepstral_padding_followup.common import (
    BOOT_SEED,
    FN_CALLS,
    MIN_N_CI,
    N_BOOT,
    OUTPUT,
    PREEMPHASIS,
    RMS_TARGET,
    SAMPLE_RATE,
    SPECTRA_MODEL_WINDOW_S,
    SPECTRA_THRESHOLD,
    TARGET_SAMPLES,
    WINDOWS,
    read_table,
    spearman,
)


def _num(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _fmt(value, digits: int = 3) -> str:
    number = _num(value)
    if not np.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def _pick(frame: pd.DataFrame, **query) -> pd.Series | None:
    mask = pd.Series(True, index=frame.index)
    for key, value in query.items():
        mask &= frame[key] == value
    hit = frame[mask]
    if hit.empty:
        return None
    return hit.iloc[0]


def _effect(record: pd.Series | None) -> str:
    if record is None:
        return "строка не посчитана"
    ci = str(record["ci_status"])
    interval = ""
    if ci in {"ok", "pack2_small"}:
        interval = f", 95% интервал {_fmt(record['ci_low'], 2)}…{_fmt(record['ci_high'], 2)}"
        if ci == "pack2_small":
            interval += " (PACK2, синтетики мало)"
    elif ci == "insufficient":
        interval = f", интервал не строился: в одной из сторон меньше {MIN_N_CI} конечных значений"
    return (
        f"n {_fmt(record['n_reference_finite'], 0)}/{_fmt(record['n_other_finite'], 0)}, "
        f"медианы {_fmt(record['reference_median'])} (IQR {_fmt(record['reference_q25'])}–{_fmt(record['reference_q75'])}) "
        f"и {_fmt(record['other_median'])} (IQR {_fmt(record['other_q25'])}–{_fmt(record['other_q75'])}), "
        f"Cliff's delta {_fmt(record['cliffs_delta'], 2)}{interval}"
    )


def _reading(overall: pd.Series | None, stratum: pd.Series | None) -> str:
    if stratum is None or overall is None:
        return "нет оценки"
    if int(stratum["n_reference_finite"]) < MIN_N_CI or int(stratum["n_other_finite"]) < MIN_N_CI:
        return "мало наблюдений"
    overall_delta = _num(overall["cliffs_delta"])
    delta = _num(stratum["cliffs_delta"])
    if not np.isfinite(overall_delta) or not np.isfinite(delta):
        return "нет оценки"
    same = overall_delta * delta > 0
    separated = str(stratum["ci_status"]) in {"ok", "pack2_small"} and _num(stratum["ci_low"]) * _num(stratum["ci_high"]) > 0
    if str(stratum["stratum"]) == "PACK2" or str(stratum["ci_status"]) == "pack2_small":
        return "мало синтетики в PACK2"
    if same and separated:
        return "направление то же, интервал без нуля"
    if same:
        return "направление то же, интервал содержит ноль"
    if separated:
        return "направление другое, интервал без нуля"
    return "направление другое, интервал содержит ноль"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join([head, rule, body])


def _endpoint_table(effects: pd.DataFrame, endpoint: str) -> str:
    overall = _pick(effects, comparison="synthetic_vs_human", endpoint=endpoint, stratum_family="overall", stratum="all")
    rows = []
    wanted = effects[
        (effects["comparison"] == "synthetic_vs_human")
        & (effects["endpoint"] == endpoint)
        & (effects["stratum_family"] != "normalization")
    ]
    order = {"overall": 0, "rms": 1, "call_duration": 2, "speech_duration": 3, "speech_ratio": 4, "campaign": 5, "n_segments": 6}
    wanted = wanted.assign(_order=wanted["stratum_family"].map(order).fillna(9)).sort_values(["_order", "stratum"])
    for record in wanted.to_dict(orient="records"):
        current = pd.Series(record)
        rows.append(
            [
                f"{record['stratum_family']} / {record['stratum']}",
                str(int(record["n_reference_finite"])),
                str(int(record["n_other_finite"])),
                _fmt(record["reference_median"]),
                f"{_fmt(record['reference_q25'])}–{_fmt(record['reference_q75'])}",
                _fmt(record["other_median"]),
                f"{_fmt(record['other_q25'])}–{_fmt(record['other_q75'])}",
                _fmt(record["cliffs_delta"], 2),
                "—" if not np.isfinite(_num(record["ci_low"])) else f"{_fmt(record['ci_low'], 2)}…{_fmt(record['ci_high'], 2)}",
                _reading(overall, current),
            ]
        )
    return _md_table(
        ["Группа", "n human", "n synth", "медиана human", "IQR human", "медиана synth", "IQR synth", "delta", "95% интервал", "чтение"],
        rows,
    )


def _paired(padding: pd.DataFrame, scope: str, column: str) -> tuple[np.ndarray, np.ndarray]:
    keys = ["call_id", "segment_index", "scope"]
    left = padding[(padding["scope"] == scope) & (padding["condition"] == "repeat")]
    right = padding[(padding["scope"] == scope) & (padding["condition"] == "zero")]
    merged = left.merge(right, on=keys, suffixes=("_repeat", "_zero"))
    delta = pd.to_numeric(merged[f"{column}_repeat"], errors="coerce") - pd.to_numeric(merged[f"{column}_zero"], errors="coerce")
    fraction = pd.to_numeric(merged["real_audio_fraction_repeat"], errors="coerce")
    valid = np.isfinite(delta) & np.isfinite(fraction)
    return delta[valid].to_numpy(dtype=np.float64), fraction[valid].to_numpy(dtype=np.float64)


def _delta_sentence(values: np.ndarray) -> str:
    if values.size == 0:
        return "нет пар"
    q25, median, q75 = np.percentile(values, [25, 50, 75])
    return (
        f"n {values.size}, медиана {median:.3f}, IQR {q25:.3f}–{q75:.3f}, "
        f"медиана модуля {np.median(np.abs(values)):.3f}"
    )


def _window_summary(calls: pd.DataFrame) -> str:
    labels = calls.set_index("call_id")[["binary_group", "spectra_error_role"]]
    margins = {"human_voice": {False: [], True: []}, "synthetic": {False: [], True: []}}
    paired = []
    by_call: dict[str, list[tuple[bool, float]]] = {}
    for row in read_csv(WINDOWS):
        call_id = row["call_id"]
        if call_id not in labels.index:
            continue
        group = labels.at[call_id, "binary_group"]
        if group not in margins:
            continue
        padded = row["padded_to_model_length"] == "True"
        margin = float(row["margin"])
        margins[group][padded].append(margin)
        by_call.setdefault(call_id, []).append((padded, margin))
    for call_id, items in by_call.items():
        full = [margin for padded, margin in items if not padded]
        tail = [margin for padded, margin in items if padded]
        if full and tail:
            paired.append(float(np.mean(tail) - np.mean(full)))
    lines = []
    for group, title in (("human_voice", "human"), ("synthetic", "synthetic")):
        for padded, name in ((False, "полные окна"), (True, "окна с повтором хвоста")):
            values = np.array(margins[group][padded], dtype=np.float64)
            q25, median, q75 = np.percentile(values, [25, 50, 75])
            lines.append(f"- {title}, {name}: n {values.size}, медиана margin {median:.3f}, IQR {q25:.3f}–{q75:.3f}.")
    if paired:
        values = np.array(paired)
        q25, median, q75 = np.percentile(values, [25, 50, 75])
        lines.append(
            f"- Внутри звонка, где есть и полное окно, и хвост: медиана (средний margin хвостов минус средний margin полных окон) {median:.3f}, IQR {q25:.3f}–{q75:.3f}, n звонков {values.size}."
        )
    lines.append(
        "Полное окно — это первые 4.0375 с куска, а повторённый хвост — его конец. Разница margin смешивает повтор, длину куска и положение внутри сегмента."
    )
    return "\n".join(lines)


def _fn_section(fn: pd.DataFrame, calls: pd.DataFrame) -> str:
    blocks = []
    for call_id in FN_CALLS:
        part = fn[fn["call_id"] == call_id].copy()
        call = calls[calls["call_id"] == call_id].iloc[0]
        closer = part["cqcc_c1_closer_segment_median"].value_counts(dropna=False).to_dict()
        c1 = pd.to_numeric(part["cqcc_c01_mean"], errors="coerce")
        voiced = pd.to_numeric(part["f0_voiced_fraction"], errors="coerce")
        rms = pd.to_numeric(part["raw_rms"], errors="coerce")
        margins = pd.to_numeric(part["mean_window_margin"], errors="coerce")
        blocks.append(
            "\n".join(
                [
                    f"### `{call_id}`",
                    "",
                    (
                        f"{call['primary_label']}, {call['campaign']}, сохранённый margin {_fmt(call['spectra0_margin_frozen'])}. "
                        f"Сегментов {int(call['n_speech_segments'])}, речь {_fmt(call['speech_duration_s'])} с, "
                        f"доля речи {_fmt(call['speech_ratio'])}, padding ratio {_fmt(call['spectra_padding_ratio'])}."
                    ),
                    (
                        f"Межсегментное СКО: LFCC {_fmt(call['lfcc_between_segment_std_mean'])}, "
                        f"CQCC librosa {_fmt(call['cqcc_between_segment_std_mean'])}, "
                        f"spafe {_fmt(call['spafe_cqcc_between_segment_std_mean'])}. "
                        f"Пул кадров CQCC c1 {_fmt(call['cqcc_c01_mean'])}."
                    ),
                    (
                        f"По сегментам CQCC c1: медиана {_fmt(c1.median())}, минимум {_fmt(c1.min())}, максимум {_fmt(c1.max())}. "
                        f"Ближе к медиане сегментов human: {int(closer.get('human_voice', 0))}, "
                        f"к медиане других synthetic: {int(closer.get('synthetic', 0))}. "
                        f"Это сравнение одного коэффициента, не классификатор."
                    ),
                    (
                        f"RMS сегментов, медиана {_fmt(rms.median())}. "
                        f"Доля voiced, медиана {_fmt(voiced.median())}. "
                        f"F0 звонка: медиана {_fmt(call['explanatory_f0_median_hz'])} Гц, "
                        f"IQR {_fmt(call['explanatory_f0_iqr_hz'])} Гц, voiced {_fmt(call['explanatory_f0_voiced_fraction'])}."
                    ),
                    (
                        f"Средний margin окон сегмента: медиана {_fmt(margins.median())}. "
                        f"Полные окна есть у {(pd.to_numeric(part['n_full_windows'], errors='coerce') > 0).sum()} сегментов."
                    ),
                ]
            )
        )
    return "\n\n".join(blocks)


def _edges_text(edges: dict) -> str:
    lines = []
    for family, bins in edges.items():
        parts = []
        for name in ("low", "mid", "high"):
            item = bins.get(name, {})
            parts.append(f"{name} n={item.get('n', '—')} [{_fmt(item.get('min'))} … {_fmt(item.get('max'))}]")
        lines.append(f"- {family}: " + "; ".join(parts))
    return "\n".join(lines)


def write_reports(status: dict) -> None:
    effects = read_table(OUTPUT / "matched_group_analysis.csv")
    calls = read_table(OUTPUT / "call_level_cepstral_summary.csv")
    fn = read_table(OUTPUT / "false_negative_segment_analysis.csv")
    fp = read_table(OUTPUT / "false_positive_padding_analysis.csv")
    text_columns = {
        "call_id",
        "comparison",
        "endpoint",
        "stratum_family",
        "stratum",
        "ci_status",
        "note",
        "primary_label",
        "campaign",
        "binary_group",
        "spectra_error_role",
        "row_type",
        "status",
        "segment_count_bin",
        "rms_tertile",
        "call_duration_tertile",
        "speech_duration_tertile",
        "speech_ratio_tertile",
        "padding_tertile",
        "segment_durations_s",
        "window_margins",
        "window_valid_s",
        "window_padded",
        "cqcc_c1_closer_segment_median",
        "condition",
        "scope",
        "construct_order",
        "sample_alignment",
        "bounds_note",
    }
    for frame in (effects, calls, fn, fp):
        for column in frame.columns:
            if column in text_columns:
                continue
            converted = pd.to_numeric(frame[column], errors="coerce")
            original = frame[column].astype(str).str.strip()
            comparable = original.ne("") & original.ne("nan")
            if comparable.any() and converted[comparable].notna().mean() > 0.9:
                frame[column] = converted
    padding_path = OUTPUT / "padding_sensitivity.csv"
    norm_path = OUTPUT / "normalization_sensitivity.csv"
    padding = read_table(padding_path) if padding_path.exists() else None
    norm = read_table(norm_path) if norm_path.exists() else None
    tables = status.get("tables", {})
    audio = status.get("audio", {})
    checksum = tables.get("checksum", {})

    names = {
        "lfcc_between_segment_std_mean": "Межсегментное СКО LFCC",
        "cqcc_between_segment_std_mean": "Межсегментное СКО CQCC librosa",
        "spafe_cqcc_between_segment_std_mean": "Межсегментное СКО CQCC spafe",
        "cqcc_c01_mean": "CQCC librosa c1",
    }
    hypothesis = []
    for endpoint, title in names.items():
        overall = _pick(effects, comparison="synthetic_vs_human", endpoint=endpoint, stratum_family="overall", stratum="all")
        hypothesis.append(f"### {title}\n\n{_effect(overall)}. Положительная delta значит, что у synthetic значение обычно больше, чем у human.\n\n{_endpoint_table(effects, endpoint)}")

    spearman_lines = []
    spearman_rows = effects[effects["comparison"] == "spearman"]
    for endpoint in names:
        for group in ("binary", "human_voice", "synthetic"):
            for covariate in ("n_speech_segments", "raw_speech_rms", "speech_duration_s", "speech_ratio"):
                record = _pick(spearman_rows, endpoint=endpoint, stratum_family=group, stratum=covariate)
                if record is None:
                    continue
                spearman_lines.append(
                    f"- {names[endpoint]}, {group}, {covariate}: Spearman {_fmt(record['spearman'], 2)}, n {_fmt(record['spearman_n'], 0)}."
                )

    margin_lines = []
    margin_rows = effects[effects["comparison"] == "spearman_margin"]
    for group in ("human_voice", "synthetic", "TN", "FP", "TP", "FN", "binary"):
        for covariate in ("spectra_padding_ratio", "speech_duration_s", "spectra_n_windows", "n_speech_segments"):
            record = _pick(margin_rows, stratum_family=group, stratum=covariate)
            if record is None:
                continue
            margin_lines.append(
                f"- {group}, margin и {covariate}: Spearman {_fmt(record['spearman'], 2)}, n {_fmt(record['spearman_n'], 0)}."
            )

    fp_lines = []
    for endpoint, title in (
        ("speech_duration_s", "Длительность речи, с"),
        ("n_speech_segments", "Число сегментов"),
        ("speech_ratio", "Доля речи"),
        ("spectra_padding_ratio", "Padding ratio"),
        ("lfcc_between_segment_std_mean", "Межсегментное СКО LFCC"),
        ("cqcc_between_segment_std_mean", "Межсегментное СКО CQCC"),
        ("spafe_cqcc_between_segment_std_mean", "Межсегментное СКО spafe"),
        ("cqcc_c01_mean", "CQCC c1"),
        ("raw_speech_rms", "RMS"),
        ("explanatory_f0_iqr_hz", "IQR F0, Гц"),
    ):
        record = _pick(effects, comparison="fp_vs_tn", endpoint=endpoint, stratum_family="overall", stratum="all")
        fp_lines.append(f"- {title}: {_effect(record)}. Здесь reference — TN, other — FP.")

    cross = fp[fp["row_type"] == "crosstab"]
    cross_rows = []
    for record in cross.to_dict(orient="records"):
        cross_rows.append(
            [
                str(record["speech_duration_tertile"]),
                str(record["padding_tertile"]),
                str(int(record["n_fp"])) if np.isfinite(_num(record["n_fp"])) else "—",
                str(int(record["n_tn"])) if np.isfinite(_num(record["n_tn"])) else "—",
                str(int(record["n_tp"])) if np.isfinite(_num(record["n_tp"])) else "—",
                str(int(record["n_fn"])) if np.isfinite(_num(record["n_fn"])) else "—",
            ]
        )
    cross_table = _md_table(["Речь", "Padding", "FP", "TN", "TP", "FN"], cross_rows)

    padding_text = "Файл `padding_sensitivity.csv` ещё не собран."
    if padding is not None:
        pieces = []
        for scope, title in (("tail", "Хвост одной длины"), ("segment", "Целый сегмент с дописанным хвостом")):
            pieces.append(f"#### {title}")
            for column, label in (
                ("cqcc_c01_mean", "CQCC c1"),
                ("lfcc_c01_mean", "LFCC c1"),
                ("spafe_cqcc_c01_mean", "spafe c1"),
            ):
                values, fraction = _paired(padding, scope, column)
                estimate, count = spearman(pd.Series(values), pd.Series(fraction))
                pieces.append(f"- {label}, повтор минус нули: {_delta_sentence(values)}. Spearman этой разницы с долей реального аудио {_fmt(estimate, 2)}, n {count}.")
        padding_text = "\n".join(pieces)

    norm_text = "Файл `normalization_sensitivity.csv` ещё не собран."
    if norm is not None:
        blocks = []
        for condition in ("none", "peak", "rms"):
            blocks.append(f"#### {condition}")
            for endpoint, title in list(names.items()) + [
                ("lfcc_c00_between_segment_std", "Межсегментное СКО LFCC c0, контроль уровня"),
                ("cqcc_c00_between_segment_std", "Межсегментное СКО CQCC c0, контроль уровня"),
            ]:
                record = _pick(effects, comparison="synthetic_vs_human", endpoint=endpoint, stratum_family="normalization", stratum=condition)
                blocks.append(f"- {title}: {_effect(record)}.")
        norm_text = "\n".join(blocks)

    max_coeff = checksum.get("max_abs_coefficient", {})
    max_mean = checksum.get("max_abs_mean", {})
    edges = tables.get("edges", {})
    c1_gap = _pick(effects, comparison="synthetic_vs_human", endpoint="cqcc_c01_mean", stratum_family="overall", stratum="all")
    gap = ""
    if c1_gap is not None:
        gap = (
            f"Разница медиан CQCC c1 synthetic и human на исходных сегментах: "
            f"{_fmt(_num(c1_gap['other_median']) - _num(c1_gap['reference_median']))}. "
            "Сдвиг c1 от замены повтора нулями стоит сравнивать с этой величиной по модулю, а не объявлять причиной ошибки модели."
        )

    methodology = f"""# Методика: кепстр и padding, день 2026-11-13

Это продолжение `reports/day_20261113_acoustic_representations`. Новый каталог ничего в том отчёте не меняет. Spectra-0 и AASIST3 заново не запускались. Порог 1.02 не менялся. Классификатор не обучался. Исходные WAV не переписывались.

## Что переиспользовано

Речевые границы и окна взяты из `evaluation/day_20261113/20261113T175850/window_results/spectra0.csv`. Сегментные средние LFCC, CQCC librosa и CQCC spafe взяты из `segment_level_features.csv` прошлого отчёта. Они уже посчитаны на канале абонента, 8 кГц, с одним pre-emphasis {PREEMPHASIS} и без нормировки громкости.

Межсегментное СКО пересчитано из этих сегментных средних: для каждого коэффициента это СКО средних по сегментам с `ddof=1`, если конечных сегментов не меньше двух. Сегмент с пустым кепстром в это СКО не входит. Внутрисегментный показатель — прежний `temporal_mad`. Он записан отдельными колонками `within_segment_*` и в межсегментное СКО не подмешивается.

Максимальное абсолютное расхождение с прошлым call-level по коэффициентам: LFCC {max_coeff.get("lfcc", "—")}, CQCC {max_coeff.get("cqcc", "—")}, spafe {max_coeff.get("spafe_cqcc", "—")}. По среднему СКО: {max_mean}.

Сохранённый margin звонка сверен со средним margin его окон. Максимальный модуль разницы: {tables.get("max_margin_gap", "—")}.

## Чего в старых файлах нет

Соответствие «окно ↔ сегмент» в CSV есть: `segment_index`, `window_index`, флаг padding, `valid_speech_audio_sec`, margin. Индексы отсчётов 16 кГц восстановить нельзя: времена округлены до 0.001 с. Поэтому опыт с повтором хвоста помечен как аналог на 8 кГц, а не как вход Spectra-0.

Сегменты короче 0.5 с в таблицу не входят. Spectra-0 отбросила их до построения окон, и отдельный список отброшенных сегментов не сохранялся. В этой таблице такие сегменты не восстановить и не назвать их числом.

Внутрисегментное СКО коэффициентов в сегментном CSV прошлого отчёта не хранилось. Здесь внутрисегментная величина — покадровый mean absolute difference, уже посчитанный на звонке.

## Группы

Терцили считаются только на 251 бинарном звонке. Звонки сортируются по ковариате, при равенстве по `call_id`, и режутся на три почти равные части. Границы не подбирались по кепстральному результату. Если одно и то же значение ковариаты стоит на разрезе, соседние терцили соприкасаются по краю.

Число сегментов режется заранее: `1`, `2-4`, `5+`. Межсегментное СКО для одного сегмента не определено, поэтому группа `1` есть только у CQCC c1.

{_edges_text(edges)}

Cliff's delta положительна, когда вторая группа обычно больше первой. Для human/synthetic вторая группа — synthetic. Для FP/TN вторая группа — FP. Интервал — перцентильный bootstrap, {N_BOOT} повторов, seed {BOOT_SEED}, только если в каждой стороне не меньше {MIN_N_CI} конечных значений. Это интервал для той же выборки, не для нового дня. Поправка на число сравнений не делается.

PACK2 содержит 10 синтетических звонков. Интервал там, где он построен, не оправдание сильного вывода.

## Padding без нового inference

Длина окна Spectra-0 — {SPECTRA_MODEL_WINDOW_S:.4f} с, {TARGET_SAMPLES} отсчётов на 8 кГц. Для каждого речевого сегмента берутся три сигнала:

- реальный хвост последнего окна;
- тот же хвост, повторённый до {TARGET_SAMPLES} отсчётов;
- тот же хвост, дополненный нулями до той же длины.

Отдельно то же самое делается для целого сегмента: префикс остаётся реальным, а дописывается только хвост. Сравнивать повтор и нули можно напрямую: длина одна. Реальный хвост короче. Его кепстр зависит от другого числа кадров и другого разрешения constant-Q, поэтому он хранится, но не является парой к дополненным сигналам.

Порядок: собрать сигнал, затем pre-emphasis {PREEMPHASIS}, затем LFCC и оба CQCC. Логарифм стоит внутри кепстра, не на громкости. Порог и logits не трогаются.

## Нормировка громкости

На сыром сегменте, до pre-emphasis:

- `none` — без изменения амплитуды;
- `peak` — пик к 0.99;
- `rms` — RMS к {RMS_TARGET}.

Клиппинг не чинится. Если RMS почти ноль, сегмент не масштабируется. Межсегментное СКО снова считается по средним сегментов. CQCC c1 в этой таблице — среднее по кадрам, как в прошлом отчёте.

## Идентификаторы

В таблицах идентификатор звонка — только `call_id`.
"""

    summary = f"""# Кепстр и padding, день 2026-11-13

Это тот же день и те же 251 бинарный звонок. Результат ниже проверяет, держится ли уже увиденное различие внутри заранее заданных групп. Это не подтверждение на новом дне и не новая точность детектора.

Spectra-0 по сохранённым счетам: 63 TP, 32 FP, 151 TN, 5 FN. Порог {SPECTRA_THRESHOLD:g}. Ниже него звонок считается spoof.

Межсегментное СКО — разброс средних LFCC или CQCC между речевыми сегментами одного звонка. Покадровые скачки внутри сегмента в это число не входят.

## Гипотеза A. Межсегментная стабильность

На всём бинарном дне синтетика по-прежнему имеет меньшее межсегментное СКО LFCC и обоих CQCC. CQCC c1 у синтетики больше. Это описание той же выборки, на которой кандидаты уже были отобраны.

{chr(10).join(hypothesis)}

### Связь с числом сегментов и с громкостью

{chr(10).join(spearman_lines)}

Внутри каждого класса межсегментное СКО LFCC растёт с числом сегментов: Spearman 0.36 и у human, и у synthetic. На смеси классов коэффициент только 0.19, потому что у классов разное число сегментов. Различие классов при этом остаётся и в группе 2–4 сегмента, и в группе 5+.

CQCC c1 на смеси классов имеет Spearman 0.27 с длительностью звонка. Внутри human это 0.12, внутри synthetic 0.10. Связь 0.27 в основном собрана из того, что синтетические звонки длиннее и c1 у них выше, а не из роста c1 вдоль длительности внутри класса.

Конечное межсегментное СКО LFCC есть у {tables.get("n_between_human", "—")} из 183 human и {tables.get("n_between_synth", "—")} из 68 synthetic. У остальных бинарных звонков один речевой сегмент, и межсегментное СКО для них не определено.

Где интервал содержит ноль или синтетики меньше восьми, вывод не держится. Это короткие звонки для СКО LFCC и CQCC librosa (12 synthetic, интервал доходит до нуля), нижний терциль RMS (6 synthetic, интервал не строился) и PACK2 (9–10 synthetic). В speech_ratio / mid synthetic тоже около десяти: интервал c1 там без нуля, но группа тонкая.

## Гипотеза B. Повтор хвоста и сохранённый margin

{_window_summary(calls)}

### Spearman сохранённого margin

{chr(10).join(margin_lines)}

Графики: `visualizations/margin_vs_padding_ratio.png`, `margin_vs_speech_duration.png`, `margin_vs_n_windows.png`, `margin_vs_n_segments.png`, `window_margin_full_vs_padded.png`.

У human более высокий padding ratio идёт вместе с более низким margin (Spearman −0.37). У synthetic эта связь около нуля. Медиана margin полных человеческих окон 6.51, медиана их хвостов 1.48: хвост ближе к порогу, но его медиана всё ещё выше 1.02. У synthetic медиана полных окон 0.15, медиана хвостов −4.62. Внутри 153 звонков, где есть оба типа окон, медиана разницы «хвост минус полное» −1.82, а верхний квартиль уже положительный: не у каждого звонка хвост ниже.

Корреляции по пяти FN не используются: n = 5.

Ассоциация не отделяет повтор от того, что хвост короткий и стоит в конце сегмента. Logits заново не считались.

## Насколько padding меняет сам кепстр

Сравнение одной длины — повтор хвоста минус нули. {gap}

{padding_text}

Нули — это тишина, повтор — ещё одна копия речи. Поэтому c0 сдвигается сильно: у хвоста медиана LFCC c0 «повтор минус нули» +42.7, медиана CQCC c0 +68.8. Это уровень энергии дополнения, не тембр.

У CQCC c1 знак сдвига не устойчив: медиана −0.93, IQR −4.80…+2.40, медиана модуля 3.51. Разрыв медиан c1 между synthetic и human на исходных сегментах около 8.6. Дополнение той же длины может сдвинуть c1 на несколько единиц, но не в одну сторону. Spearman этого сдвига с долей реального аудио у CQCC c1 всего 0.12. У LFCC c1 и spafe c1 сдвиг положительный и связан с долей реального аудио сильнее (−0.45 и −0.59): чем меньше в окне настоящей речи, тем сильнее повтор отличается от нулей.

Реальный хвост короче дополненного. Медиана числа кадров CQCC: 147 против 404. Эти две длины не сравниваются как один и тот же кепстр. Четыре реальных хвоста короче одного кадра LFCC, 160 отсчётов, и помечены `shorter_than_one_frame`. Их повтор и нули уже имеют длину окна и посчитаны. Все 11322 строки помечены `rounded_1ms`.

Рисунок: `visualizations/padding_cqcc_c1_repeat_vs_zero.png`.

Расхождение условия `none` с уже сохранённым сегментным CQCC c1, максимальный модуль: {audio.get("cqcc_c1_max_abs_diff_none_vs_stored", "—")} на {audio.get("n_checksum_segments", "—")} сегментах. Новый срез попал в те же куски.

## Нормировка громкости

{norm_text}

Рисунок: `visualizations/normalization_between_segment_std.png`.

RMS каждого сегмента к 0.1 убирает большую часть классового разрыва уровня LFCC c0: Cliff's delta среднего по кадрам c0 падает с 0.72 до 0.10 (медианы −33.5 против −26.8 без нормировки и −27.7 против −27.0 после RMS). У CQCC c0 delta падает с 0.82 до 0.41. Пиковая нормировка ослабляет LFCC c0 слабее, до 0.45.

Межсегментное СКО и CQCC c1 при этом остаются: после RMS delta СКО LFCC −0.72, CQCC librosa −0.64, spafe −0.73, c1 +0.59. Межсегментное СКО самого c0 тоже не сжимается. Выравнивание RMS сегмента меняет общий уровень и почти не меняет разброс сегментных средних. Нормировка здесь не доказывает причину. Она показывает, какой показатель живёт в громкости, а какой остаётся после неё.

## Пять пропусков

Медиана сегментного CQCC c1 на картинке — медиана всех сегментов класса. Для расстояния до synthetic медиана считается без сегментов самого этого звонка.

{_fn_section(fn, calls)}

`8b34a750` не сводится к среднему звонка. У 43 из 50 сегментов CQCC c1 ближе к медиане человеческих сегментов, у 7 — к медиане других synthetic. Медиана сегментного c1 −3.48, при этом размах сегментов от −7.16 до +6.77. Межсегментное СКО низкое, как у типичной синтетики этого дня, а c1 и доля voiced (медиана сегментов 0.14, по звонку 0.26) — нет. F0 IQR 72 Гц, RMS сегментов около 0.032.

Сохранённые окна этого звонка расходятся между собой. Средний margin 46 полных окон 2.70, средний margin 50 хвостов 0.33. Хвосты тянут средний margin вниз, к spoof, а не вверх. Среднее полных окон само по себе выше 1.02. Пропуск не объясняется тем, что повтор хвоста сделал звонок bonafide: полные окна уже на стороне bonafide.

`a037652f` — другой случай. Полных окон нет, все четыре сохранённых margin около 8.4, средний 8.44. Весь счёт собран из хвостов короче окна. CQCC c1 всех четырёх реальных сегментов ближе к медиане synthetic, пул кадров c1 +4.71. Кепстр реального аудио и сохранённый margin смотрят в разные стороны. Отделить короткий хвост от повтора здесь нельзя: другого окна в сегменте нет. Это один звонок PACK2.

`60f772ee` и `cf7c2727` почти без повтора (padding ratio 0.044 и 0.058). Средний margin их полных окон 2.19 и 1.85, тоже выше порога. c1 сегментов ближе к human. Низкое межсегментное СКО здесь есть, а padding как объяснение пропуска — нет.

`7eb5d6cd`: и полное окно (4.78), и хвосты (3.81) выше порога. Три сегмента по c1 ближе к synthetic, но сами значения от −1.21 до +7.38.

Рисунки: `visualizations/false_negative_cqcc_c1.png`, `false_negative_8b34a750_segments.png`, `false_negative_a037652f_windows.png`.

## 32 ложных spoof против 151 TN

{chr(10).join(fp_lines)}

Клетки ниже используют те же терцили речи и padding, что и остальной отчёт. Это раскладка ошибок по ковариатам, не новая модель.

{cross_table}

Кепстральные показатели FP и TN не разделяют: интервалы delta для межсегментного СКО, c1, RMS и IQR F0 содержат ноль. Разделяют длительность речи (7.04 с против 12.66 с), доля речи (0.269 против 0.433) и padding ratio (0.577 против 0.492).

15 из 32 FP лежат в клетке «короткая речь и высокий padding». В той же клетке 30 TN, так что это сочетание не равно ошибке. Ещё 17 FP стоят в других клетках. Одной причины у 32 ошибок нет. Среди самих FP более длинная речь и большее число окон идут с более высоким margin (Spearman 0.44 и 0.51, n = 32): часть ложных spoof просто ближе к порогу.

## Ответы

Межсегментное СКО LFCC и обоих CQCC на этом дне меньше у synthetic, CQCC c1 больше. Это видно после выравнивания RMS сегментов, в PACK1, в группах 2–4 и 5+ сегментов и в терцилях длительности речи. Это тот же 251 звонок, на котором кандидаты уже были видны. Нижний терциль RMS (6 synthetic) и PACK2 этот вывод не держат. У коротких звонков СКО LFCC и CQCC librosa имеет тот же знак, но интервал доходит до нуля; spafe и c1 там по-прежнему разделены.

Padding сам по себе меняет кепстр. Замена нулей повтором той же длины сдвигает c0 на десятки единиц, потому что нули — тишина. CQCC c1 сдвигается на единицы, без устойчивого знака. Это не пересчёт Spectra-0.

Сохранённый margin связан с долей повтора у человеческих звонков и ниже на хвостах, чем на полных окнах. Хвост — это ещё и конец сегмента. Для `8b34a750` хвосты как раз тянут margin к spoof, а полные окна уже выше 1.02. Для `a037652f` счёт целиком с хвостов и сильно bonafide, тогда как c1 реальных сегментов на стороне synthetic. Для двух длинных FN padding почти нет, и полные окна тоже выше порога. Padding может быть частью истории отдельных ошибок и не является общей причиной пяти пропусков.

32 FP ближе к TN по кепстру. Часть из них короче и с большим повтором хвоста, часть нет, а в самой «короткой и дополненной» клетке TN больше, чем FP.

## Что проверять на другом дне

Тот же код, без подгонки порога на дне 2026-11-13.

- Межсегментное СКО LFCC, CQCC librosa и CQCC spafe внутри заранее нарезанных терцилей RMS и длительности речи. Отдельно смотреть звонки с 2–4 сегментами: внутри класса СКО растёт с числом сегментов.
- CQCC c1 в тех же терцилях. На смеси этого дня Spearman с длительностью звонка 0.27, внутри класса около 0.1.
- PACK2 и любой терциль, где synthetic меньше десяти, не читать как проверку.
- Ошибки Spectra-0 заранее делить на короткую речь с высоким padding и остальные. Клетку сравнивать с TN в той же клетке.
- Для длинного многосегментного пропуска отдельно записывать margin полных окон и хвостов. У `8b34a750` эти две цифры разные.
"""

    readme = """# Кепстр и padding, день 2026-11-13

Проверка двух вопросов на уже посчитанном дне: межсегментная стабильность LFCC/CQCC и повтор хвоста в окнах Spectra-0. Модели заново не запускались и не обучались.

Читать сначала `summary.md`, затем `methodology.md`. CSV содержат только `call_id` как идентификатор звонка.

| Файл | Содержание |
| --- | --- |
| `summary.md` | Что видно на этом дне и чего из этого не следует |
| `methodology.md` | Откуда числа, как нарезаны группы, как устроен опыт с padding |
| `segment_level_cepstral_features.csv` | Средние кепстра каждого речевого сегмента и окна этого сегмента |
| `call_level_cepstral_summary.csv` | Межсегментное СКО, отдельный внутрисегментный MAD, margin и padding |
| `matched_group_analysis.csv` | Размеры групп, медианы, IQR, Cliff's delta и интервалы |
| `false_negative_segment_analysis.csv` | Пять пропусков по сегментам |
| `false_positive_padding_analysis.csv` | 32 FP, 151 TN и раскладка по терцилям |
| `padding_sensitivity.csv` | Реальный хвост, повтор и нули |
| `normalization_sensitivity.csv` | Без нормировки, пик и RMS |
| `visualizations/` | Margin, пять FN, padding и нормировка |

Повторный запуск из каталога `synthetic-speech-dataset` этого репозитория. Интерпретатор — его сосед `classical-anti-spoofing/.venv`, не копия вне репозитория:

```text
..\classical-anti-spoofing\.venv\Scripts\python.exe -m src.research.cepstral_padding_followup.run
```

`--resume` продолжает аудио, если прогон оборвался. Код не подключён к worker, backend и evaluation.
"""
    (OUTPUT / "methodology.md").write_text(methodology, encoding="utf-8")
    (OUTPUT / "summary.md").write_text(summary, encoding="utf-8")
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")
