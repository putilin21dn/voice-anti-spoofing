"""Правила очередей. Пороги и ширина пограничной зоны здесь зафиксированы."""

from __future__ import annotations

import math
import random


# Порог и направление не подбираются по 11–12 ноября.
# Synthetic для Spectra-0 и AASIST3, если score ниже порога.
# Synthetic для classical, если score не ниже порога.
THRESHOLDS = {
    "spectra0": 0.22710928916931153,
    "aasist3": 0.11397590637207031,
    "logistic_new": 0.7354826103642409,
    "svm_new": 0.0,
    "lightgbm_new": 0.3460832258410117,
    "logistic_old": 0.49265447022723596,
    "svm_old": -0.22144920850405767,
    "lightgbm_old": 0.04007147659674559,
}
BELOW = {"spectra0", "aasist3"}

# 0.10 * IQR score на 51 звонке валидации 13 ноября. На новых днях не пересчитывается.
WIDTHS = {
    "spectra0": 0.7157936337300472,
    "aasist3": 0.49268652335903246,
    "logistic_new": 0.06791828669328902,
    "svm_new": 0.1601107867594651,
    "lightgbm_new": 0.08333314010774318,
    "logistic_old": 0.06029181450819642,
    "svm_old": 0.14311365281200866,
    "lightgbm_old": 0.01511498268002695,
}

ANTISPOOF = (
    "spectra0",
    "aasist3",
    "logistic_new",
    "svm_new",
    "lightgbm_new",
    "logistic_old",
    "svm_old",
    "lightgbm_old",
)
WINDOW_SECONDS = 64600 / 16000
CONTROL_TARGET = 150
CONTROL_SEED = 20261111
PRIORITY_RANK = {"A": 1, "C": 2, "B": 3, "D": 4}


def finite_score(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def prediction(model: str, score: object, status: object) -> str:
    """Пустая строка, если решения нет. Отсутствие не заменяется на human."""

    if str(status or "") != "ok":
        return ""
    number = finite_score(score)
    if number is None:
        return ""
    threshold = THRESHOLDS[model]
    synthetic = number < threshold if model in BELOW else number >= threshold
    return "synthetic" if synthetic else "human"


def borderline(model: str, score: object, status: object) -> bool:
    if prediction(model, score, status) == "":
        return False
    number = finite_score(score)
    if number is None:
        return False
    return abs(number - THRESHOLDS[model]) <= WIDTHS[model]


def campaign_amd(classification: object) -> str:
    tokens = [item.strip() for item in str(classification or "").split("|") if item.strip()]
    if not tokens:
        return ""
    return "yes" if "автоответчик" in tokens else "no"


def flag(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return "yes"
    if text in {"0", "false", "no", "n"}:
        return "no"
    return ""


def disagreement_codes(predictions: dict[str, str], campaign: str, live: str) -> list[str]:
    codes: list[str] = []
    spectra = predictions.get("spectra0", "")
    light = predictions.get("lightgbm_new", "")
    if spectra and light and spectra != light:
        codes.append("spectra_lightgbm")
    if spectra and campaign and (spectra == "synthetic") != (campaign == "yes"):
        codes.append("spectra_campaign_amd")
    if light and campaign and (light == "synthetic") != (campaign == "yes"):
        codes.append("lightgbm_campaign_amd")
    if spectra and live and (spectra == "synthetic") != (live == "yes"):
        codes.append("spectra_live_amd")
    if light and live and (light == "synthetic") != (live == "yes"):
        codes.append("lightgbm_live_amd")
    available = [predictions[name] for name in ANTISPOOF if predictions.get(name)]
    if len(available) >= 2 and len(set(available)) > 1:
        codes.append("antispoof_models")
    for family in ("logistic", "svm", "lightgbm"):
        new = predictions.get(f"{family}_new", "")
        old = predictions.get(f"{family}_old", "")
        if new and old and new != old:
            codes.append(f"old_new_{family}")
    return codes


def technical_codes(row: dict) -> list[str]:
    codes: list[str] = []
    spectra_status = str(row.get("spectra_status") or "")
    classical_status = str(row.get("classical_status") or "")
    if spectra_status != "ok" or classical_status != "ok":
        codes.append("technical_status")
    segments = finite_score(row.get("spectra_n_segments"))
    if spectra_status == "ok" and segments is not None and segments <= 1:
        codes.append("technical_few_segments")
    speech = finite_score(row.get("spectra_speech_duration_s"))
    if spectra_status == "ok" and speech is not None and speech < WINDOW_SECONDS:
        codes.append("technical_short_speech")
    windows = finite_score(row.get("spectra_n_windows"))
    full_windows = finite_score(row.get("spectra_full_windows"))
    if spectra_status == "ok" and windows is not None and windows > 0 and full_windows == 0:
        codes.append("technical_all_padded")
    duration = finite_score(row.get("duration_seconds"))
    if duration is not None and duration < 15:
        codes.append("technical_short_audio")
    if str(row.get("client_channel") or "") == "":
        codes.append("technical_channel")
    return codes


def public_reasons(codes: list[str]) -> str:
    labels: list[str] = []
    if any(code in {"spectra_lightgbm", "antispoof_models"} or code.startswith("old_new_") for code in codes):
        labels.append("расхождение моделей")
    if any(code.endswith("_amd") for code in codes):
        labels.append("расхождение с определителем автоответчика")
    if any(code.startswith("borderline_") for code in codes):
        labels.append("пограничный score")
    technical = {
        "technical_status": "ошибка или неполная обработка",
        "technical_few_segments": "мало речевых сегментов",
        "technical_short_speech": "короткая речь",
        "technical_all_padded": "окна Spectra-0 собраны из повторённого хвоста",
        "technical_short_audio": "короткая запись",
        "technical_channel": "канал абонента не определён",
    }
    for code, label in technical.items():
        if code in codes:
            labels.append(label)
    if "control_sample" in codes:
        labels.append("контрольная выборка")
    return "; ".join(labels)


def priority_of(queues: set[str]) -> str:
    if not queues:
        return ""
    return min(queues, key=lambda name: PRIORITY_RANK[name])


def duration_band(duration: object) -> str:
    number = finite_score(duration)
    if number is None:
        return "unknown"
    if number < 30:
        return "lt30"
    if number < 60:
        return "30_59"
    return "ge60"


def sample_control(
    strata: dict[tuple, list[str]],
    *,
    target: int = CONTROL_TARGET,
    seed: int = CONTROL_SEED,
) -> dict[str, dict]:
    """Стратифицированная выборка. Малая страта берётся целиком, затем добор до target.

    Возвращает call_id -> вес. Вес = размер страты / число взятых из неё.
    """

    rng = random.Random(seed)
    chosen: dict[str, list[str]] = {}
    for key, members in sorted(strata.items()):
        ordered = sorted(set(members))
        if len(ordered) < 5:
            chosen[key] = ordered
            continue
        chosen[key] = []
    already = sum(len(items) for items in chosen.values())
    goal = max(target, already)
    leftovers = {
        key: [call_id for call_id in sorted(set(strata[key])) if call_id not in chosen[key]]
        for key in chosen
    }
    need = goal - already
    sizes = {key: len(leftovers[key]) for key in leftovers if leftovers[key]}
    allocation = {key: 0 for key in sizes}
    if sizes and need > 0:
        total = sum(sizes.values())
        if need >= total:
            allocation = dict(sizes)
        else:
            raw = {key: need * sizes[key] / total for key in sizes}
            for key, value in raw.items():
                allocation[key] = int(math.floor(value))
            remainder = need - sum(allocation.values())
            order = sorted(sizes, key=lambda key: (raw[key] - allocation[key], key), reverse=True)
            for key in order[:remainder]:
                allocation[key] += 1
            for key, count in list(allocation.items()):
                allocation[key] = min(count, sizes[key])
            # Если усечение по размеру страты уменьшило сумму, отдаём остаток тем, у кого ещё есть место.
            short = need - sum(allocation.values())
            if short > 0:
                room = sorted(
                    (key for key in sizes if allocation[key] < sizes[key]),
                    key=lambda key: (sizes[key] - allocation[key], key),
                    reverse=True,
                )
                for key in room:
                    if short == 0:
                        break
                    free = sizes[key] - allocation[key]
                    take = min(free, short)
                    allocation[key] += take
                    short -= take
    weights: dict[str, dict] = {}
    for key, members in chosen.items():
        extra = leftovers.get(key, [])
        take = allocation.get(key, 0)
        picked = rng.sample(extra, take) if take else []
        selected = members + picked
        population = len(set(strata[key]))
        weight = population / len(selected) if selected else ""
        for call_id in selected:
            weights[call_id] = {
                "sampling_weight": weight,
                "stratum_population": population,
                "stratum_sampled": len(selected),
                "stratum": "|".join(str(part) for part in key),
            }
    return weights
