"""Зафиксированные константы unified-priority-v1.

Пороги совпадают с shadow-спецификацией и с rules.THRESHOLDS.
Окно Spectra-0 — 64600 сэмплов после перевода речи на 16 кГц.
"""

from __future__ import annotations

RULE_VERSION = "unified-priority-v1"
SCHEMA_VERSION = 1
TEXT_RULES_VERSION = "observed-predicates-v1"
MODEL_VERSION = "spectra_0+svm_new+lightgbm_new"
THRESHOLD_VERSION = "spectra0<0.22710928916931153,svm>=0,lightgbm>=0.3460832258410117"

SPECTRA0_THRESHOLD = 0.22710928916931153
SVM_NEW_THRESHOLD = 0.0
LIGHTGBM_NEW_THRESHOLD = 0.3460832258410117

# 0.10 IQR на валидации 13 ноября, те же числа, что rules.WIDTHS.
# Полоса не меняет голос A. Она только решает, достаточно ли synthetic
# уверен, чтобы заменить чип на «Автоответчик».
BORDER_WIDTHS = {
    "spectra0": 0.7157936337300472,
    "svm": 0.1601107867594651,
    "lightgbm": 0.08333314010774318,
}

WINDOW_SAMPLES = 64600
SAMPLE_RATE = 16000

FORBIDDEN_KEYS = frozenset(
    {
        "phone",
        "phone_number",
        "external_id",
        "s3_key",
        "s3_bucket",
        "msisdn",
        "room_name",
        "room",
        "text",
        "transcript",
        "normalized_text",
        "legacy",
        "question_text",
        "answer_text",
    }
)
