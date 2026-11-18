"""Отметка прогона в подписях интерфейса B2X.

Чужая отметка сохраняется. Synthetic меняет её на «Автоответчик» или
«Антиспам-бот», если текстовая классификация уже назвала один из этих двух
исходов. Если текст их не разделил, чип становится «Автоответчик» только при
высокой уверенности голоса: три модели за synthetic и каждая дальше
пограничной полосы.
"""

from __future__ import annotations

UI_BY_STORED = {
    "лид": "Лид",
    "интерес_без_данных": "Интерес без данных",
    "отказ_лпр": "Отказ ЛПР",
    "не_лпр": "Не ЛПР",
    "не_та_организация": "Не та организация",
    "перезвон": "Перезвон",
    "асинхронная_связь": "Связь без звонка",
    "автоответчик": "Автоответчик",
    "answering_machine": "Автоответчик",
    "anti_spam_bot": "Антиспам-бот",
    "антиспам-бот": "Антиспам-бот",
    "human_check_unverified": "Не удалось подтвердить",
    "абонент_молчит": "Абонент молчит",
    "положил_трубку": "Положил трубку на приветствии",
    "не_взял_трубку": "Не взял трубку",
    "неизвестно": "Неизвестно",
}

ANSWERING_MACHINE = "Автоответчик"
ANTI_SPAM_BOT = "Антиспам-бот"
_MACHINE = {ANSWERING_MACHINE}
_ANTI_SPAM = {ANTI_SPAM_BOT}


def stored_tokens(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split("|") if item.strip()]


def interface_label(stored: str) -> str:
    labels = []
    for token in stored_tokens(stored):
        key = token.casefold().replace(" ", "_")
        label = UI_BY_STORED.get(key) or UI_BY_STORED.get(token.casefold())
        labels.append(label or token)
    if not labels:
        return "Неизвестно"
    return " | ".join(dict.fromkeys(labels))


def text_split(stored: str) -> str:
    """anti_spam, answering_machine или пустая строка, если текст их не разделил."""

    labels = set(interface_label(stored).split(" | "))
    if labels & _ANTI_SPAM:
        return "anti_spam"
    if labels & _MACHINE:
        return "answering_machine"
    return ""


def resulting_mark(stored: str, ensemble: str, *, high_confidence: bool = False) -> str:
    current = interface_label(stored)
    if ensemble != "synthetic":
        return current
    split = text_split(stored)
    if split == "anti_spam":
        return ANTI_SPAM_BOT
    if split == "answering_machine":
        return ANSWERING_MACHINE
    if high_confidence:
        return ANSWERING_MACHINE
    return current
