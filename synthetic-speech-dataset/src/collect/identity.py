from __future__ import annotations

import re


SUFFIX_RE = re.compile(r"^(?=.*[A-Za-z])[A-Za-z0-9]{12}$")


def room_suffix(external_id: str) -> str:
    """Хвост комнаты после телефона. Чистый набор цифр не сохраняется."""

    token = (external_id or "").rsplit("_", 1)[-1]
    if SUFFIX_RE.fullmatch(token):
        return token
    return ""


def client_channel(external_id: str, channels: int) -> str:
    if channels <= 1:
        return "0"
    if (external_id or "").startswith("excel-"):
        return "1"
    return "0"
