"""Каталог сигналов по уже наблюдаемым предикатам.

Список фраз здесь не хранится. Лента приносит факт, что существующий предикат
совпал, и индексы строк кортежа. Текст реплики в снимок не входит.
repeat и stall читаются как состояние после детектора, второй раз не считаются.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TextSignal:
    signal_id: int
    type: str
    source: str
    strength: str
    is_final: bool
    finals_checked: int
    live_message_taker: bool
    reason_code: str
    pattern_indexes: list[int] = field(default_factory=list)
    elapsed_s: float = 0.0
    last_elapsed_s: float = 0.0
    count: int = 1
    repeat_count: int | None = None
    repeat_words: int | None = None
    weak_marker_indexes: list[int] = field(default_factory=list)
    strong_marker_indexes: list[int] = field(default_factory=list)

    def public(self, blocks_business: bool) -> dict:
        hangup_eligible = self.type == "operator_voicemail" and self.is_final and not blocks_business
        gates: list[str] = []
        if blocks_business:
            gates.append("blocked_by_human_check")
        if not self.is_final:
            gates.append("not_final")
        payload = {
            "signal_id": self.signal_id,
            "type": self.type,
            "source": self.source,
            "strength": self.strength,
            "is_final": self.is_final,
            "finals_checked": self.finals_checked,
            "live_message_taker": self.live_message_taker,
            "gates": gates,
            "hangup_eligible": hangup_eligible,
            "detector_kind": None,
            "reason_code": self.reason_code,
            "pattern_indexes": list(self.pattern_indexes),
            "elapsed_s": self.elapsed_s,
            "last_elapsed_s": self.last_elapsed_s,
            "count": self.count,
        }
        if self.type == "repeat":
            payload["repeat_count"] = self.repeat_count
            payload["repeat_words"] = self.repeat_words
        if self.type == "stalling_bot":
            payload["weak_marker_indexes"] = list(self.weak_marker_indexes)
            payload["strong_marker_indexes"] = list(self.strong_marker_indexes)
        return payload


@dataclass
class HumanCheck:
    mode: str = "off"
    state: str | None = None
    blocks_business: bool | None = None
    trigger: str | None = None
    suspicion_verdict: str | None = None
    verdict: str | None = None
    attempts: int | None = None
    snapshot_status: str = "ok"

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "state": self.state,
            "blocks_business": self.blocks_business,
            "trigger": self.trigger,
            "suspicion_verdict": self.suspicion_verdict,
            "verdict": self.verdict,
            "attempts": self.attempts,
            "snapshot_status": self.snapshot_status,
        }

    def update(self, payload: dict | None) -> None:
        if not payload:
            return
        for key in self.as_dict():
            if key in payload:
                setattr(self, key, payload[key])


class SignalLog:
    def __init__(self) -> None:
        self._signals: dict[str, TextSignal] = {}
        self._order: list[str] = []
        self._next_id = 1
        self.finals_checked = 0
        self.client_final_seen = False

    def observe(self, event: dict) -> str:
        elapsed_s = float(event["elapsed_s"])
        is_final = bool(event.get("is_final", False))
        if is_final:
            self.finals_checked += 1
            self.client_final_seen = True
        live = bool(event.get("live_message_taker", False))
        before = self._fingerprint()
        self._mark(
            "operator_voicemail",
            event.get("operator_voicemail_indexes") or [],
            "is_operator_voicemail_text",
            "strong",
            "operator_phrase_index",
            is_final,
            elapsed_s,
            live,
        )
        self._mark(
            "machine_phrase",
            event.get("machine_phrase_indexes") or [],
            "is_answering_machine_text",
            "weak",
            "machine_phrase_index",
            is_final,
            elapsed_s,
            live,
        )
        if event.get("subscriber_mention"):
            self._add(
                "subscriber_mention",
                "mentions_subscriber",
                "weak",
                "subscriber_word",
                is_final,
                elapsed_s,
                live,
                [],
            )
        self._mark(
            "ai_assistant",
            event.get("ai_assistant_indexes") or [],
            "is_ai_assistant_text",
            "strong",
            "ai_phrase_index",
            is_final,
            elapsed_s,
            live,
        )
        if event.get("automated_self_identification"):
            self._add(
                "automated_self_identification",
                "is_automated_self_identification",
                "strong",
                "self_identification_passed",
                is_final,
                elapsed_s,
                live,
                [],
            )
        for name in (
            "explicit_assistant_claim",
            "secretary_role_claim",
            "repeated_nonresponsive_answers",
            "repeated_unanswered_question",
        ):
            if event.get(name):
                self._add(name, "SuspicionTriggers", "suspicion", name, is_final, elapsed_s, live, [])
        repeat = event.get("repeat")
        if isinstance(repeat, dict):
            self._add(
                "repeat",
                "_repeated_final",
                "weak",
                "repeat_count",
                is_final,
                elapsed_s,
                live,
                [],
                repeat_count=repeat.get("count"),
                repeat_words=repeat.get("words"),
            )
        stall = event.get("stall")
        if isinstance(stall, dict):
            self._add(
                "stalling_bot",
                "_stalling_bot",
                "weak",
                "stall_marker_indexes",
                is_final,
                elapsed_s,
                live,
                [],
                weak_marker_indexes=list(stall.get("weak_marker_indexes") or []),
                strong_marker_indexes=list(stall.get("strong_marker_indexes") or []),
            )
        if event.get("human_check") and self._fingerprint() == before:
            return "human_check"
        return "text_signal"

    def public(self, blocks_business: bool) -> list[dict]:
        blocked = bool(blocks_business)
        return [self._signals[name].public(blocked) for name in self._order]

    def types(self) -> set[str]:
        return set(self._signals)

    def final_operator_voicemail(self) -> bool:
        signal = self._signals.get("operator_voicemail")
        return bool(signal and signal.is_final)

    def live_message_taker(self) -> bool:
        signal = self._signals.get("machine_phrase")
        return bool(signal and signal.live_message_taker)

    def _fingerprint(self) -> tuple:
        rows = []
        for name in self._order:
            signal = self._signals[name]
            rows.append((name, signal.count, signal.is_final, tuple(signal.pattern_indexes), signal.live_message_taker))
        return tuple(rows)

    def _mark(
        self,
        name: str,
        indexes: object,
        source: str,
        strength: str,
        reason_code: str,
        is_final: bool,
        elapsed_s: float,
        live: bool,
    ) -> None:
        if not isinstance(indexes, list) or not indexes:
            return
        self._add(name, source, strength, reason_code, is_final, elapsed_s, live, [int(item) for item in indexes])

    def _add(
        self,
        name: str,
        source: str,
        strength: str,
        reason_code: str,
        is_final: bool,
        elapsed_s: float,
        live: bool,
        indexes: list[int],
        repeat_count: int | None = None,
        repeat_words: int | None = None,
        weak_marker_indexes: list[int] | None = None,
        strong_marker_indexes: list[int] | None = None,
    ) -> None:
        current = self._signals.get(name)
        if current is None:
            current = TextSignal(
                signal_id=self._next_id,
                type=name,
                source=source,
                strength=strength,
                is_final=is_final,
                finals_checked=self.finals_checked,
                live_message_taker=live,
                reason_code=reason_code,
                pattern_indexes=list(indexes),
                elapsed_s=elapsed_s,
                last_elapsed_s=elapsed_s,
                repeat_count=repeat_count,
                repeat_words=repeat_words,
                weak_marker_indexes=list(weak_marker_indexes or []),
                strong_marker_indexes=list(strong_marker_indexes or []),
            )
            self._next_id += 1
            self._signals[name] = current
            self._order.append(name)
            return
        current.count += 1
        current.last_elapsed_s = elapsed_s
        current.is_final = current.is_final or is_final
        current.live_message_taker = current.live_message_taker or live
        current.finals_checked = self.finals_checked
        merged = set(current.pattern_indexes)
        merged.update(indexes)
        current.pattern_indexes = sorted(merged)
        if repeat_count is not None:
            current.repeat_count = repeat_count
            current.repeat_words = repeat_words
        if weak_marker_indexes is not None:
            current.weak_marker_indexes = list(weak_marker_indexes)
        if strong_marker_indexes is not None:
            current.strong_marker_indexes = list(strong_marker_indexes)
