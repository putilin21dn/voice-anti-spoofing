"""Три оси снимка. Приоритет их не перезаписывает."""

from __future__ import annotations

from dataclasses import dataclass

AUTOMATED = {"operator_voicemail", "automated_self_identification", "ai_assistant", "machine_phrase"}
SUSPICIOUS = {
    "subscriber_mention",
    "repeat",
    "stalling_bot",
    "explicit_assistant_claim",
    "secretary_role_claim",
    "repeated_nonresponsive_answers",
    "repeated_unanswered_question",
}


@dataclass(frozen=True)
class Axes:
    voice_origin: str
    behavior: str
    scenario_types: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "voice_origin": self.voice_origin,
            "behavior": self.behavior,
            "scenario_types": list(self.scenario_types),
        }


def voice_origin(ensemble_decision: str) -> str:
    if ensemble_decision == "synthetic":
        return "synthetic"
    if ensemble_decision == "human":
        return "human"
    return "unknown"


def behavior(signal_types: set[str], client_final_seen: bool) -> str:
    if signal_types & AUTOMATED:
        return "automated"
    if signal_types & SUSPICIOUS:
        return "suspicious"
    if client_final_seen and not signal_types:
        return "normal"
    return "unknown"


def scenario_types(signal_types: set[str], live_message_taker: bool) -> tuple[str, ...]:
    labels: list[str] = []
    if "operator_voicemail" in signal_types:
        labels.append("voicemail_ivr")
    if signal_types & {"automated_self_identification", "explicit_assistant_claim", "ai_assistant"}:
        labels.append("ai_assistant")
    if "machine_phrase" in signal_types and live_message_taker:
        labels.append("message_taker")
    if "secretary_role_claim" in signal_types:
        labels.append("secretary")
    if signal_types & {
        "repeat",
        "stalling_bot",
        "repeated_nonresponsive_answers",
        "repeated_unanswered_question",
    }:
        labels.append("behavioral")
    return tuple(labels)


def axes_of(ensemble_decision: str, signal_types: set[str], client_final_seen: bool, live_message_taker: bool) -> Axes:
    return Axes(
        voice_origin=voice_origin(ensemble_decision),
        behavior=behavior(signal_types, client_final_seen),
        scenario_types=scenario_types(signal_types, live_message_taker),
    )
