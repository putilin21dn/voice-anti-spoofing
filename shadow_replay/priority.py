"""priority_of. Чистая функция снимка, без аудио, LLM и сессии звонка."""

from __future__ import annotations

from dataclasses import dataclass

from shadow_replay.constants import RULE_VERSION

HIGH = {
    "operator_voicemail",
    "automated_self_identification",
    "acoustic_synthetic_and_ai_assistant",
}
STRONG = {"operator_voicemail", "automated_self_identification", "ai_assistant"}
WEAK = {"machine_phrase", "subscriber_mention", "repeat", "stalling_bot"}
SUSPICION = {
    "explicit_assistant_claim",
    "secretary_role_claim",
    "repeated_nonresponsive_answers",
    "repeated_unanswered_question",
}


@dataclass(frozen=True)
class PriorityDecision:
    priority: str
    primary_reason: str
    reason_codes: tuple[str, ...]
    conflicts: tuple[str, ...]
    rule_version: str = RULE_VERSION

    def advice(self, shadow_session_id: str) -> dict:
        return {
            "shadow_session_id": shadow_session_id,
            "priority": self.priority,
            "additional_check": False,
            "termination_candidate": False,
            "reason_codes": list(self.reason_codes),
            "rule_version": self.rule_version,
            "action": "none",
        }


def priority_of(
    signal_types: set[str],
    voice_origin: str,
    *,
    final_operator_voicemail: bool,
    verdict: str | None,
    blocks_business: bool,
) -> PriorityDecision:
    has_self = "automated_self_identification" in signal_types
    has_ai = "ai_assistant" in signal_types
    has_voicemail = "operator_voicemail" in signal_types
    has_suspicion = bool(signal_types & SUSPICION)
    has_weak = bool(signal_types & WEAK)
    has_strong = bool(signal_types & STRONG)

    high: list[str] = []
    if final_operator_voicemail:
        high.append("operator_voicemail")
    if has_self:
        high.append("automated_self_identification")
    if has_ai and voice_origin == "synthetic":
        high.append("acoustic_synthetic_and_ai_assistant")

    codes = list(high)
    if not high:
        if has_ai and voice_origin in {"human", "unknown"}:
            codes.append("strong_text_without_acoustic_agreement")
        if voice_origin == "synthetic" and not has_voicemail and not has_self and not has_ai:
            codes.append("acoustic_synthetic_without_strong_text")
        if has_suspicion and not has_self:
            codes.append("human_check_suspicion")
        if has_weak:
            codes.append("weak_text")
    elif has_suspicion and not has_self:
        codes.append("human_check_suspicion")

    if codes and codes[0] in HIGH:
        priority = "high"
        primary = codes[0]
    elif codes:
        priority = "medium"
        primary = codes[0]
    elif voice_origin == "human" and not has_strong and not has_weak and not has_suspicion:
        priority = "low"
        primary = "acoustic_human_no_suspicion"
        codes = [primary]
    else:
        priority = "undetermined"
        primary = "insufficient_information"
        codes = [primary]

    conflicts: list[str] = []
    if has_self and verdict in {"no_suspicion", "human_claim"}:
        conflicts.append("C3")
    if final_operator_voicemail and blocks_business:
        conflicts.append("C7")
    return PriorityDecision(priority, primary, tuple(codes), tuple(conflicts))
