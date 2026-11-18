"""Проигрывание записанного звонка по часам.

На секунде t видны только события с elapsed_s <= t. Конец файла не подглядывается.
Фразы и веса в этот модуль не встроены: лента приносит индексы уже совпавших
предикатов и score уже закрытого префикса. Действие всегда none.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

from shadow_replay.acoustic import AcousticTrack
from shadow_replay.constants import (
    FORBIDDEN_KEYS,
    MODEL_VERSION,
    RULE_VERSION,
    SCHEMA_VERSION,
    TEXT_RULES_VERSION,
    THRESHOLD_VERSION,
)
from shadow_replay.fusion import axes_of
from shadow_replay.priority import priority_of
from shadow_replay.signals import HumanCheck, SignalLog


def reject_forbidden(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN_KEYS:
                raise ValueError(
                    f"Поле {key} в ленту и в shadow-событие не кладётся. "
                    "Нужны индексы предикатов и score, без текста реплики и без идентификатора абонента."
                )
            reject_forbidden(item)
    elif isinstance(value, list):
        for item in value:
            reject_forbidden(item)


def order_events(events: list[dict]) -> list[dict]:
    indexed = list(enumerate(events))
    indexed.sort(key=lambda item: (float(item[1]["elapsed_s"]), item[0]))
    return [item[1] for item in indexed]


class Replay:
    def __init__(
        self,
        call_id: str | None = None,
        shadow_session_id: str | None = None,
        started_at: str = "2026-01-01T00:00:00+00:00",
    ) -> None:
        self.call_id = call_id
        self.shadow_session_id = shadow_session_id or str(uuid.uuid4())
        self.started_at = datetime.fromisoformat(started_at)
        self.signals = SignalLog()
        self.human = HumanCheck()
        self.acoustic = AcousticTrack()
        self.events: list[dict] = []
        self._seq = 0
        self._signature: tuple | None = None
        self.dropped_signal = False

    def play(self, events: list[dict]) -> list[dict]:
        for event in order_events(events):
            self.push(event)
        return list(self.events)

    def push(self, event: dict) -> dict | None:
        reject_forbidden(event)
        kind = event.get("kind")
        elapsed_s = float(event["elapsed_s"])
        if kind == "audio":
            recording = float(event.get("recording_elapsed_s", elapsed_s))
            if event.get("dropped"):
                self.acoustic.dropped_audio = True
                self.acoustic.client_audio_seen = True
                self.acoustic.recording_elapsed_s = recording
                return self._emit(elapsed_s, "spectra_window", force=False)
            closed = self.acoustic.add_speech(
                int(event.get("speech_samples", 0)),
                event.get("spectra_margin"),
                str(event.get("spectra_status", "ok")),
                recording,
            )
            if not closed:
                return None
            self.acoustic.note(elapsed_s)
            return self._emit(elapsed_s, "spectra_window", force=False)
        if kind == "classical":
            self.acoustic.set_classical(
                event.get("svm_score"),
                str(event.get("svm_status", "ok")),
                event.get("lightgbm_score"),
                str(event.get("lightgbm_status", "ok")),
            )
            self.acoustic.note(elapsed_s)
            return self._emit(elapsed_s, "classical_segment", force=False)
        if kind == "final":
            update_kind = self.signals.observe(event)
            self.human.update(event.get("human_check"))
            return self._emit(elapsed_s, update_kind, force=False)
        if kind == "call_end":
            return self._emit(elapsed_s, "call_end", force=True)
        raise ValueError(f"Неизвестный кадр ленты: {kind}")

    def advice(self) -> dict:
        return self._decision().advice(self.shadow_session_id)

    def _decision(self):
        acoustic = self.acoustic.snapshot(padded_tail=None)
        built = axes_of(
            acoustic["ensemble_decision"],
            self.signals.types(),
            self.signals.client_final_seen,
            self.signals.live_message_taker(),
        )
        blocks = bool(self.human.blocks_business)
        decision = priority_of(
            self.signals.types(),
            built.voice_origin,
            final_operator_voicemail=self.signals.final_operator_voicemail(),
            verdict=self.human.verdict,
            blocks_business=blocks,
        )
        return decision

    def _emit(self, elapsed_s: float, update_kind: str, force: bool) -> dict | None:
        acoustic = self.acoustic.snapshot(padded_tail=self.acoustic.pending_samples > 0 if update_kind == "call_end" else None)
        built = axes_of(
            acoustic["ensemble_decision"],
            self.signals.types(),
            self.signals.client_final_seen,
            self.signals.live_message_taker(),
        )
        blocks = bool(self.human.blocks_business)
        decision = priority_of(
            self.signals.types(),
            built.voice_origin,
            final_operator_voicemail=self.signals.final_operator_voicemail(),
            verdict=self.human.verdict,
            blocks_business=blocks,
        )
        signature = (
            built.voice_origin,
            built.behavior,
            built.scenario_types,
            decision.priority,
            decision.reason_codes,
            decision.conflicts,
        )
        if not force and signature == self._signature:
            return None
        self._signature = signature
        self._seq += 1
        row = {
            "schema_version": SCHEMA_VERSION,
            "shadow_session_id": self.shadow_session_id,
            "call_id": self.call_id,
            "event_timestamp": (self.started_at + timedelta(seconds=elapsed_s)).isoformat(),
            "elapsed_s": elapsed_s,
            "recording_elapsed_s": self.acoustic.recording_elapsed_s,
            "speech_elapsed_s": self.acoustic.speech_elapsed_s,
            "update_kind": update_kind,
            "update_seq": self._seq,
            "text_signals": self.signals.public(blocks),
            "human_check": self.human.as_dict(),
            "acoustic": acoustic,
            "fusion": {**built.as_dict(), **self.acoustic.stability.as_dict()},
            "priority": decision.priority,
            "primary_reason": decision.primary_reason,
            "reason_codes": list(decision.reason_codes),
            "conflicts": list(decision.conflicts),
            "rule_version": RULE_VERSION,
            "text_rules_version": TEXT_RULES_VERSION,
            "model_version": MODEL_VERSION,
            "threshold_version": THRESHOLD_VERSION,
            "processing_delay_ms": None,
            "dropped_audio": self.acoustic.dropped_audio,
            "dropped_signal": self.dropped_signal,
            "errors": self._errors(),
        }
        reject_forbidden(row)
        self.events.append(row)
        return row

    def _errors(self) -> list[dict]:
        found = []
        for component, status in (
            ("spectra0", self.acoustic.spectra_status),
            ("svm", self.acoustic.svm_status),
            ("lightgbm", self.acoustic.lightgbm_status),
        ):
            if status in {"error", "timeout"}:
                found.append({"component": component, "code": status})
        return found


def play_tape(tape: dict) -> list[dict]:
    reject_forbidden(tape)
    replay = Replay(
        call_id=tape.get("call_id"),
        shadow_session_id=tape.get("shadow_session_id"),
        started_at=str(tape.get("started_at", "2026-01-01T00:00:00+00:00")),
    )
    return replay.play(list(tape.get("events") or []))
