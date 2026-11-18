"""Голос A по уже закрытым окнам. Будущие кадры сюда не входят."""

from __future__ import annotations

from dataclasses import dataclass, field

from shadow_replay.constants import (
    BORDER_WIDTHS,
    LIGHTGBM_NEW_THRESHOLD,
    SAMPLE_RATE,
    SPECTRA0_THRESHOLD,
    SVM_NEW_THRESHOLD,
    WINDOW_SAMPLES,
)

_THRESHOLDS = {
    "spectra0": SPECTRA0_THRESHOLD,
    "svm": SVM_NEW_THRESHOLD,
    "lightgbm": LIGHTGBM_NEW_THRESHOLD,
}


def _finite(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def model_decision(model: str, score: float | None, status: str) -> str | None:
    """synthetic / human, либо None если оценки нет. Пустое не заменяется на human."""

    if status != "ok":
        return None
    number = _finite(score)
    if number is None:
        return None
    if model == "spectra0":
        return "synthetic" if number < SPECTRA0_THRESHOLD else "human"
    if model == "svm":
        return "synthetic" if number >= SVM_NEW_THRESHOLD else "human"
    if model == "lightgbm":
        return "synthetic" if number >= LIGHTGBM_NEW_THRESHOLD else "human"
    raise KeyError(model)


def high_confidence_synthetic(scores: dict[str, tuple[object, str]]) -> bool:
    """Три модели сказали synthetic, и каждая дальше пограничной полосы.

    Два голоса из трёх и голос внутри полосы сюда не входят.
    """

    if set(scores) != set(_THRESHOLDS):
        return False
    for model, (score, status) in scores.items():
        if model_decision(model, score, status) != "synthetic":
            return False
        number = _finite(score)
        if number is None or abs(number - _THRESHOLDS[model]) <= BORDER_WIDTHS[model]:
            return False
    return True


@dataclass
class Stability:
    decision_changes: int = 0
    no_decision_gaps: int = 0
    updates_with_two_votes: int = 0
    updates_since_last_change: int = 0
    first_decision_elapsed_s: float | None = None
    first_synthetic_elapsed_s: float | None = None
    current_decision_since_elapsed_s: float | None = None
    last_binary: str | None = None
    in_gap: bool = False

    def observe(self, decision: str, valid: int, elapsed_s: float) -> None:
        if valid >= 2:
            self.updates_with_two_votes += 1
            if self.first_decision_elapsed_s is None:
                self.first_decision_elapsed_s = elapsed_s
        if decision == "synthetic" and self.first_synthetic_elapsed_s is None:
            self.first_synthetic_elapsed_s = elapsed_s
        if decision == "no_decision":
            if self.last_binary is not None:
                self.in_gap = True
            return

        came_from_gap = self.in_gap
        self.in_gap = False
        if came_from_gap:
            self.no_decision_gaps += 1
        if self.last_binary is None:
            self.last_binary = decision
            self.current_decision_since_elapsed_s = elapsed_s
            self.updates_since_last_change = 0
            return
        if decision != self.last_binary:
            if not came_from_gap:
                self.decision_changes += 1
            self.last_binary = decision
            self.current_decision_since_elapsed_s = elapsed_s
            self.updates_since_last_change = 0
            return
        if valid >= 2:
            self.updates_since_last_change += 1

    def as_dict(self) -> dict:
        return {
            "decision_changes": self.decision_changes,
            "no_decision_gaps": self.no_decision_gaps,
            "updates_with_two_votes": self.updates_with_two_votes,
            "updates_since_last_change": self.updates_since_last_change,
            "first_decision_elapsed_s": self.first_decision_elapsed_s,
            "first_synthetic_elapsed_s": self.first_synthetic_elapsed_s,
            "current_decision_since_elapsed_s": self.current_decision_since_elapsed_s,
        }


@dataclass
class AcousticTrack:
    """Копит речь и публикует score только по уже закрытому префиксу."""

    pending_samples: int = 0
    closed_samples: int = 0
    client_audio_seen: bool = False
    recording_elapsed_s: float | None = None
    margins: list[float] = field(default_factory=list)
    publish_spectra: bool = False
    spectra_status: str = "missing"
    svm_score: float | None = None
    svm_status: str = "missing"
    lightgbm_score: float | None = None
    lightgbm_status: str = "missing"
    dropped_audio: bool = False
    stability: Stability = field(default_factory=Stability)

    @property
    def speech_elapsed_s(self) -> float:
        return self.closed_samples / SAMPLE_RATE

    def add_speech(self, samples: int, margin: object, status: str, recording_elapsed_s: float) -> bool:
        if samples < 0:
            raise ValueError("speech_samples не может быть отрицательным")
        self.client_audio_seen = True
        self.recording_elapsed_s = recording_elapsed_s
        self.pending_samples += samples
        if self.pending_samples < WINDOW_SAMPLES:
            if margin is not None:
                raise ValueError("spectra_margin можно передавать только вместе с закрытием окна в 64600 сэмплов")
            return False
        windows = self.pending_samples // WINDOW_SAMPLES
        if windows > 1:
            raise ValueError(
                "Один кадр закрывает больше одного окна Spectra-0. Разбейте ленту по 64600 речевых сэмплов."
            )
        self.pending_samples -= WINDOW_SAMPLES
        self.closed_samples += WINDOW_SAMPLES
        self._apply_spectra(margin, status)
        return True

    def _apply_spectra(self, margin: object, status: str) -> None:
        if status in {"error", "timeout"}:
            self.publish_spectra = False
            self.spectra_status = status
            return
        number = _finite(margin)
        if status != "ok" or number is None:
            self.publish_spectra = False
            self.spectra_status = "missing"
            return
        self.margins.append(number)
        self.publish_spectra = True
        self.spectra_status = "ok"

    def set_classical(
        self,
        svm_score: object,
        svm_status: str,
        lightgbm_score: object,
        lightgbm_status: str,
    ) -> None:
        self.svm_score, self.svm_status = self._publish(svm_score, svm_status)
        self.lightgbm_score, self.lightgbm_status = self._publish(lightgbm_score, lightgbm_status)

    @staticmethod
    def _publish(score: object, status: str) -> tuple[float | None, str]:
        if status in {"error", "timeout", "missing"}:
            return None, status
        number = _finite(score)
        if status != "ok" or number is None:
            return None, "missing"
        return number, "ok"

    def spectra_score(self) -> float | None:
        if not self.publish_spectra or not self.margins:
            return None
        return sum(self.margins) / len(self.margins)

    def note(self, elapsed_s: float) -> None:
        """Один акустический апдейт. Повторный снимок для текста стабильность не двигает."""

        _spectra, _spectra_decision, _svm, _lgbm, ensemble, _reason, votes = self._parts()
        self.stability.observe(ensemble, votes, elapsed_s)

    def snapshot(self, padded_tail: bool | None) -> dict:
        spectra_score, spectra_decision, svm_decision, lightgbm_decision, ensemble, reason, votes = self._parts()
        update_end_ms = None if self.recording_elapsed_s is None else int(round(self.recording_elapsed_s * 1000))
        return {
            "spectra0_score": spectra_score,
            "spectra0_decision": spectra_decision,
            "svm_score": self.svm_score,
            "svm_decision": svm_decision,
            "lightgbm_score": self.lightgbm_score,
            "lightgbm_decision": lightgbm_decision,
            "ensemble_decision": ensemble,
            "no_decision_reason": reason,
            "valid_vote_count": votes,
            "padded_tail": padded_tail,
            "update_end_ms": update_end_ms,
        }

    def _parts(self) -> tuple:
        spectra_score = self.spectra_score()
        spectra_decision = model_decision(
            "spectra0",
            spectra_score,
            "ok" if spectra_score is not None else self.spectra_status,
        )
        svm_decision = model_decision("svm", self.svm_score, self.svm_status)
        lightgbm_decision = model_decision("lightgbm", self.lightgbm_score, self.lightgbm_status)
        decisions = (spectra_decision, svm_decision, lightgbm_decision)
        votes = len([item for item in decisions if item is not None])
        statuses = (self.spectra_status, self.svm_status, self.lightgbm_status)
        if votes >= 2:
            synthetic_votes = sum(item == "synthetic" for item in decisions if item is not None)
            ensemble = "synthetic" if synthetic_votes >= 2 else "human"
            reason = None
        elif (
            not self.client_audio_seen
            and self.svm_score is None
            and self.lightgbm_score is None
            and "error" not in statuses
            and "timeout" not in statuses
        ):
            ensemble = "no_decision"
            reason = "no_client_audio"
        elif "timeout" in statuses and votes < 2:
            ensemble = "no_decision"
            reason = "timeout"
        elif "error" in statuses and votes < 2:
            ensemble = "no_decision"
            reason = "model_error"
        elif not self.margins and self.svm_score is None and self.lightgbm_score is None:
            ensemble = "no_decision"
            reason = "insufficient_audio"
        else:
            ensemble = "no_decision"
            reason = "insufficient_models"
        return spectra_score, spectra_decision, svm_decision, lightgbm_decision, ensemble, reason, votes
