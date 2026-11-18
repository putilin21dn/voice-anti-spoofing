import unittest

from shadow_replay.acoustic import high_confidence_synthetic
from shadow_replay.ui_marks import interface_label, resulting_mark, text_split


class InterfaceMarkTests(unittest.TestCase):
    def test_stored_codes_use_ui_captions(self) -> None:
        self.assertEqual(interface_label("отказ_лпр"), "Отказ ЛПР")
        self.assertEqual(interface_label("асинхронная_связь"), "Связь без звонка")
        self.assertEqual(interface_label("anti_spam_bot"), "Антиспам-бот")
        self.assertEqual(interface_label("автоответчик"), "Автоответчик")
        self.assertEqual(interface_label(""), "Неизвестно")

    def test_non_synthetic_keeps_the_interface_mark(self) -> None:
        self.assertEqual(resulting_mark("лид", "human"), "Лид")
        self.assertEqual(resulting_mark("не_лпр", "no_decision"), "Не ЛПР")
        self.assertEqual(resulting_mark("абонент_молчит", "human"), "Абонент молчит")

    def test_synthetic_uses_text_split(self) -> None:
        self.assertEqual(text_split("anti_spam_bot"), "anti_spam")
        self.assertEqual(text_split("автоответчик"), "answering_machine")
        self.assertEqual(text_split("отказ_лпр"), "")
        self.assertEqual(resulting_mark("anti_spam_bot", "synthetic"), "Антиспам-бот")
        self.assertEqual(resulting_mark("автоответчик", "synthetic"), "Автоответчик")
        self.assertEqual(resulting_mark("лид", "synthetic"), "Лид")
        self.assertEqual(resulting_mark("лид", "synthetic", high_confidence=True), "Автоответчик")
        self.assertEqual(resulting_mark("anti_spam_bot", "synthetic", high_confidence=True), "Антиспам-бот")

    def test_high_confidence_needs_three_clear_synthetic_votes(self) -> None:
        clear = {
            "spectra0": (-1.0, "ok"),
            "svm": (1.0, "ok"),
            "lightgbm": (0.9, "ok"),
        }
        self.assertTrue(high_confidence_synthetic(clear))
        split = {**clear, "svm": (-0.5, "ok")}
        self.assertFalse(high_confidence_synthetic(split))
        near = {**clear, "lightgbm": (0.35, "ok")}
        self.assertFalse(high_confidence_synthetic(near))
        missing = {**clear, "spectra0": (-1.0, "no_speech")}
        self.assertFalse(high_confidence_synthetic(missing))

    def test_anti_spam_wins_when_both_are_present(self) -> None:
        stored = "автоответчик|anti_spam_bot"
        self.assertEqual(resulting_mark(stored, "synthetic"), "Антиспам-бот")


if __name__ == "__main__":
    unittest.main()
