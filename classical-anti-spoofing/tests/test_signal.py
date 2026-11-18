import unittest

from src.utils.config import load_config
from src.utils.validation import run_synthetic_checks


class SignalChecks(unittest.TestCase):
    def test_synthetic_signal_and_bandlimit(self):
        checks = run_synthetic_checks(load_config())
        failed = [item for item in checks if not item["passed"]]
        self.assertEqual(failed, [])


if __name__ == "__main__":
    unittest.main()
