import unittest

from src.data.outcome_check import outcome_bucket


class OutcomeBucketTests(unittest.TestCase):
    def test_agreement_and_conflict(self):
        self.assertEqual(outcome_bucket("machine", "answering_machine"), "consistent")
        self.assertEqual(outcome_bucket("human", "refused"), "consistent")
        self.assertEqual(outcome_bucket("machine", "b2x_lead"), "conflict")
        self.assertEqual(outcome_bucket("human", "answering_machine"), "conflict")
        self.assertEqual(outcome_bucket("human", ""), "empty")
        self.assertEqual(outcome_bucket("machine", "silence"), "ambiguous")


if __name__ == "__main__":
    unittest.main()
