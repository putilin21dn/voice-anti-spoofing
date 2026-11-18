from __future__ import annotations

import unittest

from src.collect.identity import client_channel, room_suffix


class IdentityTest(unittest.TestCase):
    def test_suffix_keeps_mixed_token_and_drops_digits(self) -> None:
        self.assertEqual(room_suffix("room::79001234567_5M3ivKRBtBtp"), "5M3ivKRBtBtp")
        self.assertEqual(room_suffix("room::79001234567"), "")
        self.assertEqual(room_suffix("79001234567"), "")

    def test_excel_prefix_uses_the_other_channel(self) -> None:
        self.assertEqual(client_channel("excel-room", 2), "1")
        self.assertEqual(client_channel("d85904ea-room", 2), "0")
        self.assertEqual(client_channel("excel-room", 1), "0")


if __name__ == "__main__":
    unittest.main()
