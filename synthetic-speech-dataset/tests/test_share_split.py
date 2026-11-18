from __future__ import annotations

import unittest

from src.annotation.share import build_priority_assignment


def row(call_id: str, priority: str, index: int) -> dict[str, str]:
    return {
        "call_id": call_id,
        "review_priority": priority,
        "date": "2026-11-11",
        "campaign": "B2X-11-11-2026-PACK1",
        "sort_index": str(index),
    }


class ShareSplitTest(unittest.TestCase):
    def test_listened_calls_stay_local_and_the_rest_alternates(self) -> None:
        queue = [row(f"00000000-0000-4000-8000-00000000000{index}", "A", index) for index in range(1, 6)]
        labeled = {queue[0]["call_id"]}
        assigned = build_priority_assignment(queue, labeled, [item["call_id"] for item in queue])
        self.assertEqual(assigned[queue[0]["call_id"]], "local")
        self.assertEqual(
            [assigned[item["call_id"]] for item in queue[1:]],
            ["local", "guest", "local", "guest"],
        )

    def test_existing_guest_call_is_not_taken_back(self) -> None:
        item = row("00000000-0000-4000-8000-0000000000aa", "C", 3)
        assigned = build_priority_assignment(
            [item],
            {item["call_id"]},
            [item["call_id"]],
            {item["call_id"]: "guest"},
        )
        self.assertEqual(assigned[item["call_id"]], "guest")

    def test_calls_outside_the_queue_stay_local(self) -> None:
        extra = "00000000-0000-4000-8000-0000000000bb"
        assigned = build_priority_assignment([], set(), [extra])
        self.assertEqual(assigned[extra], "local")


if __name__ == "__main__":
    unittest.main()
