"""Offline checks of the Room 2 stability summary's helpers."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from summarize_room2_stability import (  # noqa: E402
    first_window_reaching,
    fisher_one_sided,
    training_windows,
    window_index,
)


def episode(steps: int, ending: str) -> dict:
    return {"accepted_steps": steps, "ending": ending}


class WindowTests(unittest.TestCase):
    def test_windows_are_closed_on_the_right_and_the_tail_joins_the_last(self):
        self.assertEqual(window_index(100_000, 100_000), 0)
        self.assertEqual(window_index(100_001, 100_000), 1)
        self.assertEqual(window_index(501_760, 100_000), 4)

    def test_training_windows_count_endings(self):
        windows = training_windows([episode(2048, "success"), episode(4096, "stalled"), episode(150_000, "death"),
                                    episode(501_760, "timeout")])
        self.assertEqual(windows[0], {"episodes": 2, "success": 1, "stalled": 1, "timeout": 0, "death": 0})
        self.assertEqual(windows[1]["death"], 1)
        self.assertEqual(windows[4]["timeout"], 1)

    def test_onset_is_the_end_of_the_first_window_at_half(self):
        episodes = [episode(10_000, "death"), episode(60_000, "success"), episode(70_000, "death"),
                    episode(120_000, "success")]
        self.assertEqual(first_window_reaching(episodes), 100_000)
        self.assertIsNone(first_window_reaching([episode(10_000, "death"), episode(300_000, "stalled")]))


class FisherTests(unittest.TestCase):
    def test_matches_the_recorded_values(self):
        self.assertAlmostEqual(fisher_one_sided(5, 8, 1, 8), 0.05944055944055944)  # stall pilot screen, group A
        self.assertAlmostEqual(fisher_one_sided(4, 4, 0, 4), 0.014285714285714285)  # stall pilot, exploratory B
        self.assertAlmostEqual(fisher_one_sided(2, 8, 0, 8), 28 / 120)
        self.assertAlmostEqual(fisher_one_sided(3, 16, 0, 12), 560 / 3276)

    def test_no_events_gives_one(self):
        self.assertEqual(fisher_one_sided(0, 8, 0, 8), 1.0)


if __name__ == "__main__":
    unittest.main()
