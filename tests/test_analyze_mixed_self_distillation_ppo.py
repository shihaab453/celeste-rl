"""Offline checks of the mixed self-distillation PPO-stage reading (K-hold, K-decay, K-cost)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_mixed_self_distillation_ppo import decide_ppo  # noqa: E402


class PpoDecisionTests(unittest.TestCase):
    def test_hold_needs_a_small_median_drop_and_three_room2_finals_at_45(self):
        self.assertEqual(decide_ppo([0.1, 0.2, 0.25, 0.3], [50, 48, 45, 40])["branch"], "K-hold")

    def test_decay_when_the_median_drop_exceeds_a_quarter(self):
        self.assertEqual(decide_ppo([0.3, 0.4, 0.2, 0.5], [50, 50, 50, 50])["branch"], "K-decay")

    def test_cost_when_room1_holds_but_room2_finals_fall_short(self):
        decision = decide_ppo([0.0, 0.1, 0.05, 0.2], [50, 44, 40, 45])
        self.assertEqual((decision["branch"], decision["room2_finals_clearing_45_of_50"]), ("K-cost", "2 of 4"))


if __name__ == "__main__":
    unittest.main()
