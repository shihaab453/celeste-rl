"""Offline checks of the mixed self-distillation analyzer's declared reading."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_mixed_self_distillation import decide_clone, ppo_drop, retained_share  # noqa: E402


class CloneDecisionTests(unittest.TestCase):
    def test_k_when_the_median_sd_share_reaches_one_half(self):
        self.assertEqual(decide_clone([0.5, 0.6, 0.4, 0.7])["branch"], "K")

    def test_partial_from_one_fifth_to_below_one_half(self):
        self.assertEqual(decide_clone([0.2, 0.3, 0.1, 0.45])["branch"], "Partial")
        self.assertEqual(decide_clone([0.49, 0.49, 0.49, 0.49])["branch"], "Partial")

    def test_lost_below_one_fifth(self):
        self.assertEqual(decide_clone([0.1, 0.05, 0.19, 0.3])["branch"], "Lost")

    def test_retained_share(self):
        self.assertEqual(retained_share(0.142, 0.7643, 0.142), 0.0)
        self.assertEqual(retained_share(0.7643, 0.7643, 0.142), 1.0)


class PpoDropTests(unittest.TestCase):
    def test_the_drop_is_in_clone_floor_units_and_ignores_the_final_floor(self):
        # A donor at 0.7643: a clone at 0.70 that falls to 0.50 drops 0.2 / (0.7643 - 0.142).
        self.assertEqual(ppo_drop(0.70, 0.50, 0.7643, 0.142), round(0.2 / (0.7643 - 0.142), 4))
        self.assertEqual(ppo_drop(0.70, 0.70, 0.7643, 0.142), 0.0)


if __name__ == "__main__":
    unittest.main()
