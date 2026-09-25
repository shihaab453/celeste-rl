"""Offline checks of the mixed-imitation pilot's pre-stated decision branches."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_mixed_imitation_pilot import decide  # noqa: E402


class DecisionBranchTests(unittest.TestCase):
    def test_a_keeps_the_skill_and_ppo_does_not_erode_it(self):
        result = decide([0.60, 0.62, 0.55, 0.50], [0.7, 0.75, 0.65, 0.6], [0.6, 0.7, 0.6, 0.55], reference_mean=0.30)
        self.assertEqual((result["branch"], result["ppo_erodes"]), ("A", False))

    def test_a_with_erosion_during_ppo_points_to_mixed_ppo(self):
        result = decide([0.60, 0.62, 0.55, 0.50], [0.7, 0.75, 0.65, 0.6], [0.3, 0.4, 0.35, 0.3], reference_mean=0.30)
        self.assertEqual((result["branch"], result["next"]), ("A", "mixed PPO"))

    def test_b_when_the_mixed_clones_sit_at_the_imitation_only_level(self):
        result = decide([0.32, 0.28, 0.35, 0.30], [0.3, 0.25, 0.35, 0.3], [0.2, 0.2, 0.2, 0.2], reference_mean=0.30)
        self.assertEqual(result["branch"], "B")

    def test_c_when_near_the_floor_and_far_from_the_reference(self):
        result = decide([0.15, 0.16, 0.14, 0.17], [0.05, 0.1, 0.0, 0.1], [0.0, 0.0, 0.0, 0.0], reference_mean=0.40)
        self.assertEqual(result["branch"], "C")

    def test_d_for_anything_else(self):
        result = decide([0.45, 0.20, 0.60, 0.33], [0.5, 0.1, 0.7, 0.3], [0.4, 0.1, 0.5, 0.2], reference_mean=0.30)
        self.assertEqual(result["branch"], "D")

    def test_a_needs_three_runs_clearly_above_the_reference(self):
        # Median share 0.6 but only 2 of 4 runs more than 10 points above the reference: not A.
        result = decide([0.60, 0.62, 0.38, 0.39], [0.6, 0.65, 0.6, 0.55], [0.5, 0.5, 0.5, 0.5], reference_mean=0.30)
        self.assertNotEqual(result["branch"], "A")


if __name__ == "__main__":
    unittest.main()
