"""Offline checks of the stability-pilot analysis building blocks."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_stall_pilot import collapsed, fisher_one_sided, pairing_check, window_metrics  # noqa: E402


def episode(steps, ending="death", length=100, max_x=300, **extra):
    return {"accepted_steps": steps, "ending": ending, "length": length, "return": -1.389, "max_x": max_x,
            "end_x": max_x - 5, "end_y": -24, "max_potential": 0.2, **extra}


class StallPilotAnalysisTests(unittest.TestCase):
    def test_fisher_matches_the_review_s_figures(self):
        self.assertAlmostEqual(fisher_one_sided(1, 4, 8, 8), 616 / 4368, places=12)   # about 0.14
        self.assertAlmostEqual(fisher_one_sided(0, 4, 8, 8), 70 / 1820, places=12)    # about 0.038
        self.assertAlmostEqual(fisher_one_sided(4, 4, 8, 8), 1 - fisher_one_sided_upper(4, 4), places=12)
        # Against the primary measure's 5/8 control count (second review): about 0.059 and 0.013.
        self.assertAlmostEqual(fisher_one_sided(1, 5, 8, 8), 476 / 8008, places=12)
        self.assertAlmostEqual(fisher_one_sided(0, 5, 8, 8), 56 / 4368, places=12)

    def test_collapse_is_strictly_below_the_threshold(self):
        self.assertTrue(collapsed(0.48, 0.5))
        self.assertFalse(collapsed(0.5, 0.5))

    def test_pairs_match_up_to_the_first_stalled_episode(self):
        control = [episode(2048), episode(2048, "timeout", 1800), episode(4096)]
        treatment = [episode(2048), episode(2048, "stalled", 300), episode(4096, length=90)]
        self.assertEqual(pairing_check(treatment, control), {"shared_episodes": 1, "matches": True,
                                                             "first_mismatch": None})

    def test_a_difference_before_any_stall_is_a_pairing_failure(self):
        control = [episode(2048), episode(4096)]
        treatment = [episode(2048, length=101), episode(4096)]
        self.assertEqual(pairing_check(treatment, control)["first_mismatch"], 0)

    def test_window_metrics_count_wasted_frames_and_crossing_attempts(self):
        episodes = [episode(50_000, "timeout", 1800, 320), episode(60_000, "stalled", 300, 350),
                    episode(70_000, "success", 500, 540), episode(150_000, "death", 400, 410)]
        progress = [{"accepted_steps": "51200", "entropy_loss": "-0.5"}]
        windows = window_metrics(episodes, progress, crossing_x=344)
        first = windows["0k-100k"]
        self.assertEqual(first["episodes"], 3)
        self.assertAlmostEqual(first["mean_frames_per_episode"], 2600 / 3)
        self.assertAlmostEqual(first["share_of_frames_in_timeout_or_stalled_episodes"], 2100 / 2600)
        self.assertEqual(first["attempts_reaching_crossing"], 2)
        self.assertAlmostEqual(first["mean_entropy"], 0.5)
        self.assertEqual(windows["100k-200k"]["attempts_reaching_crossing"], 1)


def fisher_one_sided_upper(treatment_collapses: int, control_collapses: int) -> float:
    """P(treatment collapses > observed), computed independently for the complement check."""
    from math import comb
    total, population = treatment_collapses + control_collapses, 16
    return sum(comb(8, k) * comb(8, total - k) for k in range(treatment_collapses + 1, 9)
               if 0 <= total - k <= 8) / comb(population, total)


if __name__ == "__main__":
    unittest.main()
