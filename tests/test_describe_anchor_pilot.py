"""Offline checks of the anchor pilot's randomness and confident-choice calculations."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from describe_anchor_pilot import confident_breakdown, entropy_bits  # noqa: E402


class RandomnessTests(unittest.TestCase):
    def test_bits_count_fair_coin_tosses(self):
        self.assertAlmostEqual(entropy_bits(np.full((4, 5), 0.5)), 5.0)
        self.assertAlmostEqual(entropy_bits(np.zeros((4, 5))), 0.0, places=6)


class ConfidentChoiceTests(unittest.TestCase):
    def test_same_uncertain_and_flipped_shares(self):
        donor = np.array([[0.95, 0.05, 0.95, 0.05, 0.5]])  # four confident choices; 0.5 is not one
        policy = np.array([[0.97, 0.5, 0.02, 0.01, 0.99]])  # same, uncertain, flipped, same
        result = confident_breakdown(donor, policy)
        self.assertEqual(result["donor_confident_choices"], 4)
        self.assertEqual((result["still_confident_same"], result["now_uncertain"], result["flipped_confident_opposite"]),
                         (0.5, 0.25, 0.25))


if __name__ == "__main__":
    unittest.main()
