"""Offline checks of the coverage-ceiling analyzer's reading and decoding."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_coverage_ceiling import decide, exact_integers, retained_share  # noqa: E402


class DecisionTests(unittest.TestCase):
    def test_p_when_the_median_s_play_share_reaches_one_half(self):
        self.assertEqual(decide([0.5, 0.6, 0.4, 0.7])["branch"], "P")

    def test_below_one_half_coverage_comes_first_with_a_provisional_q_or_r(self):
        partial = decide([0.3, 0.25, 0.4, 0.1])
        self.assertEqual((partial["branch"], partial["provisional_after_s_play_wide"]), ("coverage first", "Q"))
        low = decide([0.1, 0.05, 0.15, 0.3])
        self.assertEqual((low["branch"], low["provisional_after_s_play_wide"]), ("coverage first", "R"))

    def test_retained_share(self):
        self.assertEqual(retained_share(0.142, 0.7643, 0.142), 0.0)
        self.assertEqual(retained_share(0.7643, 0.7643, 0.142), 1.0)


class DecodingTests(unittest.TestCase):
    def test_exact_integers_round_trip_and_refuse_inexact_values(self):
        x = np.array([19, 144, 300], dtype=np.float64)
        normalised = (x / 320).astype(np.float32)
        self.assertTrue(np.array_equal(exact_integers(normalised, 0.0, 320.0), x.astype(np.int64)))
        with self.assertRaises(ValueError):
            exact_integers(np.array([0.123456], dtype=np.float32), 0.0, 320.0)


if __name__ == "__main__":
    unittest.main()
