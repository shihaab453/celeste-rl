"""Offline checks of the stability gate and the held-out overlap sensitivity helpers."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_stability_gate import gate  # noqa: E402
from heldout_overlap_sensitivity import without  # noqa: E402


class StabilityGateTests(unittest.TestCase):
    def test_a_collapse_is_strictly_below_the_threshold(self):
        result = gate({"A": {"a0": 0.5, "a1": 0.49}}, collapse_below=0.5, max_collapses=1)
        self.assertEqual(result["arms"]["A"]["collapsed"], ["a1"])
        self.assertTrue(result["passes"])

    def test_every_arm_must_pass(self):
        result = gate({"A": {"a0": 0.9, "a1": 0.8}, "B": {"b0": 0.1, "b1": 0.2}}, collapse_below=0.5,
                      max_collapses=1)
        self.assertTrue(result["arms"]["A"]["passes"])
        self.assertFalse(result["arms"]["B"]["passes"])
        self.assertFalse(result["passes"])


class OverlapSensitivityTests(unittest.TestCase):
    def test_excluded_states_are_removed_from_every_run(self):
        rows = {(30, "A"): [{"state_id": "s1"}, {"state_id": "s2"}], (30, "B"): [{"state_id": "s2"}]}
        self.assertEqual(without(rows, {"s2"}), {(30, "A"): [{"state_id": "s1"}], (30, "B"): []})


if __name__ == "__main__":
    unittest.main()
