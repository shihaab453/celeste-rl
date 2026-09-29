"""Offline checks of the PPO anchor tie-break: the declared decision rule and the plan generator's command rewriting."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze_ppo_anchor_pilot import one_unit_drop  # noqa: E402
from analyze_ppo_tiebreak import decide_tiebreak, drops_along  # noqa: E402
from make_tiebreak_plans import tiebreak_command, value_after, with_checkpoint  # noqa: E402

PILOT_E0 = [0.010, 0.1162, 0.3047, 0.3625]
PILOT_A1 = [0.1062, 0.0212, 0.0861, 0.0584]


class DecisionRuleTests(unittest.TestCase):
    def test_the_pilot_alone_is_not_enough_to_prefer_a1(self):
        decision = decide_tiebreak(PILOT_E0, PILOT_A1)
        self.assertEqual((decision["carry_forward"], decision["count_gap"]), ("E0", 2))
        # The median gap (0.138) alone would pass; the count gap of 2 is what keeps the pilot from deciding.
        self.assertTrue(decision["median_condition_met"])
        self.assertFalse(decision["count_condition_met"])

    def test_a_count_gap_of_three_alone_is_not_enough(self):
        # E0 has 3 runs above 0.25 and A1 none, but the medians are only 0.05 apart.
        e0 = [0.30, 0.30, 0.30, 0.10, 0.10, 0.10, 0.10, 0.10]
        a1 = [0.05] * 8
        decision = decide_tiebreak(e0, a1)
        self.assertEqual((decision["count_gap"], decision["count_condition_met"]), (3, True))
        self.assertEqual((decision["median_condition_met"], decision["carry_forward"]), (False, "E0"))

    def test_both_conditions_together_prefer_a1(self):
        e0 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.375, 0.375, 0.375]  # median 0.5, six runs above 0.25
        a1 = [0.25, 0.25, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]  # median 0.125, none above 0.25 (0.25 is not above)
        decision = decide_tiebreak(e0, a1)
        self.assertEqual((decision["a1_runs_above_0.25"], decision["e0_runs_above_0.25"]), (0, 8))
        self.assertEqual(decision["carry_forward"], "A1")

    def test_a_median_gap_alone_is_not_enough(self):
        e0 = [0.2] * 8  # a big median gap, but no run above 0.25
        a1 = [0.0] * 8
        decision = decide_tiebreak(e0, a1)
        self.assertEqual((decision["median_condition_met"], decision["count_condition_met"]), (True, False))
        self.assertEqual(decision["carry_forward"], "E0")

    def test_the_count_gap_is_inclusive_at_three_and_a_drop_of_exactly_a_quarter_is_not_above(self):
        a1 = [0.0] * 8
        self.assertEqual(decide_tiebreak([0.5] * 3 + [0.25] * 5, a1)["count_gap"], 3)
        self.assertEqual(decide_tiebreak([0.5] * 2 + [0.25] * 6, a1)["count_gap"], 2)

    def test_more_a1_runs_above_the_line_can_cancel_the_gap(self):
        e0 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1, 0.1]  # 6 above
        a1 = [0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1]  # 3 above: gap 3
        self.assertEqual(decide_tiebreak(e0, a1)["count_gap"], 3)
        a1_worse = [0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1]  # 4 above: gap 2
        self.assertEqual(decide_tiebreak(e0, a1_worse)["count_gap"], 2)

    def test_drops_along_uses_the_pilots_one_unit_formula(self):
        curve = [0.6447, 0.6, 0.5, 0.4]
        expected = [one_unit_drop(0.6447, value, 0.7643, 0.142) for value in curve[1:]]
        self.assertEqual(drops_along(curve, 0.7643, 0.142), expected)


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        self.command = ["scripts/train_anchored.py", "--anchor-coef", "1.0", "--anchor-seed", "1", "--seed", "41",
                        "--run-dir", "runs/train/ppo-anchor-pilot-A1-k1", "--total-timesteps", "500000"]

    def test_only_the_seed_and_the_run_dir_change(self):
        new = tiebreak_command(self.command, 1, "runs/train/ppo-tiebreak-A1-k1", "runs/train/ppo-anchor-pilot-A1-k1")
        changed = [i for i, (a, b) in enumerate(zip(self.command, new)) if a != b]
        self.assertEqual([self.command[i] for i in changed], ["41", "runs/train/ppo-anchor-pilot-A1-k1"])
        self.assertEqual((new[value_after(new, "--seed")], new[value_after(new, "--run-dir")]),
                         ("51", "runs/train/ppo-tiebreak-A1-k1"))
        self.assertEqual(len(new), len(self.command))

    def test_a_pilot_command_with_another_seed_or_folder_is_refused(self):
        with self.assertRaises(SystemExit):
            tiebreak_command(self.command, 0, "runs/train/ppo-tiebreak-A1-k0", "runs/train/ppo-anchor-pilot-A1-k1")
        with self.assertRaises(SystemExit):
            tiebreak_command(self.command, 1, "runs/train/ppo-tiebreak-A1-k1", "runs/train/somewhere-else")

    def test_an_option_that_is_missing_or_repeated_is_refused(self):
        with self.assertRaises(SystemExit):
            value_after(["x", "--seed", "1", "--seed", "2"], "--seed")
        with self.assertRaises(SystemExit):
            value_after(["x"], "--seed")

    def test_with_checkpoint_changes_only_the_checkpoint(self):
        template = ["scripts/evaluate_heldout.py", "--checkpoint", "old.zip", "--starts", "config/heldout_starts.json"]
        self.assertEqual(with_checkpoint(template, "new.zip"),
                         ["scripts/evaluate_heldout.py", "--checkpoint", "new.zip", "--starts",
                          "config/heldout_starts.json"])
        self.assertEqual(template[2], "old.zip")


if __name__ == "__main__":
    unittest.main()
