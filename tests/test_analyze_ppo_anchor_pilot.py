"""Offline checks of the PPO anchor pilot's declared reading and record checks."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze_ppo_anchor_pilot import (  # noqa: E402
    anchored_run_problems,
    decide_arm,
    e0_problems,
    one_unit_drop,
    pin_problems,
    read_together,
    replayed_anchor_rows,
    session_commits,
)
from celeste_rl.texthash import text_sha256  # noqa: E402

THRESHOLD = 0.8176 - 0.05


class ArmBranchTests(unittest.TestCase):
    def test_holds_needs_a_small_drop_and_both_room2_conditions(self):
        self.assertEqual(decide_arm([0.1, 0.2, 0.25, 0.3], [0.8] * 4, [50, 48, 45, 40], THRESHOLD)["branch"], "Holds")

    def test_a_small_drop_with_a_room2_failure_is_a_room2_cost(self):
        self.assertEqual(decide_arm([0.1] * 4, [0.7] * 4, [50] * 4, THRESHOLD)["branch"], "Holds with a Room 2 cost")
        self.assertEqual(decide_arm([0.1] * 4, [0.8] * 4, [50, 44, 44, 45], THRESHOLD)["branch"],
                         "Holds with a Room 2 cost")

    def test_partial_and_does_not_hold(self):
        self.assertEqual(decide_arm([0.3, 0.5, 0.4, 0.6], [0.8] * 4, [50] * 4, THRESHOLD)["branch"], "Partial")
        self.assertEqual(decide_arm([0.9, 0.8, 0.7, 0.6], [0.8] * 4, [50] * 4, THRESHOLD)["branch"], "Does not hold")

    def test_the_drop_is_in_the_original_margin_unit(self):
        self.assertEqual(one_unit_drop(0.645, 0.081, 0.7643, 0.142), 0.9063)


class ReadingTests(unittest.TestCase):
    def decisions(self, **branches):
        return {arm: {"branch": branch, "median_drop": drop, "room2_v1_median": v1}
                for arm, (branch, drop, v1) in branches.items()}

    def test_a_clean_hold_beats_a_hold_with_a_room2_cost(self):
        reading = read_together(self.decisions(A1=("Holds", 0.2, 0.78), A10=("Partial", 0.4, 0.8),
                                               E0=("Holds with a Room 2 cost", 0.05, 0.70)))
        self.assertEqual((reading["preferred_arm"], reading["tie"]), ("A1", False))

    def test_within_a_branch_e0_is_preferred_unless_more_than_005_worse_on_room2(self):
        # E0 0.03 below the anchor on Room 2: still preferred, and far enough apart on drop not to tie.
        close = read_together(self.decisions(A1=("Holds", 0.24, 0.80), A10=("Does not hold", 0.9, 0.8),
                                             E0=("Holds", 0.05, 0.77)))
        self.assertEqual(close["preferred_arm"], "E0")
        # E0 0.08 below: the anchor arm ranks first.
        worse = read_together(self.decisions(A1=("Holds", 0.24, 0.85), A10=("Does not hold", 0.9, 0.8),
                                             E0=("Holds", 0.05, 0.77)))
        self.assertEqual(worse["preferred_arm"], "A1")

    def test_close_top_two_are_a_tie(self):
        reading = read_together(self.decisions(A1=("Holds", 0.10, 0.80), A10=("Holds", 0.20, 0.79),
                                               E0=("Partial", 0.4, 0.8)))
        self.assertTrue(reading["tie"])
        self.assertEqual(sorted(reading["tied"]), ["A1", "A10"])
        self.assertIsNone(reading["preferred_arm"])

    def test_anchor_arms_rank_by_drop_and_nothing_holding_has_no_lever(self):
        only = read_together(self.decisions(A1=("Holds", 0.24, 0.8), A10=("Holds", 0.02, 0.9),
                                            E0=("Does not hold", 0.9, 0.8)))
        self.assertEqual(only["preferred_arm"], "A10")
        nothing = read_together(self.decisions(A1=("Partial", 0.4, 0.8), A10=("Does not hold", 0.7, 0.8),
                                               E0=("Partial", 0.5, 0.8)))
        self.assertIsNone(nothing["lever"])


class RecordCheckTests(unittest.TestCase):
    def setUp(self):
        self.run = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.run)
        self.anchor = {"coef": 1.0, "seed": 0}

    def write(self, sessions, minibatches, config=None, updates=None, grads=None, accepted_steps=None,
              resumed_at=None):
        """A run folder: one anchor.csv row per entry of `minibatches`. The update column steps by n_epochs (4) as
        SB3's counter does, unless `updates` says otherwise; the manifest's accepted steps default to one 2,048-step
        rollout per row; `resumed_at` gives each session's resumed_at steps (None leaves it out)."""
        updates = updates if updates is not None else [4 * index for index in range(len(minibatches))]
        grads = grads if grads is not None else [0.02] * len(minibatches)
        steps = len(minibatches) * 2048 if accepted_steps is None else accepted_steps
        resumed_at = resumed_at or [None] * len(sessions)
        (self.run / "manifest.json").write_text(json.dumps(
            {"config": config or {"ent_coef": 0.01}, "accepted_steps": steps,
             "sessions": [{"provenance": p, **({"resumed_at": r} if r is not None else {})}
                          for p, r in zip(sessions, resumed_at)]}), encoding="utf-8")
        (self.run / "anchor.csv").write_text(
            "update,minibatches,anchor_grad_norm\n"
            + "".join(f"{u},{m},{g}\n" for u, m, g in zip(updates, minibatches, grads)), encoding="utf-8")

    def test_a_clean_anchored_run_passes(self):
        self.write([{"anchor": self.anchor}, {"anchor": self.anchor}], [16, 16, 16])
        self.assertEqual(anchored_run_problems(self.run, 1.0), [])

    def test_an_unanchored_resume_a_wrong_lambda_or_short_update_fails(self):
        self.write([{"anchor": self.anchor}, {}], [16])
        self.assertTrue(anchored_run_problems(self.run, 1.0))
        self.write([{"anchor": self.anchor}], [16])
        self.assertTrue(anchored_run_problems(self.run, 10.0))
        self.write([{"anchor": self.anchor}], [16, 15])
        self.assertTrue(anchored_run_problems(self.run, 1.0))

    def test_updates_no_resume_explains_are_caught_as_repeated_rows(self):
        # The second session says it resumed at the very start, but the file drops back to update 4.
        self.write([{"anchor": self.anchor}, {"anchor": self.anchor}], [16] * 5, updates=[0, 4, 8, 4, 8],
                   accepted_steps=3 * 2048, resumed_at=[0, 3 * 2048])
        problems = anchored_run_problems(self.run, 1.0)
        self.assertEqual(len(problems), 1)
        self.assertIn("repeated update numbers: [4, 8]", problems[0])
        # A drop-back with a single session (nothing resumed) is refused too.
        self.write([{"anchor": self.anchor}], [16] * 5, updates=[0, 4, 8, 4, 8], accepted_steps=3 * 2048)
        self.assertTrue(anchored_run_problems(self.run, 1.0))

    def test_a_resume_that_replayed_updates_is_accepted_and_the_replays_counted(self):
        # Session 1 wrote updates 0 to 16 (5 rollouts) and was killed; its latest checkpoint was at 3 rollouts, so
        # session 2 resumed at 3 * 2048 steps (update 12) and rewrote 12, 16 before going on to 20.
        rows = [16] * 8
        updates = [0, 4, 8, 12, 16, 12, 16, 20]
        self.write([{"anchor": self.anchor}, {"anchor": self.anchor}], rows, updates=updates,
                   accepted_steps=6 * 2048, resumed_at=[0, 3 * 2048])
        self.assertEqual(anchored_run_problems(self.run, 1.0), [])
        parsed = [{"update": str(u), "minibatches": "16", "anchor_grad_norm": "0.02"} for u in updates]
        manifest = json.loads((self.run / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(replayed_anchor_rows(parsed, manifest), 2)
        # The same rows with a session that resumed somewhere else are not explained.
        self.write([{"anchor": self.anchor}, {"anchor": self.anchor}], rows, updates=updates,
                   accepted_steps=6 * 2048, resumed_at=[0, 4 * 2048])
        self.assertTrue(anchored_run_problems(self.run, 1.0))

    def test_a_replay_still_has_to_end_at_every_update(self):
        # Explained drop-back, but the last update never got a row.
        self.write([{"anchor": self.anchor}, {"anchor": self.anchor}], [16] * 7,
                   updates=[0, 4, 8, 12, 16, 12, 16], accepted_steps=6 * 2048, resumed_at=[0, 3 * 2048])
        self.assertTrue(anchored_run_problems(self.run, 1.0))

    def test_blank_or_non_numeric_values_are_refusals_not_crashes(self):
        self.write([{"anchor": self.anchor}], [16, 16], grads=["", 0.02])
        self.assertTrue(anchored_run_problems(self.run, 1.0))
        self.write([{"anchor": self.anchor}], [16, ""], accepted_steps=2 * 2048)
        self.assertTrue(anchored_run_problems(self.run, 1.0))
        self.write([{"anchor": self.anchor}], [16, 16], updates=[0, ""], accepted_steps=2 * 2048)
        self.assertTrue(anchored_run_problems(self.run, 1.0))

    def test_a_missing_row_or_a_gap_in_the_update_numbers_fails(self):
        self.write([{"anchor": self.anchor}], [16, 16, 16], accepted_steps=4 * 2048)  # 4 updates, 3 rows
        self.assertTrue(anchored_run_problems(self.run, 1.0))
        self.write([{"anchor": self.anchor}], [16, 16, 16], updates=[0, 4, 12])
        self.assertTrue(anchored_run_problems(self.run, 1.0))

    def test_the_anchor_gradient_must_be_above_zero_when_lambda_is(self):
        self.write([{"anchor": self.anchor}], [16, 16, 16], grads=[0.02, 0.0, 0.03])
        problems = anchored_run_problems(self.run, 1.0)
        self.assertEqual(len(problems), 1)
        self.assertIn("anchor gradient missing or not above 0 in updates ['4']", problems[0])
        self.write([{"anchor": self.anchor}], [16, 16], grads=[0.02, "nan"])
        self.assertTrue(anchored_run_problems(self.run, 1.0))
        # Lambda 0 has no anchor gradient to require.
        self.write([{"anchor": {"coef": 0.0, "seed": 0}}], [16, 16], grads=[0.0, 0.0])
        self.assertEqual(anchored_run_problems(self.run, 0.0), [])

    def test_a_manifest_with_more_epochs_expects_a_wider_step(self):
        self.write([{"anchor": self.anchor}], [16, 16], config={"ent_coef": 0.01, "n_epochs": 10, "n_steps": 1024},
                   updates=[0, 10], accepted_steps=2 * 1024)
        self.assertEqual(anchored_run_problems(self.run, 1.0), [])

    def test_the_evaluation_plans_pins_must_match_their_files(self):
        summary, train, pilot = self.run / "summary.json", self.run / "train.json", self.run / "pilot.json"
        summary.write_bytes(b'{"a": 1}')
        train.write_text("{}", encoding="utf-8")
        pilot.write_text('{"label": "x"}', encoding="utf-8")
        plan = {"training_campaign_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
                "training_plan_sha256": text_sha256(train), "pilot_plan_sha256": text_sha256(pilot)}
        self.assertEqual(pin_problems(plan, summary, train, pilot), [])
        pilot.write_text('{"label": "changed"}', encoding="utf-8")
        summary.write_bytes(b'{"a": 2}')
        problems = pin_problems(plan, summary, train, pilot)
        self.assertEqual(len(problems), 2)
        self.assertTrue(pin_problems({}, summary, train, pilot))

    def test_session_commits_lists_every_session_shortened(self):
        manifest = {"sessions": [{"provenance": {"commit": "54c955695001e7732f427aba93495b27a7e5a4ac"}},
                                 {"provenance": {"commit": "9a3ee81f67d4970c519e04494eb951bc0242573a"}}]}
        self.assertEqual(session_commits(manifest), ["54c95569", "9a3ee81f"])

    def test_e0_must_have_no_entropy_bonus_and_no_anchor(self):
        self.write([{}], [], config={"ent_coef": 0.0})
        self.assertEqual(e0_problems(self.run), [])
        self.write([{}], [], config={"ent_coef": 0.01})
        self.assertTrue(e0_problems(self.run))


if __name__ == "__main__":
    unittest.main()
