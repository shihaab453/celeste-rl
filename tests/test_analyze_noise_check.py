"""The noise-check analysis on synthetic rows: validation fails closed, and the declared quantities are exact."""
from __future__ import annotations

import unittest

from scripts.analyze_noise_check import (
    CANONICAL_ID,
    AnalysisError,
    band_of,
    checkpoint_measures,
    start_list_sha256,
    summarise,
    validate_rows,
)

TASK = {"name": "chapter-1-room-2", "task_definition": "config/room2.json", "task_definition_sha256": "a" * 64,
        "source_route_sha256": "b" * 64, "start_room": "2", "target_room": "3"}
PROTOCOL = {"canonical_stochastic_episodes": 4, "deterministic_repeats_per_start": 2,
            "stochastic_episodes_per_demonstration_start": 2}
ENTRY = {"id": "noise-room2-A-seed20", "checkpoint_sha256": "c" * 64, "arm": "A"}
STARTS = [
    {"start_id": "r1@40", "route": "r1", "position": [300, -40]},
    {"start_id": "r1@80", "route": "r1", "position": [400, -60]},
    {"start_id": "r1@120", "route": "r1", "position": [500, -60]},
    {"start_id": "r2@40", "route": "r2", "position": [350, -40]},
]


def row(start_id, mode, repeat, ending="success", end_x=500, max_x=520, **extra):
    return {"start_id": start_id, "mode": mode, "repeat": repeat, "ending": ending, "end_x": end_x,
            "max_x": max_x, "problem": None, "checkpoint_sha256": ENTRY["checkpoint_sha256"], "task": TASK, **extra}


def complete_rows(outcomes: dict | None = None) -> list[dict]:
    """Every declared row; `outcomes[(start_id, mode)]` lists endings in repeat order (default all success)."""
    outcomes = outcomes or {}
    rows = []
    plan = [(CANONICAL_ID, "stochastic", 4), (CANONICAL_ID, "deterministic", 2)]
    for start in STARTS:
        plan += [(start["start_id"], "stochastic", 2), (start["start_id"], "deterministic", 2)]
    for start_id, mode, count in plan:
        endings = outcomes.get((start_id, mode), ["success"] * count)
        for repeat, ending in enumerate(endings):
            rows.append(row(start_id, mode, repeat, ending=ending, end_x=None if ending == "death" else 380))
    return rows


class ValidationTests(unittest.TestCase):
    def test_complete_rows_pass(self):
        validate_rows(ENTRY, PROTOCOL, STARTS, complete_rows(), TASK)

    def test_a_missing_row_is_refused(self):
        rows = [r for r in complete_rows() if not (r["start_id"] == "r2@40" and r["mode"] == "stochastic"
                                                    and r["repeat"] == 1)]
        with self.assertRaisesRegex(AnalysisError, r"r2@40', 'stochastic'\) has repeats \[0\]"):
            validate_rows(ENTRY, PROTOCOL, STARTS, rows, TASK)

    def test_a_duplicated_row_is_refused(self):
        rows = complete_rows() + [row("r1@40", "deterministic", 1)]
        with self.assertRaisesRegex(AnalysisError, "has repeats"):
            validate_rows(ENTRY, PROTOCOL, STARTS, rows, TASK)

    def test_a_problem_row_is_refused(self):
        rows = complete_rows()
        rows[7] = {**rows[7], "problem": "the prefix arrived elsewhere", "ending": None}
        with self.assertRaisesRegex(AnalysisError, "problem row"):
            validate_rows(ENTRY, PROTOCOL, STARTS, rows, TASK)

    def test_a_wrong_checkpoint_hash_is_refused(self):
        rows = complete_rows()
        rows[0] = {**rows[0], "checkpoint_sha256": "d" * 64}
        with self.assertRaisesRegex(AnalysisError, "checkpoint hash"):
            validate_rows(ENTRY, PROTOCOL, STARTS, rows, TASK)

    def test_an_undeclared_start_is_refused(self):
        rows = complete_rows() + [row("r9@40", "stochastic", 0)]
        with self.assertRaisesRegex(AnalysisError, "undeclared row"):
            validate_rows(ENTRY, PROTOCOL, STARTS, rows, TASK)

    def test_start_list_hash_ignores_recipes(self):
        with_lines = [{**start, "lines": ["1,R"]} for start in STARTS]
        self.assertEqual(start_list_sha256(with_lines), start_list_sha256(STARTS))


class MeasureTests(unittest.TestCase):
    def test_primary_uses_starts_below_the_threshold_with_equal_route_weight(self):
        rows = complete_rows({
            ("r1@40", "stochastic"): ["death", "death"],        # r1@40: 1 - 0 = +1
            ("r1@80", "stochastic"): ["success", "death"],      # r1@80: 1 - 0.5 = +0.5
            ("r1@120", "stochastic"): ["death", "death"],       # r1@120 is x 500: secondary only
            ("r2@40", "deterministic"): ["death", "death"],     # r2@40: 0 - 1 = -1
        })

        measures = checkpoint_measures(STARTS, rows, 460)

        self.assertEqual(measures["primary"]["starts"], 3)
        self.assertAlmostEqual(measures["primary"]["routes"]["r1"], 0.75)
        self.assertAlmostEqual(measures["primary"]["routes"]["r2"], -1.0)
        self.assertAlmostEqual(measures["primary"]["difference"], (0.75 - 1.0) / 2)
        self.assertAlmostEqual(measures["all_starts"]["routes"]["r1"], (1 + 0.5 + 1) / 3)
        self.assertEqual(measures["primary"]["starts_per_route"], {"r1": 2, "r2": 1})

    def test_disagreeing_deterministic_repeats_use_the_mean_and_are_counted(self):
        rows = complete_rows({("r1@80", "deterministic"): ["success", "death"],
                              (CANONICAL_ID, "deterministic"): ["death", "success"]})

        measures = checkpoint_measures(STARTS, rows, 460)

        self.assertAlmostEqual(measures["by_start"]["r1@80"]["deterministic"], 0.5)
        self.assertTrue(measures["by_start"]["r1@80"]["deterministic_repeats_disagree"])
        self.assertEqual(measures["deterministic_disagreements"], {"count": 2, "starts": [CANONICAL_ID, "r1@80"]})
        self.assertAlmostEqual(measures["canonical"]["difference"], 0.5 - 1.0)

    def test_failures_are_banded_by_end_x_and_true_max_x_separately(self):
        rows = complete_rows({("r2@40", "stochastic"): ["timeout", "death"]})
        rows = [{**r, "end_x": 325, "max_x": 410} if r["start_id"] == "r2@40" and r["mode"] == "stochastic" else r
                for r in rows]

        failures = checkpoint_measures(STARTS, rows, 460)["failures"]["stochastic"]

        self.assertEqual(failures["by_end_x"], {"before 360": 2})
        self.assertEqual(failures["by_max_x"], {"360 to 459": 2})
        self.assertEqual([band_of(359), band_of(360), band_of(459), band_of(460), band_of(None)],
                         ["before 360", "360 to 459", "360 to 459", "after 459", "no position"])


class DecisionTests(unittest.TestCase):
    PLAN = {
        "decision_rule": {
            "thresholds": {"favour_points": 10, "favour_min_checkpoints": 9, "null_points": 3,
                           "null_min_checkpoints": 9},
            "outcomes": {"favour": "f", "null": "n", "inconclusive": "i"},
        },
        "runs": [{"id": f"run{i}", "arm": "A" if i < 6 else "B"} for i in range(12)] + [{"id": "clone", "arm": None}],
    }

    def measures(self, points: list[float]) -> dict:
        values = {f"run{i}": value for i, value in enumerate(points)}
        values["clone"] = 50.0
        return {run_id: {"primary": {"difference": value / 100}, "deterministic_disagreements": {"count": 0}}
                for run_id, value in values.items()}

    def test_nine_checkpoints_at_exactly_ten_points_favour(self):
        result = summarise(self.PLAN, self.measures([10.0] * 9 + [0.0] * 3))
        self.assertEqual(result["decision"], "favour")
        self.assertEqual(result["checkpoints_at_least_favour_points"], 9)

    def test_nine_checkpoints_below_three_points_are_null(self):
        result = summarise(self.PLAN, self.measures([2.9] * 9 + [20.0] * 3))
        self.assertEqual(result["decision"], "null")

    def test_three_points_exactly_is_not_below_three(self):
        result = summarise(self.PLAN, self.measures([3.0] * 9 + [20.0] * 3))
        self.assertEqual(result["decision"], "inconclusive")

    def test_the_clone_reference_never_counts_towards_the_decision(self):
        result = summarise(self.PLAN, self.measures([10.0] * 8 + [0.0] * 4))
        self.assertEqual(result["decision"], "inconclusive")
        self.assertNotIn("clone", result["primary_points_by_checkpoint"])
        self.assertEqual(set(result["by_arm_descriptive"]), {"A", "B"})


if __name__ == "__main__":
    unittest.main()
