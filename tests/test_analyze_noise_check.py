"""The noise-check analysis on synthetic rows: validation fails closed, and the declared quantities are exact."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_noise_check import (
    CANONICAL_ID,
    REPO,
    AnalysisError,
    analyse,
    band_of,
    checkpoint_measures,
    file_sha256,
    load_campaign,
    script_git_blob,
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


PINNED_ANALYSIS_BLOB = "e" * 40
TOOL_BLOB = "f" * 40


def fake_git_blob(blobs: dict[str, str]):
    """A stand-in for the git lookup: the tool's blob at each commit."""
    return lambda commit, path: blobs[commit]


class CampaignFixture:
    """A complete synthetic campaign on disk for one declared checkpoint: plan, summary, result, rows, starts."""

    def __init__(self, root: Path):
        self.root = root
        run_dir = root / "run"
        run_dir.mkdir()
        self.starts_path = run_dir / "starts.json"
        self.result_path = run_dir / "result.json"
        self.episodes_path = run_dir / "episodes.jsonl"
        self.plan_path = root / "plan.json"
        self.summary_path = root / "summary.json"
        starts_sha = start_list_sha256(STARTS)
        self.starts = {"starts": [{**start, "lines": ["1,R"]} for start in STARTS], "sha256": starts_sha}
        self.result = {
            "checkpoint_sha256": ENTRY["checkpoint_sha256"], "evaluation_seed": 7,
            "deterministic_repeats": PROTOCOL["deterministic_repeats_per_start"], "aborted": None, "task": TASK,
            "attributable": True,
            "demonstration_starts": {"starts_file": "starts.json", "starts_sha256": starts_sha,
                                     "stochastic_per_start": PROTOCOL["stochastic_episodes_per_demonstration_start"]},
        }
        self.rows = complete_rows()
        self.plan = {
            "name": "noise-test",
            "task": TASK,
            "tool": {"script": "scripts/evaluate_checkpoint.py", "commit": "pinned-commit", "git_blob": TOOL_BLOB},
            "starts": {"demonstration": {"expected_start_list_sha256": starts_sha,
                                         "primary_route_counts": {"r1": 2, "r2": 1, "total": 3}}},
            "evaluation_protocol": {**PROTOCOL, "evaluation_seed": 7},
            "analysis": {"label": "diagnostic", "primary_start_x_below": 460, "primary_starts": 3,
                         "analysis_code": {"script": "scripts/analyze_noise_check.py",
                                           "git_blob": PINNED_ANALYSIS_BLOB}},
            "decision_rule": {**DecisionTests.PLAN["decision_rule"], "scope": "diagnostic"},
            "runs": [{**ENTRY, "role": "declared checkpoint"}],
        }
        self.status = "ok"
        self.git_blobs = {"pinned-commit": TOOL_BLOB, "campaign-commit": TOOL_BLOB}

    def write(self) -> None:
        """Write every file, then a summary that records their hashes as the campaign runner would."""
        self.starts_path.write_text(json.dumps(self.starts), encoding="utf-8")
        self.result_path.write_text(json.dumps(self.result), encoding="utf-8")
        self.episodes_path.write_text("".join(json.dumps(r) + "\n" for r in self.rows), encoding="utf-8")
        self.plan_path.write_text(json.dumps(self.plan), encoding="utf-8")
        artifact = {"result_file": str(self.result_path), "episodes_file": str(self.episodes_path),
                    "result_sha256": file_sha256(self.result_path),
                    "episodes_sha256": file_sha256(self.episodes_path)}
        summary = {"plan": self.plan["name"], "plan_sha256": file_sha256(self.plan_path), "commit": "campaign-commit",
                   "results": [{"id": ENTRY["id"], "status": self.status, "artifact": artifact}]}
        self.summary_path.write_text(json.dumps(summary), encoding="utf-8")

    def load(self, script_blob: str = PINNED_ANALYSIS_BLOB):
        return load_campaign(self.plan_path, self.summary_path, script_blob=script_blob,
                             git_blob=fake_git_blob(self.git_blobs))


class LoadCampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.campaign = CampaignFixture(Path(self.temp.name))

    def test_a_complete_campaign_loads_and_analyses(self):
        self.campaign.write()
        plan, data = self.campaign.load()
        report = analyse(plan, data)
        self.assertEqual(report["checkpoints"][ENTRY["id"]]["primary"]["starts_per_route"], {"r1": 2, "r2": 1})

    def test_a_changed_plan_file_is_refused(self):
        self.campaign.write()
        self.campaign.plan_path.write_text(json.dumps({**self.campaign.plan, "note": "edited"}), encoding="utf-8")
        with self.assertRaisesRegex(AnalysisError, "not produced from this plan file"):
            self.campaign.load()

    def test_a_different_evaluate_checkpoint_blob_is_refused(self):
        self.campaign.git_blobs["campaign-commit"] = "0" * 40
        self.campaign.write()
        with self.assertRaisesRegex(AnalysisError, "different evaluate_checkpoint.py"):
            self.campaign.load()

    def test_a_run_that_did_not_finish_ok_is_refused(self):
        self.campaign.status = "failed"
        self.campaign.write()
        with self.assertRaisesRegex(AnalysisError, "did not finish ok: failed"):
            self.campaign.load()

    def test_a_result_or_episodes_file_changed_after_the_campaign_is_refused(self):
        for path_name in ("result_path", "episodes_path"):
            with self.subTest(path_name):
                self.campaign.write()
                path = getattr(self.campaign, path_name)
                path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
                with self.assertRaisesRegex(AnalysisError, "changed after the campaign"):
                    self.campaign.load()

    def test_a_wrong_start_list_is_refused(self):
        self.campaign.plan["starts"]["demonstration"]["expected_start_list_sha256"] = "9" * 64
        self.campaign.write()
        with self.assertRaisesRegex(AnalysisError, "did not use the declared start list"):
            self.campaign.load()

    def test_a_start_list_edited_on_disk_is_refused(self):
        self.campaign.starts["starts"][0]["position"] = [301, -40]
        self.campaign.write()
        with self.assertRaisesRegex(AnalysisError, "did not use the declared start list"):
            self.campaign.load()

    def test_an_aborted_or_non_attributable_result_is_refused(self):
        complete = self.campaign.result
        for field, value in (("aborted", "the game closed"), ("attributable", False)):
            with self.subTest(field):
                self.campaign.result = {**complete, field: value}
                self.campaign.write()
                with self.assertRaisesRegex(AnalysisError, f"result {field} is"):
                    self.campaign.load()

    def test_primary_counts_per_route_must_match_the_plan(self):
        for counts in ({"r1": 1, "r2": 2, "total": 3}, {"r1": 3, "total": 3}, {"r1": 2, "r2": 1, "r3": 0, "total": 3}):
            with self.subTest(counts):
                self.campaign.plan["starts"]["demonstration"]["primary_route_counts"] = counts
                self.campaign.write()
                plan, data = self.campaign.load()
                with self.assertRaisesRegex(AnalysisError, "primary starts per route"):
                    analyse(plan, data)

    def test_an_analysis_script_other_than_the_pinned_blob_is_refused(self):
        self.campaign.write()
        with self.assertRaisesRegex(AnalysisError, "the plan pins " + PINNED_ANALYSIS_BLOB):
            self.campaign.load(script_blob="1" * 40)

    def test_the_script_blob_is_the_line_ending_normalised_git_blob(self):
        content = (REPO / "scripts" / "analyze_noise_check.py").read_bytes().replace(b"\r\n", b"\n")
        expected = hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()
        self.assertEqual(script_git_blob(), expected)


if __name__ == "__main__":
    unittest.main()
