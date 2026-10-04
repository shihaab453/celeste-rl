"""The stage 1 plan generator (scripts/make_confirmation_stage1_plans.py; amendment 3). Synthetic fixtures and
stand-in repositories only: no game, no training, and no fresh v2 contents."""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

from celeste_rl.tasks import resolve_task_definition, task_identity
from scripts import confirmation_historical_donors as hx
from scripts import make_confirmation_stage1_plans as g
from scripts import run_overnight as ro

REPO = Path(__file__).resolve().parents[1]
DECLARATION = json.loads((REPO / "config" / "retention-confirmation.json").read_text(encoding="utf-8"))
ROOM1 = task_identity(resolve_task_definition(None))
ROOM2 = task_identity(resolve_task_definition(REPO / "config" / "room2.json"))
PINS = {"path": g.V1, "file_sha256": "f" * 64, "manifest_sha256": "e" * 64, "states": 200}
COMMIT = "c" * 40
GAME = str(Path(g.GAME_COPIES[0][0]).resolve())


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(root: Path, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args], check=True,
                   capture_output=True)


def donor(j: int, seed: int) -> dict:
    return {"j": j, "seed": seed, "checkpoint": f"runs/train/donor-{seed}/checkpoints/latest.zip",
            "sha256": f"{seed:02d}" * 32}


class Temp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patches = [unittest.mock.patch.object(g, "REPO", self.root),
                        unittest.mock.patch.object(g, "CONFIG", self.root / "config"),
                        unittest.mock.patch.object(g, "STAGE1", self.root / "runs/confirmation/stage1"),
                        unittest.mock.patch.object(g, "CAMPAIGNS", self.root / "runs/campaign")]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()


# ------------------------------------------------------------------------------------------------- fixtures

def write_evaluation(root: Path, stamp: str, checkpoint_sha: str, value: float = 0.6, commit: str = COMMIT,
                     **changes) -> dict:
    folder = root / "runs/heldout-evaluation" / stamp
    folder.mkdir(parents=True)
    result = {"commit": commit, "uncommitted_changes": False, "changed_paths": [], "git_error": None,
              "runtime_problems": [], "attributable": True, "task": ROOM1, "checkpoint": "x.zip",
              "checkpoint_sha256": checkpoint_sha, "heldout_set": "config\\heldout_starts.json",
              "heldout_sha256": PINS["manifest_sha256"], "heldout_file_sha256": PINS["file_sha256"],
              "heldout_format_version": 3, "heldout_states": 200, "repeats": 1, "deterministic": False,
              "evaluation_seed": 20260920, "attempts": 200, "episodes": 200, "stale_starts": 0, "successes": 120,
              "success_rate": 0.9, "episodes_file": "episodes.jsonl", "route_macro_success_rate": value, **changes}
    (folder / "results.json").write_text(json.dumps(result), encoding="utf-8")
    (folder / "episodes.jsonl").write_text("{}\n", encoding="utf-8")
    return {"result_file": f"runs/heldout-evaluation/{stamp}/results.json",
            "result_sha256": sha(folder / "results.json"),
            "episodes_file": f"runs/heldout-evaluation/{stamp}/episodes.jsonl",
            "episodes_sha256": sha(folder / "episodes.jsonl")}


def write_recording(root: Path, stamp: str, item: dict, commit: str = COMMIT, frames_per: int = 2,
                    **changes) -> str:
    folder = root / "runs/policy-play/chapter-1-room-1" / stamp
    folder.mkdir(parents=True)
    frames = 25 * frames_per
    np.savez_compressed(folder / "dataset.npz", actions=np.zeros((frames, 24), np.int8),
                        trajectory=np.repeat(np.arange(25), frames_per), obs_context=np.zeros((frames, 2)))
    play = {"commit": commit, "uncommitted_changes": False, "changed_paths": [], "git_error": None,
            "runtime_problems": [], "attributable": True, "task": ROOM1, "checkpoint": item["checkpoint"],
            "checkpoint_sha256": item["sha256"], "seed": 20260926, "episodes": 25,
            "starts": "canonical only; no held-out state is read", "frames": frames, "endings": {"success": 25},
            "summaries": [{"episode": i, "ending": "success", "frames": frames_per} for i in range(25)],
            "dataset": "dataset.npz", "dataset_sha256": sha(folder / "dataset.npz"), **changes}
    (folder / "play.json").write_text(json.dumps(play), encoding="utf-8")
    return f"runs/policy-play/chapter-1-room-1/{stamp}"


def write_copy(root: Path, stamp: str, item: dict, entry: dict, commit: str = COMMIT, last_epoch: int = 299,
               **changes) -> dict:
    folder = root / "runs/clone/chapter-1-room-2" / stamp
    folder.mkdir(parents=True)
    command = entry["command"]
    args = {command[i][2:].replace("-", "_"): command[i + 1].replace("/", "\\") for i in range(1, len(command) - 1)
            if command[i].startswith("--") and not command[i + 1].startswith("--") and command[i] != "--mix-play"}
    recording = item["recording"]
    args.update(game_dir=GAME, routes_only="True", no_play="True", mix_room="None", allow_dirty="False",
                allow_runtime_mismatch="False", eval_episodes="50",
                mix_play=f"[WindowsPath('default'), WindowsPath('{recording['folder']}')]")
    result = {"commit": commit, "uncommitted_changes": False, "changed_paths": [], "git_error": None,
              "task": ROOM2, "args": args,
              "manifests": {"heldout": "config\\heldout_starts-room2-v2.json",
                            "heldout_sha256": g.ROOM2_V2_MANIFEST_SHA256},
              "init_from": {"path": item["checkpoint"], "sha256": item["sha256"]},
              "mix": {"room_weighting": "equal", "targets": "donor", "task": ROOM1, "states": "donor play",
                      "play": {"kind": "donor play", "play": recording["folder"],
                               "dataset_sha256": recording["dataset_sha256"], "episodes": 25, "seed": 20260926,
                               "recorded_at": recording["commit"]}},
              "cloning": {"epochs": 300, "history": [{"epoch": 0}, {"epoch": last_epoch}]}, **changes}
    (folder / "results.json").write_text(json.dumps(result), encoding="utf-8")
    (folder / "cloned.zip").write_bytes(b"copy " + stamp.encode())
    return {"result_file": f"runs/clone/chapter-1-room-2/{stamp}/results.json",
            "result_sha256": sha(folder / "results.json")}


def runner_record(entry: dict, status: str = "ok", **fields) -> dict:
    return {**entry, "status": status, "command": entry["command"] + ["--game-dir", GAME], **fields}


# ------------------------------------------------------------------------------------------------- the rules

class MarginTest(unittest.TestCase):
    """Amendment 3: route-macro minus the literal 0.142, rounded to 6 decimals, then >= 0.20."""

    def test_rounding_happens_before_the_comparison(self):
        self.assertFalse(g.passes(0.142 + 0.1999994))
        self.assertTrue(g.passes(0.142 + 0.1999996))
        self.assertTrue(g.passes(0.342))
        self.assertFalse(g.passes(0.341999))
        self.assertTrue(g.passes(0.342001))
        self.assertEqual(g.margin(0.5), 0.358)

    def test_route_macro_not_state_weighted(self):
        artifact = write_evaluation(self.tmp_root(), "20261005-100000", "a" * 64, value=0.30, success_rate=0.95)
        with unittest.mock.patch.object(g, "REPO", self.root):
            outcome, problems = g.evaluation_outcome({"artifact": artifact}, "a" * 64, COMMIT, PINS)
        self.assertEqual(problems, [])
        self.assertFalse(outcome["passes"])  # 0.30 - 0.142 < 0.20, whatever the state-weighted rate

    def tmp_root(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        return self.root


class TemplateTest(unittest.TestCase):
    def test_commands_are_the_declared_ones(self):
        commands = g.templates(DECLARATION)
        self.assertIn(commands["v1"], DECLARATION["donors"]["inclusion"]["stage1_v1"])
        self.assertTrue(DECLARATION["pipeline"]["record"].startswith(commands["record"]))
        self.assertTrue(DECLARATION["pipeline"]["copy"].startswith(commands["copy"]))
        for command in g.CODE_CHECKS.values():
            self.assertTrue(any(" ".join(command) in text for text in DECLARATION["code_checks"].values()))
        self.assertIn(g.ROOM2_V2_MANIFEST_SHA256, DECLARATION["running"]["v2_guard"])

    def test_plans_pass_the_runner_guard_and_a_v2_evaluation_does_not(self):
        item = {**donor(2, 9), "recording": {"folder": "runs/policy-play/chapter-1-room-1/20261005-100000",
                                             "dataset_sha256": "d" * 64, "commit": COMMIT}}
        with unittest.mock.patch.object(g, "v1_pins", return_value=PINS):
            for phase, items in (("donor-v1", [donor(0, 7)]), ("record", [donor(0, 7)]), ("copy", [item]),
                                 ("copy-v1", [{**item, "clone": {"checkpoint": "runs/clone/x/cloned.zip",
                                                                 "sha256": "b" * 64}}])):
                with self.subTest(phase=phase):
                    plan = g.build_plan(phase, 0, 1, items, [], DECLARATION)
                    self.assertEqual(plan["runs"][0]["limit_minutes"], g.LIMITS[phase])
                    self.assertEqual(len(plan["runner"]["game_copies"]), 1 if phase == "copy" else 3)
            plan = g.build_plan("donor-v1", 0, 1, [donor(0, 7)], [], DECLARATION)
        plan["runs"][0]["command"][plan["runs"][0]["command"].index("--starts") + 1] = \
            "config/heldout_starts-room1-v2.json"
        self.assertTrue(ro.fresh_set_problems(plan, [Path(g.GAME_COPIES[0][0])]))
        copy = g.build_plan("copy", 0, 1, [item], [], DECLARATION)
        self.assertEqual(copy["runs"][0]["command"][copy["runs"][0]["command"].index("--seed") + 1], "12")


class FreshRoleTest(unittest.TestCase):
    def test_only_the_allowed_fields_may_name_room2_v2(self):
        allowed = {("args", "heldout"): g.ROOM2_V2, ("manifests", "heldout"): g.ROOM2_V2}
        good = {"args": {"heldout": "config\\heldout_starts-room2-v2.json"},
                "manifests": {"heldout": "config/heldout_starts-room2-v2.json"}}
        self.assertEqual(g.fresh_mentions(good, allowed), [])
        for bad in ({"args": {"heldout": "config/heldout_starts-room1-v2.json"}},
                    {"args": {"dataset": "config/heldout_starts-room2-v2.json"}},
                    {"notes": ["x", "see config/heldout_starts-room2-v2.json"]}):
            with self.subTest(bad=bad):
                self.assertTrue(g.fresh_mentions(bad, allowed))
        self.assertTrue(g.fresh_mentions(good, {}))  # an evaluation or recording allows none


# -------------------------------------------------------------------------------------- outcome validation

class EvaluationOutcomeTest(Temp):
    def outcome(self, **changes):
        artifact = write_evaluation(self.root, f"20261005-1{len(list(self.root.rglob('results.json'))):05d}",
                                    "a" * 64, **changes)
        return g.evaluation_outcome({"artifact": artifact}, "a" * 64, COMMIT, PINS)

    def test_valid(self):
        outcome, problems = self.outcome(value=0.5)
        self.assertEqual(problems, [])
        self.assertEqual((outcome["margin"], outcome["passes"]), (0.358, True))

    def test_each_wrong_field_is_a_technical_failure(self):
        for name, changes in (("checkpoint", {"checkpoint_sha256": "b" * 64}), ("seed field", {"evaluation_seed": 1}),
                              ("repeats", {"repeats": 3}), ("deterministic", {"deterministic": True}),
                              ("task", {"task": ROOM2}), ("states", {"heldout_states": 199}),
                              ("attempts", {"attempts": 199}), ("file hash", {"heldout_file_sha256": "0" * 64}),
                              ("manifest", {"heldout_sha256": "0" * 64}), ("set", {"heldout_set": "config/other.json"}),
                              ("dirty", {"uncommitted_changes": True}), ("other commit", {"commit": "d" * 40}),
                              ("runtime", {"runtime_problems": ["x"]}), ("not attributable", {"attributable": False}),
                              ("nan", {"route_macro_success_rate": float("nan")}),
                              ("missing score", {"route_macro_success_rate": None}),
                              ("int score", {"route_macro_success_rate": 1})):
            with self.subTest(case=name):
                outcome, problems = self.outcome(**changes)
                self.assertIsNone(outcome)
                self.assertTrue(problems)

    def test_stale_starts_stop_for_review(self):
        with self.assertRaises(g.Refused):
            self.outcome(stale_starts=2, episodes=198)

    def test_changed_files_and_runner_problems(self):
        artifact = write_evaluation(self.root, "20261005-110000", "a" * 64)
        (self.root / artifact["episodes_file"]).write_text("changed\n", encoding="utf-8")
        self.assertIsNone(g.evaluation_outcome({"artifact": artifact}, "a" * 64, COMMIT, PINS)[0])
        self.assertIsNone(g.evaluation_outcome({"artifact": {**artifact, "problem": "x"}}, "a" * 64, COMMIT, PINS)[0])
        self.assertIsNone(g.evaluation_outcome({}, "a" * 64, COMMIT, PINS)[0])


class RecordingOutcomeTest(Temp):
    def setUp(self):
        super().setUp()
        self.item = donor(2, 9)

    def test_bound_by_its_own_announcement(self):
        folder = write_recording(self.root, "20261005-120000", self.item)
        outcome, problems = g.recording_outcome({"announced": [folder]}, self.item, COMMIT)
        self.assertEqual(problems, [])
        self.assertEqual(outcome["folder"], folder)
        self.assertEqual(outcome["dataset_sha256"], sha(self.root / folder / "dataset.npz"))
        for announced in ([], [folder, folder], [""], ["runs/elsewhere/20261005-120000"], None):
            with self.subTest(announced=announced):
                self.assertIsNone(g.recording_outcome({"announced": announced}, self.item, COMMIT)[0])

    def test_metadata_and_arrays_must_match(self):
        cases = {"other donor": {"checkpoint_sha256": "0" * 64}, "seed": {"seed": 1}, "episodes": {"episodes": 24},
                 "starts": {"starts": "held-out"}, "dirty": {"uncommitted_changes": True},
                 "frames": {"frames": 49}, "endings": {"endings": {"success": 24}},
                 "dataset hash": {"dataset_sha256": "0" * 64}, "task": {"task": ROOM2}}
        for index, (name, changes) in enumerate(cases.items()):
            with self.subTest(case=name):
                folder = write_recording(self.root, f"20261005-12{index:04d}", self.item, **changes)
                self.assertIsNone(g.recording_outcome({"announced": [folder]}, self.item, COMMIT)[0])


class CopyOutcomeTest(Temp):
    def setUp(self):
        super().setUp()
        recording = {"folder": "runs/policy-play/chapter-1-room-1/20261005-120000", "dataset_sha256": "d" * 64,
                     "commit": "e" * 40}
        self.item = {**donor(2, 9), "recording": recording}
        with unittest.mock.patch.object(g, "v1_pins", return_value=PINS):
            self.entry = g.build_plan("copy", 0, 1, [self.item], [], DECLARATION)["runs"][0]

    def outcome(self, stamp, **changes):
        artifact = write_copy(self.root, stamp, self.item, self.entry, **changes)
        return g.copy_outcome({"artifact": artifact}, self.item, COMMIT, self.entry)

    def test_valid_with_string_args_and_recorded_at_commit(self):
        outcome, problems = self.outcome("20261005-130000")
        self.assertEqual(problems, [])
        self.assertEqual(outcome["sha256"], sha(self.root / outcome["checkpoint"]))

    def test_wrong_fits_are_technical_failures(self):
        cases = {"not 300 epochs": {"last_epoch": 298},
                 "seed": {"args": None}, "recorded at a timestamp": {"mix": None},
                 "dirty": {"uncommitted_changes": True}, "task": {"task": ROOM1},
                 "v2 elsewhere": {"notes": "config/heldout_starts-room2-v2.json"}}
        for index, (name, changes) in enumerate(cases.items()):
            with self.subTest(case=name):
                stamp = f"20261005-13{index + 1:04d}"
                if name == "seed":
                    artifact = write_copy(self.root, stamp, self.item, self.entry)
                    result = json.loads((self.root / artifact["result_file"]).read_text(encoding="utf-8"))
                    result["args"]["seed"] = "13"
                    (self.root / artifact["result_file"]).write_text(json.dumps(result), encoding="utf-8")
                    artifact["result_sha256"] = sha(self.root / artifact["result_file"])
                    self.assertIsNone(g.copy_outcome({"artifact": artifact}, self.item, COMMIT, self.entry)[0])
                elif name == "recorded at a timestamp":
                    artifact = write_copy(self.root, stamp, self.item, self.entry)
                    result = json.loads((self.root / artifact["result_file"]).read_text(encoding="utf-8"))
                    result["mix"]["play"]["recorded_at"] = "20261005-120000"
                    (self.root / artifact["result_file"]).write_text(json.dumps(result), encoding="utf-8")
                    artifact["result_sha256"] = sha(self.root / artifact["result_file"])
                    self.assertIsNone(g.copy_outcome({"artifact": artifact}, self.item, COMMIT, self.entry)[0])
                else:
                    self.assertIsNone(self.outcome(stamp, **changes)[0])


# ------------------------------------------------------------------------- campaigns, phases and recovery

class PhaseTest(Temp):
    """donor-v1 in a stand-in repository: plan, audit, recovery revision, completion."""

    def setUp(self):
        super().setUp()
        (self.root / "config").mkdir()
        shutil.copy(REPO / "config" / "retention-confirmation.json", self.root / "config")
        git(self.root.parent, "init", "-q", str(self.root))
        self.donors = [donor(j, seed) for j, seed in enumerate(g.ORIGINAL_SEEDS)]
        self.more = [unittest.mock.patch.object(hx, "REPO", self.root),
                     unittest.mock.patch.object(g, "round_donors", return_value=self.donors),
                     unittest.mock.patch.object(g, "v1_pins", return_value=PINS),
                     unittest.mock.patch.object(g, "run_code_checks", return_value={})]
        for p in self.more:
            p.start()

    def tearDown(self):
        for p in reversed(self.more):
            p.stop()
        super().tearDown()

    def commit(self):
        git(self.root, "add", "-A")
        git(self.root, "commit", "-qm", "step")
        return subprocess.run(["git", "-C", str(self.root), "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip()

    def campaign(self, version: int, stamp: str, records: list[dict], commit: str, finished=True, extra_log=""):
        path = g.plan_path("donor-v1", 0, version)
        plan = json.loads(path.read_text(encoding="utf-8"))
        folder = self.root / "runs/campaign" / f"{stamp}-{plan['name']}"
        folder.mkdir(parents=True)
        starts = "".join(f"[t] start {r['id']}: x\n" for r in records)
        log = f"[t] campaign {plan['name']}, {len(plan['runs'])} runs, commit {commit[:8]}\n{starts}{extra_log}"
        (folder / "campaign.log").write_text(log + ("[t] campaign finished: 8 of 8\n" if finished else ""),
                                             encoding="utf-8")
        from celeste_rl.texthash import text_sha256
        (folder / "summary.json").write_text(json.dumps({"plan": plan["name"], "plan_sha256": text_sha256(path),
                                                         "commit": commit, "results": records}), encoding="utf-8")
        return plan

    def evaluated(self, plan, commit, fail_j=(), stamp="2026100512"):
        records = []
        for entry in plan["runs"]:
            if entry["j"] in fail_j:
                records.append(runner_record(entry, status="exit_1"))
                continue
            artifact = write_evaluation(self.root, f"{stamp}{entry['j']:04d}", self.donors[entry["j"]]["sha256"],
                                        value=0.5, commit=commit)
            records.append(runner_record(entry, artifact=artifact))
        return records

    def test_plan_recovery_and_completion(self):
        self.assertIn("written (8 entries)", g.advance("donor-v1", 0))
        self.assertIn("waiting to run", g.advance("donor-v1", 0))  # identical rerun: nothing new
        commit = self.commit()
        plan = json.loads(g.plan_path("donor-v1", 0, 1).read_text(encoding="utf-8"))
        self.campaign(1, "20261005-120000", self.evaluated(plan, commit, fail_j=(3,)), commit)
        self.assertIn("written (1 entries)", g.advance("donor-v1", 0))  # recovery of j 3 only
        recovery = json.loads(g.plan_path("donor-v1", 0, 2).read_text(encoding="utf-8"))
        self.assertEqual([e["j"] for e in recovery["runs"]], [3])
        self.assertEqual(recovery["runs"][0]["command"], plan["runs"][3]["command"])  # same command and seed
        commit = self.commit()
        self.campaign(2, "20261005-130000", self.evaluated(recovery, commit, stamp="2026100513"), commit)
        self.assertIn("written (8 items)", g.advance("donor-v1", 0))
        completion = json.loads(g.completion_path("donor-v1", 0).read_text(encoding="utf-8"))
        self.assertEqual([i["j"] for i in completion["items"]], list(range(8)))
        self.assertEqual([f["j"] for f in completion["failures"]], [3])  # the failed attempt is kept
        self.assertEqual(completion["items"][3]["evidence"]["plan"], "config/campaign-confirmation-stage1-donor-v1-r0-v2.json")
        self.assertIn("exists", g.advance("donor-v1", 0))

    def test_uncertain_or_inconsistent_campaigns_stop(self):
        g.advance("donor-v1", 0)
        commit = self.commit()
        plan = json.loads(g.plan_path("donor-v1", 0, 1).read_text(encoding="utf-8"))
        records = self.evaluated(plan, commit)
        cases = {"still running or interrupted": dict(finished=False),
                 "missing entry": dict(records=records[:7]),
                 "started without outcome": dict(records=records[:7], extra_log="[t] start s1-r0-donor-v1-j7-seed14: x\n"),
                 "wrong commit": dict(commit="0" * 40)}
        for index, (name, options) in enumerate(cases.items()):
            with self.subTest(case=name):
                shutil.rmtree(self.root / "runs/campaign", ignore_errors=True)
                self.campaign(1, f"20261005-14{index:04d}", options.get("records", records),
                              options.get("commit", commit), finished=options.get("finished", True),
                              extra_log=options.get("extra_log", ""))
                with self.assertRaises(g.Refused):
                    g.advance("donor-v1", 0)

    def test_dry_runs_are_ignored_and_a_second_execution_stops(self):
        g.advance("donor-v1", 0)
        commit = self.commit()
        plan = json.loads(g.plan_path("donor-v1", 0, 1).read_text(encoding="utf-8"))
        dry = self.root / "runs/campaign" / f"20261005-110000-{plan['name']}"
        dry.mkdir(parents=True)
        (dry / "campaign.log").write_text(f"[t] campaign {plan['name']}, 8 runs\n[t] dry run: nothing executed\n",
                                          encoding="utf-8")
        self.assertIn("waiting to run", g.advance("donor-v1", 0))
        records = self.evaluated(plan, commit)
        self.campaign(1, "20261005-120000", records, commit)
        self.campaign(1, "20261005-130000", records, commit)
        with self.assertRaises(g.Refused):
            g.advance("donor-v1", 0)

    def test_an_edited_unexecuted_plan_stops(self):
        g.advance("donor-v1", 0)
        path = g.plan_path("donor-v1", 0, 1)
        path.write_text(path.read_text(encoding="utf-8").replace('"limit_minutes": 30', '"limit_minutes": 31', 1),
                        encoding="utf-8")
        with self.assertRaises(g.Refused):
            g.advance("donor-v1", 0)

    def test_record_phase_takes_only_passing_donors(self):
        items = []
        for d in self.donors:
            value = 0.30 if d["j"] in (1, 5) else 0.5
            items.append({**d, "outcome": {"route_macro_success_rate": value, "margin": g.margin(value),
                                           "passes": g.passes(value)}, "evidence": {}})
        g.write_new(g.completion_path("donor-v1", 0), {"items": items})
        record_items, _ = g.phase_items("record", 0, DECLARATION)
        self.assertEqual([i["j"] for i in record_items], [0, 2, 3, 4, 6, 7])


# ------------------------------------------------------------------------------------------- inclusion

class InclusionTest(Temp):
    """Amendment 3: donor-line exclusions before copy-line exclusions, by j; seeds 15 then 16; j inherited."""

    def complete(self, round_: int, donors: dict[int, tuple], copies: dict[int, float]):
        def outcome(value):
            return {"route_macro_success_rate": value, "margin": g.margin(value), "passes": g.passes(value)}
        g.write_new(g.completion_path("donor-v1", round_), {"items": [
            {"j": j, "seed": seed, "sha256": f"{seed:02d}" * 32, "outcome": outcome(value)}
            for j, (seed, value) in donors.items()]})
        for phase in ("record", "copy"):
            g.write_new(g.completion_path(phase, round_), {"items": []})
        g.write_new(g.completion_path("copy-v1", round_), {"items": [
            {"j": j, "outcome": outcome(value)} for j, value in copies.items()]})

    def test_allocation_order_shared_budget_and_end_state(self):
        donors = {j: (seed, 0.30 if j in (5, 3) else 0.5) for j, seed in enumerate(g.ORIGINAL_SEEDS)}
        copies = {j: (0.30 if j == 1 else 0.5) for j in donors if j not in (5, 3)}
        self.complete(0, donors, copies)
        first = g.inclusion(0)
        self.assertEqual([(p["seed"], p["j"], p["replaces"]) for p in first["proposals"]], [(15, 3, 10), (16, 5, 12)])
        self.assertEqual(first["slots"]["1"]["active"]["status"], "excluded: copy line; no replacement left")
        self.assertEqual(first["reserved_seeds"], [15, 16])
        self.assertEqual(g.inclusion(0), first)  # an identical rerun gives the same record
        self.complete(1, {3: (15, 0.5), 5: (16, 0.5)}, {3: 0.5, 5: 0.30})
        second = g.inclusion(1)
        self.assertEqual(second["proposals"], [])
        self.assertEqual(second["slots"]["5"]["active"]["status"], "excluded: copy line; no replacement left")
        self.assertEqual(second["slots"]["3"]["history"][0]["seed"], 10)  # the superseded donor is kept
        self.assertEqual(second["qualified"], [0, 2, 3, 4, 6, 7])
        self.assertTrue(second["status"].startswith("stage 1 complete with 6"))

    def test_fewer_than_six_stops_and_technical_failures_never_allocate(self):
        donors = {j: (seed, 0.30 if j < 4 else 0.5) for j, seed in enumerate(g.ORIGINAL_SEEDS)}
        self.complete(0, donors, {j: 0.5 for j in range(4, 8)})
        record = g.inclusion(0)
        self.assertEqual([p["j"] for p in record["proposals"]], [0, 1])
        self.assertIn("replacements proposed", record["status"])
        self.complete(1, {0: (15, 0.30), 1: (16, 0.5)}, {1: 0.5})
        self.assertTrue(g.inclusion(1)["status"].startswith("stop for review: 5 qualified"))

    def test_needs_every_phase_complete(self):
        g.write_new(g.completion_path("donor-v1", 0), {"items": []})
        with self.assertRaises(g.Refused):
            g.inclusion(0)


class WriteNewTest(Temp):
    def test_exclusive_identical_or_refused(self):
        path = self.root / "x.json"
        self.assertEqual(g.write_new(path, {"a": 1}), "written")
        self.assertEqual(g.write_new(path, {"a": 1}), "exists")
        with self.assertRaises(g.Refused):
            g.write_new(path, {"a": True})  # type-preserving: true is not 1


# --------------------------------------------------------------------------------------------- preflight

class PreflightTest(Temp):
    """The reviewed control code, the plan on the remote branch, and no earlier execution."""

    def setUp(self):
        super().setUp()
        (self.root / "config").mkdir()
        (self.root / "scripts").mkdir()
        shutil.copy(REPO / "config" / "retention-confirmation.json", self.root / "config")
        (self.root / "scripts" / "control.py").write_text("RULE = 1\n", encoding="utf-8")
        git(self.root.parent, "init", "-q", str(self.root))
        self.approved = self.commit("reviewed code")
        self.plan_file = g.plan_path("donor-v1", 0, 1)
        self.plan = {"name": g.plan_name("donor-v1", 0, 1), "runs": []}
        self.plan_file.write_text(json.dumps(self.plan), encoding="utf-8")
        self.plan_commit = self.commit("plan")
        git(self.root, "update-ref", "refs/remotes/origin/proposal/ppo-anchor", self.plan_commit)
        invocation = [sys.executable, "-c", "print('dry')", "--copy", "C:/x:1:2"]
        self.more = [unittest.mock.patch.object(hx, "REPO", self.root),
                     unittest.mock.patch.object(g, "control_files",
                                                return_value=["config/retention-confirmation.json",
                                                              "scripts/control.py"]),
                     unittest.mock.patch.object(g, "expected_plan", side_effect=lambda *a: dict(self.plan)),
                     unittest.mock.patch.object(g, "run_code_checks", return_value={"evaluation": "PASS"}),
                     unittest.mock.patch.object(g, "runner_invocation", return_value=invocation),
                     unittest.mock.patch.object(g.rc, "thread_probe", return_value={"torch_intra_op_threads": 10})]
        for p in self.more:
            p.start()

    def tearDown(self):
        for p in reversed(self.more):
            p.stop()
        super().tearDown()

    def commit(self, message):
        git(self.root, "add", "-A")
        git(self.root, "commit", "-qm", message)
        return subprocess.run(["git", "-C", str(self.root), "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip()

    def test_passes_and_records(self):
        record = g.preflight(self.plan_file, self.approved, "origin/proposal/ppo-anchor")
        self.assertEqual(record["approved_commit"], self.approved)
        self.assertEqual(record["plan"]["commit"], self.plan_commit)
        self.assertTrue((self.root / record["record"]).exists())

    def test_unpushed_plan_is_refused(self):
        git(self.root, "update-ref", "refs/remotes/origin/proposal/ppo-anchor", self.approved)
        with self.assertRaises(g.Refused):
            g.preflight(self.plan_file, self.approved, "origin/proposal/ppo-anchor")

    def test_unreviewed_helper_change_is_refused(self):
        (self.root / "scripts" / "control.py").write_text("RULE = 2\n", encoding="utf-8")
        with self.assertRaises(g.Refused):  # a working change
            g.preflight(self.plan_file, self.approved, "origin/proposal/ppo-anchor")
        head = self.commit("helper changed after review")
        git(self.root, "update-ref", "refs/remotes/origin/proposal/ppo-anchor", head)
        with self.assertRaises(g.Refused):  # committed and pushed, but not the approved version
            g.preflight(self.plan_file, self.approved, "origin/proposal/ppo-anchor")

    def test_a_plan_that_has_run_is_refused(self):
        with unittest.mock.patch.object(g, "audit", return_value={"kind": "executed"}), \
                unittest.mock.patch.object(g, "campaign_folders", return_value=[self.root]):
            with self.assertRaises(g.Refused):
                g.preflight(self.plan_file, self.approved, "origin/proposal/ppo-anchor")

    def test_a_plan_not_derived_by_the_generator_is_refused(self):
        with unittest.mock.patch.object(g, "expected_plan", return_value={"name": "other"}):
            with self.assertRaises(g.Refused):
                g.preflight(self.plan_file, self.approved, "origin/proposal/ppo-anchor")


if __name__ == "__main__":
    unittest.main()
