"""The stage 1 plan generator (scripts/make_confirmation_stage1_plans.py; amendment 3). Synthetic fixtures and
stand-in git repositories only: no game, no training, and no fresh v2 contents."""
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from celeste_rl.tasks import resolve_task_definition, task_identity
from celeste_rl.texthash import text_sha256
from scripts import confirmation_historical_donors as hx
from scripts import make_confirmation_stage1_plans as g
from scripts import run_overnight as ro

REPO = Path(__file__).resolve().parents[1]
DECLARATION = json.loads((REPO / "config" / "retention-confirmation.json").read_text(encoding="utf-8"))
ROOM1 = task_identity(resolve_task_definition(None))
ROOM2 = task_identity(resolve_task_definition(REPO / "config" / "room2.json"))
PINS = {"path": g.V1, "file_sha256": "f" * 64, "manifest_sha256": "e" * 64, "states": 200}
SIDECAR = {"format_version": 1, "dataset_sha256": "a" * 64, "demonstrations_manifest_sha256": "b" * 64,
           "heldout_manifest_sha256": g.ROOM2_V2_MANIFEST_SHA256, "demonstration_route_sha256s": ["c" * 64],
           "task": ROOM2}
COPY_INPUTS = {"room2_task": {"path": g.ROOM2_TASK, "text_sha256": "1" * 64, "identity": ROOM2},
               "source_route": {"path": "tests/fixtures/route.json", "sha256": "2" * 64},
               "dataset": {"path": g.COPY_DATASET, "sha256": "a" * 64},
               "dataset_sidecar": {"path": "runs/clone/x/dataset.manifest.json", "sha256": "3" * 64,
                                   "content": SIDECAR},
               "demonstrations": {"path": g.COPY_DEMONSTRATIONS, "file_sha256": "4" * 64, "manifest_sha256": "b" * 64},
               "room2_v2": {"path": g.ROOM2_V2, "manifest_sha256": g.ROOM2_V2_MANIFEST_SHA256,
                            "file_sha256": g.ROOM2_V2_FILE_SHA256, "role": "audit only"}}
COMMIT = "c" * 40
GAME = str(Path(g.GAME_COPIES[0][0]).resolve())


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(root: Path, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args], check=True,
                   capture_output=True)


def donor(j: int, seed: int) -> dict:
    return {"j": j, "seed": seed, "checkpoint": f"runs/train/donor-{seed}/checkpoints/latest.zip",
            "sha256": f"{seed:02d}" * 32, "donor_source": {"declared pin": f"test {seed}"}}


def donor_manifest(seed: int) -> dict:
    return {"path": f"runs/train/donor-{seed}/manifest.json", "sha256": "5" * 64, "status": "finished",
            "accepted_steps": 501760, "commits": [hx.HISTORICAL_COMMIT], "schema_fingerprint": "b710a27fa4f1d96f",
            "disabled_inputs": ["S", "Q", "N"]}


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


def recorder_arrays(frames_per: int = 2, drop: str | None = None, absent_episode: int | None = None) -> dict:
    trajectory = np.repeat(np.arange(25), frames_per).astype("<i8")
    if absent_episode is not None:
        trajectory[trajectory == absent_episode] = absent_episode - 1 if absent_episode else 1
    frames = len(trajectory)
    arrays = {"actions": np.zeros((frames, 24), np.int8), "trajectory": trajectory,
              "obs_player": np.zeros((frames, 4, 75), "<f4"), "obs_actions": np.zeros((frames, 4, 24), np.int8),
              "obs_history_valid": np.zeros((frames, 4), np.int8), "obs_grid": np.zeros((frames, 11, 32, 32), np.uint8),
              "obs_context": np.zeros((frames, 2), "<f4")}
    if drop:
        del arrays[drop]
    return arrays


def write_recording(root: Path, stamp: str, item: dict, commit: str = COMMIT, arrays: dict | None = None,
                    **changes) -> str:
    folder = root / "runs/policy-play/chapter-1-room-1" / stamp
    folder.mkdir(parents=True)
    arrays = arrays or recorder_arrays()
    np.savez_compressed(folder / "dataset.npz", **arrays)
    frames = 50
    play = {"commit": commit, "uncommitted_changes": False, "changed_paths": [], "git_error": None,
            "runtime_problems": [], "attributable": True, "task": ROOM1, "checkpoint": item["checkpoint"],
            "checkpoint_sha256": item["sha256"], "seed": 20260926, "episodes": 25, "starts": g.STARTS,
            "frames": frames, "endings": {"success": 25},
            "summaries": [{"episode": i, "ending": "success", "frames": 2} for i in range(25)],
            "dataset": "dataset.npz", "dataset_sha256": sha(folder / "dataset.npz"), **changes}
    (folder / "play.json").write_text(json.dumps(play), encoding="utf-8")
    return f"runs/policy-play/chapter-1-room-1/{stamp}"


def write_copy(root: Path, stamp: str, item: dict, entry: dict, inputs: dict = COPY_INPUTS, commit: str = COMMIT,
               last_epoch: int = 299, edit=None) -> dict:
    folder = root / "runs/clone/chapter-1-room-2" / stamp
    folder.mkdir(parents=True)
    command = entry["command"]
    args = {command[i][2:].replace("-", "_"): command[i + 1].replace("/", "\\") for i in range(1, len(command) - 1)
            if command[i].startswith("--") and not command[i + 1].startswith("--") and command[i] != "--mix-play"}
    recording = item["recording"]
    args.update(game_dir=GAME, routes_only="True", no_play="True", mix_room="None", allow_dirty="False",
                allow_runtime_mismatch="False", eval_episodes="50",
                mix_play=f"[WindowsPath('default'), WindowsPath('{recording['folder']}')]")
    play = {"kind": "donor play", "play": recording["folder"], "dataset_sha256": recording["dataset_sha256"],
            "episodes": 25, "seed": 20260926, "recorded_at": recording["commit"]}
    manifest = item["donor_manifest"]
    result = {"commit": commit, "uncommitted_changes": False, "changed_paths": [], "git_error": None,
              "task": inputs["room2_task"]["identity"], "args": args,
              "manifests": {"demonstrations": "config\\demonstrations-room2-v2.json",
                            "demonstrations_sha256": inputs["demonstrations"]["manifest_sha256"],
                            "demonstrations_file_sha256": inputs["demonstrations"]["file_sha256"],
                            "heldout": "config\\heldout_starts-room2-v2.json",
                            "heldout_sha256": g.ROOM2_V2_MANIFEST_SHA256, "heldout_file_sha256": g.ROOM2_V2_FILE_SHA256},
              "provenance": [{"dataset": inputs["dataset"]["path"].replace("/", "\\"),
                              **inputs["dataset_sidecar"]["content"]}, dict(play)],
              "init_from": {"path": item["checkpoint"], "sha256": item["sha256"],
                            "provenance": {"kind": "training run", "manifest": manifest["path"],
                                           "status": manifest["status"], "commits": manifest["commits"],
                                           "schema_fingerprint": manifest["schema_fingerprint"],
                                           "disabled_inputs": manifest["disabled_inputs"],
                                           "accepted_steps": manifest["accepted_steps"]}},
              "mix": {"room_weighting": "equal", "targets": "donor", "task": ROOM1, "states": "donor play",
                      "play": dict(play)},
              "cloning": {"epochs": 300, "history": [{"epoch": 0}, {"epoch": last_epoch}]}}
    if edit:
        edit(result)
    (folder / "results.json").write_text(json.dumps(result), encoding="utf-8")
    (folder / "cloned.zip").write_bytes(b"copy " + stamp.encode())
    return {"result_file": f"runs/clone/chapter-1-room-2/{stamp}/results.json",
            "result_sha256": sha(folder / "results.json")}


def runner_record(entry: dict, status: str = "ok", game: str = GAME, **fields) -> dict:
    return {**entry, "status": status, "command": entry["command"] + ["--game-dir", game], **fields}


def build(phase, items, **kw):
    with unittest.mock.patch.object(g, "v1_pins", return_value=PINS), \
            unittest.mock.patch.object(g, "copy_input_pins", return_value=COPY_INPUTS):
        return g.build_plan(phase, 0, 1, items, [], [], DECLARATION, kw.get("stop_time"))


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


class TemplateTest(unittest.TestCase):
    def test_commands_and_pins_are_the_declared_ones(self):
        commands = g.templates(DECLARATION)
        self.assertIn(commands["v1"], DECLARATION["donors"]["inclusion"]["stage1_v1"])
        self.assertTrue(DECLARATION["pipeline"]["record"].startswith(commands["record"]))
        self.assertTrue(DECLARATION["pipeline"]["copy"].startswith(commands["copy"]))
        for command in g.CODE_CHECKS.values():
            self.assertTrue(any(" ".join(command) in text for text in DECLARATION["code_checks"].values()))
        self.assertIn(g.ROOM2_V2_MANIFEST_SHA256, DECLARATION["running"]["v2_guard"])

    def test_plans_pass_the_runner_guard_and_a_v2_evaluation_does_not(self):
        item = {**donor(2, 9), "donor_manifest": donor_manifest(9),
                "recording": {"folder": "runs/policy-play/chapter-1-room-1/20261005-100000",
                              "dataset_sha256": "d" * 64, "commit": COMMIT}}
        for phase, items in (("donor-v1", [donor(0, 7)]), ("record", [donor(0, 7)]), ("copy", [item]),
                             ("copy-v1", [{**item, "clone": {"checkpoint": "runs/clone/x/cloned.zip", "sha256": "b" * 64}}])):
            with self.subTest(phase=phase):
                plan = build(phase, items)
                self.assertEqual(plan["runs"][0]["limit_minutes"], g.LIMITS[phase])
                self.assertEqual(len(plan["runner"]["game_copies"]), 1 if phase == "copy" else 3)
        plan = build("donor-v1", [donor(0, 7)])
        plan["runs"][0]["command"][plan["runs"][0]["command"].index("--starts") + 1] = "config/heldout_starts-room1-v2.json"
        self.assertTrue(ro.fresh_set_problems(plan, [Path(g.GAME_COPIES[0][0])]))
        self.assertEqual(build("copy", [item])["runs"][0]["command"][
            build("copy", [item])["runs"][0]["command"].index("--seed") + 1], "12")

    def test_stop_time(self):
        now = datetime(2026, 10, 5, 9, 0)
        self.assertEqual(g.stop_time_problems(None, "donor-v1", now), [])
        self.assertEqual(g.stop_time_problems("2026-10-05T18:00", "copy", now), [])
        for bad in ("tomorrow", "2026-10-05T09:20", "2026-10-05T18:00+01:00"):
            with self.subTest(bad=bad):
                self.assertTrue(g.stop_time_problems(bad, "donor-v1", now))
        self.assertEqual(build("donor-v1", [donor(0, 7)], stop_time="2026-10-05T18:00")["stop_time"], "2026-10-05T18:00:00")


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
        self.assertTrue(g.fresh_mentions(good, {}))
        plan = build("copy", [{**donor(2, 9), "donor_manifest": donor_manifest(9), "recording": {
            "folder": "runs/policy-play/chapter-1-room-1/20261005-100000", "dataset_sha256": "d" * 64, "commit": COMMIT}}])
        self.assertEqual(g.fresh_mentions(plan, g.plan_roles("copy")), [])
        self.assertTrue(g.fresh_mentions(plan, g.plan_roles("donor-v1")))


# -------------------------------------------------------------------------------------- outcome validation

class EvaluationOutcomeTest(Temp):
    def outcome(self, **changes):
        artifact = write_evaluation(self.root, f"20261005-1{len(list(self.root.rglob('results.json'))):05d}",
                                    "a" * 64, **changes)
        return g.evaluation_outcome({"artifact": artifact}, "a" * 64, COMMIT, PINS)

    def test_valid_and_route_macro_not_state_weighted(self):
        outcome, problems = self.outcome(value=0.5, success_rate=0.1)
        self.assertEqual((problems, outcome["margin"], outcome["passes"]), ([], 0.358, True))
        self.assertFalse(self.outcome(value=0.30, success_rate=0.95)[0]["passes"])

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

    def test_changed_or_missing_files(self):
        artifact = write_evaluation(self.root, "20261005-110000", "a" * 64)
        (self.root / artifact["episodes_file"]).write_text("changed\n", encoding="utf-8")
        self.assertIsNone(g.evaluation_outcome({"artifact": artifact}, "a" * 64, COMMIT, PINS)[0])
        self.assertIsNone(g.evaluation_outcome({"artifact": {**artifact, "problem": "x"}}, "a" * 64, COMMIT, PINS)[0])
        (self.root / artifact["result_file"]).unlink()
        item = {"j": 0, "seed": 7, "sha256": "a" * 64}
        entry = {"id": "x", "command": ["scripts/evaluate_heldout.py"]}
        plan = {"inputs": {"v1": PINS}, "runner": {"game_copies": [{"game_dir": g.GAME_COPIES[0][0], "ports": [1, 2]}]}}
        outcome, problems = g.outcome_for("donor-v1", runner_record(entry, artifact=artifact), entry, item, COMMIT, plan)
        self.assertIsNone(outcome)  # a missing result is a technical failure to recover, not a crash
        self.assertIn("FileNotFoundError", problems[0])


class RecordingOutcomeTest(Temp):
    def setUp(self):
        super().setUp()
        self.item = donor(2, 9)

    def test_bound_by_its_own_announcement(self):
        folder = write_recording(self.root, "20261005-120000", self.item)
        outcome, problems = g.recording_outcome({"announced": [folder]}, self.item, COMMIT)
        self.assertEqual(problems, [])
        self.assertEqual(outcome["dataset_sha256"], sha(self.root / folder / "dataset.npz"))
        for announced in ([], [folder, folder], [""], ["runs/elsewhere/20261005-120000"], None):
            with self.subTest(announced=announced):
                self.assertIsNone(g.recording_outcome({"announced": announced}, self.item, COMMIT)[0])

    def test_metadata_must_match(self):
        cases = {"other donor": {"checkpoint_sha256": "0" * 64}, "seed": {"seed": 1}, "episodes": {"episodes": 24},
                 "starts": {"starts": "canonical starts"}, "dirty": {"uncommitted_changes": True},
                 "endings": {"endings": {"success": 24}}, "dataset hash": {"dataset_sha256": "0" * 64},
                 "task": {"task": ROOM2}}
        for index, (name, changes) in enumerate(cases.items()):
            with self.subTest(case=name):
                folder = write_recording(self.root, f"20261005-12{index:04d}", self.item, **changes)
                self.assertIsNone(g.recording_outcome({"announced": [folder]}, self.item, COMMIT)[0])

    def test_the_full_recorder_schema_is_required(self):
        """Review B6: every array, its shape and dtype, and episodes 0 to 24 each present."""
        bad_dtype = recorder_arrays()
        bad_dtype["obs_grid"] = bad_dtype["obs_grid"].astype(np.float32)
        bad_shape = recorder_arrays()
        bad_shape["obs_player"] = np.zeros((50, 4, 74), "<f4")
        cases = {"missing obs_grid": recorder_arrays(drop="obs_grid"), "missing obs_player": recorder_arrays(drop="obs_player"),
                 "absent last episode": recorder_arrays(absent_episode=24),
                 "absent first episode": recorder_arrays(absent_episode=0), "dtype": bad_dtype, "shape": bad_shape}
        for index, (name, arrays) in enumerate(cases.items()):
            with self.subTest(case=name):
                folder = write_recording(self.root, f"20261005-13{index:04d}", self.item, arrays=arrays)
                self.assertIsNone(g.recording_outcome({"announced": [folder]}, self.item, COMMIT)[0])

    def test_malformed_dataset_is_a_technical_failure(self):
        folder = write_recording(self.root, "20261005-140000", self.item)
        (self.root / folder / "dataset.npz").write_bytes(b"not a zip")
        play = json.loads((self.root / folder / "play.json").read_text(encoding="utf-8"))
        play["dataset_sha256"] = sha(self.root / folder / "dataset.npz")
        (self.root / folder / "play.json").write_text(json.dumps(play), encoding="utf-8")
        entry = {"id": "x", "command": ["scripts/record_policy_play.py"]}
        plan = {"runner": {"game_copies": [{"game_dir": g.GAME_COPIES[0][0], "ports": [1, 2]}]}}
        outcome, problems = g.outcome_for("record", runner_record(entry, announced=[folder]), entry, self.item,
                                          COMMIT, plan)
        self.assertIsNone(outcome)
        self.assertIn("artifact could not be validated", problems[0])


class CopyOutcomeTest(Temp):
    def setUp(self):
        super().setUp()
        recording = {"folder": "runs/policy-play/chapter-1-room-1/20261005-120000", "dataset_sha256": "d" * 64,
                     "commit": "e" * 40}
        self.item = {**donor(2, 9), "donor_manifest": donor_manifest(9), "recording": recording}
        self.entry = build("copy", [self.item])["runs"][0]

    def outcome(self, stamp, **kw):
        artifact = write_copy(self.root, stamp, self.item, self.entry, **kw)
        return g.copy_outcome({"artifact": artifact}, self.item, COMMIT, self.entry, COPY_INPUTS)

    def test_valid_with_complete_provenance(self):
        outcome, problems = self.outcome("20261005-130000")
        self.assertEqual(problems, [])
        self.assertEqual(outcome["sha256"], sha(self.root / outcome["checkpoint"]))

    def test_missing_or_wrong_audits_are_technical_failures(self):
        """Review B5: the clone's consumed-input audit must equal the pinned inputs."""
        def drop(*path):
            def edit(result):
                target = result
                for key in path[:-1]:
                    target = target[key]
                del target[path[-1]]
            return edit

        def change(value, *path):
            def edit(result):
                target = result
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
            return edit
        cases = {"no provenance": drop("provenance"), "no demonstrations audit": drop("manifests", "demonstrations_sha256"),
                 "other dataset": change("0" * 64, "provenance", 0, "dataset_sha256"),
                 "other route hashes": change(["9" * 64], "provenance", 0, "demonstration_route_sha256s"),
                 "other demonstrations file": change("0" * 64, "manifests", "demonstrations_file_sha256"),
                 "other v2 audit": change("0" * 64, "manifests", "heldout_file_sha256"),
                 "other donor manifest": change("runs/train/x/manifest.json", "init_from", "provenance", "manifest"),
                 "newer donor commit": change(["0" * 40], "init_from", "provenance", "commits"),
                 "play mismatch": change("1" * 64, "provenance", 1, "dataset_sha256"),
                 "seed": change("13", "args", "seed"), "recorded at a timestamp": change("20261005-120000", "mix", "play", "recorded_at"),
                 "dirty": change(True, "uncommitted_changes"), "task": change(ROOM1, "task"),
                 "v2 elsewhere": change("config/heldout_starts-room2-v2.json", "notes")}
        for index, (name, edit) in enumerate(cases.items()):
            with self.subTest(case=name):
                self.assertIsNone(self.outcome(f"20261005-13{index + 1:04d}", edit=edit)[0])
        self.assertIsNone(self.outcome("20261005-139999", last_epoch=298)[0])


# ------------------------------------------------------------------------- campaigns, phases and recovery

class Chain(Temp):
    """A stand-in repository where whole phases are generated, committed, preflighted, run and audited."""

    def setUp(self):
        super().setUp()
        (self.root / "config").mkdir()
        shutil.copy(REPO / "config" / "retention-confirmation.json", self.root / "config")
        git(self.root.parent, "init", "-q", str(self.root))
        self.donors = [donor(j, seed) for j, seed in enumerate(g.ORIGINAL_SEEDS)]
        for d in self.donors:
            (self.root / Path(d["checkpoint"]).parent.parent).mkdir(parents=True)
        self.clock = 0
        self.more = [unittest.mock.patch.object(hx, "REPO", self.root),
                     unittest.mock.patch.object(g, "round_donors", side_effect=lambda d, r: copy.deepcopy(self.donors)),
                     unittest.mock.patch.object(g, "v1_pins", return_value=PINS),
                     unittest.mock.patch.object(g, "copy_input_pins", return_value=COPY_INPUTS),
                     unittest.mock.patch.object(g, "donor_manifest",
                                                side_effect=lambda item: donor_manifest(item["seed"])),
                     unittest.mock.patch.object(g, "run_code_checks", return_value={})]
        for p in self.more:
            p.start()

    def tearDown(self):
        for p in reversed(self.more):
            p.stop()
        super().tearDown()

    def stamp(self) -> str:
        self.clock += 1
        return (datetime(2026, 10, 5, 8) + timedelta(minutes=self.clock)).strftime("%Y%m%d-%H%M%S")

    def commit(self) -> str:
        git(self.root, "add", "-A")
        git(self.root, "commit", "-qm", "step", "--allow-empty")
        return subprocess.run(["git", "-C", str(self.root), "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip()

    def receipt(self, plan_file: Path, plan: dict, commit: str, **changes):
        blob = subprocess.run(["git", "-C", str(self.root), "rev-parse", f"{commit}:config/retention-confirmation.json"],
                              capture_output=True, text=True).stdout.strip()
        record = {"plan": {"path": g.rel(plan_file), "text_sha256": text_sha256(plan_file)}, "approved_commit": commit,
                  "head": commit, "control_files": {"config/retention-confirmation.json": blob},
                  "invocation": g.runner_invocation(plan_file, plan["phase"]), **changes}
        g.write_new(g.STAGE1 / f"preflight-{plan['name']}-{self.stamp()}.json", record)

    def run_campaign(self, plan_file: Path, records_for, commit: str, receipt: bool = True, threads: int = 10,
                     finished: bool = True, extra_log: str = "", drop: int = 0, copies=None) -> Path:
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
        if receipt:
            self.receipt(plan_file, plan, commit)
        folder = g.CAMPAIGNS / f"{self.stamp()}-{plan['name']}"
        folder.mkdir(parents=True)
        records = records_for(plan, commit)[:len(plan["runs"]) - drop]
        starts = "".join(f"[t] start {r['id']}: x\n" for r in records if r["status"] != "skipped_out_of_time")
        log = f"[t] campaign {plan['name']}, {len(plan['runs'])} runs\n{starts}{extra_log}"
        (folder / "campaign.log").write_text(log + ("[t] campaign finished: n\n" if finished else ""), encoding="utf-8")
        parallel = {"copies": copies if copies is not None else plan["runner"]["game_copies"], "memory_floor_gb": 2.5}
        (folder / "summary.json").write_text(json.dumps({
            "plan": plan["name"], "plan_sha256": text_sha256(plan_file), "commit": commit, "results": records,
            "threads_per_job": threads, "parallel": parallel}), encoding="utf-8")
        return folder

    def evaluations(self, fail=(), skip=(), value=lambda j: 0.5, sha_of=None):
        def records_for(plan, commit):
            out = []
            for entry in plan["runs"]:
                if entry["j"] in skip:
                    out.append({**entry, "status": "skipped_out_of_time", "seconds": 0})
                elif entry["j"] in fail:
                    out.append(runner_record(entry, status="exit_1"))
                else:
                    expected = sha_of(entry) if sha_of else self.donors[entry["j"]]["sha256"]
                    artifact = write_evaluation(self.root, self.stamp(), expected, value=value(entry["j"]), commit=commit)
                    out.append(runner_record(entry, artifact=artifact))
            return out
        return records_for

    def done_phase(self, phase, records_for):
        self.assertIn("written", g.advance(phase, 0))
        plan_file = g.plan_path(phase, 0, 1)
        commit = self.commit()
        self.run_campaign(plan_file, records_for, commit)
        self.assertIn("complete.json written", g.advance(phase, 0))


class PhaseTest(Chain):
    def test_plan_recovery_and_completion(self):
        self.assertIn("written (8 entries)", g.advance("donor-v1", 0))
        self.assertIn("waiting to run", g.advance("donor-v1", 0))
        commit = self.commit()
        plan_file = g.plan_path("donor-v1", 0, 1)
        self.run_campaign(plan_file, self.evaluations(fail=(3,)), commit)
        self.assertIn("written (1 entries)", g.advance("donor-v1", 0, stop_time=(datetime.now() + timedelta(hours=5))
                                                       .isoformat(timespec="minutes")))
        recovery = json.loads(g.plan_path("donor-v1", 0, 2).read_text(encoding="utf-8"))
        original = json.loads(plan_file.read_text(encoding="utf-8"))
        self.assertEqual(recovery["runs"][0]["command"], original["runs"][3]["command"])
        self.assertEqual([a["j"] for a in recovery["accepted"]], [0, 1, 2, 4, 5, 6, 7])  # frozen accepted outcomes
        self.assertTrue(any("summary" in p for p in recovery["predecessors"]))  # audited summary and log pinned
        self.assertIn("stop_time", recovery)
        commit = self.commit()
        self.run_campaign(g.plan_path("donor-v1", 0, 2), self.evaluations(), commit)
        self.assertIn("written (8 items)", g.advance("donor-v1", 0))
        completion = json.loads(g.completion_path("donor-v1", 0).read_text(encoding="utf-8"))
        self.assertEqual([f["j"] for f in completion["failures"]], [3])
        self.assertTrue(all("receipt" in c for c in completion["campaigns"]))
        self.assertIn("reconstructs", g.advance("donor-v1", 0))

    def test_cutoff_skips_and_missing_artifacts_are_recovered(self):
        g.advance("donor-v1", 0)
        commit = self.commit()
        plan_file = g.plan_path("donor-v1", 0, 1)

        def records_for(plan, commit_):
            records = self.evaluations(skip=(6, 7))(plan, commit_)
            (self.root / records[0]["artifact"]["result_file"]).unlink()  # j 0's result disappears
            return records
        self.run_campaign(plan_file, records_for, commit)
        self.assertIn("written (3 entries)", g.advance("donor-v1", 0))
        recovery = json.loads(g.plan_path("donor-v1", 0, 2).read_text(encoding="utf-8"))
        self.assertEqual([e["j"] for e in recovery["runs"]], [0, 6, 7])

    def test_unverified_execution_stops(self):
        """Review B3: no receipt, a receipt at another commit, other threads or copies, unreviewed code."""
        g.advance("donor-v1", 0)
        commit = self.commit()
        plan_file = g.plan_path("donor-v1", 0, 1)
        cases = {"no receipt": dict(receipt=False), "one thread": dict(threads=1),
                 "other copies": dict(copies=[{"game_dir": "C:/elsewhere", "ports": [1, 2]}]),
                 "still running": dict(finished=False), "missing entry": dict(drop=1),
                 "started without outcome": dict(drop=1, extra_log="[t] start s1-r0-donor-v1-j7-seed14: x\n")}
        for name, options in cases.items():
            with self.subTest(case=name):
                shutil.rmtree(g.CAMPAIGNS, ignore_errors=True)
                shutil.rmtree(g.STAGE1, ignore_errors=True)
                self.run_campaign(plan_file, self.evaluations(), commit, **options)
                with self.assertRaises(g.Refused):
                    g.advance("donor-v1", 0)
        shutil.rmtree(g.CAMPAIGNS)
        shutil.rmtree(g.STAGE1)
        self.receipt(plan_file, json.loads(plan_file.read_text(encoding="utf-8")), commit)
        (self.root / "config" / "retention-confirmation.json").write_text("{}", encoding="utf-8")
        later = self.commit()  # the campaign ran at a commit whose code is not the receipt's approved code
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
        folder = g.CAMPAIGNS / f"{self.stamp()}-{plan['name']}"
        folder.mkdir(parents=True)
        (folder / "campaign.log").write_text(f"[t] campaign {plan['name']}, 8 runs\n[t] campaign finished: 8\n",
                                             encoding="utf-8")
        (folder / "summary.json").write_text(json.dumps({"plan": plan["name"], "plan_sha256": text_sha256(plan_file),
                                                         "commit": later, "results": self.evaluations()(plan, later),
                                                         "threads_per_job": 10,
                                                         "parallel": {"copies": plan["runner"]["game_copies"]}}),
                                             encoding="utf-8")
        with self.assertRaises(g.Refused):
            g.advance("donor-v1", 0)

    def test_dry_runs_are_ignored_and_a_second_execution_stops(self):
        g.advance("donor-v1", 0)
        commit = self.commit()
        plan_file = g.plan_path("donor-v1", 0, 1)
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
        dry = g.CAMPAIGNS / f"20261005-070000-{plan['name']}"
        dry.mkdir(parents=True)
        (dry / "campaign.log").write_text(f"[t] campaign {plan['name']}, 8 runs\n[t] dry run: nothing executed\n",
                                          encoding="utf-8")
        self.assertIn("waiting to run", g.advance("donor-v1", 0))
        self.run_campaign(plan_file, self.evaluations(), commit)
        self.run_campaign(plan_file, self.evaluations(), commit)
        with self.assertRaises(g.Refused):
            g.advance("donor-v1", 0)

    def test_edited_plans_and_records_stop(self):
        """Review B2: an edited completion is not reused; an executed plan must still be the derivation."""
        self.done_phase("donor-v1", self.evaluations())
        path = g.completion_path("donor-v1", 0)
        original = path.read_text(encoding="utf-8")
        edited = json.loads(original)
        edited["items"][0]["outcome"]["passes"] = False  # score and margin unchanged
        path.write_text(json.dumps(edited, indent=2), encoding="utf-8")
        with self.assertRaises(g.Refused):
            g.advance("record", 0)
        self.assertFalse(g.plan_path("record", 0, 1).exists())
        path.write_text(original, encoding="utf-8")
        episodes = next(self.root.glob("runs/heldout-evaluation/*/episodes.jsonl"))
        episodes.write_text("changed\n", encoding="utf-8")  # a leaf pin behind the completion
        with self.assertRaises(g.Refused):
            g.advance("record", 0)


class FullRoundTest(Chain):
    """donor-v1, record, copy, copy-v1 and inclusion end to end; then a changed cloned.zip is caught."""

    def test_round(self):
        self.done_phase("donor-v1", self.evaluations(value=lambda j: 0.30 if j == 5 else 0.5))

        def recordings(plan, commit):
            out = []
            for entry in plan["runs"]:
                folder = write_recording(self.root, self.stamp(), self.donors[entry["j"]], commit=commit)
                out.append(runner_record(entry, announced=[folder]))
            return out
        self.done_phase("record", recordings)
        record_completion = json.loads(g.completion_path("record", 0).read_text(encoding="utf-8"))
        self.assertNotIn(5, [i["j"] for i in record_completion["items"]])  # below the donor line: no recording

        def copies(plan, commit):
            out = []
            for entry, item in zip(plan["runs"], plan["items"]):
                artifact = write_copy(self.root, self.stamp(), item, entry, commit=commit)
                out.append(runner_record(entry, artifact=artifact))
            return out
        self.done_phase("copy", copies)
        clones = {i["j"]: i["outcome"]["sha256"]
                  for i in json.loads(g.completion_path("copy", 0).read_text(encoding="utf-8"))["items"]}
        self.done_phase("copy-v1", self.evaluations(value=lambda j: 0.30 if j == 2 else 0.5,
                                                    sha_of=lambda entry: clones[entry["j"]]))
        record = g.inclusion(0)
        self.assertEqual([(p["seed"], p["j"]) for p in record["proposals"]], [(15, 5), (16, 2)])
        self.assertEqual(g.inclusion(0), record)
        clone = next(self.root.glob("runs/clone/chapter-1-room-2/*/cloned.zip"))
        clone.write_bytes(b"a changed policy")  # review B2: a changed cloned.zip is caught before any use
        with self.assertRaises(g.Refused):
            g.inclusion(0)


# ------------------------------------------------------------------------------------------- inclusion

class InclusionTest(Temp):
    """Amendment 3: donor-line exclusions before copy-line exclusions, by j; seeds 15 then 16; j inherited."""

    def records(self, rounds: dict):
        def outcome(value):
            return {"route_macro_success_rate": value, "margin": g.margin(value), "passes": g.passes(value)}

        def completion(phase, round_, declaration):
            donors, copies, failures = rounds[round_]
            if phase == "donor-v1":
                return {"items": [{"j": j, "seed": s, "sha256": f"{s:02d}" * 32, "outcome": outcome(v)}
                                  for j, (s, v) in donors.items()], "failures": failures}
            if phase == "copy-v1":
                return {"items": [{"j": j, "outcome": outcome(v)} for j, v in copies.items()], "failures": []}
            return {"items": [], "failures": []}
        stored = {}

        def inclusion_of(round_):
            return stored[round_]
        return completion, inclusion_of, stored

    def test_allocation_order_shared_budget_and_end_state(self):
        donors = {j: (seed, 0.30 if j in (5, 3) else 0.5) for j, seed in enumerate(g.ORIGINAL_SEEDS)}
        copies = {j: (0.30 if j == 1 else 0.5) for j in donors if j not in (5, 3)}
        completion, inclusion_of, stored = self.records({0: (donors, copies, []),
                                                         1: ({3: (15, 0.5), 5: (16, 0.5)}, {3: 0.5, 5: 0.30}, [])})
        with unittest.mock.patch.object(g, "verified_completion", side_effect=completion), \
                unittest.mock.patch.object(g, "verified_inclusion", side_effect=inclusion_of), \
                unittest.mock.patch.object(g, "pin", side_effect=lambda path: {"path": str(path)}):
            first = g.inclusion_record(0)
            stored[0] = first
            self.assertEqual([(p["seed"], p["j"], p["replaces"]) for p in first["proposals"]], [(15, 3, 10), (16, 5, 12)])
            self.assertEqual(first["slots"]["1"]["active"]["status"], "excluded: copy line; no replacement left")
            second = g.inclusion_record(1)
        self.assertEqual(second["proposals"], [])
        self.assertEqual(second["slots"]["5"]["active"]["status"], "excluded: copy line; no replacement left")
        self.assertEqual(second["slots"]["3"]["history"][0]["seed"], 10)
        self.assertEqual(second["qualified"], [0, 2, 3, 4, 6, 7])
        self.assertTrue(second["status"].startswith("stage 1 complete with 6"))

    def test_fewer_than_six_stops_and_technical_failures_never_allocate(self):
        donors = {j: (seed, 0.30 if j < 4 else 0.5) for j, seed in enumerate(g.ORIGINAL_SEEDS)}
        failures = [{"j": 6, "seed": 13, "problems": ["status exit_1"]}]  # recovered in the round; no seed for it
        completion, inclusion_of, stored = self.records({0: (donors, {j: 0.5 for j in range(4, 8)}, failures),
                                                         1: ({0: (15, 0.30), 1: (16, 0.5)}, {1: 0.5}, [])})
        with unittest.mock.patch.object(g, "verified_completion", side_effect=completion), \
                unittest.mock.patch.object(g, "verified_inclusion", side_effect=inclusion_of), \
                unittest.mock.patch.object(g, "pin", side_effect=lambda path: {"path": str(path)}):
            record = g.inclusion_record(0)
            stored[0] = record
            self.assertEqual([p["j"] for p in record["proposals"]], [0, 1])
            self.assertEqual(record["slots"]["6"]["active"]["status"], "qualified")
            self.assertTrue(g.inclusion_record(1)["status"].startswith("stop for review: 5 qualified"))


class WriteNewTest(Temp):
    def test_exclusive_identical_or_refused(self):
        path = self.root / "x.json"
        self.assertEqual(g.write_new(path, {"a": 1}), "written")
        self.assertEqual(g.write_new(path, {"a": 1}), "exists")
        with self.assertRaises(g.Refused):
            g.write_new(path, {"a": True})


class GuardTest(Temp):
    """Review B1: hashes and the task loader are reached only after the identity check."""

    def test_aliases_are_refused_before_hashing_or_loading(self):
        from celeste_rl.fresh_sets import FreshSet
        from celeste_rl import fresh_sets
        v1 = "config/heldout_starts.json"
        spec = {"room1": FreshSet(v1, fresh_sets.git_blob(REPO / v1), None, None)}
        alias = self.root / "alias.json"
        os.link(REPO / v1, alias)
        spy = unittest.mock.MagicMock()
        with unittest.mock.patch.object(hx, "FRESH", spec), \
                unittest.mock.patch.object(g.fresh_sets, "text_sha256", spy), \
                unittest.mock.patch.object(g.hashlib, "sha256", spy):
            for reader in (g.text_hash, g.guarded_hash, g.read_meta):
                with self.subTest(reader=reader.__name__):
                    with self.assertRaises(g.Refused):
                        reader(alias)
        spy.assert_not_called()
        loader = unittest.mock.MagicMock()
        with unittest.mock.patch.object(hx, "FRESH", spec), unittest.mock.patch.object(g, "CODE_ROOT", self.root), \
                unittest.mock.patch.object(g, "resolve_task_definition", loader):
            (self.root / "config").mkdir()
            os.link(REPO / v1, self.root / g.ROOM2_TASK)
            with self.assertRaises(g.Refused):
                g.room2_identity()
        loader.assert_not_called()


# --------------------------------------------------------------------------------------------- preflight

class PreflightTest(Chain):
    """Preflight with the real derivation: reviewed code, the plan on the remote branch, no earlier execution."""

    def setUp(self):
        super().setUp()
        (self.root / "scripts").mkdir()
        (self.root / "scripts" / "control.py").write_text("RULE = 1\n", encoding="utf-8")
        self.approved = self.commit()
        g.advance("donor-v1", 0)
        self.plan_file = g.plan_path("donor-v1", 0, 1)
        self.plan_commit = self.commit()
        git(self.root, "update-ref", "refs/remotes/origin/proposal/ppo-anchor", self.plan_commit)
        invocation = [sys.executable, "-c", "print('dry')", "--copy", "C:/x:1:2"]
        self.pre = [unittest.mock.patch.object(g, "control_files",
                                               return_value=["config/retention-confirmation.json", "scripts/control.py"]),
                    unittest.mock.patch.object(g, "runner_invocation", return_value=invocation),
                    unittest.mock.patch.object(g.rc, "thread_probe", return_value={"torch_intra_op_threads": 10})]
        for p in self.pre:
            p.start()

    def tearDown(self):
        for p in reversed(self.pre):
            p.stop()
        super().tearDown()

    def preflight(self):
        return g.preflight(self.plan_file, self.approved, "origin/proposal/ppo-anchor")

    def test_passes_and_records(self):
        record = self.preflight()
        self.assertEqual((record["approved_commit"], record["plan"]["commit"]), (self.approved, self.plan_commit))
        self.assertTrue((self.root / record["record"]).exists())

    def test_unpushed_plan_is_refused(self):
        git(self.root, "update-ref", "refs/remotes/origin/proposal/ppo-anchor", self.approved)
        with self.assertRaises(g.Refused):
            self.preflight()

    def test_unreviewed_helper_change_is_refused(self):
        (self.root / "scripts" / "control.py").write_text("RULE = 2\n", encoding="utf-8")
        with self.assertRaises(g.Refused):
            self.preflight()
        head = self.commit()
        git(self.root, "update-ref", "refs/remotes/origin/proposal/ppo-anchor", head)
        with self.assertRaises(g.Refused):
            self.preflight()

    def test_a_plan_that_has_run_is_refused(self):
        self.run_campaign(self.plan_file, self.evaluations(), self.plan_commit)
        with self.assertRaises(g.Refused):
            self.preflight()

    def test_a_hand_edited_plan_is_refused(self):
        text = self.plan_file.read_text(encoding="utf-8").replace('"limit_minutes": 30', '"limit_minutes": 31', 1)
        self.plan_file.write_text(text, encoding="utf-8")
        head = self.commit()
        git(self.root, "update-ref", "refs/remotes/origin/proposal/ppo-anchor", head)
        with self.assertRaises(g.Refused):
            self.preflight()


if __name__ == "__main__":
    unittest.main()
