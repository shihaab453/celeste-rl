"""The confirmation's recipe check (scripts/confirmation_recipe_check.py): what counts as identical, invalid or
different, and what a launch records. Nothing here starts the game or training."""
import base64
import io
import json
import pathlib
import pickle
import shutil
import subprocess
import tempfile
import unittest
import unittest.mock
import zipfile
from pathlib import Path

import torch

from scripts import confirmation_recipe_check as rc

REPO = Path(__file__).resolve().parents[1]
DECLARATION = json.loads((REPO / "config" / "retention-confirmation.json").read_text(encoding="utf-8"))
GAME = Path("C:/games/celeste")


def tensor_bytes(value) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def stored_path(*parts) -> dict:
    """A path field as Stable-Baselines3 serializes it."""
    return {":type:": "<class 'pathlib.WindowsPath'>",
            ":serialized:": base64.b64encode(pickle.dumps(pathlib.WindowsPath(*parts))).decode()}


def checkpoint(path: Path, start_time=1, weight=1.0, lr=0.0003, system="Windows", run="runs/train/a",
               optimizer=None, variables=None, data_extra=None, drop=()):
    data = {"start_time": start_time, "num_timesteps": 51200, "learning_rate": lr, "target_kl": None,
            "abort_checkpoint_path": stored_path(run, "checkpoints", "aborted.zip"), **(data_extra or {})}
    members = {
        "data": json.dumps(data).encode(),
        "policy.pth": tensor_bytes({"layer.weight": torch.full((2, 2), weight)}),
        "policy.optimizer.pth": tensor_bytes(optimizer if optimizer is not None else
                                             {"state": {0: {"step": torch.tensor(3.0)}}, "param_groups": [{"lr": lr}]}),
        "pytorch_variables.pth": tensor_bytes(variables if variables is not None else {"buffers": [torch.zeros(1)]}),
        "_stable_baselines3_version": b"2.9.0",
        "system_info.txt": system.encode(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, raw in members.items():
            if name not in drop:
                archive.writestr(name, raw)
    return path


class Temp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()


class CommandAndEnvironmentTest(unittest.TestCase):
    def test_declared_seed_7_command_with_only_seed_and_run_dir_changed(self):
        command = rc.declared_command(DECLARATION, GAME)
        original = json.loads((REPO / "config" / "campaign-finetune-variance.json").read_text(encoding="utf-8"))
        seed7 = [r for r in original["runs"] if r["id"] == "B-seed7"][0]["command"]
        differing = [i for i, (a, b) in enumerate(zip(seed7, command)) if a != b]
        self.assertEqual([seed7[i - 1] for i in differing], ["--run-dir"])
        self.assertEqual(command[command.index("--run-dir") + 1], "runs/train/confirm-recipe-check-B-seed7")
        self.assertEqual(command[len(seed7):], ["--game-dir", str(GAME)])

    def test_every_thread_variable_is_removed_and_nothing_else(self):
        env = {"OMP_NUM_THREADS": "10", "OMP_THREAD_LIMIT": "4", "KMP_AFFINITY": "x", "mkl_num_threads": "4",
               "OPENBLAS_NUM_THREADS": "2", "BLIS_NUM_THREADS": "2", "TORCH_NUM_THREADS": "2", "PATH": "p", "HOME": "h"}
        self.assertEqual(rc.stripped_env(env), {"PATH": "p", "HOME": "h"})
        self.assertEqual(set(rc.thread_variables(env)), set(env) - {"PATH", "HOME"})


class StructureTest(Temp):
    """Review B2: key presence, key types, container types and empty containers all count."""

    def test_structure_changes_are_reported(self):
        base = {"state": {0: {"step": torch.tensor(3.0)}}, "param_groups": [{"lr": 0.1}]}
        for name, changed in (("int key to str", {"state": {"0": {"step": torch.tensor(3.0)}}, "param_groups": [{"lr": 0.1}]}),
                              ("extra empty state", {"state": {0: {"step": torch.tensor(3.0)}, 1: {}},
                                                     "param_groups": [{"lr": 0.1}]}),
                              ("list to tuple", {"state": {0: {"step": torch.tensor(3.0)}}, "param_groups": ({"lr": 0.1},)}),
                              ("dtype", {"state": {0: {"step": torch.tensor(3.0, dtype=torch.float64)}},
                                         "param_groups": [{"lr": 0.1}]})):
            with self.subTest(name=name):
                self.assertTrue(rc.structural_differences(base, changed))
        self.assertEqual(rc.structural_differences(base, {"state": {0: {"step": torch.tensor(3.0)}},
                                                          "param_groups": [{"lr": 0.1}]}), [])

    def test_missing_null_data_field_is_a_difference(self):
        a = checkpoint(self.dir / "a.zip")
        with zipfile.ZipFile(a) as archive:
            members = {n: archive.read(n) for n in archive.namelist()}
        data = json.loads(members["data"])
        del data["target_kl"]  # null in the original
        members["data"] = json.dumps(data).encode()
        b = self.dir / "b.zip"
        with zipfile.ZipFile(b, "w") as archive:
            for name, raw in members.items():
                archive.writestr(name, raw)
        self.assertTrue(rc.compare_checkpoint(a, b))

    def test_only_start_time_system_info_and_own_run_folder_may_differ(self):
        runs = (Path("runs/train/overnight-B-seed7"), Path("runs/train/confirm-recipe-check-B-seed7"))
        a = checkpoint(self.dir / "a.zip", start_time=1, system="A", run=runs[0])
        b = checkpoint(self.dir / "b.zip", start_time=2, system="B", run=runs[1])
        self.assertEqual(rc.compare_checkpoint(a, b, *runs), [])
        self.assertTrue(rc.compare_checkpoint(a, checkpoint(self.dir / "c.zip", run="runs/train/elsewhere"), *runs))
        for name, kwargs in (("weight", {"weight": 1.0000001}), ("saved field", {"lr": 0.0004}),
                             ("new field", {"data_extra": {"x": 1}}),
                             ("variables list to tuple", {"variables": {"buffers": (torch.zeros(1),)}})):
            with self.subTest(name=name):
                self.assertTrue(rc.compare_checkpoint(a, checkpoint(self.dir / f"d-{name}.zip", run=runs[1], **kwargs),
                                                      *runs))

    def test_missing_member_or_start_time_is_invalid(self):
        a = checkpoint(self.dir / "a.zip")
        with self.assertRaises(rc.Invalid):
            rc.compare_checkpoint(a, checkpoint(self.dir / "b.zip", drop=("policy.optimizer.pth",)))
        with self.assertRaises(rc.Invalid):
            rc.compare_checkpoint(a, checkpoint(self.dir / "c.zip", start_time="yesterday"))

    def test_restricted_path_reader(self):
        self.assertIsNone(rc.stored_path_parts({":serialized:": base64.b64encode(pickle.dumps(len)).decode()}))
        self.assertEqual(rc.stored_path_parts(stored_path("runs", "x")), ("runs", "x"))


class RecordsTest(Temp):
    def test_rows_ignore_only_volatile_columns_and_count_key_presence(self):
        a = [{"accepted_steps": "2048", "success_rate": "0.1", "env_steps_per_second": "400", "x": None}]
        self.assertEqual(rc.compare_rows(a, [{**a[0], "env_steps_per_second": "380"}], rc.VOLATILE_PROGRESS, "p"), [])
        self.assertTrue(rc.compare_rows(a, [{**a[0], "success_rate": "0.2"}], rc.VOLATILE_PROGRESS, "p"))
        self.assertTrue(rc.compare_rows(a, a + a, rc.VOLATILE_PROGRESS, "p"))
        self.assertTrue(rc.compare_rows(a, [{k: v for k, v in a[0].items() if k != "x"}], (), "p"))

    def test_evaluation_references_are_verified_per_run(self):
        """Review B3: a reference to another checkpoint, or a false hash, is reported."""
        run = self.dir / "run"
        target = checkpoint(run / "checkpoints" / "step_000501760.zip")
        other = checkpoint(run / "checkpoints" / "step_000051200.zip", weight=2.0)
        good = {"checkpoint": "step_000501760.zip", "checkpoint_sha256": rc._sha256(target), "accepted_steps": 501760}
        self.assertEqual(rc.evaluation_problems(run, [good], "new"), [])
        self.assertTrue(rc.evaluation_problems(run, [{**good, "checkpoint_sha256": "0" * 64}], "new"))
        self.assertTrue(rc.evaluation_problems(run, [{**good, "checkpoint": "missing.zip"}], "new"))
        swapped = {**good, "checkpoint": "step_000051200.zip", "checkpoint_sha256": rc._sha256(other)}
        self.assertEqual(rc.evaluation_problems(run, [swapped], "new"), [])  # truthful on its own...
        self.assertTrue(rc.compare_rows(rc.normalized_evaluations([good]), rc.normalized_evaluations([swapped]), (),
                                        "e"))  # ...but not the same checkpoint as the original's row
        prefixed = {**good, "checkpoint": "runs/train/x/checkpoints/step_000501760.zip"}
        self.assertEqual(rc.normalized_evaluations([prefixed])[0]["checkpoint"], "step_000501760.zip")

    def test_config_allows_only_documented_defaults(self):
        original = {"seed": 7, "ent_coef": 0.01}
        self.assertEqual(rc.config_problems(original, {**original, "task_definition": "", "stall_frames": 0}), [])
        self.assertTrue(rc.config_problems(original, {**original, "stall_frames": 296}))
        self.assertTrue(rc.config_problems(original, {**original, "seed": 8}))
        self.assertTrue(rc.config_problems(original, {"seed": 7}))


class AcceptanceTest(Temp):
    """Review B1: an incomplete or unbound pair is INVALID, never IDENTICAL."""

    def test_empty_or_incomplete_runs_are_invalid(self):
        for run in (self.dir / "a", self.dir / "b"):
            (run / "checkpoints").mkdir(parents=True)
            row = {"accepted_steps": 501760, "stochastic_episodes": 50}
            (run / "evaluations.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            (run / "progress.csv").write_text("accepted_steps\n501760\n", encoding="utf-8")
        report = rc.acceptance(self.dir / "a", self.dir / "b", DECLARATION, self.dir / "records", GAME)
        self.assertEqual(report["verdict"], "INVALID")

    def test_real_original_with_and_without_a_bound_run_record(self):
        original = REPO / rc.ORIGINAL
        if not (original / "checkpoints").exists():
            self.skipTest("original seed 7 run not present")
        self.assertEqual(rc.validate_run(original, "original")["accepted_steps"], 501760)
        self.assertEqual(rc.evaluation_problems(original, rc._rows_jsonl(original / "evaluations.jsonl"), "o"), [])
        records = self.dir / "records"
        records.mkdir()
        unbound = rc.acceptance(original, original, DECLARATION, records, GAME)
        self.assertEqual(unbound["verdict"], "INVALID")  # no successful run record
        commit = json.loads((original / "manifest.json").read_text(encoding="utf-8"))["sessions"][-1]["provenance"]["commit"]
        record = {"exit_code": 0, "command": rc.declared_command(DECLARATION, GAME), "problems": [],
                  "declaration_text_sha256": rc.fresh_sets.text_sha256(rc.DECLARATION), "git": {"commit": commit}}
        (records / "run-20261001-000000.json").write_text(json.dumps(record), encoding="utf-8")
        bound = rc.acceptance(original, original, DECLARATION, records, GAME)
        self.assertEqual(bound["verdict"], "IDENTICAL", bound.get("reason") or bound.get("primary"))
        self.assertEqual(len(bound["artifacts"]["new"]), 13)
        (records / "run-20261001-000001.json").write_text(json.dumps({**record, "exit_code": 1}), encoding="utf-8")
        self.assertEqual(rc.acceptance(original, original, DECLARATION, records, GAME)["verdict"], "IDENTICAL")

    def test_seed_8_is_different_not_invalid(self):
        original, other = REPO / rc.ORIGINAL, REPO / "runs/train/overnight-B-seed8"
        if not other.exists():
            self.skipTest("seed 8 run not present")
        report = rc.compare_runs(original, other)
        self.assertTrue(report["primary"])
        self.assertFalse([d for v in report["primary"].values() for d in v if "abort_checkpoint_path" in d])


class LaunchTest(Temp):
    """Review R1 to R3: exit status, full logs on timeout, and the alone conditions, with every process mocked."""

    def patch_launch(self, check_code=0, check_text="PASS", games=(), others=()):
        check = subprocess.CompletedProcess([], check_code, stdout=check_text, stderr="")
        return [unittest.mock.patch.object(rc, "RECORDS", self.dir),
                unittest.mock.patch.object(rc.runtime, "refusal", return_value=None),
                unittest.mock.patch.object(rc, "all_game_processes", return_value=list(games)),
                unittest.mock.patch.object(rc, "other_experiments", return_value=list(others)),
                unittest.mock.patch.object(rc, "thread_probe", return_value={"torch_intra_op_threads": 10}),
                unittest.mock.patch.object(rc.subprocess, "run", return_value=check)]

    def run_with(self, patches, dry_run=True):
        for p in patches:
            p.start()
        try:
            return rc.run(GAME, dry_run)
        finally:
            for p in reversed(patches):
                p.stop()

    def record(self, stem):
        return json.loads(sorted(self.dir.glob(f"{stem}-*.json"))[-1].read_text(encoding="utf-8"))

    def test_code_check_needs_exit_status_zero(self):
        self.assertEqual(self.run_with(self.patch_launch(check_code=1, check_text="PASS")), 1)
        self.assertTrue(any("exited 1" in p for p in self.record("dry-run")["problems"]))

    def test_other_games_or_experiments_refuse(self):
        self.assertEqual(self.run_with(self.patch_launch(games=["123 C:/other/Celeste.exe"])), 1)
        self.assertEqual(self.run_with(self.patch_launch(others=["python scripts/run_overnight.py --plan x"])), 1)
        self.assertEqual(self.run_with(self.patch_launch()), 0)
        self.assertIn("alone", self.record("dry-run")["conditions"])

    def test_timeout_keeps_full_logs_and_cleans_only_its_own_game(self):
        def fake_run(args, **kwargs):
            if args[1:2] == ["scripts/train_room1.py"]:
                raise subprocess.TimeoutExpired(args, 5400, output="partial out\n" * 50, stderr="partial err")
            return subprocess.CompletedProcess(args, 0, stdout="PASS", stderr="")

        patches = self.patch_launch()[:-1] + [unittest.mock.patch.object(rc.subprocess, "run", side_effect=fake_run),
                                              unittest.mock.patch.object(rc.game_process, "running_game_pids",
                                                                         side_effect=[[4242], []])]
        self.assertEqual(self.run_with(patches, dry_run=False), 1)
        record = self.record("run")
        self.assertTrue(record["timed_out"])
        self.assertEqual([c["pid"] for c in record["game_cleanup"]], [4242])
        self.assertEqual(record["game_still_running"], [])
        stdout = sorted(self.dir.glob("run-*.stdout.txt"))[-1].read_text(encoding="utf-8")
        self.assertEqual(stdout.count("partial out"), 50)  # complete, not a tail

    def test_records_are_never_overwritten(self):
        with unittest.mock.patch.object(rc, "RECORDS", self.dir), \
                unittest.mock.patch.object(rc, "datetime") as clock:
            clock.now.return_value.strftime.return_value = "20261001-000000"
            first = rc._write({"a": 1}, "x", {"stdout.txt": "one"})
            second = rc._write({"a": 2}, "x", {"stdout.txt": "two"})
        self.assertNotEqual(first, second)
        self.assertEqual(json.loads(first.read_text(encoding="utf-8")), {"a": 1})
        self.assertEqual((self.dir / "x-20261001-000000.stdout.txt").read_text(encoding="utf-8"), "one")
        self.assertEqual((self.dir / "x-20261001-000000-2.stdout.txt").read_text(encoding="utf-8"), "two")


if __name__ == "__main__":
    unittest.main()
