"""The confirmation's recipe check (scripts/confirmation_recipe_check.py): what counts as identical."""
import base64
import io
import json
import pathlib
import pickle
import tempfile
import unittest
import zipfile
from pathlib import Path

import torch

from scripts import confirmation_recipe_check as rc

REPO = Path(__file__).resolve().parents[1]


def tensor_bytes(value) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def stored_path(*parts) -> dict:
    """A path field as Stable-Baselines3 serializes it."""
    return {":type:": "<class 'pathlib.WindowsPath'>",
            ":serialized:": base64.b64encode(pickle.dumps(pathlib.WindowsPath(*parts))).decode()}


def checkpoint(path: Path, start_time=1, weight=1.0, lr=0.0003, system="Windows", extra=None, run="runs/train/a"):
    members = {
        "data": json.dumps({"start_time": start_time, "num_timesteps": 51200, "learning_rate": lr,
                            "abort_checkpoint_path": stored_path(run, "checkpoints", "aborted.zip")}).encode(),
        "policy.pth": tensor_bytes({"layer.weight": torch.full((2, 2), weight)}),
        "policy.optimizer.pth": tensor_bytes({"state": {0: {"step": torch.tensor(3.0)}}, "param_groups": [{"lr": lr}]}),
        "pytorch_variables.pth": tensor_bytes({}),
        "_stable_baselines3_version": b"2.9.0",
        "system_info.txt": system.encode(),
        **(extra or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path


class CommandTest(unittest.TestCase):
    def test_declared_seed_7_command_with_only_seed_and_run_dir_changed(self):
        declaration = json.loads((REPO / "config" / "retention-confirmation.json").read_text(encoding="utf-8"))
        command = rc.declared_command(declaration, Path("C:/games/celeste"))
        self.assertEqual(command[:2], ["scripts/train_room1.py", "--reward-version"])
        self.assertEqual(command[command.index("--seed") + 1], "7")
        self.assertEqual(command[command.index("--run-dir") + 1], "runs/train/confirm-recipe-check-B-seed7")
        self.assertEqual(command[-2:], ["--game-dir", str(Path("C:/games/celeste"))])
        original = json.loads((REPO / "config" / "campaign-finetune-variance.json").read_text(encoding="utf-8"))
        seed7 = [r for r in original["runs"] if r["id"] == "B-seed7"][0]["command"]
        differing = {i for i, (a, b) in enumerate(zip(seed7, command)) if a != b}
        self.assertEqual([seed7[i - 1] for i in differing], ["--run-dir"])  # only the run folder value differs
        self.assertEqual(len(command), len(seed7) + 2)

    def test_thread_variables_are_removed_and_nothing_else(self):
        env = {"OMP_NUM_THREADS": "10", "mkl_num_threads": "4", "TORCH_NUM_THREADS": "2", "PATH": "x", "HOME": "y"}
        self.assertEqual(rc.stripped_env(env), {"PATH": "x", "HOME": "y"})


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_only_start_time_and_system_info_may_differ(self):
        a = checkpoint(self.dir / "a.zip", start_time=1, system="Windows A")
        b = checkpoint(self.dir / "b.zip", start_time=2, system="Windows B")
        self.assertEqual(rc.compare_checkpoint(a, b), [])

    def test_run_folder_field_names_each_runs_own_folder(self):
        a = checkpoint(self.dir / "a.zip", run="runs/train/overnight-B-seed7")
        b = checkpoint(self.dir / "b.zip", run="runs/train/confirm-recipe-check-B-seed7")
        runs = (Path("runs/train/overnight-B-seed7"), Path("runs/train/confirm-recipe-check-B-seed7"))
        self.assertEqual(rc.compare_checkpoint(a, b, *runs), [])
        self.assertTrue(rc.compare_checkpoint(a, b))  # without run folders the field must simply be equal
        wrong = checkpoint(self.dir / "c.zip", run="runs/train/somewhere-else")
        self.assertTrue(rc.compare_checkpoint(a, wrong, *runs))
        self.assertIsNone(rc.stored_path_parts({":serialized:": base64.b64encode(pickle.dumps(len)).decode()}))

    def test_changes_that_matter_are_reported(self):
        a = checkpoint(self.dir / "a.zip")
        for name, kwargs in (("weight", {"weight": 1.0000001}), ("saved field", {"lr": 0.0004}),
                             ("extra member", {"extra": {"new.pth": b"x"}})):
            with self.subTest(name=name):
                b = checkpoint(self.dir / f"b-{name}.zip", **kwargs)
                self.assertTrue(rc.compare_checkpoint(a, b))


class RowsTest(unittest.TestCase):
    def test_volatile_columns_ignored_others_compared(self):
        a = [{"accepted_steps": "2048", "success_rate": "0.1", "env_steps_per_second": "400", "game_private_mb": "900"}]
        same = [{**a[0], "env_steps_per_second": "380", "game_private_mb": "950"}]
        self.assertEqual(rc.compare_rows(a, same, rc.VOLATILE_PROGRESS, "progress"), [])
        self.assertTrue(rc.compare_rows(a, [{**a[0], "success_rate": "0.2"}], rc.VOLATILE_PROGRESS, "progress"))
        self.assertTrue(rc.compare_rows(a, a + a, rc.VOLATILE_PROGRESS, "progress"))  # row count

    def test_evaluation_checkpoint_path_and_hash_ignored(self):
        a = [{"checkpoint": "runs/train/overnight-B-seed7/checkpoints/x.zip", "checkpoint_sha256": "aa",
              "stochastic_success_rate": 0.96}]
        b = [{**a[0], "checkpoint": "runs/train/confirm-recipe-check-B-seed7/checkpoints/x.zip", "checkpoint_sha256": "bb"}]
        self.assertEqual(rc.compare_rows(a, b, rc.VOLATILE_EVALUATION, "evaluations"), [])
        self.assertTrue(rc.compare_rows(a, [{**b[0], "stochastic_success_rate": 0.94}], rc.VOLATILE_EVALUATION, "e"))


class RealRunTest(unittest.TestCase):
    def test_the_original_run_matches_itself_and_a_missing_run_is_reported(self):
        original = REPO / rc.ORIGINAL
        if not (original / "checkpoints").exists():
            self.skipTest("original seed 7 run not present")
        report = rc.compare_runs(original, original)
        self.assertTrue(report["identical"], report["differences"])
        with zipfile.ZipFile(original / "checkpoints" / "step_000051200.zip") as archive:
            stored = json.loads(archive.read("data"))["abort_checkpoint_path"]
        self.assertEqual(rc.stored_path_parts(stored), ("runs", "train", "overnight-B-seed7", "checkpoints", "aborted.zip"))
        self.assertIn("step_000501760.zip", report["checkpoints_compared"])
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp)
            (empty / "checkpoints").mkdir()
            for name in ("progress.csv", "evaluations.jsonl"):
                (empty / name).write_text("", encoding="utf-8")
            self.assertFalse(rc.compare_runs(original, empty)["identical"])


if __name__ == "__main__":
    unittest.main()
