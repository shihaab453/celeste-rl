"""The fresh-set guard of the retention confirmation (celeste_rl/fresh_sets.py) and its runner hook.

No test reads either fresh set: shapes are checked on command lists; aliases (short names, links, copies) and the
identity check run on the reused v1 development sets through injected specs and altered pins.
"""
import ctypes
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from celeste_rl.fresh_sets import (EVAL_SEED, FRESH, FreshSet, FreshSetRefused, classify, plan_problems,
                                   refers_to_fresh, refuse_unless_declared, verify_identity)
from celeste_rl.heldout import HeldoutManifestError
from scripts import run_overnight

REPO = Path(__file__).resolve().parents[1]
DECLARATION = json.loads((REPO / "config" / "retention-confirmation.json").read_text(encoding="utf-8"))
CKPT = "runs/train/confirm-A1-j0/checkpoints/step_000501760.zip"
R1 = "config/heldout_starts-room1-v2.json"
R2 = "config/heldout_starts-room2-v2.json"


def room1(repeats="3", seed=EVAL_SEED, extra=()):
    return ["scripts/evaluate_heldout.py", "--checkpoint", CKPT, "--starts", R1, "--repeats", repeats, "--seed", seed,
            *extra]


def room2(extra=()):
    return ["scripts/evaluate_heldout.py", "--task-definition", "config/room2.json", "--checkpoint", CKPT, "--starts",
            R2, "--repeats", "1", "--seed", EVAL_SEED, "--reward-version", "rew-v2", "--shaping-scale", "2.0", *extra]


def copy(seed="10", heldout=R2, mix=("default", "runs/policy-play/chapter-1-room-1/x"), drop=(), extra=()):
    command = ["scripts/clone_room1.py", "--task-definition", "config/room2.json", "--dataset",
               "runs/clone/chapter-1-room-2/20260924-014351/dataset.npz", "--demonstrations",
               "config/demonstrations-room2-v2.json", "--heldout", heldout, "--mix-play", *mix, "--mix-targets", "donor",
               "--room-weighting", "equal", "--routes-only", "--holdout", "0.25", "--epochs", "300", "--batch-size",
               "64", "--learning-rate", "0.001", "--no-play", "--seed", seed, "--init-from",
               "runs/train/overnight-B-seed7/checkpoints/latest.zip", "--init-from-sha256", "ab33", *extra]
    for option in drop:
        index = command.index(option)
        del command[index:index + 1 + (0 if option in ("--routes-only", "--no-play") else 1)]
    return command


class ShapeTest(unittest.TestCase):
    def test_ordinary_commands_are_not_fresh(self):
        self.assertIsNone(classify(["scripts/evaluate_heldout.py", "--checkpoint", CKPT, "--starts",
                                    "config/heldout_starts.json", "--repeats", "1", "--seed", "20260920"]))
        self.assertIsNone(classify(["scripts/train_room1.py", "--seed", "60"]))

    def test_declared_shapes_are_named(self):
        self.assertEqual(classify(room1("3")), "room1_eval")
        self.assertEqual(classify(room1("1")), "room1_eval")
        self.assertEqual(classify(room1(extra=("--game-dir", "C:/copy-2"))), "room1_eval")
        self.assertEqual(classify(room2()), "room2_eval")
        self.assertEqual(classify(room2(("--game-dir", "C:/copy-3"))), "room2_eval")
        self.assertEqual(classify(copy("10")), "copy_overlap_guard")
        self.assertEqual(classify(copy("17")), "copy_overlap_guard")
        self.assertEqual(classify(["scripts\\evaluate_heldout.py", *room1()[1:]]), "room1_eval")

    def test_near_misses_are_refused(self):
        refused = [
            room1(seed="20260920"),                                   # wrong seed
            room1(repeats="2"),                                       # undeclared repeats
            room1(extra=("--deterministic",)),                        # extra option
            room1(extra=("--seed", EVAL_SEED)),                       # option twice
            room2()[:3] + ["--checkpoint", CKPT, "--starts", R1] + room2()[7:],  # Room 1 set in the Room 2 shape
            room1()[:4] + ["./" + R1] + room1()[5:],                  # path variant
            room1()[:4] + [R1.replace("/", "\\")] + room1()[5:],     # backslash variant
            room1()[:4] + [R1.upper()] + room1()[5:],                 # case variant
            room1()[:2] + [R2] + room1()[3:],                         # a fresh set as the checkpoint
            ["scripts/evaluate_checkpoint.py", "--starts", R1, "--seed", EVAL_SEED],  # another script
            copy("9"), copy("18"),                                    # clone seed outside 10 to 17
            copy(heldout=R1),                                         # wrong set in the copy
            copy(mix=("room", "x")),                                  # mix-play not 'default <recording>'
            copy(mix=("default", R1)),                                # fresh set as the recording
            copy(drop=("--no-play",)),                                # missing flag
            copy(extra=("--epochs-extra", "1")),                      # extra option
            ["scripts/train_room1.py", "--run-dir", "runs/train/heldout_starts-room1-v2"],  # name elsewhere
        ]
        for command in refused:
            with self.subTest(command=command):
                with self.assertRaises(FreshSetRefused):
                    classify(command)

    def test_runner_built_commands_keep_their_shapes(self):
        self.assertEqual(classify(copy(extra=("--game-dir", "C:/Projects/celeste-research-scratch/game-copy-2"))),
                         "copy_overlap_guard")
        entry = {"id": "x", "run_dir": "runs/clone/x", "command": copy()}
        self.assertEqual(classify(run_overnight.build_command(entry, Path("C:/games/celeste"))[1:]),
                         "copy_overlap_guard")

    def test_option_spellings_argparse_would_accept_are_refused(self):
        flagged = copy()
        flagged.insert(flagged.index("--no-play") + 1, "yes")              # a flag given a value
        for command in (room1()[:3] + [f"--starts={R1}"] + room1()[5:],     # equals form
                        room1()[:3] + ["--start", R1] + room1()[5:],        # abbreviation
                        flagged):
            with self.subTest(command=command):
                with self.assertRaises(FreshSetRefused):
                    classify(command)

    def test_refuse_unless_declared_counts_and_restricts(self):
        commands = [room1("3"), room1("1"), room2(), ["scripts/train_room1.py", "--seed", "60"]]
        self.assertEqual(refuse_unless_declared(commands, ("room1_eval", "room2_eval")),
                         {"room1_eval": 2, "room2_eval": 1})
        with self.assertRaises(FreshSetRefused):   # a stage 1 plan may copy, never evaluate on a fresh set
            refuse_unless_declared([copy(), room1()], ("copy_overlap_guard",))


class DeclarationAgreementTest(unittest.TestCase):
    """The module's constants and shapes are the declaration's, so neither can drift alone."""

    def fill(self, template, values):
        command = template.split(" (")[0].replace(" [--game-dir <copy>]", "")
        for placeholder, value in values.items():
            command = command.replace(placeholder, value)
        self.assertNotIn("<", command, f"unfilled placeholder in {command}")
        return command.split()

    def test_declared_commands_classify_as_their_shapes(self):
        measurements = DECLARATION["measurements"]
        for repeats in ("1", "3"):
            self.assertEqual(classify(self.fill(measurements["room1_v2"]["command"],
                                                {"<zip>": CKPT, "<R>": repeats})), "room1_eval")
        self.assertEqual(classify(self.fill(measurements["room2_v2"]["command"], {"<zip>": CKPT})), "room2_eval")
        for j in range(8):
            command = self.fill(DECLARATION["pipeline"]["copy"],
                                {"<recording>": "runs/policy-play/chapter-1-room-1/x", "<10+j>": str(10 + j),
                                 "<donor>": "runs/train/overnight-B-seed7/checkpoints/latest.zip", "<sha>": "ab33"})
            self.assertEqual(classify(command), "copy_overlap_guard")

    def test_pins_and_seed_match_the_declaration(self):
        guard = DECLARATION["running"]["v2_guard"]
        self.assertIn(FRESH["room2"].git_blob, guard)
        self.assertIn(FRESH["room2"].manifest_sha256, guard)
        self.assertIn(FRESH["room1"].git_blob, DECLARATION["measurements"]["room1_v2"]["pins"])
        self.assertIn(FRESH["room2"].git_blob, DECLARATION["measurements"]["room2_v2"]["pins"])
        self.assertIn(FRESH["room2"].manifest_sha256, DECLARATION["measurements"]["room2_v2"]["pins"])
        self.assertIsNone(FRESH["room1"].manifest_sha256)  # declared as computed at plan time
        self.assertIn(f"--seed {EVAL_SEED}", DECLARATION["measurements"]["room1_v2"]["command"])
        self.assertIn(f"--seed {EVAL_SEED}", DECLARATION["measurements"]["room2_v2"]["command"])

    def test_committed_blobs_are_the_declared_ones(self):
        tree = subprocess.run(["git", "-C", str(REPO), "ls-tree", "HEAD", R1, R2], capture_output=True, text=True,
                              check=True).stdout  # git metadata only; the files are not read
        self.assertIn(FRESH["room1"].git_blob, tree)
        self.assertIn(FRESH["room2"].git_blob, tree)


class IdentityTest(unittest.TestCase):
    """verify_identity on the reused v1 development sets (never on a fresh set)."""

    def committed_blob(self, path):
        return subprocess.run(["git", "-C", str(REPO), "ls-tree", "HEAD", path], capture_output=True, text=True,
                              check=True).stdout.split()[2]

    def test_identity_returns_hashes_and_count_only(self):
        path = "config/heldout_starts.json"
        spec = {"room1": FreshSet(path, self.committed_blob(path), None, None)}
        identity = verify_identity("room1", fresh=spec)
        self.assertEqual(set(identity), {"path", "git_blob", "manifest_sha256", "text_sha256", "states"})
        self.assertEqual(identity["states"], 200)

    def test_wrong_blob_refuses_before_parsing(self):
        spec = {"room1": FreshSet("config/heldout_starts.json", "0" * 40, None, None)}
        with self.assertRaises(FreshSetRefused):
            verify_identity("room1", fresh=spec)

    def test_wrong_manifest_pin_refuses(self):
        path = "config/heldout_starts-room2.json"
        spec = {"room2": FreshSet(path, self.committed_blob(path), "0" * 64, "config/room2.json")}
        with self.assertRaises(FreshSetRefused):
            verify_identity("room2", fresh=spec)

    def test_wrong_task_refuses(self):
        path = "config/heldout_starts-room2.json"  # a Room 2 set checked against the Room 1 task
        spec = {"room1": FreshSet(path, self.committed_blob(path), None, None)}
        with self.assertRaises(HeldoutManifestError):
            verify_identity("room1", fresh=spec)


V1 = "config/heldout_starts.json"  # a reused development set standing in for a fresh one


def committed_blob(path):
    return subprocess.run(["git", "-C", str(REPO), "ls-tree", "HEAD", path], capture_output=True, text=True,
                          check=True).stdout.split()[2]


class AliasTest(unittest.TestCase):
    """A fresh set reached by another name is still recognised (review G1), shown with V1 as the injected set."""

    def setUp(self):
        self.spec = {"room1": FreshSet(V1, committed_blob(V1), None, None)}
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def assert_recognised(self, token):
        self.assertTrue(refers_to_fresh(token, self.spec))
        with self.assertRaises(FreshSetRefused):  # and an undeclared use of it is refused
            classify(["scripts/evaluate_heldout.py", "--checkpoint", CKPT, "--starts", token, "--repeats", "3",
                      "--seed", EVAL_SEED], self.spec)

    def test_windows_short_name(self):
        buffer = ctypes.create_unicode_buffer(1024)
        if not ctypes.windll.kernel32.GetShortPathNameW(str(REPO / V1), buffer, 1024):
            self.skipTest("short names unavailable")
        short = buffer.value
        if "heldout_starts" in short.lower():
            self.skipTest("this volume has no 8.3 names")
        self.assert_recognised(short)

    def test_hard_link(self):
        link = self.dir / "innocent.json"
        try:
            os.link(REPO / V1, link)
        except OSError:
            self.skipTest("hard links unavailable here")
        self.assert_recognised(str(link))

    def test_renamed_copies_in_either_line_ending(self):
        lf = (REPO / V1).read_bytes().replace(b"\r\n", b"\n")
        for name, data in (("lf.txt", lf), ("crlf.dat", lf.replace(b"\n", b"\r\n"))):
            (self.dir / name).write_bytes(data)
            with self.subTest(name=name):
                self.assert_recognised(str(self.dir / name))

    def test_symlink(self):
        link = self.dir / "pointer.json"
        try:
            os.symlink(REPO / V1, link)
        except OSError:
            self.skipTest("symlinks need privileges here")
        self.assert_recognised(str(link))

    def test_unrelated_files_are_ordinary(self):
        (self.dir / "other.json").write_text('{"x": 1}', encoding="utf-8")
        self.assertFalse(refers_to_fresh(str(self.dir / "other.json"), self.spec))
        self.assertFalse(refers_to_fresh("config/room2.json", self.spec))


class StageTest(unittest.TestCase):
    """What each plan may do with the fresh sets, checked on the commands as they will run (review: biggest gap)."""

    def test_stage_rules(self):
        cases = [
            ({"fresh_set_stage": "evaluation"}, [room1("3"), room2()], True),
            ({"fresh_set_stage": "stage1"}, [copy(extra=("--game-dir", "C:/g"))], True),
            ({"fresh_set_stage": "night"}, [["scripts/train_room1.py", "--seed", "60"]], True),
            ({}, [["scripts/train_room1.py", "--seed", "60"]], True),
            ({"fresh_set_stage": "stage1"}, [room1("3")], False),
            ({"fresh_set_stage": "night"}, [copy()], False),
            ({}, [room1("3")], False),                                   # no stage: no fresh-set use at all
            ({"fresh_set_stage": "evaluations"}, [], False),             # unknown stage
            ({"fresh_set_stage": "evaluation"}, [room1(seed="1")], False),
        ]
        for plan, commands, ok in cases:
            with self.subTest(plan=plan, commands=commands):
                self.assertEqual(plan_problems(plan, commands) == [], ok)

    def test_runner_checks_commands_as_built(self):
        evaluation = {"fresh_set_stage": "evaluation",
                      "runs": [{"id": "e", "run_dir": "runs/evaluation/x", "command": room1("3")}]}
        self.assertEqual(run_overnight.fresh_set_problems(evaluation, Path("C:/games/celeste")), [])
        unstaged = {"runs": evaluation["runs"]}
        self.assertTrue(run_overnight.fresh_set_problems(unstaged, Path("C:/games/celeste")))
        night = {"fresh_set_stage": "night", "runs": [{"id": "c", "run_dir": "runs/clone/x", "command": copy()}]}
        self.assertTrue(run_overnight.fresh_set_problems(night, Path("C:/games/celeste")))


if __name__ == "__main__":
    unittest.main()
