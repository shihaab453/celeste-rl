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
import unittest.mock
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
        games = [Path("C:/games/celeste")]
        evaluation = {"fresh_set_stage": "evaluation",
                      "runs": [{"id": "e", "run_dir": "runs/evaluation/x", "command": room1("3")}]}
        self.assertEqual(run_overnight.fresh_set_problems(evaluation, games), [])
        unstaged = {"runs": evaluation["runs"]}
        self.assertTrue(run_overnight.fresh_set_problems(unstaged, games))
        night = {"fresh_set_stage": "night", "runs": [{"id": "c", "run_dir": "runs/clone/x", "command": copy()}]}
        self.assertTrue(run_overnight.fresh_set_problems(night, games))


class BuiltCommandTest(unittest.TestCase):
    """Fresh paths that appear only when the runner builds a command (review B4)."""

    TRAIN = {"id": "t", "run_dir": "runs/train/confirm-control-j0",
             "command": ["scripts/train_room1.py", "--seed", "60", "--run-dir", "runs/train/confirm-control-j0"]}

    def test_every_game_copy_is_checked(self):
        plan = {"fresh_set_stage": "night", "runs": [self.TRAIN]}
        self.assertEqual(run_overnight.fresh_set_problems(plan, [Path("C:/games/a")]), [])
        self.assertTrue(run_overnight.fresh_set_problems(plan, [Path("C:/games/a"),
                                                                Path("C:/games/heldout_starts-room1-v2")]))

    def test_automatic_resume_form_is_checked(self):
        entry = {**self.TRAIN, "run_dir": "runs/train/heldout_starts-room2-v2"}  # only the resume form uses run_dir
        self.assertNotIn("heldout_starts-room2-v2", " ".join(entry["command"]))
        self.assertTrue(run_overnight.fresh_set_problems({"fresh_set_stage": "night", "runs": [entry]},
                                                         [Path("C:/games/a")]))

    def test_launch_rechecks_the_built_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "campaign.log"
            with unittest.mock.patch.object(run_overnight.subprocess, "run", side_effect=AssertionError("launched")):
                result = run_overnight.execute(self.TRAIN, log, Path("C:/games/heldout_starts-room1-v2"),
                                               fresh_set_stage="night")
        self.assertEqual(result["status"], "refused_fresh_set")

    def test_main_refuses_before_any_entry_check_reads_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = Path(tmp) / "plan.json"
            plan.write_text(json.dumps({"name": "x", "runs": [{"id": "e", "run_dir": "runs/evaluation/x",
                                                               "command": room1("3")}]}), encoding="utf-8")
            with unittest.mock.patch.object(run_overnight, "entry_problems", side_effect=AssertionError("read")), \
                    unittest.mock.patch("sys.argv", ["run_overnight.py", "--plan", str(plan)]):
                self.assertEqual(run_overnight.main(), 2)


class SlotTest(unittest.TestCase):
    """A fresh set is allowed only in its shape's designated slot (review B1)."""

    def test_protected_data_in_a_fixed_slot_is_refused(self):
        dataset = "runs/clone/chapter-1-room-2/20260924-014351/dataset.npz"
        if not (REPO / dataset).exists():
            self.skipTest("declared dataset not present")
        spec = {"room1": FreshSet(dataset, "f" * 40, None, None), "room2": FRESH["room2"]}  # dataset as protected
        with self.assertRaises(FreshSetRefused):
            classify(copy(), spec)

    def test_designated_slot_must_hold_its_own_room(self):
        with self.assertRaises(FreshSetRefused):  # Room 2 shape with the Room 1 set in --starts
            classify(room2()[:5] + ["--starts", R1] + room2()[7:])


class IndirectTest(unittest.TestCase):
    """Inputs a command makes its child read (review B2), with V1 as the injected fresh set."""

    def setUp(self):
        self.spec = {"room1": FreshSet(V1, committed_blob(V1), None, None)}
        self.tmp = tempfile.TemporaryDirectory(dir=REPO / "runs")  # inside the repo, where indirect inputs live
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def lf_copy(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((REPO / V1).read_bytes().replace(b"\r\n", b"\n"))

    def test_recording_folder_holding_a_fresh_set(self):
        self.lf_copy(self.dir / "play.json")
        with self.assertRaises(FreshSetRefused):
            classify(["scripts/train_anchored.py", "--anchor-play", str(self.dir)], self.spec)

    def test_task_definition_naming_a_fresh_source(self):
        definition = self.dir / "task.json"
        definition.write_text(json.dumps({"source_route": V1}), encoding="utf-8")
        with self.assertRaises(FreshSetRefused):
            classify(["scripts/train_room1.py", "--task-definition", str(definition)], self.spec)

    def test_resume_folder_holding_a_fresh_manifest(self):
        self.lf_copy(self.dir / "manifest.json")
        entry = {"id": "t", "run_dir": str(self.dir), "command": ["scripts/train_room1.py", "--seed", "60"]}
        commands = [run_overnight.build_command(entry, Path("C:/g"), resume=True)[1:]]  # --resume <run folder>
        with self.assertRaises(FreshSetRefused):
            refuse_unless_declared(commands, (), self.spec)

    def test_checkpoint_provenance_beside_it(self):
        (self.dir / "checkpoints").mkdir()
        (self.dir / "checkpoints" / "latest.zip").write_bytes(b"not a fresh set")
        self.lf_copy(self.dir / "manifest.json")
        with self.assertRaises(FreshSetRefused):
            classify(["scripts/evaluate_heldout.py", "--checkpoint", str(self.dir / "checkpoints" / "latest.zip")],
                     self.spec)

    def test_declared_commands_stay_allowed(self):
        self.assertEqual(classify(copy()), "copy_overlap_guard")
        self.assertEqual(classify(room1("3")), "room1_eval")


class RoundThreeTest(unittest.TestCase):
    """Review round 3: limits refuse (R1), known dependencies are followed anywhere (R2), the runner reads nothing
    of a refused or invalid entry (R3). V1 stands in for a fresh set."""

    def setUp(self):
        self.v1 = {"room1": FreshSet(V1, committed_blob(V1), None, None)}
        self.both = {**self.v1, "room2": FRESH["room2"]}
        self.tmp = tempfile.TemporaryDirectory()  # outside the repository on purpose
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def protected_copy(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((REPO / V1).read_bytes().replace(b"\r\n", b"\n"))
        return path

    def standin_repo(self):
        """A fake repository whose stand-in protected set has a name no real path contains, so a copy command's real
        Room 2 v2 slot cannot be mistaken for it (V1's name, heldout_starts, is inside every held-out file name)."""
        repo = self.dir / "repo"
        self.protected_copy(repo / "protected" / "standin.json")
        spec = {"room1": FreshSet("protected/standin.json", committed_blob(V1), None, None), "room2": FRESH["room2"]}
        self.assertEqual(classify(copy(), spec, repo=repo), "copy_overlap_guard")  # the plain copy stays allowed
        return repo, spec

    def test_folder_limit_refuses_instead_of_skipping(self):
        repo, spec = self.standin_repo()
        folder = self.dir / "recording"
        folder.mkdir()
        for i in range(2001):
            (folder / f"filler{i:04d}.txt").write_text("x", encoding="utf-8")
        self.protected_copy(folder / "play.json")
        for command in (copy(mix=("default", str(folder))), ["scripts/train_anchored.py", "--anchor-play", str(folder)]):
            with self.subTest(command=command[0]):
                with self.assertRaises(FreshSetRefused):
                    classify(command, spec, repo=repo)
        (folder / "play.json").unlink()  # even with nothing protected, an unchecked folder is refused
        with self.assertRaises(FreshSetRefused):
            classify(["scripts/train_anchored.py", "--anchor-play", str(folder)], spec, repo=repo)

    def test_external_recording_folder_is_scanned(self):
        repo, spec = self.standin_repo()
        folder = self.dir / "external-recording"
        self.protected_copy(folder / "play.json")
        with self.assertRaises(FreshSetRefused):
            classify(copy(mix=("default", str(folder))), spec, repo=repo)

    def test_resume_manifest_chain_to_a_source_route(self):
        source = self.protected_copy(self.dir / "source.JSO")
        task = self.dir / "task.json"
        task.write_text(json.dumps({"source_route": str(source)}), encoding="utf-8")
        run = self.dir / "run"
        run.mkdir()
        (run / "manifest.json").write_text(json.dumps({"config": {"task_definition": str(task),
                                                                  "disabled_inputs": []}}), encoding="utf-8")
        with self.assertRaises(FreshSetRefused):
            classify(["scripts/train_room1.py", "--resume", str(run)], self.v1)

    def test_bare_spelling_in_a_known_field(self):
        repo = self.dir / "repo"
        canonical = self.protected_copy(repo / V1)
        try:
            os.link(canonical, repo / "ALIAS-1.JSO")  # no slash, upper-case extension
        except OSError:
            self.skipTest("hard links unavailable here")
        (repo / "task.json").write_text(json.dumps({"source_route": "ALIAS-1.JSO"}), encoding="utf-8")
        with self.assertRaises(FreshSetRefused):
            classify(["scripts/train_room1.py", "--task-definition", "task.json"], self.v1, repo=repo)

    def test_declared_commands_still_pass_the_walk(self):
        self.assertEqual(classify(copy()), "copy_overlap_guard")
        self.assertEqual(classify(room1("3")), "room1_eval")
        self.assertEqual(classify(room2()), "room2_eval")

    def test_runner_checks_run_folders_and_refuses_invalid_resume_before_reading(self):
        run = self.dir / "eval-run"
        run.mkdir()
        manifest = self.protected_copy(run / "manifest.json")
        entry = {"id": "e", "run_dir": str(run), "resume": True, "command": room1("3")}  # evaluations cannot resume
        self.assertTrue(run_overnight.fresh_set_problems({"runs": [entry]}, [Path("C:/g")], fresh=self.v1))
        real_read = Path.read_text

        def guarded_read(path, *args, **kwargs):
            if Path(path) == manifest:
                raise AssertionError("entry_problems read the manifest of an entry that cannot resume")
            return real_read(path, *args, **kwargs)

        with unittest.mock.patch.object(Path, "read_text", guarded_read):
            problems = run_overnight.entry_problems(entry)
        self.assertTrue(any("does not support --resume" in p for p in problems))

    def run_entry_with(self, outcomes):
        calls = []

        def fake_run(entry, logfile, game_dir, resume=False, env=None):
            calls.append(resume)
            return outcomes[len(calls) - 1]

        run = self.dir / "trained"
        run.mkdir(exist_ok=True)
        (run / "manifest.json").write_text("{}", encoding="utf-8")  # would tempt a resume retry
        entry = {"id": "t", "run_dir": str(run), "command": ["scripts/train_room1.py", "--seed", "60"]}
        with unittest.mock.patch.object(run_overnight, "clear_game"), \
                unittest.mock.patch.object(run_overnight, "outcome_of", side_effect=AssertionError("read outcome")):
            record = run_overnight.run_entry(entry, run_overnight.GameCopy(Path("C:/g")), self.dir / "log.txt", 10,
                                             run=fake_run, free_memory=lambda: 16.0)
        return record, calls

    def test_refused_launch_is_final(self):
        record, calls = self.run_entry_with([{"status": "refused_fresh_set", "seconds": 0}])
        self.assertEqual((record["status"], calls), ("refused_fresh_set", [False]))

    def test_refused_retry_reads_nothing_after(self):
        record, calls = self.run_entry_with([{"status": "aborted", "seconds": 5},
                                             {"status": "refused_fresh_set", "seconds": 0}])
        self.assertEqual((record["status"], calls), ("refused_fresh_set", [False, True]))


class RoundFourTest(unittest.TestCase):
    """Review round 4: per-file checks by role (B1), JSON by role not name, refusing when too large (B2), the
    game-folder exemption per occurrence (B3), the runner's own files (B4). A uniquely named stand-in in a fake
    repository plays the protected set, so no real path can be mistaken for it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.repo = self.dir / "repo"
        self.data = (REPO / V1).read_bytes().replace(b"\r\n", b"\n")
        (self.repo / "protected").mkdir(parents=True)
        (self.repo / "protected" / "standin.json").write_bytes(self.data)
        self.spec = {"room1": FreshSet("protected/standin.json", committed_blob(V1), None, None),
                     "room2": FRESH["room2"]}
        self.src = self.dir / "elsewhere" / "src.json"  # a renamed copy of the protected set, outside the run folders
        self.src.parent.mkdir()
        self.src.write_bytes(self.data)
        self.ordinary = self.dir / "elsewhere" / "route.json"
        self.ordinary.write_text('{"lines": []}', encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def refused(self, command):
        with self.assertRaises(FreshSetRefused):
            classify(command, self.spec, repo=self.repo)

    def allowed(self, command, shape=None):
        self.assertEqual(classify(command, self.spec, repo=self.repo), shape)

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def run_folder(self, task_name="task.json", source=None):
        run = self.dir / f"run-{task_name.replace('.', '_')}"
        self.write_json(run / task_name, {"source_route": str(source or self.src)})
        self.write_json(run / "manifest.json", {"config": {"task_definition": str(run / task_name)}})
        (run / "checkpoints").mkdir()
        (run / "checkpoints" / "latest.zip").write_bytes(b"PK\x03\x04 not json")
        return run

    # B1
    def test_task_inside_the_resume_folder(self):
        self.refused(["scripts/train_room1.py", "--resume", str(self.run_folder())])

    def test_file_seen_first_as_provenance_then_as_a_known_path(self):
        run = self.run_folder(task_name="play.json")  # a task definition that is also a provenance file name
        self.refused(["scripts/evaluate_heldout.py", "--checkpoint", str(run / "checkpoints" / "latest.zip")])

    def test_file_seen_first_as_a_leaf_then_as_a_known_path(self):
        task = self.write_json(self.dir / "t" / "task.json", {"source_route": str(self.src)})
        direct = self.write_json(self.dir / "t" / "direct.json", {"config": {"task_definition": str(task)},
                                                                  "note": str(task)})
        self.refused(["scripts/train_room1.py", "--task-definition", str(direct)])

    def test_clone_provenance_stays_identity_only(self):
        clone = self.dir / "clone"
        clone.mkdir()
        (clone / "cloned.zip").write_bytes(b"PK\x03\x04 not json")
        self.write_json(clone / "results.json", {"heldout": "config/heldout_starts-room2-v2.json", "x": str(self.src)})
        self.allowed(["scripts/train_room1.py", "--init-from", str(clone / "cloned.zip")])

    # B2
    def test_required_json_whatever_its_name(self):
        for name in ("task.JSO", "task"):
            with self.subTest(name=name):
                task = self.write_json(self.dir / "named" / name, {"source_route": str(self.src)})
                self.refused(["scripts/train_room1.py", "--task-definition", str(task)])
                self.refused(["scripts/train_room1.py", "--resume", str(self.run_folder(task_name=name))])

    def test_required_json_too_large_refuses_at_the_exact_boundary(self):
        task = self.write_json(self.dir / "big" / "task.json", {"source_route": str(self.ordinary)})
        run = self.run_folder(task_name="task.json", source=self.ordinary)
        cases = [  # each command and every required JSON it reads (direct or by role)
            (["scripts/train_room1.py", "--task-definition", str(task)], [task, self.ordinary]),
            (["scripts/train_room1.py", "--resume", str(run)], [run / "manifest.json", run / "task.json", self.ordinary]),
        ]
        for command, required in cases:
            largest = max(path.stat().st_size for path in required)
            for limit, ok in ((largest, True), (largest - 1, False)):  # at the limit parsed; one byte over refused
                with self.subTest(command=command[-1], limit=limit), \
                        unittest.mock.patch("celeste_rl.fresh_sets.MAX_JSON_BYTES", limit):
                    if ok:
                        self.allowed(command)
                    else:
                        self.refused(command)

    # B3
    def test_game_dir_exemption_is_per_occurrence(self):
        recording = self.dir / "recording"
        self.write_json(recording / "dataset_note.json", {})
        self.allowed(copy(mix=("default", str(recording)), extra=("--game-dir", str(recording))),
                     "copy_overlap_guard")  # control: nothing protected
        (recording / "play.json").write_bytes(self.data)
        self.refused(copy(mix=("default", str(recording)), extra=("--game-dir", str(self.dir / "game"))))
        self.refused(copy(mix=("default", str(recording)), extra=("--game-dir", str(recording))))
        entry = {"id": "c", "run_dir": "runs/clone/x", "command": copy(mix=("default", str(recording)))}
        self.refused(run_overnight.build_command(entry, recording)[1:])  # the runner-built form

    # B4
    def test_runner_files_checked_by_identity_before_reading(self):
        v1 = {"room1": FreshSet(V1, committed_blob(V1), None, None)}  # the runner's checks use the real repository
        plan_alias, settings_dir = self.dir / "plan.json", self.dir / "game" / run_overnight.SETTINGS_RELATIVE.parent
        settings_dir.mkdir(parents=True)
        try:
            os.link(REPO / V1, plan_alias)
            os.link(REPO / V1, settings_dir / run_overnight.SETTINGS_RELATIVE.name)
        except OSError:
            self.skipTest("hard links unavailable here")
        ordinary_plan = self.write_json(self.dir / "ordinary-plan.json", {"name": "x", "runs": []})
        with unittest.mock.patch.object(run_overnight.fresh_sets, "FRESH", v1):
            game = run_overnight.GameCopy(self.dir / "game")
            self.assertTrue(run_overnight.runner_file_problems(plan_alias, []))
            self.assertTrue(run_overnight.runner_file_problems(ordinary_plan, [game]))
            self.assertEqual(run_overnight.runner_file_problems(ordinary_plan, [run_overnight.GameCopy(self.dir)]), [])
            real_read = Path.read_text

            def trap(path, *args, **kwargs):
                if os.path.samefile(path, REPO / V1):
                    raise AssertionError(f"read {path} before the guard")
                return real_read(path, *args, **kwargs)

            with unittest.mock.patch.object(Path, "read_text", trap), \
                    unittest.mock.patch("sys.argv", ["run_overnight.py", "--plan", str(plan_alias), "--dry-run"]):
                self.assertEqual(run_overnight.main(), 2)


class MetadataFirstTest(unittest.TestCase):
    """A file that IS a fresh set is recognised without reading it (review B3), and no answer outlives its check."""

    def setUp(self):
        self.spec = {"room1": FreshSet(V1, committed_blob(V1), None, None)}
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def no_reading(path):
        """Content reader that fails if asked for the protected file's contents (other files read normally)."""
        if os.path.samefile(path, REPO / V1):
            raise AssertionError(f"read the protected file's contents through {path}")
        return Path(path).read_bytes()

    def recognised_without_reading(self, token):
        command = ["scripts/evaluate_heldout.py", "--checkpoint", CKPT, "--starts", token, "--repeats", "3",
                   "--seed", EVAL_SEED]
        with self.assertRaises(FreshSetRefused):
            classify(command, self.spec, read_content=self.no_reading)

    def test_short_name_without_reading(self):
        buffer = ctypes.create_unicode_buffer(1024)
        if not ctypes.windll.kernel32.GetShortPathNameW(str(REPO / V1), buffer, 1024) or \
                "heldout_starts" in buffer.value.lower():
            self.skipTest("no 8.3 name")
        self.recognised_without_reading(buffer.value)
        self.recognised_without_reading(f"--starts={buffer.value}")

    def test_hard_link_without_reading(self):
        link = self.dir / "innocent.json"
        try:
            os.link(REPO / V1, link)
        except OSError:
            self.skipTest("hard links unavailable here")
        self.recognised_without_reading(str(link))

    def test_junction_whose_real_path_names_the_set(self):
        """Only the real-path step can catch this: a folder alias (identity compares against the fresh files)."""
        target = self.dir / "heldout_starts_folder"  # its real name carries the injected set's marker
        target.mkdir()
        link = self.dir / "innocent"
        made = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, text=True)
        if made.returncode != 0:
            self.skipTest("junctions unavailable here")
        self.assertNotIn("heldout_starts", str(link).lower())
        self.assertTrue(refers_to_fresh(str(link), self.spec))

    def test_equals_form_renamed_copy(self):
        copy_path = self.dir / "renamed.txt"
        copy_path.write_bytes((REPO / V1).read_bytes().replace(b"\r\n", b"\n"))
        self.assertTrue(refers_to_fresh(f"--starts={copy_path}", self.spec))

    def test_no_answer_survives_between_checks(self):
        target = self.dir / "same.txt"
        data = (REPO / V1).read_bytes().replace(b"\r\n", b"\n")
        target.write_bytes(b"x" * len(data))
        stamp = (1_700_000_000, 1_700_000_000)
        os.utime(target, stamp)
        self.assertFalse(refers_to_fresh(str(target), self.spec))
        target.write_bytes(data)  # same size, same mtime, now a copy of the protected set
        os.utime(target, stamp)
        self.assertTrue(refers_to_fresh(str(target), self.spec))


if __name__ == "__main__":
    unittest.main()
