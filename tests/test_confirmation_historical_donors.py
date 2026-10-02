"""The historical donor executor (scripts/confirmation_historical_donors.py): authorization, plan, source check,
fresh-set checks, what counts as ok, and the run order. Nothing here starts the game or training."""
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from celeste_rl import fresh_sets
from celeste_rl.fresh_sets import FreshSet
from scripts import confirmation_historical_donors as h
from scripts import confirmation_recipe_check as rc

REPO = Path(__file__).resolve().parents[1]
DECLARATION = json.loads((REPO / "config" / "retention-confirmation.json").read_text(encoding="utf-8"))
REVIEWED = h.REVIEWED_COMPARATORS[0]
V1 = "config/heldout_starts.json"  # a development set, standing in for a fresh one


class Temp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()


def comparator_record(verdict, kind="compare", version=2, commit=REVIEWED, dirty=False, bound=None):
    record = {"comparator": {"version": version, "script_sha256": "0" * 64,
                             "git": {"commit": commit, "uncommitted_changes": dirty, "git_error": None}},
              "verdict": verdict}
    if kind == "compare":
        record["run_record"] = {"path": f"C:/x/runs/confirmation/recipe-check/{bound or h.RECIPE_CHECK_RUN}"}
    else:
        record["run_record"] = {"sha256": bound or h.DIAGNOSTIC_SHA256}
    return record


def write(directory: Path, name: str, record) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


class DeclarationTest(unittest.TestCase):
    def test_amendment_2_is_present(self):
        self.assertEqual(h.amendment_problems(DECLARATION), [])
        stripped = copy.deepcopy(DECLARATION)
        del stripped["donors"]["historical_training"]
        del stripped["owner_decisions"]["historical_donors"]
        self.assertEqual(len(h.amendment_problems(stripped)), 2)

    def test_donor_command_is_the_declared_command(self):
        command = h.donor_command(DECLARATION, 9, h.run_folder(9, 1))
        campaign = json.loads((REPO / "config" / "campaign-finetune-variance.json").read_text(encoding="utf-8"))
        seed7 = [r for r in campaign["runs"] if r["id"] == "B-seed7"][0]["command"]  # the original donor command
        differing = {seed7[i - 1] for i, (a, b) in enumerate(zip(seed7, command)) if a != b}
        self.assertEqual(differing, {"--seed", "--run-dir"})
        self.assertEqual(command[len(seed7):], ["--game-dir", h.GAME.as_posix()])
        self.assertEqual(command[command.index("--run-dir") + 1], "runs/train/confirm-donor-B-seed9")
        self.assertEqual(h.run_folder(9, 3), "runs/train/confirm-donor-B-seed9-rerun2")

    def test_forbidden_options_are_refused(self):
        for option in ("--resume x", "--allow-dirty", "--allow-runtime-mismatch"):
            with self.subTest(option=option):
                bad = copy.deepcopy(DECLARATION)
                bad["donors"]["command"] += f" {option}"
                with self.assertRaises(h.Refused):
                    h.donor_command(bad, 9, h.run_folder(9, 1))

    def test_donor_index(self):
        self.assertEqual(h.donor_index(DECLARATION), {7: 0, 8: 1, 9: 2, 10: 3, 11: 4, 12: 5, 13: 6, 14: 7})


class AuthorizationTest(Temp):
    """Amendment 2: the corrected diagnostic plus the replay's DIFFERENT permit training; nothing else does."""

    def good(self):
        write(self.dir, "compare-20261003-090000.json", comparator_record("DIFFERENT"))
        write(self.dir, "compare-diagnostic-20261003-090100.json", comparator_record("IDENTICAL", "diagnostic"))

    def test_good_records_authorize(self):
        self.good()
        auth = h.authorization(self.dir)
        self.assertEqual(auth["compare"]["verdict"], "DIFFERENT")
        self.assertEqual(auth["compare_diagnostic"]["verdict"], "IDENTICAL")
        self.assertTrue(auth["compare_diagnostic"]["path"].endswith("compare-diagnostic-20261003-090100.json"))

    def test_missing_or_unreviewed_records_refuse(self):
        cases = {
            "nothing": [],
            "only the replay": [("compare-20261003-090000.json", comparator_record("DIFFERENT"))],
            "version 1": [("compare-20261003-090000.json", comparator_record("DIFFERENT", version=1)),
                          ("compare-diagnostic-20261003-090100.json", comparator_record("IDENTICAL", "d", version=1))],
            "unreviewed commit": [("compare-20261003-090000.json", comparator_record("DIFFERENT", commit="abc")),
                                  ("compare-diagnostic-20261003-090100.json",
                                   comparator_record("IDENTICAL", "d", commit="abc"))],
            "dirty tree": [("compare-20261003-090000.json", comparator_record("DIFFERENT", dirty=True)),
                           ("compare-diagnostic-20261003-090100.json", comparator_record("IDENTICAL", "d", dirty=True))],
            "diagnostic file only matches the diagnostic pattern": [
                ("compare-diagnostic-20261003-090100.json", comparator_record("IDENTICAL", "d"))]}
        for name, files in cases.items():
            with self.subTest(case=name):
                folder = self.dir / name.replace(" ", "-")
                folder.mkdir()
                for file, record in files:
                    write(folder, file, record)
                with self.assertRaises(h.Refused):
                    h.authorization(folder)

    def test_every_reviewed_record_must_agree(self):
        """No IDENTICAL record of the current code may authorize, and a disagreeing diagnostic record refuses."""
        for name, extra in (
                ("current code identical", ("compare-20261003-100000.json", comparator_record("IDENTICAL"))),
                ("bound to another run", ("compare-20261003-100000.json",
                                          comparator_record("DIFFERENT", bound="run-20261003-000000.json"))),
                ("diagnostic different", ("compare-diagnostic-20261003-100000.json",
                                          comparator_record("DIFFERENT", "d"))),
                ("diagnostic bound elsewhere", ("compare-diagnostic-20261003-100000.json",
                                                comparator_record("IDENTICAL", "d", bound="1" * 64)))):
            with self.subTest(case=name):
                folder = self.dir / name.replace(" ", "-")
                write(folder, "compare-20261003-090000.json", comparator_record("DIFFERENT"))
                write(folder, "compare-diagnostic-20261003-090100.json", comparator_record("IDENTICAL", "d"))
                write(folder, *extra)
                with self.assertRaises(h.Refused):
                    h.authorization(folder)

    def test_unreadable_record_authorizes_nothing(self):
        write(self.dir, "compare-20261003-090000.json", comparator_record("DIFFERENT"))
        (self.dir / "compare-diagnostic-20261003-090100.json").write_text("{not json", encoding="utf-8")
        with self.assertRaises(h.Refused):
            h.authorization(self.dir)


AUTH = {"compare": {"path": "runs/confirmation/recipe-check/compare-x.json", "sha256": "a" * 64,
                    "verdict": "DIFFERENT", "comparator_commit": REVIEWED},
        "compare_diagnostic": {"path": "runs/confirmation/recipe-check/compare-diagnostic-x.json", "sha256": "b" * 64,
                               "verdict": "IDENTICAL", "comparator_commit": REVIEWED}}


def attempt_record(directory: Path, seed: int, attempt: int, outcome: str, stamp: str):
    write(directory, f"attempt-{stamp}.json", {"entry": {"seed": seed, "attempt": attempt}, "outcome": outcome})


class PlanTest(Temp):
    def setUp(self):
        super().setUp()
        self.plan_path = self.dir / "plan.json"
        self.patch = unittest.mock.patch.object(h, "PLAN", self.plan_path)
        self.patch.start()
        self.records = self.dir / "records"
        self.records.mkdir()

    def tearDown(self):
        self.patch.stop()
        super().tearDown()

    def initial(self):
        plan = h.initial_plan(DECLARATION, AUTH)
        self.plan_path.write_text(json.dumps(plan), encoding="utf-8")
        return plan

    def test_initial_plan_is_the_six_new_donors_in_order(self):
        plan = self.initial()
        self.assertEqual([(e["j"], e["seed"], e["attempt"]) for e in plan["entries"]],
                         [(2, 9, 1), (3, 10, 1), (4, 11, 1), (5, 12, 1), (6, 13, 1), (7, 14, 1)])
        self.assertEqual(plan["source"], {"commit": h.HISTORICAL_COMMIT, "tree": h.HISTORICAL_TREE})
        self.assertEqual(plan["declaration"]["text_sha256"], fresh_sets.text_sha256(h.DECLARATION))
        self.assertEqual(h.plan_problems(plan, DECLARATION, AUTH), [])

    def test_any_change_to_a_plan_is_reported(self):
        plan = self.initial()
        other_auth = {**AUTH, "compare": {**AUTH["compare"], "sha256": "c" * 64}}
        self.assertTrue(h.plan_problems(plan, DECLARATION, other_auth))
        for name, change in (
                ("command flag", lambda p: p["entries"][0]["command"].append("--allow-dirty")),
                ("seed", lambda p: p["entries"][0].update(seed=99)),
                ("j", lambda p: p["entries"][1].update(j=0)),
                ("order", lambda p: p["entries"].reverse()),
                ("dropped donor", lambda p: p["entries"].pop()),
                ("worktree", lambda p: p.update(worktree="C:/elsewhere")),
                ("limit", lambda p: p.update(limit_minutes=240)),
                ("declaration hash", lambda p: p["declaration"].update(text_sha256="0" * 64)),
                ("skipped attempt", lambda p: p["entries"].append(h.entry(DECLARATION, 2, 9, 3)))):
            with self.subTest(change=name):
                changed = copy.deepcopy(plan)
                change(changed)
                self.assertTrue(h.plan_problems(changed, DECLARATION, AUTH))

    def test_rerun_only_after_a_failure_and_at_most_twice(self):
        plan = self.initial()
        with self.assertRaises(h.Refused):
            h.revised_plan(plan, DECLARATION, self.records, rerun=10)  # no failed attempt
        attempt_record(self.records, 10, 1, "failed", "20261004-010000")
        revised = h.revised_plan(plan, DECLARATION, self.records, rerun=10)
        self.assertEqual(revised["entries"][:6], plan["entries"])  # earlier entries kept exactly
        self.assertEqual(revised["entries"][6]["run_dir"], "runs/train/confirm-donor-B-seed10-rerun1")
        self.assertEqual((revised["entries"][6]["j"], revised["revision"]), (3, 1))
        self.assertEqual(h.plan_problems(revised, DECLARATION, AUTH), [])
        for attempt, stamp in ((2, "20261004-020000"), (3, "20261004-030000")):
            attempt_record(self.records, 10, attempt, "failed", stamp)
            revised = h.revised_plan(revised, DECLARATION, self.records, rerun=10) if attempt == 2 else revised
        with self.assertRaises(h.Refused):  # attempt 3 failed: two reruns used
            h.revised_plan(revised, DECLARATION, self.records, rerun=10)

    def test_replacements_take_the_replaced_j_within_the_budget(self):
        plan = self.initial()
        with self.assertRaises(h.Refused):
            h.revised_plan(plan, DECLARATION, self.records, replace=11)  # no reason
        first = h.revised_plan(plan, DECLARATION, self.records, replace=11, reason="copy v1 margin below 0.20")
        self.assertEqual((first["entries"][-1]["seed"], first["entries"][-1]["j"]), (15, 4))
        second = h.revised_plan(first, DECLARATION, self.records, replace=7, reason="donor v1 margin below 0.20")
        self.assertEqual((second["entries"][-1]["seed"], second["entries"][-1]["j"]), (16, 0))
        self.assertEqual(h.plan_problems(second, DECLARATION, AUTH), [])
        with self.assertRaises(h.Refused):
            h.revised_plan(second, DECLARATION, self.records, replace=12, reason="x")
        wrong = copy.deepcopy(first)
        wrong["entries"][-1]["j"] = 5
        self.assertTrue(h.plan_problems(wrong, DECLARATION, AUTH))


class RunOrderTest(Temp):
    def plan(self, extra=()):
        plan = h.initial_plan(DECLARATION, AUTH)
        plan["entries"] += list(extra)
        return plan

    def test_pending_entries_follow_the_records(self):
        plan = self.plan()
        self.assertEqual(len(h.next_entries(plan, self.dir)), 6)
        attempt_record(self.dir, 9, 1, "ok", "20261004-010000")
        write(self.dir, "refused-20261004-010500.json", {"entry": {"seed": 10, "attempt": 1}})  # does not count
        write(self.dir, "dry-run-20261004-010600.json", {"entry": {"seed": 10, "attempt": 1}})
        self.assertEqual([e["seed"] for e in h.next_entries(plan, self.dir)], [10, 11, 12, 13, 14])

    def test_a_failure_stops_until_the_plan_is_revised(self):
        attempt_record(self.dir, 9, 1, "failed", "20261004-010000")
        with self.assertRaises(h.Refused):
            h.next_entries(self.plan(), self.dir)
        rerun = h.entry(DECLARATION, 2, 9, 2, "rerun after a failed attempt")
        self.assertEqual(h.next_entries(self.plan([rerun]), self.dir)[-1], rerun)

    def test_no_second_attempt_after_an_ok_one(self):
        attempt_record(self.dir, 9, 1, "ok", "20261004-010000")
        with self.assertRaises(h.Refused):
            h.next_entries(self.plan([h.entry(DECLARATION, 2, 9, 2, "rerun after a failed attempt")]), self.dir)
        too_many = self.plan([h.entry(DECLARATION, 2, 9, n, "rerun after a failed attempt") for n in (2, 3, 4)])
        self.assertTrue([p for p in h.plan_problems(too_many, DECLARATION, AUTH) if "reruns" in p])

    def test_unreadable_or_contradicting_records_are_failures(self):
        (self.dir / "attempt-20261004-010000.json").write_text("{", encoding="utf-8")
        self.assertEqual(h.entry_status(self.dir, {"seed": 9, "attempt": 1}), "failed")
        other = self.dir / "other"
        attempt_record(other, 9, 1, "ok", "20261004-010000")
        attempt_record(other, 9, 1, "maybe", "20261004-020000")
        self.assertEqual(h.entry_status(other, {"seed": 9, "attempt": 1}), "failed")


def git(root: Path, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args], check=True,
                   capture_output=True)


class SourceCheckTest(Temp):
    """code_checks.historical_donors on a small repository standing in for 8423081."""

    def make_repo(self, with_package=True) -> tuple[Path, str, str]:
        repo = self.dir / "worktree"
        (repo / "scripts").mkdir(parents=True)
        (repo / ".gitignore").write_text("runs/\n__pycache__/\n", encoding="utf-8")
        (repo / "scripts" / "train_room1.py").write_text(
            "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parents[1]))\n"
            "import celeste_rl.reward\n", encoding="utf-8")
        if with_package:
            (repo / "celeste_rl").mkdir()
            (repo / "celeste_rl" / "__init__.py").write_text("", encoding="utf-8")
            (repo / "celeste_rl" / "reward.py").write_text("SCALE = 2.0\n", encoding="utf-8")
        git(repo.parent, "init", "-q", str(repo))
        git(repo, "add", ".")
        git(repo, "commit", "-qm", "historical")
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD", "HEAD^{tree}"], capture_output=True,
                              text=True).stdout.split()
        return repo, head[0], head[1]

    def problems(self, repo, commit, tree):
        return h.source_problems(repo, commit, tree, Path(sys.executable))

    def test_clean_checkout_passes_with_its_closure(self):
        repo, commit, tree = self.make_repo()
        problems, evidence = self.problems(repo, commit, tree)
        self.assertEqual(problems, [])
        self.assertEqual(sorted(evidence["closure"]),
                         ["celeste_rl/__init__.py", "celeste_rl/reward.py", "scripts/train_room1.py"])

    def test_changes_strays_flags_and_wrong_commits_fail(self):
        repo, commit, tree = self.make_repo()
        reward = repo / "celeste_rl" / "reward.py"
        cases = (
            ("edited module", lambda: reward.write_text("SCALE = 3.0\n", encoding="utf-8"), lambda: git(repo, "checkout", "--", ".")),
            ("hidden edit", lambda: (reward.write_text("SCALE = 3.0\n", encoding="utf-8"),
                                     git(repo, "update-index", "--assume-unchanged", "celeste_rl/reward.py")),
             lambda: (git(repo, "update-index", "--no-assume-unchanged", "celeste_rl/reward.py"),
                      git(repo, "checkout", "--", "."))),
            ("untracked file", lambda: (repo / "notes.txt").write_text("x", encoding="utf-8"),
             lambda: (repo / "notes.txt").unlink()),
            ("stray ignored file", lambda: ((repo / "runs" / "other").mkdir(parents=True),
                                            (repo / "runs" / "other" / "x.json").write_text("{}", encoding="utf-8")),
             lambda: (repo / "runs" / "other" / "x.json").unlink()))
        for name, damage, repair in cases:
            with self.subTest(case=name):
                damage()
                try:
                    self.assertTrue(self.problems(repo, commit, tree)[0])
                finally:
                    repair()
        self.assertEqual(self.problems(repo, commit, tree)[0], [])
        self.assertTrue(self.problems(repo, "0" * 40, tree)[0])
        self.assertTrue(h.source_problems(self.dir / "nowhere", commit, tree, Path(sys.executable))[0])

    def test_declared_ignored_files_are_allowed(self):
        repo, commit, tree = self.make_repo()
        for relative in ("runs/clone/20260918-183508/cloned.zip", "runs/train/confirm-donor-B-seed9/checkpoints/a.zip",
                         "runs/train/confirm-donor-B-seed9-rerun1/manifest.json", "celeste_rl/__pycache__/r.pyc"):
            (repo / relative).parent.mkdir(parents=True, exist_ok=True)
            (repo / relative).write_text("x", encoding="utf-8")
        self.assertEqual(self.problems(repo, commit, tree)[0], [])

    def test_a_module_imported_from_outside_fails(self):
        repo, commit, tree = self.make_repo(with_package=False)
        outside = self.dir / "outside" / "celeste_rl"
        outside.mkdir(parents=True)
        (outside / "__init__.py").write_text("", encoding="utf-8")
        (outside / "reward.py").write_text("SCALE = 2.0\n", encoding="utf-8")
        with unittest.mock.patch.dict(os.environ, {"PYTHONPATH": str(outside.parent)}):
            problems = self.problems(repo, commit, tree)[0]
        self.assertTrue([p for p in problems if "outside the worktree" in p])


class InitCloneTest(Temp):
    def test_copy_must_equal_the_source_and_the_pin(self):
        source = REPO / h.INIT_CLONE
        if not source.exists():
            self.skipTest("init clone not present")
        import shutil
        shutil.copytree(source, self.dir / h.INIT_CLONE)
        problems, hashes = h.init_clone_problems(self.dir)
        self.assertEqual((problems, hashes["cloned.zip"]), ([], h.INIT_CLONE_SHA256))
        (self.dir / h.INIT_CLONE / "results.json").write_text("{}", encoding="utf-8")
        self.assertTrue(h.init_clone_problems(self.dir)[0])
        self.assertTrue(h.init_clone_problems(self.dir / "missing")[0])


class GuardTest(Temp):
    """The fresh-set guard sees the command as the child resolves it inside the worktree."""

    def test_alias_reachable_only_inside_the_worktree_is_refused(self):
        spec = {"room1": FreshSet(V1, fresh_sets.git_blob(REPO / V1), None, None)}
        (self.dir / "inputs").mkdir()
        os.link(REPO / V1, self.dir / "inputs" / "starts.json")  # a hard link: identity, no content read
        command = ["scripts/train_room1.py", "--seed", "9", "--extra", "inputs/starts.json"]
        self.assertIsNone(fresh_sets.classify(command, fresh=spec))  # as given, it names nothing in this repository
        self.assertTrue(h.guard_problems(command, self.dir, fresh=spec))
        self.assertEqual(h.guard_problems(command[:3], self.dir, fresh=spec), [])

    def test_resolution_keeps_options_and_absent_paths(self):
        (self.dir / "runs" / "clone").mkdir(parents=True)
        command = ["--seed", "9", "runs/clone", "runs/train/new", "C:/abs"]
        self.assertEqual(h.resolved(command, self.dir),
                         ["--seed", "9", str(self.dir / "runs/clone"), "runs/train/new", "C:/abs"])


class OutcomeTest(unittest.TestCase):
    """donors.historical_training.ok, with the diagnostic's own run (8423081, seed 7) as the positive control."""

    def setUp(self):
        if not rc.DIAGNOSTIC_RUN.exists() or not (REPO / rc.ORIGINAL).exists():
            self.skipTest("diagnostic run copy or original not present")
        self.manifest = json.loads((rc.DIAGNOSTIC_RUN / "manifest.json").read_text(encoding="utf-8"))

    def test_the_diagnostic_run_counts(self):
        problems, faults = h.outcome_problems(rc.DIAGNOSTIC_RUN, 7)
        self.assertEqual(problems, [])
        self.assertEqual(faults["faults"], [])

    def test_the_seed_must_match_the_config(self):
        self.assertTrue(h.outcome_problems(rc.DIAGNOSTIC_RUN, 9)[0])

    def test_every_ok_condition_counts(self):
        def resumed(m):
            m["sessions"].append(copy.deepcopy(m["sessions"][0]))

        def provenance(**changes):
            return lambda m: m["sessions"][0]["provenance"].update(changes)

        cases = {"resumed": resumed, "other commit": provenance(commit="2bd1fc2"),
                 "uncommitted": provenance(uncommitted_changes=True),
                 "runtime problem": provenance(runtime_problems=["mod hash"]),
                 "runtime differs": lambda m: m["sessions"][0]["provenance"]["runtime"]["python"].update(torch="x"),
                 "config differs": lambda m: m["config"].update(ent_coef=0.0),
                 "config field added": lambda m: m["config"].update(stall_frames=0)}
        for name, change in cases.items():
            with self.subTest(case=name):
                manifest = copy.deepcopy(self.manifest)
                change(manifest)
                with unittest.mock.patch.object(rc, "validate_run", return_value=manifest):
                    self.assertTrue(h.outcome_problems(rc.DIAGNOSTIC_RUN, 7)[0])

    def test_incomplete_run_is_not_ok(self):
        with unittest.mock.patch.object(rc, "validate_run", side_effect=rc.Invalid("lacks checkpoints")):
            self.assertTrue(h.outcome_problems(rc.DIAGNOSTIC_RUN, 7)[0])


class CopyAndRecordTest(Temp):
    def test_copy_is_verified_and_never_overwrites(self):
        source = self.dir / "source"
        (source / "checkpoints").mkdir(parents=True)
        (source / "checkpoints" / "a.zip").write_bytes(b"abc")
        (source / "manifest.json").write_text("{}", encoding="utf-8")
        copied = h.copy_run(source, self.dir / "target")
        self.assertTrue(copied["verified"])
        self.assertEqual(set(copied["files"]), {"checkpoints/a.zip", "manifest.json"})
        with self.assertRaises(h.Refused):
            h.copy_run(source, self.dir / "target")

    def test_records_are_never_overwritten(self):
        with unittest.mock.patch.object(h, "datetime") as clock:
            clock.now.return_value.strftime.return_value = "20261004-000000"
            first = h.write_record({"a": 1}, "attempt", {"stdout.txt": "one"}, self.dir)
            second = h.write_record({"a": 2}, "attempt", {"stdout.txt": "two"}, self.dir)
        self.assertNotEqual(first, second)
        self.assertEqual(json.loads(first.read_text(encoding="utf-8"))["a"], 1)
        self.assertEqual((self.dir / "attempt-20261004-000000-2.stdout.txt").read_text(encoding="utf-8"), "two")


class LaunchTest(Temp):
    """run: every launch condition refuses, and a refused attempt writes a refusal, never an attempt record."""

    def patches(self, **overrides):
        values = {"source_problems": ([], {}), "init_clone_problems": ([], {}), "guard_problems": [],
                  "authorization": AUTH, "committed": [], "plan_problems": [], "steam_running": True}
        values.update(overrides)
        patches = [unittest.mock.patch.object(h, name, return_value=value) for name, value in values.items()]
        plan = self.dir / "plan.json"
        plan.write_text(json.dumps(h.initial_plan(DECLARATION, AUTH)), encoding="utf-8")
        return patches + [
            unittest.mock.patch.object(h, "PLAN", plan),
            unittest.mock.patch.object(h, "RECORDS", self.dir / "records"),
            unittest.mock.patch.object(h, "WORKTREE", self.dir / "worktree"),
            unittest.mock.patch.object(h.runtime, "refusal", return_value=None),
            unittest.mock.patch.object(h.sys, "executable", str(REPO / h.INTERPRETER)),
            unittest.mock.patch.object(rc, "all_game_processes", return_value=[]),
            unittest.mock.patch.object(rc, "other_experiments", return_value=[]),
            unittest.mock.patch.object(rc, "thread_probe", return_value={"torch_intra_op_threads": 10})]

    def launch(self, patches, item=None, dry_run=True):
        item = item or h.entry(DECLARATION, 2, 9, 1)
        for p in patches:
            p.start()
        try:
            return h.run_entry(h.initial_plan(DECLARATION, AUTH), item, DECLARATION, dry_run)
        finally:
            for p in reversed(patches):
                p.stop()

    def test_all_clear_dry_run(self):
        self.assertEqual(self.launch(self.patches()), "dry-run")
        self.assertEqual(len(list((self.dir / "records").glob("dry-run-*.json"))), 1)

    def test_each_condition_refuses(self):
        for name, override in (("source", {"source_problems": (["not 8423081"], {})}),
                               ("init clone", {"init_clone_problems": (["differs"], {})}),
                               ("fresh set", {"guard_problems": ["fresh-set guard: x"]}),
                               ("plan", {"plan_problems": ["plan entry differs"]}),
                               ("plan not committed", {"committed": ["not committed"]}),
                               ("steam", {"steam_running": False})):
            with self.subTest(case=name):
                self.assertEqual(self.launch(self.patches(**override)), "refused")
        for name, patch in (("game", unittest.mock.patch.object(rc, "all_game_processes", return_value=["Celeste"])),
                            ("experiment", unittest.mock.patch.object(rc, "other_experiments", return_value=["x.py"]))):
            with self.subTest(case=name):
                self.assertEqual(self.launch(self.patches() + [patch]), "refused")
        (self.dir / "worktree" / "runs/train/confirm-donor-B-seed9").mkdir(parents=True)
        self.assertEqual(self.launch(self.patches()), "refused")
        self.assertFalse(list((self.dir / "records").glob("attempt-*.json")))

    def test_failed_training_is_recorded_and_stops(self):
        real_run = subprocess.run

        def fake_run(args, **kwargs):
            if "scripts/train_room1.py" in args:
                self.assertEqual(Path(kwargs["cwd"]), self.dir / "worktree")
                self.assertFalse([k for k in kwargs["env"] if k.upper().startswith(rc.THREAD_PREFIXES)])
                return subprocess.CompletedProcess(args, 3, stdout="out", stderr="err")
            return real_run(args, **kwargs)  # git state queries

        with unittest.mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "4"}):
            status = self.launch(self.patches() + [unittest.mock.patch.object(h.subprocess, "run", side_effect=fake_run)],
                                 dry_run=False)
        self.assertEqual(status, "failed")
        record = json.loads(next((self.dir / "records").glob("attempt-*.json")).read_text(encoding="utf-8"))
        self.assertEqual((record["outcome"], record["exit_code"]), ("failed", 3))
        self.assertEqual(record["thread_variables_removed"], {"OMP_NUM_THREADS": "4"})
        self.assertEqual(Path(record["log_files"]["stderr.txt"]).read_text(encoding="utf-8"), "err")


if __name__ == "__main__":
    unittest.main()
