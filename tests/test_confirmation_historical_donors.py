"""The historical donor executor (scripts/confirmation_historical_donors.py): authorization, plan, source check,
fresh-set checks, what counts as ok, and the run order. Nothing here starts the game or training."""
import copy
import json
import os
import shutil
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

    def test_replacement_is_for_inclusion_never_for_a_training_failure(self):
        """Review B2: a failed or untrained donor is recovered with its own seed, not replaced."""
        plan = self.initial()
        with self.assertRaises(h.Refused):  # not trained yet: no inclusion outcome exists
            h.revised_plan(plan, DECLARATION, self.records, replace=9, reason="copy v1 margin below 0.20")
        attempt_record(self.records, 9, 1, "failed", "20261004-010000")
        with self.assertRaises(h.Refused):
            h.revised_plan(plan, DECLARATION, self.records, replace=9, reason="training timed out")
        rerun = h.revised_plan(plan, DECLARATION, self.records, rerun=9)
        attempt_record(self.records, 9, 2, "ok", "20261004-020000")
        replaced = h.revised_plan(rerun, DECLARATION, self.records, replace=9, reason="donor v1 margin below 0.20")
        self.assertEqual((replaced["entries"][-1]["seed"], replaced["entries"][-1]["j"]), (15, 2))

    def test_replacement_order_is_validated_in_hand_edited_plans(self):
        plan = self.initial()
        attempt_record(self.records, 11, 1, "ok", "20261004-010000")
        sixteen_first = copy.deepcopy(plan)
        sixteen_first["entries"].append({**h.entry(DECLARATION, 4, 16, 1, "replaces seed 11: x")})
        self.assertTrue([p for p in h.plan_problems(sixteen_first, DECLARATION, AUTH) if "order" in p])

    def test_replacements_take_the_replaced_j_within_the_budget(self):
        plan = self.initial()
        attempt_record(self.records, 11, 1, "ok", "20261004-010000")
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


class PlanActionTest(Temp):
    """The plan command: written once, checked, revised only from a committed plan."""

    def run_plan(self, check=False, rerun=None, replace=None, reason=None, committed=()):
        with unittest.mock.patch.object(h, "PLAN", self.dir / "plan.json"), \
                unittest.mock.patch.object(h, "RECORDS", self.dir / "records"), \
                unittest.mock.patch.object(h, "authorization", return_value=AUTH), \
                unittest.mock.patch.object(h, "committed", return_value=list(committed)):
            return h.plan_action(check, rerun, replace, reason)

    def test_write_once_check_and_revise(self):
        self.assertEqual(self.run_plan(), 0)
        first = (self.dir / "plan.json").read_text(encoding="utf-8")
        self.assertEqual(self.run_plan(), 1)  # never overwritten
        self.assertEqual((self.dir / "plan.json").read_text(encoding="utf-8"), first)
        self.assertEqual(self.run_plan(check=True), 0)
        attempt_record(self.dir / "records", 12, 1, "ok", "20261004-010000")
        self.assertEqual(self.run_plan(replace=12, reason="x", committed=["plan.json is not committed"]), 1)
        self.assertEqual((self.dir / "plan.json").read_text(encoding="utf-8"), first)
        self.assertEqual(self.run_plan(replace=12, reason="copy v1 margin below 0.20"), 0)
        revised = json.loads((self.dir / "plan.json").read_text(encoding="utf-8"))
        self.assertEqual((revised["revision"], revised["entries"][-1]["seed"]), (1, 15))
        self.assertEqual(self.run_plan(check=True), 0)
        revised["entries"][0]["command"].append("--resume")
        (self.dir / "plan.json").write_text(json.dumps(revised), encoding="utf-8")
        self.assertEqual(self.run_plan(check=True), 1)

    def test_no_plan_without_authorization(self):
        with unittest.mock.patch.object(h, "PLAN", self.dir / "plan.json"), \
                unittest.mock.patch.object(h, "authorization", side_effect=h.Refused("no records")):
            self.assertEqual(h.plan_action(False, None, None, None), 1)
        self.assertFalse((self.dir / "plan.json").exists())


class RunOrderTest(Temp):
    def plan(self, extra=()):
        plan = h.initial_plan(DECLARATION, AUTH)
        plan["entries"] += list(extra)
        return plan

    RERUN_9 = staticmethod(lambda n=2: h.entry(DECLARATION, 2, 9, n, "rerun after a failed attempt"))

    def test_next_entry_follows_the_records(self):
        plan = self.plan()
        self.assertEqual(h.next_entry(plan, self.dir)["seed"], 9)
        attempt_record(self.dir, 9, 1, "ok", "20261004-010000")
        write(self.dir, "refused-20261004-010500.json", {"entry": {"seed": 10, "attempt": 1}})  # not an attempt
        write(self.dir, "dry-run-20261004-010600.json", {"entry": {"seed": 10, "attempt": 1}})
        self.assertEqual(h.next_entry(plan, self.dir)["seed"], 10)

    def test_a_failure_stops_until_a_rerun_is_planned(self):
        attempt_record(self.dir, 9, 1, "failed", "20261004-010000")
        with self.assertRaises(h.Refused):
            h.next_entry(self.plan(), self.dir)
        replacement = h.entry(DECLARATION, 2, 15, 1, "replaces seed 9: x")
        with self.assertRaises(h.Refused):  # a replacement does not recover a training failure
            h.next_entry(self.plan([replacement]), self.dir)
        self.assertEqual(h.next_entry(self.plan([self.RERUN_9()]), self.dir)["seed"], 10)  # rerun resolves seed 9

    def test_eligibility_at_launch(self):
        """Review B2: rules that depend on the records are checked again immediately before each launch."""
        rerun = self.RERUN_9()
        plan = self.plan([rerun])
        self.assertTrue(h.eligibility_problems(plan, rerun, self.dir, preflight=True))  # no failed attempt 1
        self.assertTrue(h.eligibility_problems(plan, plan["entries"][1], self.dir))  # seed 9 unresolved
        self.assertEqual(h.eligibility_problems(plan, plan["entries"][1], self.dir, preflight=True), [])
        attempt_record(self.dir, 9, 1, "ok", "20261004-010000")
        self.assertTrue([p for p in h.eligibility_problems(plan, rerun, self.dir, preflight=True) if "ok attempt" in p])
        self.assertTrue(h.eligibility_problems(plan, plan["entries"][0], self.dir))  # already has a record
        replacement = h.entry(DECLARATION, 3, 15, 1, "replaces seed 10: x")
        plan = self.plan([replacement])
        self.assertTrue(h.eligibility_problems(plan, replacement, self.dir, preflight=True))  # seed 10 not ok
        attempt_record(self.dir, 10, 1, "ok", "20261004-020000")
        self.assertEqual(h.eligibility_problems(plan, replacement, self.dir, preflight=True), [])

    def test_batch_rereads_records_before_every_launch(self):
        """Review B2: after six successes in one batch, a hand-added attempt 2 of seed 9 is refused, not trained."""
        plan = self.plan([self.RERUN_9()])
        self.assertEqual(h.plan_problems(plan, DECLARATION, AUTH), [])  # statically valid
        plan_path = self.dir / "plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        launched = []

        def fake_entry(plan, item, declaration, dry_run):
            if h.eligibility_problems(plan, item, self.dir / "records"):
                launched.append(("refused", item["seed"], item["attempt"]))
                return "refused"
            launched.append(("ok", item["seed"], item["attempt"]))
            attempt_record(self.dir / "records", item["seed"], item["attempt"], "ok",
                           f"20261004-0{len(launched)}0000")
            return "ok"

        (self.dir / "records").mkdir()
        with unittest.mock.patch.object(h, "PLAN", plan_path), \
                unittest.mock.patch.object(h, "RECORDS", self.dir / "records"), \
                unittest.mock.patch.object(h, "run_entry", side_effect=fake_entry):
            self.assertEqual(h.run_action(False, None), 1)
        self.assertEqual(launched[-1], ("refused", 9, 2))
        self.assertEqual([l for l in launched if l[0] == "ok"], [("ok", s, 1) for s in range(9, 15)])

    def test_rerun_limit_in_the_plan(self):
        too_many = self.plan([self.RERUN_9(n) for n in (2, 3, 4)])
        self.assertTrue([p for p in h.plan_problems(too_many, DECLARATION, AUTH) if "reruns" in p])

    def test_markers_contradictions_and_unreadable_files(self):
        """Review B3: a started marker without its record is a failed attempt; an unreadable file stops for review."""
        write(self.dir, "attempt-20261004-010000.started.json", {"entry": {"seed": 9, "attempt": 1}})
        self.assertEqual(h.entry_status(self.dir, {"seed": 9, "attempt": 1}), "failed")
        attempt_record(self.dir / "finished", 9, 1, "ok", "20261004-010000")
        write(self.dir / "finished", "attempt-20261004-010000.started.json", {"entry": {"seed": 9, "attempt": 1}})
        self.assertEqual(h.entry_status(self.dir / "finished", {"seed": 9, "attempt": 1}), "ok")
        other = self.dir / "other"
        attempt_record(other, 9, 1, "ok", "20261004-010000")
        attempt_record(other, 9, 1, "maybe", "20261004-020000")
        self.assertEqual(h.entry_status(other, {"seed": 9, "attempt": 1}), "failed")
        broken = self.dir / "broken"
        broken.mkdir()
        (broken / "attempt-20261004-010000.json").write_text("{", encoding="utf-8")
        with self.assertRaises(h.Refused):
            h.entry_status(broken, {"seed": 9, "attempt": 1})

    def test_dry_run_preflights_every_pending_entry(self):
        plan_path = self.dir / "plan.json"
        plan_path.write_text(json.dumps(self.plan()), encoding="utf-8")
        calls = []
        with unittest.mock.patch.object(h, "PLAN", plan_path), \
                unittest.mock.patch.object(h, "RECORDS", self.dir / "records"), \
                unittest.mock.patch.object(h, "run_entry",
                                           side_effect=lambda p, item, d, dry_run: calls.append(item["seed"]) or "dry-run"):
            self.assertEqual(h.run_action(True, None), 0)
        self.assertEqual(calls, [9, 10, 11, 12, 13, 14])


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
                         "runs/train/confirm-donor-B-seed9-rerun1/manifest.json"):
            (repo / relative).parent.mkdir(parents=True, exist_ok=True)
            (repo / relative).write_text("x", encoding="utf-8")
        self.assertEqual(self.problems(repo, commit, tree)[0], [])

    def import_children(self):
        """A spy on subprocess.run that records whether the import child started (git calls pass through)."""
        real, children = subprocess.run, []

        def spy(args, **kwargs):
            if h.CLOSURE_CODE in args:
                children.append(kwargs.get("env", {}))
            return real(args, **kwargs)
        return spy, children

    def test_stale_bytecode_is_refused_before_any_import(self):
        """Review B1: a cache compiled from edited source, valid for the restored source, would run instead of it."""
        import py_compile
        repo, commit, tree = self.make_repo()
        reward = repo / "celeste_rl" / "reward.py"
        stamp = reward.stat().st_mtime
        reward.write_text("SCALE = 3.0\n", encoding="utf-8")  # same size as the committed SCALE = 2.0
        os.utime(reward, (stamp, stamp))
        py_compile.compile(str(reward), cfile=str(repo / "celeste_rl" / "__pycache__" /
                                                  f"reward.{sys.implementation.cache_tag}.pyc"), doraise=True)
        reward.write_text("SCALE = 2.0\n", encoding="utf-8")
        os.utime(reward, (stamp, stamp))
        self.assertEqual(subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True,
                                        text=True).stdout, "")  # the source and git look clean
        env = {k: v for k, v in os.environ.items() if k != "PYTHONDONTWRITEBYTECODE"}
        ran = subprocess.run([sys.executable, "-c", "import celeste_rl.reward as r; print(r.SCALE)"], cwd=repo,
                             capture_output=True, text=True, env=env).stdout.strip()
        self.assertEqual(ran, "3.0")  # the gap: the cached edit runs, not the checked source
        spy, children = self.import_children()
        with unittest.mock.patch.object(h.subprocess, "run", side_effect=spy):
            problems = self.problems(repo, commit, tree)[0]
        self.assertTrue([p for p in problems if "bytecode" in p])
        self.assertEqual(children, [])  # refused before the import child started

    def test_failed_preconditions_start_no_import_and_imports_write_no_bytecode(self):
        repo, commit, tree = self.make_repo()
        (repo / "notes.txt").write_text("x", encoding="utf-8")
        spy, children = self.import_children()
        with unittest.mock.patch.object(h.subprocess, "run", side_effect=spy):
            self.assertTrue(self.problems(repo, commit, tree)[0])
            self.assertEqual(children, [])
            (repo / "notes.txt").unlink()
            self.assertEqual(self.problems(repo, commit, tree)[0], [])
        self.assertEqual(children[0].get("PYTHONDONTWRITEBYTECODE"), "1")
        self.assertFalse(list(repo.rglob("*.pyc")))

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
        values = {"identity_problems": [], "guard_problems": [], "source_problems": ([], {}),
                  "init_clone_problems": ([], {}), "authorization": AUTH, "committed": [], "plan_problems": [],
                  "eligibility_problems": [], "steam_running": True}
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

    def test_guard_refusal_stops_everything_else(self):
        """Review B4: after a fresh-set refusal no other input is read and no child starts."""
        for override in ({"guard_problems": ["fresh-set guard: x"]}, {"identity_problems": ["fresh-set guard: y"]}):
            with self.subTest(override=override):
                spies = {name: unittest.mock.MagicMock(return_value=([], {}))
                         for name in ("source_problems", "init_clone_problems")}
                spies["authorization"] = unittest.mock.MagicMock(return_value=AUTH)
                probe = unittest.mock.MagicMock()
                patches = self.patches(**override) + [unittest.mock.patch.object(h, n, s) for n, s in spies.items()]
                patches.append(unittest.mock.patch.object(rc, "thread_probe", probe))
                self.assertEqual(self.launch(patches, dry_run=False), "refused")
                for spy in [*spies.values(), probe]:
                    spy.assert_not_called()
        self.assertFalse(list((self.dir / "records").glob("attempt-*")))

    def test_all_clear_dry_run(self):
        self.assertEqual(self.launch(self.patches()), "dry-run")
        record = json.loads(next((self.dir / "records").glob("dry-run-*.json")).read_text(encoding="utf-8"))
        self.assertEqual(record["environment_set"], {"PYTHONDONTWRITEBYTECODE": "1"})

    def test_each_condition_refuses(self):
        for name, override in (("source", {"source_problems": (["not 8423081"], {})}),
                               ("init clone", {"init_clone_problems": (["differs"], {})}),
                               ("fresh set", {"guard_problems": ["fresh-set guard: x"]}),
                               ("game file", {"identity_problems": ["fresh-set guard: settings"]}),
                               ("plan", {"plan_problems": ["plan entry differs"]}),
                               ("eligibility", {"eligibility_problems": ["already has an ok attempt"]}),
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
        self.assertFalse(list((self.dir / "records").glob("attempt-*")))

    def test_failed_training_is_recorded_with_streamed_logs(self):
        real_run = subprocess.run

        def fake_run(args, **kwargs):
            if "scripts/train_room1.py" in args:
                self.assertEqual(Path(kwargs["cwd"]), self.dir / "worktree")
                self.assertFalse([k for k in kwargs["env"] if k.upper().startswith(rc.THREAD_PREFIXES)])
                self.assertEqual(kwargs["env"]["PYTHONDONTWRITEBYTECODE"], "1")
                kwargs["stdout"].write("out")
                kwargs["stderr"].write("err")
                return subprocess.CompletedProcess(args, 3)
            return real_run(args, **kwargs)  # git state queries

        with unittest.mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "4"}):
            status = self.launch(self.patches() + [unittest.mock.patch.object(h.subprocess, "run", side_effect=fake_run)],
                                 dry_run=False)
        self.assertEqual(status, "failed")
        record = final_record(self.dir / "records")
        self.assertEqual((record["outcome"], record["exit_code"]), ("failed", 3))
        self.assertEqual(record["thread_variables_removed"], {"OMP_NUM_THREADS": "4"})
        self.assertEqual(Path(record["log_files"]["stderr.txt"]).read_text(encoding="utf-8"), "err")
        self.assertEqual(h.entry_status(self.dir / "records", {"seed": 9, "attempt": 1}), "failed")


def final_record(records: Path) -> dict:
    finals = [p for p in records.glob("attempt-*.json") if not p.name.endswith(h.STARTED)]
    assert len(finals) == 1, finals
    return json.loads(finals[0].read_text(encoding="utf-8"))


class FinalizationTest(Temp):
    """Review B3: every way out of an attempt leaves its record and logs (trainer and game mocked)."""

    def setUp(self):
        super().setUp()
        self.records, self.worktree = self.dir / "records", self.dir / "worktree"
        self.item = h.entry(DECLARATION, 2, 9, 1)
        plan = self.dir / "plan.json"
        plan.write_text("{}", encoding="utf-8")
        self.cleanup = unittest.mock.MagicMock(return_value={"stopped": [1]})
        self.base = [unittest.mock.patch.object(h, "PLAN", plan),
                     unittest.mock.patch.object(h, "RECORDS", self.records),
                     unittest.mock.patch.object(h, "WORKTREE", self.worktree),
                     unittest.mock.patch.object(h, "launch_problems", return_value=([], {})),
                     unittest.mock.patch.object(h, "worktree_problems", return_value=([], {})),
                     unittest.mock.patch.object(rc, "thread_probe", return_value={}),
                     unittest.mock.patch.object(rc, "cleanup_own_game", self.cleanup)]

    def attempt(self, child=None, **patches):
        real = subprocess.run

        def trainer(args, **kwargs):
            if "scripts/train_room1.py" not in args:
                return real(args, **kwargs)
            kwargs["stdout"].write("training output\n")
            kwargs["stdout"].flush()
            (self.worktree / self.item["run_dir"]).mkdir(parents=True)
            (self.worktree / self.item["run_dir"] / "manifest.json").write_text("{}", encoding="utf-8")
            if child:
                raise child
            return subprocess.CompletedProcess(args, 0)

        active = self.base + [unittest.mock.patch.object(h.subprocess, "run", side_effect=trainer)]
        active += [unittest.mock.patch.object(h, name, **spec) for name, spec in patches.items()]
        for p in active:
            p.start()
        try:
            return h.run_entry({"revision": 0}, self.item, DECLARATION, dry_run=False)
        finally:
            for p in reversed(active):
                p.stop()

    def logged(self, record):
        return Path(record["log_files"]["stdout.txt"]).read_text(encoding="utf-8")

    def reset(self):
        shutil.rmtree(self.records)
        shutil.rmtree(self.worktree)

    def test_postprocessing_errors_are_recorded(self):
        for name, patches in (("hashing", {"file_hashes": {"side_effect": OSError("disk")}}),
                              ("malformed metadata", {"outcome_problems": {"side_effect": KeyError("sessions")}})):
            with self.subTest(case=name):
                self.assertEqual(self.attempt(**patches), "failed")
                record = final_record(self.records)
                self.assertTrue(record["problems"])
                self.assertEqual(self.logged(record), "training output\n")
                self.assertEqual(h.entry_status(self.records, self.item), "failed")
                self.reset()

    def test_interrupted_copy_is_recorded_then_reraised(self):
        with self.assertRaises(KeyboardInterrupt):
            self.attempt(outcome_problems={"return_value": ([], {})}, copy_run={"side_effect": KeyboardInterrupt()})
        record = final_record(self.records)
        self.assertEqual(record["outcome"], "failed")
        self.assertEqual(record["copy_target"], "runs/train/confirm-donor-B-seed9")
        self.cleanup.assert_not_called()  # the child had already finished

    def test_timeout_and_interrupted_child_keep_partial_logs(self):
        self.assertEqual(self.attempt(child=subprocess.TimeoutExpired("train", 5400)), "failed")
        record = final_record(self.records)
        self.assertTrue(record["timed_out"])
        self.assertEqual(self.logged(record), "training output\n")
        self.reset()
        with self.assertRaises(KeyboardInterrupt):
            self.attempt(child=KeyboardInterrupt())
        record = final_record(self.records)
        self.assertTrue(record["interrupted_child"])
        self.assertEqual(self.logged(record), "training output\n")
        self.assertEqual(self.cleanup.call_count, 2)

    def test_ok_attempt(self):
        self.assertEqual(self.attempt(outcome_problems={"return_value": ([], {"faults": []})},
                                      copy_run={"return_value": {"verified": True}}), "ok")
        record = final_record(self.records)
        self.assertEqual((record["outcome"], record["problems"], record["copy"]), ("ok", [], {"verified": True}))
        self.assertEqual(h.entry_status(self.records, self.item), "ok")


class ReadingRuleTest(Temp):
    """Review B4: inputs the executor (or the old trainer) reads are identity-checked first; v1 stands in."""

    def setUp(self):
        super().setUp()
        self.spec = {"room1": FreshSet(V1, fresh_sets.git_blob(REPO / V1), None, None)}
        self.alias = self.dir / "alias.json"
        os.link(REPO / V1, self.alias)  # a hard link: identity, no content read

    def no_read_text(self):
        return unittest.mock.patch.object(Path, "read_text", side_effect=AssertionError("protected content read"))

    def test_plan_or_declaration_alias_is_refused_before_reading(self):
        with unittest.mock.patch.object(h, "FRESH", self.spec), unittest.mock.patch.object(h, "PLAN", self.alias):
            with self.no_read_text():
                self.assertEqual(h.run_action(False, None), 1)
                self.assertEqual(h.run_action(True, None), 1)
                self.assertEqual(h.plan_action(True, None, None, None), 1)
        with unittest.mock.patch.object(h, "FRESH", self.spec), unittest.mock.patch.object(h, "DECLARATION", self.alias):
            with self.no_read_text():
                self.assertEqual(h.plan_action(False, None, None, None), 1)

    def test_record_aliases_are_refused(self):
        records = self.dir / "records"
        records.mkdir()
        os.link(REPO / V1, records / "compare-diagnostic-20261004-000000.json")
        with unittest.mock.patch.object(h, "FRESH", self.spec), self.no_read_text():
            with self.assertRaises(h.Refused):
                h.authorization(records)
        attempts = self.dir / "attempts"
        attempts.mkdir()
        os.link(REPO / V1, attempts / "attempt-20261004-000000.json")
        with unittest.mock.patch.object(h, "FRESH", self.spec), self.no_read_text():
            with self.assertRaises(h.Refused):
                h.attempt_statuses(attempts)

    def test_game_settings_alias_stops_the_launch(self):
        game = self.dir / "game"
        (game / "probe-profile" / "Saves").mkdir(parents=True)
        os.link(REPO / V1, game / "probe-profile" / "Saves" / "modsettings-Everest.celeste")
        command = h.entry(DECLARATION, 2, 9, 1)["command"]
        self.assertEqual(h.guard_problems(command, self.dir, fresh=self.spec), [])  # the command guard alone misses it
        spies = {name: unittest.mock.MagicMock(return_value=([], {})) for name in ("source_problems", "init_clone_problems")}
        probe = unittest.mock.MagicMock()
        (self.dir / "plan.json").write_text("{}", encoding="utf-8")
        with unittest.mock.patch.object(h, "FRESH", self.spec), unittest.mock.patch.object(h, "GAME", game), \
                unittest.mock.patch.object(h, "WORKTREE", self.dir / "worktree"), \
                unittest.mock.patch.object(h, "RECORDS", self.dir / "records"), \
                unittest.mock.patch.object(h, "PLAN", self.dir / "plan.json"), \
                unittest.mock.patch.object(h, "source_problems", spies["source_problems"]), \
                unittest.mock.patch.object(h, "init_clone_problems", spies["init_clone_problems"]), \
                unittest.mock.patch.object(rc, "thread_probe", probe):
            plan, item = h.initial_plan(DECLARATION, AUTH), h.entry(DECLARATION, 2, 9, 1)
            problems, _ = h.launch_problems(plan, item, DECLARATION)
            self.assertTrue([p for p in problems if "modsettings-Everest.celeste" in p])
            self.assertEqual(h.run_entry(plan, item, DECLARATION, False), "refused")
        for spy in [*spies.values(), probe]:
            spy.assert_not_called()
        self.assertFalse(list((self.dir / "records").glob("attempt-*")))


if __name__ == "__main__":
    unittest.main()
