"""The retention confirmation's historical donor executor (config/retention-confirmation.json,
donors.historical_training; amendment 2).

The owner chose to train the new donors (seeds 9 to 14, replacements 15 and 16) with the code that trained seeds 7 and
8, commit 8423081, because the current code no longer reproduces that donor recipe (donors.recipe_check.result). This
script is the committed envelope around that old code: it runs the declared donor command inside a detached worktree
at 8423081, alone and one donor at a time, and records everything the result rests on.

Run from the repo root with the RL interpreter:
    .venv-rl/Scripts/python.exe scripts/confirmation_historical_donors.py plan
    .venv-rl/Scripts/python.exe scripts/confirmation_historical_donors.py plan --check
    .venv-rl/Scripts/python.exe scripts/confirmation_historical_donors.py plan --rerun SEED
    .venv-rl/Scripts/python.exe scripts/confirmation_historical_donors.py plan --replace SEED --reason TEXT
    .venv-rl/Scripts/python.exe scripts/confirmation_historical_donors.py prepare
    .venv-rl/Scripts/python.exe scripts/confirmation_historical_donors.py check
    .venv-rl/Scripts/python.exe scripts/confirmation_historical_donors.py run [--dry-run] [--max N]

plan: writes config/confirmation-donors-historical.json (refuses to overwrite) once the authorization exists: the
declaration carries amendment 2, and the recipe check's comparator version 2, at a reviewed commit on a clean tree,
has recorded the current replay as DIFFERENT and the 8423081 diagnostic as IDENTICAL (no record of the current code
as IDENTICAL is needed or produced). The plan pins the declaration's text hash, both records, the source commit, the
worktree, the game copy, the interpreter, the init clone, the limit and one entry per donor (j, seed, attempt, run
folder, command). --check recomputes everything and reports any difference. --rerun and --replace write a revision
that keeps every earlier entry unchanged: a rerun only after a failed attempt of that seed (at most two reruns), a
replacement only within the two-seed budget, in order 15 then 16, taking the replaced donor's j. Commit and push the
plan (and each revision) before running it.

prepare: creates the worktree (git worktree add --detach) if absent and copies the init clone folder into it,
verifying every file's sha256; records both. It never changes an existing worktree's files.

check: the declared source check (code_checks.historical_donors): HEAD and tree are 8423081's; no tracked or
untracked change and no index flag that hides one; ignored files only under the declared input and run folders or
Python caches; every module of scripts/train_room1.py's import closure, imported there with this interpreter, lies
inside the worktree and equals its blob at 8423081.

run: trains the plan's entries in order, one at a time, stopping at the first attempt that is not ok. Before each
launch it refuses unless: this repository's tracked tree is clean (so the executor and plan are committed); the plan
passes --check; the source check passes; the init clone copy matches its pin; the fresh-set guard allows no use on
the command as given and as resolved in the worktree, nor on the copied inputs; the run folder exists neither in the
worktree nor here; Steam runs; no Celeste process runs in any folder and no other script of this repository runs;
this is the declared interpreter. It removes every thread variable, probes torch's default threads, and runs the
command with the worktree as working directory for at most 90 minutes (a timeout stops only the default copy's
game). An attempt is ok only if the run is complete and finished at the historical commit with exactly one
attributable session, no runtime problems, the original seed 7 runtime record, seed 7's config apart from the seed and
truthful evaluation references; it is then copied to the same relative folder here and every file's hash is checked.
Faults are recorded by name. A failed attempt stays in the worktree with its record; a rerun needs a plan revision.
Every record (and full stdout and stderr) is created exclusively in runs/confirmation/donors/, never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from celeste_rl import fresh_sets, runtime  # noqa: E402
from scripts import confirmation_recipe_check as rc  # noqa: E402

DECLARATION = rc.DECLARATION
HISTORICAL_COMMIT = rc.HISTORICAL_COMMIT
HISTORICAL_TREE = "8ee839019d8ef4c68890a327dcfa970e903a45a1"
WORKTREE = Path("C:/Projects/celeste-research-scratch/historical-donors-8423081")
GAME = Path("C:/Projects/celeste-research-scratch/game-probe")
INTERPRETER = Path(".venv-rl/Scripts/python.exe")  # relative to this repository
INIT_CLONE = Path("runs/clone/20260918-183508")
INIT_CLONE_SHA256 = "541e58f6492196183475cd0a48ed7315c5be2754825640f08a21a5f7b2ae76d5"
PLAN = REPO / "config" / "confirmation-donors-historical.json"
RECORDS = REPO / "runs" / "confirmation" / "donors"
LIMIT_MINUTES = 90
RUN_PREFIX = "runs/train/confirm-donor-B-seed"
FORBIDDEN_OPTIONS = ("--resume", "--allow-dirty", "--allow-runtime-mismatch")
MAX_RERUNS = 2
MAX_REPLACEMENTS = 2
# The recipe check's comparator commits whose records may authorize historical donors (reviewed).
REVIEWED_COMPARATORS = ("7987f04786622ab0ded60f2914b18243cb15d236",)
RECIPE_CHECK_RUN = "run-20261001-194858.json"  # the live replay's launch record
DIAGNOSTIC_SHA256 = "36f5460cf45b474c072ffd78372acb0267a95732c53d0c804ae6cb13a8710fd3"
COMPARE_NAME = re.compile(r"compare-\d{8}-\d{6}(-\d+)?\.json")
DIAGNOSTIC_COMPARE_NAME = re.compile(r"compare-diagnostic-\d{8}-\d{6}(-\d+)?\.json")


class Refused(Exception):
    """A precondition does not hold, so nothing is written or launched."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if done.returncode != 0:
        raise Refused(f"git {' '.join(args)} in {root} exited {done.returncode}: {(done.stderr or '').strip()[:300]}")
    return done.stdout


# -------------------------------------------------------------------------------------------------- the declaration

def load_declaration() -> dict:
    return json.loads(DECLARATION.read_text(encoding="utf-8"))


def amendment_problems(declaration: dict) -> list[str]:
    """The declaration must carry amendment 2: the owner's choice and the historical training section."""
    problems = []
    if "historical_donors" not in declaration.get("owner_decisions", {}):
        problems.append("the declaration has no owner_decisions.historical_donors (amendment 2)")
    training = declaration.get("donors", {}).get("historical_training")
    if not isinstance(training, dict) or HISTORICAL_COMMIT not in training.get("source", ""):
        problems.append(f"the declaration has no donors.historical_training naming {HISTORICAL_COMMIT}")
    if "historical_donors" not in declaration.get("code_checks", {}):
        problems.append("the declaration has no code_checks.historical_donors")
    return problems


def run_folder(seed: int, attempt: int) -> str:
    """The declared folder: attempt 1 uses the command's own folder, a rerun n adds -rerun<n>."""
    return f"{RUN_PREFIX}{seed}" + (f"-rerun{attempt - 1}" if attempt > 1 else "")


def donor_command(declaration: dict, seed: int, folder: str) -> list[str]:
    """The declared donor command for one seed and folder, with the default game copy (no interpreter)."""
    command = declaration["donors"]["command"].replace("<s>", str(seed)).split()
    command[command.index("--run-dir") + 1] = folder
    command += ["--game-dir", GAME.as_posix()]
    forbidden = [token for token in command if token.split("=", 1)[0] in FORBIDDEN_OPTIONS]
    if forbidden:
        raise Refused(f"the donor command carries {forbidden}")
    return command


def donor_index(declaration: dict) -> dict[int, int]:
    """j for each seed that is a donor from the start: 7, 8, then the new seeds (pipeline.index)."""
    order = [int(s) for s in declaration["donors"]["existing"] if s.isdigit()] + list(declaration["donors"]["new_seeds"])
    return {seed: j for j, seed in enumerate(order)}


# ---------------------------------------------------------------------------------------------------- authorization

def _comparator_records(records: Path, pattern: re.Pattern) -> list[tuple[Path, dict]]:
    """Every record of that kind written by a reviewed comparator version 2 on a clean tree."""
    found = []
    for path in sorted(records.glob("compare-*.json")):
        if not pattern.fullmatch(path.name):
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue  # an unreadable record authorizes nothing
        comparator = record.get("comparator") if isinstance(record, dict) else None
        git = comparator.get("git") if isinstance(comparator, dict) else None
        if (isinstance(git, dict) and comparator.get("version") == rc.COMPARATOR_VERSION
                and git.get("commit") in REVIEWED_COMPARATORS and git.get("uncommitted_changes") is False
                and not git.get("git_error")):
            found.append((path, record))
    return found


def authorization(records: Path = rc.RECORDS) -> dict:
    """The two comparator version 2 records that permit historical donors; Refused unless every such record agrees:
    the current replay DIFFERENT (bound to the live replay) and the diagnostic IDENTICAL (bound to its record)."""
    replay = _comparator_records(records, COMPARE_NAME)
    diagnostic = _comparator_records(records, DIAGNOSTIC_COMPARE_NAME)
    if not replay or not diagnostic:
        raise Refused("no compare and compare-diagnostic records from a reviewed comparator version 2 on a clean tree")
    for path, record in replay:
        bound = Path(str((record.get("run_record") or {}).get("path", ""))).name
        if record.get("verdict") != "DIFFERENT" or bound != RECIPE_CHECK_RUN:
            raise Refused(f"{path.name}: verdict {record.get('verdict')} bound to {bound!r}; expected DIFFERENT for "
                          f"{RECIPE_CHECK_RUN}")
    for path, record in diagnostic:
        if record.get("verdict") != "IDENTICAL" or (record.get("run_record") or {}).get("sha256") != DIAGNOSTIC_SHA256:
            raise Refused(f"{path.name}: verdict {record.get('verdict')}; expected IDENTICAL bound to the diagnostic")
    newest = {"compare": replay[-1][0], "compare_diagnostic": diagnostic[-1][0]}
    return {key: {"path": _relative(path), "sha256": _sha256(path),
                  "verdict": "DIFFERENT" if key == "compare" else "IDENTICAL",
                  "comparator_commit": json.loads(path.read_text(encoding="utf-8"))["comparator"]["git"]["commit"]}
            for key, path in newest.items()}


def _relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


# --------------------------------------------------------------------------------------------------------- the plan

def fixed_fields(declaration: dict, auth: dict) -> dict:
    """Everything a plan pins apart from its entries and revision."""
    return {"name": "confirmation-donors-historical",
            "declaration": {"path": "config/retention-confirmation.json",
                            "text_sha256": fresh_sets.text_sha256(DECLARATION)},
            "rules": "donors.historical_training (amendment 2)",
            "source": {"commit": HISTORICAL_COMMIT, "tree": HISTORICAL_TREE},
            "worktree": WORKTREE.as_posix(), "game_dir": GAME.as_posix(), "interpreter": INTERPRETER.as_posix(),
            "init_clone": {"path": INIT_CLONE.as_posix(), "cloned_sha256": INIT_CLONE_SHA256},
            "limit_minutes": LIMIT_MINUTES, "authorization": auth}


def entry(declaration: dict, j: int, seed: int, attempt: int, reason: str | None = None) -> dict:
    folder = run_folder(seed, attempt)
    item = {"j": j, "seed": seed, "attempt": attempt, "run_dir": folder,
            "command": donor_command(declaration, seed, folder)}
    return {**item, "reason": reason} if reason else item


def initial_plan(declaration: dict, auth: dict) -> dict:
    index = donor_index(declaration)
    entries = [entry(declaration, index[seed], seed, 1) for seed in declaration["donors"]["new_seeds"]]
    return {**fixed_fields(declaration, auth), "revision": 0, "entries": entries}


def plan_problems(plan: dict, declaration: dict, auth: dict) -> list[str]:
    """Differences between a plan and what the declaration and records require now."""
    if not isinstance(plan, dict) or not isinstance(plan.get("entries"), list):
        return ["the plan is not a record with entries"]
    problems = [f"plan {key}: {plan.get(key)!r}, expected {value!r}"
                for key, value in fixed_fields(declaration, auth).items() if plan.get(key) != value]
    allowed = set(declaration["donors"]["new_seeds"]) | set(declaration["donors"]["replacement_seeds"])
    seen = set()
    for item in plan["entries"]:
        try:
            expected = entry(declaration, item["j"], item["seed"], item["attempt"], item.get("reason"))
        except (KeyError, TypeError, ValueError, Refused) as error:
            problems.append(f"plan entry {item!r} is malformed: {error}")
            continue
        key = (item["seed"], item["attempt"])
        if item != expected:
            problems.append(f"plan entry {key} is not the declared entry {expected}")
        if item["seed"] not in allowed:
            problems.append(f"plan entry {key}: seed {item['seed']} is not a new or replacement seed")
        if key in seen or (item["attempt"] > 1 and (item["seed"], item["attempt"] - 1) not in seen):
            problems.append(f"plan entry {key} repeats or skips an attempt")
        if item["attempt"] > MAX_RERUNS + 1:
            problems.append(f"plan entry {key} exceeds {MAX_RERUNS} reruns")
        seen.add(key)
    problems += index_problems(plan["entries"], declaration)
    initial = initial_plan(declaration, auth)["entries"]
    if plan["entries"][:len(initial)] != initial:
        problems.append("the plan's first entries are not the declared new donors in order")
    return problems


def index_problems(entries: list, declaration: dict) -> list[str]:
    """Every attempt of a seed keeps its j; a replacement takes the j of the donor its reason names."""
    known, problems = dict(donor_index(declaration)), []
    replacements = set(declaration["donors"]["replacement_seeds"])
    for item in entries:
        if not isinstance(item, dict):
            continue
        seed, j = item.get("seed"), item.get("j")
        if seed in replacements and item.get("attempt") == 1:
            named = re.match(r"replaces seed (\d+):", str(item.get("reason", "")))
            if not named or known.get(int(named.group(1))) != j:
                problems.append(f"replacement seed {seed} (j {j}) does not take the j of the donor it names")
            known[seed] = j
        elif known.get(seed) != j:
            problems.append(f"seed {seed} attempt {item.get('attempt')} has j {j}, not {known.get(seed)}")
    return problems


def revised_plan(plan: dict, declaration: dict, records: Path, rerun: int | None = None,
                 replace: int | None = None, reason: str | None = None) -> dict:
    """A revision with one more entry; every earlier entry is kept exactly."""
    entries = list(plan["entries"])
    if rerun is not None:
        attempts = [item for item in entries if item["seed"] == rerun]
        if not attempts:
            raise Refused(f"seed {rerun} is not in the plan")
        last = attempts[-1]
        if entry_status(records, last) != "failed":
            raise Refused(f"seed {rerun} attempt {last['attempt']} has no failed record; a rerun follows a failure")
        if last["attempt"] > MAX_RERUNS:
            raise Refused(f"seed {rerun} has had {MAX_RERUNS} reruns; stop for review")
        entries.append(entry(declaration, last["j"], rerun, last["attempt"] + 1, reason or "rerun after a failed attempt"))
    elif replace is not None:
        if not reason:
            raise Refused("a replacement needs --reason (the inclusion rule's outcome that requires it)")
        index = {**donor_index(declaration), **{item["seed"]: item["j"] for item in entries}}
        if replace not in index:
            raise Refused(f"seed {replace} is not a donor")
        used = [item["seed"] for item in entries if item["seed"] in declaration["donors"]["replacement_seeds"]]
        spare = [s for s in declaration["donors"]["replacement_seeds"] if s not in used]
        if len(set(used)) >= MAX_REPLACEMENTS or not spare:
            raise Refused("the two replacements are used up; stop for review (donors.inclusion.end_state)")
        entries.append(entry(declaration, index[replace], spare[0], 1, f"replaces seed {replace}: {reason}"))
    else:
        raise Refused("a revision needs --rerun or --replace")
    return {**plan, "revision": plan["revision"] + 1, "previous_text_sha256": fresh_sets.text_sha256(PLAN),
            "entries": entries}


def committed(path: Path) -> list[str]:
    """Problems unless the file is tracked and unchanged at HEAD in this repository."""
    relative = path.resolve().relative_to(REPO.resolve()).as_posix()
    if subprocess.run(["git", "-C", str(REPO), "ls-files", "--error-unmatch", relative], capture_output=True).returncode:
        return [f"{relative} is not committed"]
    status = _git(REPO, "status", "--porcelain", "--", relative).strip()
    return [f"{relative} has uncommitted changes"] if status else []


def plan_action(check: bool, rerun: int | None, replace: int | None, reason: str | None) -> int:
    declaration = load_declaration()
    try:
        problems = amendment_problems(declaration)
        if problems:
            raise Refused("; ".join(problems))
        auth = authorization()
        if check:
            problems = plan_problems(json.loads(PLAN.read_text(encoding="utf-8")), declaration, auth)
            print("\n".join(problems) if problems else f"PASS: {_relative(PLAN)} is the declared plan")
            return 1 if problems else 0
        if rerun is None and replace is None:
            if PLAN.exists():
                raise Refused(f"{_relative(PLAN)} exists; use --rerun or --replace for a revision")
            plan = initial_plan(declaration, auth)
        else:
            problems = committed(PLAN)
            current = json.loads(PLAN.read_text(encoding="utf-8"))
            problems += plan_problems(current, declaration, auth)
            if problems:
                raise Refused("; ".join(problems))
            plan = revised_plan(current, declaration, RECORDS, rerun, replace, reason)
    except Refused as error:
        print(f"Refused: {error}")
        return 1
    PLAN.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {_relative(PLAN)} revision {plan['revision']} with {len(plan['entries'])} entries; commit and push it "
          "before running it.")
    return 0


# ------------------------------------------------------------------------------------------------ the worktree source

def _ignored_allowed(path: str) -> bool:
    return (path.startswith(INIT_CLONE.as_posix() + "/") or path.startswith(RUN_PREFIX)
            or "/__pycache__/" in f"/{path}" or path.endswith((".pyc", ".pyo")))


CLOSURE_CODE = ("import os, sys; sys.path.insert(0, 'scripts'); import train_room1; "
                "print('\\n'.join(sorted({os.path.abspath(m.__file__) for n, m in list(sys.modules.items()) "
                "if (n == 'celeste_rl' or n.startswith('celeste_rl.')) and getattr(m, '__file__', None)} "
                "| {os.path.abspath(train_room1.__file__)})))")


def source_problems(worktree: Path | None = None, commit: str = HISTORICAL_COMMIT, tree: str = HISTORICAL_TREE,
                    interpreter: Path | None = None) -> tuple[list[str], dict]:
    """The declared source check (code_checks.historical_donors), with the evidence it rests on."""
    worktree = worktree or WORKTREE
    if not (worktree / ".git").exists():
        return [f"{worktree} is not a git worktree"], {}
    evidence: dict = {"worktree": worktree.as_posix()}
    try:
        evidence["head"] = _git(worktree, "rev-parse", "HEAD").strip()
        evidence["tree"] = _git(worktree, "rev-parse", "HEAD^{tree}").strip()
        changes = _git(worktree, "status", "--porcelain=v1", "--untracked-files=all").splitlines()
        listed = _git(worktree, "status", "--porcelain=v1", "--ignored=traditional", "--untracked-files=all").splitlines()
        flags = [line for line in _git(worktree, "ls-files", "-v").splitlines() if not line.startswith("H ")]
    except Refused as error:
        return [str(error)], evidence
    problems = []
    if evidence["head"] != commit or evidence["tree"] != tree:
        problems.append(f"the worktree is at {evidence['head']} (tree {evidence['tree']}), not {commit} ({tree})")
    if changes:
        problems.append(f"the worktree has changes: {changes[:10]}")
    if flags:
        problems.append(f"index flags that can hide changes: {flags[:10]}")
    ignored = [line[3:].strip('"') for line in listed if line.startswith("!! ")]
    stray = [path for path in ignored if not _ignored_allowed(path)]
    if stray:
        problems.append(f"undeclared ignored files in the worktree: {stray[:10]}")
    evidence["ignored_files"] = len(ignored)
    done = subprocess.run([str(interpreter or REPO / INTERPRETER), "-c", CLOSURE_CODE], cwd=worktree,
                          capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    if done.returncode != 0:
        return problems + [f"importing scripts/train_room1.py in the worktree failed: {done.stderr.strip()[-300:]}"], evidence
    closure, tracked = {}, set(_git(worktree, "ls-files").splitlines())
    for line in done.stdout.splitlines():
        file = Path(line)
        try:
            relative = file.resolve().relative_to(worktree.resolve()).as_posix()
        except ValueError:
            problems.append(f"{file} was imported from outside the worktree")
            continue
        blob = _git(worktree, "hash-object", f"--path={relative}", str(file)).strip()
        try:
            expected = _git(worktree, "rev-parse", f"{commit}:{relative}").strip() if relative in tracked else None
        except Refused:  # the file is not in that commit (for example the worktree is at another commit)
            expected = None
        if blob != expected:
            problems.append(f"{relative} is not its blob at {commit[:7]} ({blob} vs {expected})")
        closure[relative] = blob
    if "scripts/train_room1.py" not in closure or not any(p.startswith("celeste_rl/") for p in closure):
        problems.append(f"the import closure is incomplete: {sorted(closure)}")
    evidence["closure"] = closure
    return problems, evidence


def check_action() -> int:
    problems, evidence = source_problems()
    print(json.dumps({k: v for k, v in evidence.items() if k != "closure"}, indent=2))
    print(f"closure: {len(evidence.get('closure', {}))} files")
    print("\n".join(f"FAIL: {p}" for p in problems) if problems else
          f"PASS: the worktree is {HISTORICAL_COMMIT[:7]} with only declared extra files")
    return 1 if problems else 0


# ------------------------------------------------------------------------------------------------------- prepare

def file_hashes(folder: Path) -> dict[str, str]:
    """sha256 of every file under a folder, by relative path."""
    return {path.relative_to(folder).as_posix(): _sha256(path) for path in sorted(folder.rglob("*")) if path.is_file()}


def init_clone_problems(worktree: Path | None = None) -> tuple[list[str], dict]:
    """The worktree's copy of the init clone must equal this repository's folder, and cloned.zip its pin."""
    worktree = worktree or WORKTREE
    source, copy = REPO / INIT_CLONE, worktree / INIT_CLONE
    if not copy.is_dir():
        return [f"{copy} does not exist"], {}
    hashes, expected = file_hashes(copy), file_hashes(source)
    problems = [] if hashes == expected else [f"{copy} differs from {source}"]
    if hashes.get("cloned.zip") != INIT_CLONE_SHA256:
        problems.append(f"{copy}/cloned.zip is not the pinned {INIT_CLONE_SHA256}")
    return problems, hashes


def prepare_action() -> int:
    record: dict = {"action": "prepare", "worktree": WORKTREE.as_posix(), "commit": HISTORICAL_COMMIT,
                    "executor": executor_identity()}
    try:
        guard = fresh_sets.paths_problems([str(REPO / INIT_CLONE)])
        if guard:
            raise Refused("; ".join(guard))
        if not WORKTREE.exists():
            _git(REPO, "worktree", "add", "--detach", str(WORKTREE), HISTORICAL_COMMIT)
            record["created_worktree"] = True
        if not (WORKTREE / INIT_CLONE).exists():
            shutil.copytree(REPO / INIT_CLONE, WORKTREE / INIT_CLONE)
            record["copied_init_clone"] = True
        guard = fresh_sets.paths_problems([str(WORKTREE / INIT_CLONE)])  # the copy, before it is hashed
        if guard:
            raise Refused("; ".join(guard))
        problems, hashes = init_clone_problems()
        record["init_clone_files"] = hashes
        problems += source_problems()[0]
        record["problems"] = problems
    except (Refused, OSError, subprocess.SubprocessError) as error:
        record["problems"] = [f"{type(error).__name__}: {error}"]
    print("\n".join(record["problems"]) or "Prepared: the worktree passes the source check")
    print(f"Record: {write_record(record, 'prepare')}")
    return 1 if record["problems"] else 0


# ---------------------------------------------------------------------------------------------------------- records

def write_record(record: dict, stem: str, extra: dict[str, str] | None = None, directory: Path | None = None) -> Path:
    """Create the record (and any text files beside it) exclusively; never overwrite."""
    directory = directory or RECORDS
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for attempt in range(1, 100):
        base = f"{stem}-{stamp}" + (f"-{attempt}" if attempt > 1 else "")
        try:
            handle = open(directory / f"{base}.json", "x", encoding="utf-8")
        except FileExistsError:
            continue
        sides = {suffix: (directory / f"{base}.{suffix}").as_posix() for suffix in (extra or {})}
        with handle:
            handle.write(json.dumps({**record, "log_files": sides} if sides else record, indent=2, default=str) + "\n")
        for suffix, text in (extra or {}).items():
            with open(directory / f"{base}.{suffix}", "x", encoding="utf-8") as side:
                side.write(text)
        return directory / f"{base}.json"
    raise FileExistsError(f"no free record name for {stem}-{stamp}")


def entry_status(records: Path, item: dict) -> str | None:
    """'ok' or 'failed' from this entry's attempt records (refusals and dry runs do not count); None if never run."""
    outcomes = set()
    for path in records.glob("attempt-*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            outcomes.add("failed")  # an unreadable attempt record is never taken as success
            continue
        planned = record.get("entry") if isinstance(record, dict) else None
        if isinstance(planned, dict) and (planned.get("seed"), planned.get("attempt")) == (item["seed"], item["attempt"]):
            outcomes.add(record.get("outcome") if record.get("outcome") == "ok" else "failed")
    if "failed" in outcomes:
        return "failed"
    return "ok" if outcomes else None


def executor_identity() -> dict:
    return {"script_sha256": _sha256(Path(__file__)), "git": runtime.git_state()}


# ------------------------------------------------------------------------------------------------------------- run

def steam_running() -> bool:
    output = rc._query(["tasklist", "/FI", "IMAGENAME eq steam.exe", "/FO", "CSV", "/NH"], "Steam process query")
    return any(line.strip().lower().startswith('"steam.exe"') for line in output.splitlines())


def resolved(command: list[str], root: Path) -> list[str]:
    """The command as the child sees it: each relative path that exists under its working directory made absolute."""
    return [str(root / token) if not token.startswith("-") and not Path(token).is_absolute() and (root / token).exists()
            else token for token in command]


def guard_problems(command: list[str], worktree: Path, fresh: dict = fresh_sets.FRESH) -> list[str]:
    """The fresh-set guard with no allowed use: the command as given, as resolved in the worktree, and the copied init
    clone with its dependencies."""
    try:
        fresh_sets.refuse_unless_declared([command, resolved(command, worktree)], (), fresh=fresh)
    except fresh_sets.FreshSetRefused as error:
        return [f"fresh-set guard: {error}"]
    return fresh_sets.paths_problems([str(worktree / INIT_CLONE)], fresh=fresh)


def next_entries(plan: dict, records: Path) -> list[dict]:
    """The entries still to run, in plan order; Refused if a failed attempt has no later attempt planned."""
    pending, done = [], set()
    for item in plan["entries"]:
        status = entry_status(records, item)
        if item["seed"] in done:
            raise Refused(f"seed {item['seed']} already has an ok attempt; attempt {item['attempt']} is not allowed")
        if status == "ok":
            done.add(item["seed"])
        if status == "failed":
            later = [e for e in plan["entries"] if e["seed"] == item["seed"] and e["attempt"] > item["attempt"]]
            replaced = [e for e in plan["entries"] if f"replaces seed {item['seed']}:" in str(e.get("reason", ""))]
            if not later and not replaced:
                raise Refused(f"seed {item['seed']} attempt {item['attempt']} failed; revise the plan (plan --rerun)")
        elif status is None:
            pending.append(item)
    return pending


def launch_problems(plan: dict, item: dict, declaration: dict) -> tuple[list[str], dict]:
    """Why this attempt must not start, and the recorded conditions."""
    problems, conditions = [], {}
    refusal = runtime.refusal(runtime.git_state(), allow_dirty=False)
    if refusal:
        problems.append(refusal)
    problems += committed(PLAN) + amendment_problems(declaration)
    try:
        problems += plan_problems(plan, declaration, authorization())
    except Refused as error:
        problems.append(f"authorization: {error}")
    # The fresh-set guard comes before anything reads the command's inputs (the init clone is hashed only after it).
    guard = guard_problems(item["command"], WORKTREE)
    problems += guard
    conditions["fresh_set_check"] = "no allowed use; command as given and as resolved in the worktree; init clone"
    source, conditions["source_check"] = source_problems()
    problems += source
    if not guard:
        clone, conditions["init_clone_files"] = init_clone_problems()
        problems += clone
    for root in (WORKTREE, REPO):
        if (root / item["run_dir"]).exists():
            problems.append(f"{root / item['run_dir']} already exists")
    if Path(sys.executable).resolve() != (REPO / INTERPRETER).resolve():
        problems.append(f"running under {sys.executable}, not {REPO / INTERPRETER}")
    queries = (("steam_running", steam_running), ("game_processes", rc.all_game_processes),
               ("other_experiments", lambda: rc.other_experiments(os.getpid())))
    for key, query in queries:
        try:
            conditions[key] = query()
        except rc.ProcessEvidenceUnavailable as error:
            conditions[key] = f"unavailable: {error}"
            problems.append(f"cannot establish the launch conditions: {error}")
            continue
        if key == "steam_running" and not conditions[key]:
            problems.append("Steam is not running")
        elif key != "steam_running" and conditions[key]:
            problems.append(f"{key.replace('_', ' ')} found ({conditions[key]}); a historical donor trains alone")
    return problems, conditions


def outcome_problems(run: Path, seed: int, original: Path = REPO / rc.ORIGINAL) -> tuple[list[str], dict]:
    """Why a finished attempt does not count as a donor (donors.historical_training.ok), and its fault report."""
    try:
        manifest = rc.validate_run(run, f"seed {seed}")
        rows = rc._rows_jsonl(run / "evaluations.jsonl")
        reference = json.loads((original / "manifest.json").read_text(encoding="utf-8"))
    except (rc.Invalid, OSError, ValueError, KeyError, IndexError, TypeError) as error:
        return [f"incomplete run: {type(error).__name__}: {error}"], {}
    problems = []
    sessions = manifest["sessions"]
    if len(sessions) != 1:
        problems.append(f"{len(sessions)} sessions; a historical donor is never resumed")
    for session in sessions:
        provenance = session["provenance"]
        if provenance.get("commit") != HISTORICAL_COMMIT or provenance.get("uncommitted_changes") is not False:
            problems.append(f"session at {provenance.get('commit')} (uncommitted {provenance.get('uncommitted_changes')})")
        if provenance.get("runtime_problems") != []:
            problems.append(f"runtime problems: {provenance.get('runtime_problems')}")
        if provenance.get("runtime") != reference["sessions"][0]["provenance"]["runtime"]:
            problems.append("the session's runtime record differs from the original seed 7 run's")
    expected_config = {**reference["config"], "seed": seed}
    if manifest["config"] != expected_config:
        differing = sorted(k for k in set(manifest["config"]) | set(expected_config)
                           if manifest["config"].get(k, object()) != expected_config.get(k, object()))
        problems.append(f"config differs from seed 7's apart from the seed: {differing}")
    problems += rc.evaluation_problems(run, rows, f"seed {seed}")
    return problems, manifest.get("fault_stats", {})


def copy_run(source: Path, target: Path) -> dict:
    """Copy the run folder here and verify every file; raises Refused if anything differs."""
    if target.exists():
        raise Refused(f"{target} already exists")
    shutil.copytree(source, target)
    before, after = file_hashes(source), file_hashes(target)
    if before != after:
        raise Refused(f"the copy {target} differs from {source}")
    return {"path": _relative(target), "files": after, "verified": True}


def run_entry(plan: dict, item: dict, declaration: dict, dry_run: bool) -> str:
    """One attempt (or its dry run) with its record: 'ok', 'failed', 'refused' or 'dry-run'."""
    command = [str(REPO / INTERPRETER), *item["command"]]
    problems, conditions = launch_problems(plan, item, declaration)
    env = rc.stripped_env(dict(os.environ))
    record = {"executor": executor_identity(), "source": {"commit": HISTORICAL_COMMIT, "tree": HISTORICAL_TREE},
              "plan": {"path": _relative(PLAN), "text_sha256": fresh_sets.text_sha256(PLAN),
                       "revision": plan.get("revision")},
              "entry": item, "worktree": WORKTREE.as_posix(), "cwd": WORKTREE.as_posix(), "command": command,
              "declaration_text_sha256": fresh_sets.text_sha256(DECLARATION),
              "thread_variables_removed": rc.thread_variables(dict(os.environ)), "threads": rc.thread_probe(env),
              "limit_minutes": LIMIT_MINUTES, "conditions": conditions, "launch_problems": problems, "dry_run": dry_run}
    if problems or dry_run:
        print(("Refused:\n  " + "\n  ".join(problems)) if problems else f"Dry run passed: seed {item['seed']} "
              f"attempt {item['attempt']}: {' '.join(item['command'])}")
        status = "refused" if problems else "dry-run"
        print(f"Record: {write_record(record, status)}")
        return status
    started, output = time.time(), {"stdout": "", "stderr": ""}
    try:
        done = subprocess.run(command, cwd=WORKTREE, env=env, capture_output=True, text=True,
                              timeout=LIMIT_MINUTES * 60)
        output.update(stdout=done.stdout or "", stderr=done.stderr or "")
        record["exit_code"] = done.returncode
    except subprocess.TimeoutExpired as error:
        for key, value in (("stdout", error.stdout), ("stderr", error.stderr)):
            output[key] = value.decode(errors="replace") if isinstance(value, bytes) else (value or "")
        record.update(exit_code=None, timed_out=True, game_cleanup=rc.cleanup_own_game(GAME))
    except BaseException as error:  # stopped by hand or broken: stop the game, keep what was captured, then re-raise
        record.update(exit_code=None, launch_error=f"{type(error).__name__}: {error}", outcome="failed",
                      problems=["the launch was interrupted"], game_cleanup=rc.cleanup_own_game(GAME))
        print(f"Record: {write_record(record, 'attempt', {'stdout.txt': output['stdout'], 'stderr.txt': output['stderr']})}")
        raise
    record["seconds"] = round(time.time() - started)
    run = WORKTREE / item["run_dir"]
    outcome = [f"exit status {record['exit_code']}"] if record.get("exit_code") != 0 else []
    faults: dict = {}
    if run.is_dir():
        found, faults = outcome_problems(run, item["seed"])
        outcome += found
        record["run_files"] = file_hashes(run)
    else:
        outcome.append(f"{run} was not created")
    record["faults"] = faults
    if not outcome:
        try:
            record["copy"] = copy_run(run, REPO / item["run_dir"])
        except (Refused, OSError) as error:
            outcome.append(f"copy: {error}")
    record.update(problems=outcome, outcome="failed" if outcome else "ok",
                  stdout_tail=output["stdout"].strip().splitlines()[-5:],
                  stderr_tail=output["stderr"].strip().splitlines()[-8:])
    path = write_record(record, "attempt", {"stdout.txt": output["stdout"], "stderr.txt": output["stderr"]})
    print(f"seed {item['seed']} attempt {item['attempt']}: {record['outcome'].upper()}"
          + (f" ({'; '.join(outcome)})" if outcome else f" in {record['seconds']} s"))
    print(f"Record: {path}")
    return record["outcome"]


def run_action(dry_run: bool, limit: int | None) -> int:
    declaration = load_declaration()
    try:
        if not PLAN.exists():
            raise Refused(f"{_relative(PLAN)} does not exist; run plan first")
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        pending = next_entries(plan, RECORDS)
    except (Refused, OSError, ValueError) as error:
        print(f"Refused: {error}")
        return 1
    if not pending:
        print("Nothing to run: every planned attempt has a record.")
        return 0
    if dry_run:
        return 0 if run_entry(plan, pending[0], declaration, dry_run=True) == "dry-run" else 1
    for item in pending[:limit] if limit is not None else pending:
        if run_entry(plan, item, declaration, dry_run=False) != "ok":
            return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--check", action="store_true")
    plan.add_argument("--rerun", type=int)
    plan.add_argument("--replace", type=int)
    plan.add_argument("--reason")
    sub.add_parser("prepare")
    sub.add_parser("check")
    run = sub.add_parser("run")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--max", type=int, help="run at most this many attempts")
    args = parser.parse_args()
    if args.action == "plan":
        return plan_action(args.check, args.rerun, args.replace, args.reason)
    if args.action == "prepare":
        return prepare_action()
    if args.action == "check":
        return check_action()
    return run_action(args.dry_run, args.max)


if __name__ == "__main__":
    raise SystemExit(main())
