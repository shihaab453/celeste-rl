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
    .venv-rl/Scripts/python.exe scripts/confirmation_historical_donors.py plan --rebind-amendment-3
    .venv-rl/Scripts/python.exe scripts/confirmation_historical_donors.py prepare
    .venv-rl/Scripts/python.exe scripts/confirmation_historical_donors.py check
    .venv-rl/Scripts/python.exe scripts/confirmation_historical_donors.py run [--dry-run] [--max N]

Reading rule (fresh sets): every file this script reads itself (the declaration, the plan, the comparator and attempt
records, the original seed 7 manifest) is identity-checked by the fresh-set guard before it is read, and so are the
game copy's files the old trainer reads (its four profile settings, the lockstep mod's manifest and the hashed game
files). The command, as given and as resolved in the worktree, and the copied init clone go through the full guard.
A refusal stops at once: nothing else is read and no child process starts.

plan: writes config/confirmation-donors-historical.json (refuses to overwrite) once the authorization exists: the
declaration carries amendment 2, and every record of the recipe check's comparator version 2, at a reviewed commit on
a clean tree, agrees that the current replay is DIFFERENT and the 8423081 diagnostic IDENTICAL (no record of the
current code as IDENTICAL is needed or produced). The plan pins the declaration's text hash, both records, the source
commit, the worktree, the game copy, the interpreter, the init clone, the limit and one entry per donor (j, seed,
attempt, run folder, command). --check recomputes everything. --rerun and --replace write a revision that keeps every
earlier entry: a rerun only after a failed attempt of that seed (at most two reruns); a replacement only for the
inclusion rule, of a donor whose training succeeded (a failed training attempt is recovered with the same seed),
within the two-seed budget, in order 15 then 16, taking the replaced donor's j. Commit and push the plan (and each
revision) before running it.

plan --rebind-amendment-3 (one time): amendment 3 changes the declaration's text hash that the plan pins. This
writes a plan revision with every entry unchanged, bound to the amended declaration, only if: the plan and the
declaration are committed; the declaration text the plan pins is read from the commit holding that plan version
(identity-checked, never assumed to be HEAD) and the plan passes plan_problems against it; the new declaration is
that text plus exactly one appended entry, the reviewed amendment 3 (canonical JSON hash AMENDMENT_3_SHA256, number
the integer 3), compared type-preservingly with duplicate JSON keys refused; and every attempt record names a
committed plan version and the declaration that version pinned. Run again after it is committed: nothing to do.
Any other declaration change, a second amendment 3 or an edited plan is refused.

prepare: creates the worktree (git worktree add --detach) if absent and copies the init clone folder into it,
verifying every file's sha256; records both. It never changes an existing worktree's files.

check: the declared source check (code_checks.historical_donors): HEAD and tree are 8423081's; no tracked or
untracked change and no index flag that hides one; no Python bytecode anywhere in the worktree (a stale cache could
run other code than the checked source); ignored files only under the declared input and run folders. Only then is
scripts/train_room1.py imported there, without writing bytecode, and every module of its import closure must lie
inside the worktree and equal its blob at 8423081.

run: trains the plan's entries in order, one at a time, stopping at the first attempt that is not ok; the records are
read again before every launch, so eligibility always reflects the latest attempt. Before each launch it refuses
unless: the reading rule's checks pass (first, before anything else); this repository's tracked tree is clean and the
plan committed; the plan passes --check; the entry is eligible (never run, no ok attempt of its seed, a rerun only
after a failed attempt, a replacement only of a donor whose training succeeded, every earlier entry resolved); the
source check passes; the init clone copy matches its pin; the run folder exists neither in the worktree nor here;
this is the declared interpreter; Steam runs; no Celeste process runs in any folder and no other script of this
repository runs. It removes every thread variable, disables bytecode writing, probes torch's default threads and runs
the command with the worktree as working directory for at most 90 minutes, streaming stdout and stderr to files. A
started marker is written before launch, and the attempt's record is written on every path out (success, failure,
timeout, error, interruption); a marker without a record counts as a failed attempt. An attempt is ok only if the
run is complete and finished at the historical commit with exactly one attributable session, no runtime problems,
the original seed 7 runtime record, seed 7's config apart from the seed, truthful evaluation references and no
bytecode left in the worktree; it is then copied to the same relative folder here and every file's hash is checked.
Faults are recorded by name. A failed attempt stays in the worktree with its record; a rerun needs a plan revision.
--dry-run preflights every pending entry without launching (earlier entries need not be resolved).
Records are created exclusively in runs/confirmation/donors/, never overwritten.
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
FRESH = fresh_sets.FRESH  # the protected sets (tests substitute a stand-in)
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
# Game copy files that 8423081's runtime check reads (celeste_rl/runtime.py there: REQUIRED_SETTINGS, HASHED_FILES,
# MOD_MANIFESTS). log.txt is written by the game during the run, so it cannot be checked before launch.
GAME_FILES_READ = tuple(f"probe-profile/Saves/{name}" for name in (
    "modsettings-Everest.celeste", "settings.celeste", "modsettings-CelesteTAS.celeste",
    "modsettings-SpeedrunTool.celeste")) + (
    "Celeste.exe", "Celeste.dll", "Celeste.Mod.mm.dll", "FNA.dll", "MMHOOK_Celeste.dll", "Mods/CelesteTAS.zip",
    "Mods/SpeedrunTool.zip", "Mods/CelesteRLLockstep/CelesteRLLockstep.dll", "Mods/CelesteRLLockstep/everest.yaml")
NO_BYTECODE = {"PYTHONDONTWRITEBYTECODE": "1"}
STARTED = ".started.json"


class Refused(Exception):
    """A precondition does not hold, so nothing is written or launched."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if done.returncode != 0:
        raise Refused(f"git {' '.join(args)} in {root} exited {done.returncode}: {(done.stderr or '').strip()[:300]}")
    return done.stdout


# ----------------------------------------------------------------------------------------------- the reading rule

def identity_problems(paths) -> list[str]:
    """The fresh-set identity check for files this script (or the old trainer) reads itself, before any is read."""
    return fresh_sets.identity_problems([str(path) for path in paths], fresh=FRESH)


def read_json(path: Path):
    """A JSON file this script reads itself, identity-checked first; Refused if it refers to a fresh set."""
    problems = identity_problems([path])
    if problems:
        raise Refused("; ".join(problems))
    return json.loads(Path(path).read_text(encoding="utf-8"))


def guard_problems(command: list[str], worktree: Path, fresh: dict | None = None) -> list[str]:
    """The fresh-set guard with no allowed use: the command as given, as resolved in the worktree, and the copied init
    clone with its dependencies."""
    fresh = fresh or FRESH
    try:
        fresh_sets.refuse_unless_declared([command, resolved(command, worktree)], (), fresh=fresh)
    except fresh_sets.FreshSetRefused as error:
        return [f"fresh-set guard: {error}"]
    return fresh_sets.paths_problems([str(worktree / INIT_CLONE)], fresh=fresh)


def resolved(command: list[str], root: Path) -> list[str]:
    """The command as the child sees it: each relative path that exists under its working directory made absolute."""
    return [str(root / token) if not token.startswith("-") and not Path(token).is_absolute() and (root / token).exists()
            else token for token in command]


# -------------------------------------------------------------------------------------------------- the declaration

def load_declaration() -> dict:
    return read_json(DECLARATION)


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
            record = read_json(path)
        except (OSError, ValueError):
            continue  # an unreadable record authorizes nothing
        comparator = record.get("comparator") if isinstance(record, dict) else None
        git = comparator.get("git") if isinstance(comparator, dict) else None
        if (isinstance(git, dict) and comparator.get("version") == rc.COMPARATOR_VERSION
                and git.get("commit") in REVIEWED_COMPARATORS and git.get("uncommitted_changes") is False
                and not git.get("git_error")):
            found.append((path, record))
    return found


def authorization(records: Path | None = None) -> dict:
    """The two comparator version 2 records that permit historical donors; Refused unless every such record agrees:
    the current replay DIFFERENT (bound to the live replay) and the diagnostic IDENTICAL (bound to its record)."""
    records = records or rc.RECORDS
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
    newest = {"compare": replay[-1], "compare_diagnostic": diagnostic[-1]}
    return {key: {"path": _relative(path), "sha256": _sha256(path), "verdict": record["verdict"],
                  "comparator_commit": record["comparator"]["git"]["commit"]}
            for key, (path, record) in newest.items()}


def _relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


# --------------------------------------------------------------------------------------------------------- the plan

def fixed_fields(declaration: dict, auth: dict, declaration_sha256: str | None = None) -> dict:
    """Everything a plan pins apart from its entries and revision. declaration_sha256 names the declaration text the
    plan was written against (default: the live file's), so an earlier plan can be checked against its own text."""
    return {"name": "confirmation-donors-historical",
            "declaration": {"path": "config/retention-confirmation.json",
                            "text_sha256": declaration_sha256 or fresh_sets.text_sha256(DECLARATION)},
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


def replaced_seed(item: dict) -> int | None:
    """The donor a replacement entry names in its reason ('replaces seed N: ...'), or None."""
    named = re.match(r"replaces seed (\d+):", str(item.get("reason", "")))
    return int(named.group(1)) if named else None


def plan_problems(plan: dict, declaration: dict, auth: dict, declaration_sha256: str | None = None) -> list[str]:
    """Differences between a plan and what the declaration and records require now (static rules only; the attempt
    records are checked by eligibility_problems before each launch)."""
    if not isinstance(plan, dict) or not isinstance(plan.get("entries"), list):
        return ["the plan is not a record with entries"]
    problems = [f"plan {key}: {plan.get(key)!r}, expected {value!r}"
                for key, value in fixed_fields(declaration, auth, declaration_sha256).items() if plan.get(key) != value]
    replacements = list(declaration["donors"]["replacement_seeds"])
    allowed = set(declaration["donors"]["new_seeds"]) | set(replacements)
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
    used = [item.get("seed") for item in plan["entries"]
            if isinstance(item, dict) and item.get("seed") in replacements and item.get("attempt") == 1]
    if used != replacements[:len(used)]:
        problems.append(f"replacement seeds {used} are not used in the declared order {replacements}")
    problems += index_problems(plan["entries"], declaration)
    initial = initial_plan(declaration, auth)["entries"]
    if plan["entries"][:len(initial)] != initial:
        problems.append("the plan's first entries are not the declared new donors in order")
    return problems


def index_problems(entries: list, declaration: dict) -> list[str]:
    """Every attempt of a seed keeps its j; a replacement takes the j of a donor named earlier in its reason."""
    known, problems = dict(donor_index(declaration)), []
    replacements = set(declaration["donors"]["replacement_seeds"])
    for item in entries:
        if not isinstance(item, dict):
            continue
        seed, j = item.get("seed"), item.get("j")
        if seed in replacements and item.get("attempt") == 1:
            if known.get(replaced_seed(item)) != j:
                problems.append(f"replacement seed {seed} (j {j}) does not take the j of the donor it names")
            known[seed] = j
        elif known.get(seed) != j:
            problems.append(f"seed {seed} attempt {item.get('attempt')} has j {j}, not {known.get(seed)}")
    return problems


def revised_plan(plan: dict, declaration: dict, records: Path, rerun: int | None = None,
                 replace: int | None = None, reason: str | None = None) -> dict:
    """A revision with one more entry; every earlier entry is kept exactly."""
    entries = list(plan["entries"])
    statuses = attempt_statuses(records)
    if rerun is not None:
        attempts = [item for item in entries if item["seed"] == rerun]
        if not attempts:
            raise Refused(f"seed {rerun} is not in the plan")
        last = attempts[-1]
        if any(statuses.get((rerun, item["attempt"])) == "ok" for item in attempts):
            raise Refused(f"seed {rerun} already has an ok attempt; no further attempt")
        if statuses.get((rerun, last["attempt"])) != "failed":
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
        attempts = [item for item in entries if item["seed"] == replace]
        if attempts and statuses.get((replace, attempts[-1]["attempt"])) != "ok":
            raise Refused(f"seed {replace}'s latest training attempt is not ok: a replacement is only for the inclusion "
                          "rule; a failed training attempt is recovered with the same seed (plan --rerun)")
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
    try:
        problems = identity_problems([DECLARATION, PLAN])
        if problems:
            raise Refused("; ".join(problems))
        declaration = load_declaration()
        problems = amendment_problems(declaration)
        if problems:
            raise Refused("; ".join(problems))
        auth = authorization()
        if check:
            problems = plan_problems(read_json(PLAN), declaration, auth)
            print("\n".join(problems) if problems else f"PASS: {_relative(PLAN)} is the declared plan")
            return 1 if problems else 0
        if rerun is None and replace is None:
            if PLAN.exists():
                raise Refused(f"{_relative(PLAN)} exists; use --rerun or --replace for a revision")
            plan = initial_plan(declaration, auth)
        else:
            problems = committed(PLAN)
            current = read_json(PLAN)
            problems += plan_problems(current, declaration, auth)
            if problems:
                raise Refused("; ".join(problems))
            plan = revised_plan(current, declaration, RECORDS, rerun, replace, reason)
    except (Refused, OSError, ValueError) as error:
        print(f"Refused: {error}")
        return 1
    PLAN.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {_relative(PLAN)} revision {plan['revision']} with {len(plan['entries'])} entries; commit and push it "
          "before running it.")
    return 0


# --------------------------------------------------------------------------- the one-time amendment 3 rebind

def strict_json(text: str):
    """JSON with duplicate keys refused (a later duplicate would silently replace an earlier value)."""
    def pairs(items):
        keys = [key for key, _ in items]
        repeated = sorted({key for key in keys if keys.count(key) > 1})
        if repeated:
            raise Refused(f"duplicate JSON keys {repeated}")
        return dict(items)
    return json.loads(text, object_pairs_hook=pairs)


def same_json(a, b) -> bool:
    """Type-preserving JSON equality: true is not 1, 1 is not 1.0, and key sets must match exactly."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same_json(a[key], b[key]) for key in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same_json(x, y) for x, y in zip(a, b))
    return a == b


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


def _text_sha256_of(blob: bytes) -> str:
    """The texthash of committed bytes (CRLF taken as LF), as fresh_sets.text_sha256 computes it for a file."""
    return hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest()


def _git_bytes(*args: str) -> bytes:
    done = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True)
    if done.returncode != 0:
        raise Refused(f"git {' '.join(args)} exited {done.returncode}: {done.stderr.decode(errors='replace')[:300]}")
    return done.stdout


# The reviewed amendment 3 entry, by canonical JSON hash: the rebind accepts exactly this appended entry.
AMENDMENT_3_SHA256 = "96c5f7d54d2028fe031c4c54c575c2cd7ce7785393b3ae5367dca628484cec42"
PLAN_RELATIVE = "config/confirmation-donors-historical.json"
DECLARATION_RELATIVE = "config/retention-confirmation.json"


def plan_history() -> list[tuple[str, str, dict]]:
    """Every committed version of the plan, newest first: (commit, plan text hash, plan)."""
    commits = _git_bytes("log", "--format=%H", "--", PLAN_RELATIVE).decode().split()
    history = []
    for commit in commits:
        blob = _git_bytes("show", f"{commit}:{PLAN_RELATIVE}")
        history.append((commit, _text_sha256_of(blob), strict_json(blob.decode("utf-8"))))
    return history


def rebind_problems(old_text: bytes, new_text: bytes) -> list[str]:
    """Whether the new declaration is the old one plus exactly the reviewed amendment 3, nothing else."""
    try:
        old, new = strict_json(old_text.decode("utf-8")), strict_json(new_text.decode("utf-8"))
    except (Refused, ValueError) as error:
        return [f"declaration cannot be read strictly: {error}"]
    old_amendments, new_amendments = old.get("amendments"), new.get("amendments")
    if not isinstance(old_amendments, list) or not isinstance(new_amendments, list):
        return ["a declaration has no amendments list"]
    if any(isinstance(a, dict) and "number" in a for a in old_amendments):
        return ["the previous declaration already has a numbered amendment (amendment 3 was applied before)"]
    if len(new_amendments) != len(old_amendments) + 1:
        return [f"expected exactly one appended amendment, found {len(new_amendments) - len(old_amendments)}"]
    appended = new_amendments[-1]
    problems = []
    if not isinstance(appended, dict) or type(appended.get("number")) is not int or appended.get("number") != 3:
        problems.append("the appended entry is not numbered with the integer 3")
    if canonical_sha256(appended) != AMENDMENT_3_SHA256:
        problems.append("the appended entry is not the reviewed amendment 3")
    if not same_json({**new, "amendments": new_amendments[:-1]}, old):
        problems.append("the declaration differs from the previous one outside the appended amendment")
    return problems


def rebind_action() -> int:
    """Record amendment 3's new declaration hash as a plan revision; entries and attempt evidence unchanged."""
    try:
        problems = identity_problems([DECLARATION, PLAN]) + committed(PLAN) + committed(DECLARATION)
        if problems:
            raise Refused("; ".join(problems))
        history = plan_history()
        if not history:
            raise Refused(f"{PLAN_RELATIVE} has no committed version")
        _, current_hash, current = history[0]
        if current_hash != fresh_sets.text_sha256(PLAN) or not same_json(current, read_json(PLAN)):
            raise Refused("the plan on disk is not its last committed version")
        new_text = DECLARATION.read_bytes()
        new_hash = _text_sha256_of(new_text)
        rebind = current.get("rebind")
        if rebind is not None:
            if rebind.get("amendment") != 3 or current["declaration"]["text_sha256"] != new_hash:
                raise Refused("the plan was already rebound; another amendment 3 is not accepted")
            historical_evidence(history)  # the committed chain, the rebind revision and every attempt still hold
            problems = plan_problems(current, load_declaration(), authorization())
            if problems:
                raise Refused("; ".join(problems))
            print("Already rebound to amendment 3 (chain, attempts and plan verified); nothing to do.")
            return 0
        old_hash = current["declaration"]["text_sha256"]
        # The declaration text the plan pins, from a commit holding exactly this plan version and that declaration
        # text (never HEAD by default: HEAD may already hold the amended declaration).
        old_text, commit = None, None
        for candidate, plan_hash, _ in history:
            if plan_hash != current_hash:
                continue
            text = _git_bytes("show", f"{candidate}:{DECLARATION_RELATIVE}")
            if _text_sha256_of(text) == old_hash:
                old_text, commit = text, candidate
                break
        if old_text is None:
            raise Refused(f"no commit holds this plan version with the declaration it pins ({old_hash[:12]})")
        problems = rebind_problems(old_text, new_text)
        old_declaration = strict_json(old_text.decode("utf-8"))
        problems += plan_problems(current, old_declaration, authorization(), declaration_sha256=old_hash)
        if problems:
            raise Refused("; ".join(problems))
        historical_evidence(history)  # chain, membership, sequence and verified copies of every attempt
    except (Refused, OSError, ValueError) as error:
        print(f"Refused: {error}")
        return 1
    revised = {**current, "declaration": {"path": DECLARATION_RELATIVE, "text_sha256": new_hash},
               "revision": current["revision"] + 1, "previous_text_sha256": current_hash,
               "rebind": {"amendment": 3, "from_declaration_sha256": old_hash, "to_declaration_sha256": new_hash,
                          "amendment_sha256": AMENDMENT_3_SHA256, "from_plan_commit": commit,
                          "reason": "amendment 3 appended to the declaration; entries and attempts unchanged"}}
    PLAN.write_text(json.dumps(revised, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {PLAN_RELATIVE} revision {revised['revision']} bound to the amended declaration; commit and push it.")
    return 0


def chain_problems(history: list[tuple[str, str, dict]]) -> list[str]:
    """The committed plan versions form one chain: revisions 0..n, each naming its predecessor's text hash, earlier
    entries kept, fixed fields unchanged except one amendment 3 rebind of the declaration binding."""
    by_revision: dict[int, tuple[str, dict]] = {}
    for _, plan_hash, plan in history:
        revision = plan.get("revision")
        if type(revision) is not int or (revision in by_revision and by_revision[revision][0] != plan_hash):
            return [f"plan revision {revision!r} is missing or appears with different contents"]
        by_revision[revision] = (plan_hash, plan)
    if sorted(by_revision) != list(range(len(by_revision))):
        return [f"plan revisions are not 0..n: {sorted(by_revision)}"]
    problems, rebinds = [], 0
    for revision in range(1, len(by_revision)):
        (before_hash, before), (_, after) = by_revision[revision - 1], by_revision[revision]
        if after.get("previous_text_sha256") != before_hash:
            problems.append(f"revision {revision} does not name revision {revision - 1}")
        if not same_json(after["entries"][:len(before["entries"])], before["entries"]):
            problems.append(f"revision {revision} changed earlier entries")
        if "rebind" in after and "rebind" not in before:
            rebinds += 1
            rebind = after["rebind"]
            if (rebind.get("amendment") != 3 or rebind.get("from_declaration_sha256") != before["declaration"]["text_sha256"]
                    or rebind.get("to_declaration_sha256") != after["declaration"]["text_sha256"]
                    or rebind.get("amendment_sha256") != AMENDMENT_3_SHA256 or after["entries"] != before["entries"]):
                problems.append(f"revision {revision} is not the reviewed amendment 3 rebind")
        elif after["declaration"] != before["declaration"]:
            problems.append(f"revision {revision} changed the declaration binding outside a rebind")
        ignore = {"entries", "revision", "previous_text_sha256", "rebind", "declaration"}
        if not same_json({k: v for k, v in after.items() if k not in ignore},
                         {k: v for k, v in before.items() if k not in ignore}):
            problems.append(f"revision {revision} changed the plan's fixed fields")
    if rebinds > 1:
        problems.append("more than one rebind")
    return problems


def historical_evidence(history: list[tuple[str, str, dict]] | None = None) -> dict[int, dict]:
    """The validated historical donors, by seed, from the committed plan chain and the attempt records; Refused on
    any inconsistency. Every final record and started marker must name a committed plan version that contains its
    full entry and the declaration that version pinned, at the historical source; attempts follow the sequence (a
    rerun only after a failed attempt, nothing after an ok one, one ok attempt per seed); an ok attempt needs its
    verified copy with the checkpoint's hash."""
    history = plan_history() if history is None else history
    problems = chain_problems(history)
    versions = {plan_hash: plan for _, plan_hash, plan in history}
    statuses = attempt_statuses(RECORDS)  # also refuses conflicting or unreadable attempt files
    donors = {}
    for path in sorted(RECORDS.glob("attempt-*.json")):
        record = read_json(path)
        marker = path.name.endswith(STARTED)
        entry = record.get("entry") if isinstance(record, dict) else None
        if marker:
            if not any(any(same_json(entry, e) for e in plan["entries"]) for plan in versions.values()):
                problems.append(f"{path.name} names an entry that no committed plan version holds")
            continue
        bound = versions.get((record.get("plan") or {}).get("text_sha256"))
        if bound is None:
            problems.append(f"{path.name} names a plan version that is not committed")
            continue
        if not any(same_json(entry, e) for e in bound["entries"]):
            problems.append(f"{path.name}: its entry is not in the plan version it names")
            continue  # nothing else about an unplanned attempt can be trusted
        if record.get("declaration_text_sha256") != bound["declaration"]["text_sha256"]:
            problems.append(f"{path.name}: its declaration is not the one its plan version pinned")
        if (record.get("source") or {}).get("commit") != HISTORICAL_COMMIT:
            problems.append(f"{path.name}: not run at the historical source")
        if record.get("outcome") == "ok":
            copy = record.get("copy") or {}
            sha = (copy.get("files") or {}).get("checkpoints/latest.zip")
            if copy.get("verified") is not True or copy.get("path") != entry["run_dir"] or not sha:
                problems.append(f"{path.name}: ok without a verified copy of {entry['run_dir']}")
            elif entry["seed"] in donors:
                problems.append(f"seed {entry['seed']} has more than one ok attempt")
            else:
                donors[entry["seed"]] = {"seed": entry["seed"], "j": entry["j"], "attempt": entry["attempt"],
                                         "checkpoint": f"{entry['run_dir']}/checkpoints/latest.zip", "sha256": sha,
                                         "record": {"path": _relative(path),
                                                    "sha256": _sha256(path)},
                                         "plan_version_sha256": record["plan"]["text_sha256"]}
    by_seed: dict[int, dict[int, str]] = {}
    for (seed, attempt), status in statuses.items():
        by_seed.setdefault(seed, {})[attempt] = status
    for seed, attempts in by_seed.items():
        numbers = sorted(attempts)
        if numbers != list(range(1, len(numbers) + 1)):
            problems.append(f"seed {seed} attempts {numbers} are not 1..n")
        if any(attempts[n] != "failed" for n in numbers[:-1]):
            problems.append(f"seed {seed} has an attempt after an ok attempt")
    if problems:
        raise Refused("; ".join(problems))
    return donors


# ------------------------------------------------------------------------------------------------ the worktree source

def _ignored_allowed(path: str) -> bool:
    return path.startswith(INIT_CLONE.as_posix() + "/") or path.startswith(RUN_PREFIX)


def _bytecode(path: str) -> bool:
    return "/__pycache__/" in f"/{path}" or path.endswith((".pyc", ".pyo"))


CLOSURE_CODE = ("import os, sys; sys.path.insert(0, 'scripts'); import train_room1; "
                "print('\\n'.join(sorted({os.path.abspath(m.__file__) for n, m in list(sys.modules.items()) "
                "if (n == 'celeste_rl' or n.startswith('celeste_rl.')) and getattr(m, '__file__', None)} "
                "| {os.path.abspath(train_room1.__file__)})))")


def worktree_problems(worktree: Path, commit: str, tree: str) -> tuple[list[str], dict]:
    """The git-level part of the source check (no import): commit, tree, changes, index flags, bytecode, strays."""
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
    extra = [line[3:].strip('"') for line in listed if line.startswith(("!! ", "?? "))]
    bytecode = [path for path in extra if _bytecode(path)]
    if bytecode:
        problems.append(f"Python bytecode in the worktree (it could run instead of the checked source): {bytecode[:10]}")
    stray = [line[3:].strip('"') for line in listed if line.startswith("!! ")]
    stray = [path for path in stray if not _ignored_allowed(path) and not _bytecode(path)]
    if stray:
        problems.append(f"undeclared ignored files in the worktree: {stray[:10]}")
    evidence["ignored_files"] = sum(line.startswith("!! ") for line in listed)
    return problems, evidence


def source_problems(worktree: Path | None = None, commit: str = HISTORICAL_COMMIT, tree: str = HISTORICAL_TREE,
                    interpreter: Path | None = None) -> tuple[list[str], dict]:
    """The declared source check (code_checks.historical_donors), with the evidence it rests on. The trainer is
    imported only if every git-level check passed, and never with bytecode written."""
    worktree = worktree or WORKTREE
    problems, evidence = worktree_problems(worktree, commit, tree)
    if problems:
        return problems, evidence
    done = subprocess.run([str(interpreter or REPO / INTERPRETER), "-c", CLOSURE_CODE], cwd=worktree,
                          capture_output=True, text=True, env={**os.environ, **NO_BYTECODE})
    if done.returncode != 0:
        return [f"importing scripts/train_room1.py in the worktree failed: {done.stderr.strip()[-300:]}"], evidence
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
        except Refused:  # the file is not in that commit
            expected = None
        if blob != expected:
            problems.append(f"{relative} is not its blob at {commit[:7]} ({blob} vs {expected})")
        closure[relative] = blob
    if "scripts/train_room1.py" not in closure or not any(p.startswith("celeste_rl/") for p in closure):
        problems.append(f"the import closure is incomplete: {sorted(closure)}")
    problems += worktree_problems(worktree, commit, tree)[0]  # the import wrote nothing
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
        guard = fresh_sets.paths_problems([str(REPO / INIT_CLONE)], fresh=FRESH)
        if guard:
            raise Refused("; ".join(guard))
        if not WORKTREE.exists():
            _git(REPO, "worktree", "add", "--detach", str(WORKTREE), HISTORICAL_COMMIT)
            record["created_worktree"] = True
        if not (WORKTREE / INIT_CLONE).exists():
            shutil.copytree(REPO / INIT_CLONE, WORKTREE / INIT_CLONE)
            record["copied_init_clone"] = True
        guard = fresh_sets.paths_problems([str(WORKTREE / INIT_CLONE)], fresh=FRESH)  # the copy, before it is hashed
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


def attempt_statuses(records: Path) -> dict[tuple[int, int], str]:
    """'ok' or 'failed' per (seed, attempt) from the attempt records: a final record says which; a started marker
    without its final record is an interrupted, failed attempt. Refusals and dry runs are not attempts. An attempt
    launches at most once, so a second record for the same seed and attempt (whatever its outcome) is a conflict that
    stops everything for review, as does an unreadable attempt file: neither may become permission for another run."""
    outcomes: dict[tuple[int, int], list] = {}
    for path in sorted(records.glob("attempt-*.json")):
        marker = path.name.endswith(STARTED)
        if marker and path.with_name(path.name[:-len(STARTED)] + ".json").exists():
            continue  # its final record decides
        try:
            record = read_json(path)
            planned = record["entry"]
            key = (int(planned["seed"]), int(planned["attempt"]))
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise Refused(f"attempt file {path.name} cannot be read ({type(error).__name__}); stop for review")
        outcome = "failed" if marker else ("ok" if record.get("outcome") == "ok" else "failed")
        outcomes.setdefault(key, []).append((path.name, outcome))
    conflicts = {key: found for key, found in outcomes.items() if len(found) > 1}
    if conflicts:
        raise Refused(f"conflicting attempt records {conflicts}; an attempt launches once: stop for review")
    return {key: found[0][1] for key, found in outcomes.items()}


def entry_status(records: Path, item: dict) -> str | None:
    return attempt_statuses(records).get((item["seed"], item["attempt"]))


def executor_identity() -> dict:
    return {"script_sha256": _sha256(Path(__file__)), "git": runtime.git_state()}


# ------------------------------------------------------------------------------------------------------ eligibility

def eligibility_problems(plan: dict, item: dict, records: Path, preflight: bool = False) -> list[str]:
    """Whether the attempt records allow this entry to launch now. preflight (dry run) skips only the rule that every
    earlier entry is resolved."""
    statuses = attempt_statuses(records)
    seed, attempt = item["seed"], item["attempt"]
    problems = []
    if (seed, attempt) in statuses:
        problems.append(f"seed {seed} attempt {attempt} already has an attempt record")
    if any(statuses.get((seed, a)) == "ok" for a in range(1, attempt)):
        problems.append(f"seed {seed} already has an ok attempt")
    if attempt > 1 and statuses.get((seed, attempt - 1)) != "failed":
        problems.append(f"seed {seed} attempt {attempt} is a rerun, but attempt {attempt - 1} has no failed record")
    replaced = replaced_seed(item)
    if replaced is not None:
        tries = [e["attempt"] for e in plan["entries"] if e["seed"] == replaced]
        if tries and statuses.get((replaced, tries[-1])) != "ok":
            problems.append(f"seed {seed} replaces seed {replaced}, whose latest training attempt is not ok")
    if not preflight:
        for earlier in plan["entries"][:plan["entries"].index(item)]:
            status = statuses.get((earlier["seed"], earlier["attempt"]))
            later = [e for e in plan["entries"] if e["seed"] == earlier["seed"] and e["attempt"] > earlier["attempt"]]
            if status is None or (status == "failed" and not later):
                problems.append(f"earlier entry seed {earlier['seed']} attempt {earlier['attempt']} is unresolved")
    return problems


def next_entry(plan: dict, records: Path) -> dict | None:
    """The first entry without an attempt record; Refused if a failed attempt has no later attempt planned."""
    statuses = attempt_statuses(records)
    for item in plan["entries"]:
        status = statuses.get((item["seed"], item["attempt"]))
        if status == "failed" and not [e for e in plan["entries"]
                                       if e["seed"] == item["seed"] and e["attempt"] > item["attempt"]]:
            raise Refused(f"seed {item['seed']} attempt {item['attempt']} failed; revise the plan (plan --rerun)")
        if status is None:
            return item
    return None


# ------------------------------------------------------------------------------------------------------------- run

PROCESS_MARKER = "__QUERY_OK__"
PROCESS_QUERY = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" -ErrorAction Stop | ForEach-Object { "
                 "\"$($_.ProcessId)`t$($_.ParentProcessId)`t$($_.ExecutablePath)`t$($_.CommandLine)\" }; '"
                 + PROCESS_MARKER + "'")


def _arguments(command_line: str) -> str:
    """A Windows command line without its first token (the program, quoted or not)."""
    match = re.match(r'\s*("[^"]*"|\S+)\s*(.*)$', command_line or "", re.DOTALL)
    return match.group(2).strip() if match else ""


def other_experiments(output: str | None = None, own_pid: int | None = None) -> list[str]:
    """Command lines of running python processes that run any script of this repository (the recipe check included),
    except this process and its own venv launcher: on Windows .venv-rl's python.exe starts the real interpreter as a
    child with the same arguments, so the parent is excluded only if it is that launcher with identical arguments.
    Another executor, or anything that started this one, is still reported."""
    if output is None:
        output = rc._query(["powershell", "-NoProfile", "-Command", PROCESS_QUERY], "python process query")
    if PROCESS_MARKER not in output:
        raise rc.ProcessEvidenceUnavailable("python process query did not complete")
    rows = {}
    for line in output.splitlines():
        parts = line.split("\t", 3)
        if len(parts) == 4 and parts[0].strip().isdigit():
            parent = int(parts[1]) if parts[1].strip().isdigit() else None
            rows[int(parts[0])] = (parent, parts[2].strip(), parts[3].strip())
    own = os.getpid() if own_pid is None else own_pid
    excluded = {own}
    if own in rows:
        parent, _, own_line = rows[own]
        launcher = rows.get(parent)
        if (launcher and launcher[1] and os.path.normcase(os.path.abspath(launcher[1]))
                == os.path.normcase(os.path.abspath(REPO / INTERPRETER))
                and _arguments(launcher[2]) == _arguments(own_line)):
            excluded.add(parent)
    names = (*rc.EXPERIMENT_SCRIPTS, Path(rc.__file__).name)
    return [line[:300] for pid, (_, _, line) in sorted(rows.items())
            if pid not in excluded and any(name in line for name in names)]


def steam_running() -> bool:
    output = rc._query(["tasklist", "/FI", "IMAGENAME eq steam.exe", "/FO", "CSV", "/NH"], "Steam process query")
    return any(line.strip().lower().startswith('"steam.exe"') for line in output.splitlines())


def launch_problems(plan: dict, item: dict, declaration: dict, preflight: bool = False) -> tuple[list[str], dict]:
    """Why this attempt must not start, and the recorded conditions, in stages that each return at once on a problem:
    (1) the fresh-set checks of the inputs; (2) the records (authorization and attempt history, each record
    identity-checked before it is read) and the plan; (3) the repository state; only then (4) the source check, which
    imports the old trainer, the init clone hashes, the folders and the running processes."""
    conditions: dict = {"fresh_set_check": "identity of the original manifest and the game files the old trainer reads; "
                                           "full guard on the command as given and as resolved, and the init clone"}
    guard = identity_problems([REPO / rc.ORIGINAL / "manifest.json", *(GAME / name for name in GAME_FILES_READ)])
    guard += guard_problems(item["command"], WORKTREE)
    if guard:
        return guard, conditions
    try:
        problems = plan_problems(plan, declaration, authorization()) + amendment_problems(declaration)
        problems += eligibility_problems(plan, item, RECORDS, preflight)
    except Refused as error:  # a record refused (fresh-set identity, conflict, unreadable): stop here
        return [str(error)], conditions
    if problems:
        return problems, conditions
    refusal = runtime.refusal(runtime.git_state(), allow_dirty=False)
    problems = ([refusal] if refusal else []) + committed(PLAN)
    if problems:
        return problems, conditions
    source, conditions["source_check"] = source_problems()
    problems += source
    clone, conditions["init_clone_files"] = init_clone_problems()
    problems += clone
    for root in (WORKTREE, REPO):
        if (root / item["run_dir"]).exists():
            problems.append(f"{root / item['run_dir']} already exists")
    if Path(sys.executable).resolve() != (REPO / INTERPRETER).resolve():
        problems.append(f"running under {sys.executable}, not {REPO / INTERPRETER}")
    queries = (("steam_running", steam_running), ("game_processes", rc.all_game_processes),
               ("other_experiments", other_experiments))
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
    """Why a finished attempt does not count as a donor (donors.historical_training.ok), and its fault report.
    Unexpected structure raises; the caller records it as a failed attempt."""
    try:
        manifest = rc.validate_run(run, f"seed {seed}")
    except rc.Invalid as error:
        return [f"incomplete run: {error}"], {}
    rows = rc._rows_jsonl(run / "evaluations.jsonl")
    reference = read_json(original / "manifest.json")
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


def start_attempt(record: dict) -> Path:
    """Reserve the attempt's name with an exclusively created started marker; returns the base path (no suffix)."""
    RECORDS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for number in range(1, 100):
        base = RECORDS / (f"attempt-{stamp}" + (f"-{number}" if number > 1 else ""))
        try:
            with open(f"{base}{STARTED}", "x", encoding="utf-8") as handle:
                handle.write(json.dumps({"entry": record["entry"], "started": datetime.now().isoformat(),
                                         "command": record["command"]}, indent=2) + "\n")
            return base
        except FileExistsError:
            continue
    raise FileExistsError(f"no free attempt name for attempt-{stamp}")


def _tail(path: Path, lines: int) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-lines:]
    except OSError:
        return []


def finish_attempt(base: Path, record: dict) -> str:
    """Write the attempt's final record exclusively, with its log files; returns the outcome."""
    logs = {"stdout.txt": Path(f"{base}.stdout.txt"), "stderr.txt": Path(f"{base}.stderr.txt")}
    record.update(log_files={k: v.as_posix() for k, v in logs.items()}, stdout_tail=_tail(logs["stdout.txt"], 5),
                  stderr_tail=_tail(logs["stderr.txt"], 8), outcome="failed" if record["problems"] else "ok")
    with open(f"{base}.json", "x", encoding="utf-8") as handle:
        handle.write(json.dumps(record, indent=2, default=str) + "\n")
    print(f"seed {record['entry']['seed']} attempt {record['entry']['attempt']}: {record['outcome'].upper()}"
          + (f" ({'; '.join(map(str, record['problems']))})" if record["problems"] else f" in {record['seconds']} s"))
    print(f"Record: {base}.json")
    return record["outcome"]


def postprocess(record: dict, item: dict) -> None:
    """Judge a finished child and copy an ok run here; fills record['problems']. Unexpected errors propagate."""
    problems = record["problems"]
    if record.get("exit_code") != 0:
        problems.append(f"exit status {record.get('exit_code')}")
    after = worktree_problems(WORKTREE, HISTORICAL_COMMIT, HISTORICAL_TREE)[0]
    problems += [f"after the run: {p}" for p in after]
    run = WORKTREE / item["run_dir"]
    if not run.is_dir():
        problems.append(f"{run} was not created")
        return
    guard = fresh_sets.paths_problems([str(run)], fresh=FRESH)  # the outputs, before they are read
    if guard:
        problems += guard
        return
    record["run_files"] = file_hashes(run)
    found, record["faults"] = outcome_problems(run, item["seed"])
    problems += found
    if not problems:
        record["copy_target"] = _relative(REPO / item["run_dir"])  # names a partial copy if copying is interrupted
        record["copy"] = copy_run(run, REPO / item["run_dir"])


def run_entry(plan: dict, item: dict, declaration: dict, dry_run: bool) -> str:
    """One attempt (or its preflight) with its record: 'ok', 'failed', 'refused' or 'dry-run'."""
    command = [str(REPO / INTERPRETER), *item["command"]]
    problems, conditions = launch_problems(plan, item, declaration, preflight=dry_run)
    if problems:  # recorded from what is already known: no Git query, no further plan or declaration read
        refused = {"entry": item, "command": command, "plan": {"path": _relative(PLAN), "revision": plan.get("revision")},
                   "conditions": conditions, "launch_problems": problems, "dry_run": dry_run}
        print("Refused:\n  " + "\n  ".join(problems))
        print(f"Record: {write_record(refused, 'refused')}")
        return "refused"
    record = {"executor": executor_identity(), "source": {"commit": HISTORICAL_COMMIT, "tree": HISTORICAL_TREE},
              "plan": {"path": _relative(PLAN), "text_sha256": fresh_sets.text_sha256(PLAN),
                       "revision": plan.get("revision")},
              "entry": item, "worktree": WORKTREE.as_posix(), "cwd": WORKTREE.as_posix(), "command": command,
              "declaration_text_sha256": fresh_sets.text_sha256(DECLARATION), "limit_minutes": LIMIT_MINUTES,
              "conditions": conditions, "launch_problems": problems, "dry_run": dry_run}
    env = {**rc.stripped_env(dict(os.environ)), **NO_BYTECODE}
    record.update(thread_variables_removed=rc.thread_variables(dict(os.environ)), threads=rc.thread_probe(env),
                  environment_set=NO_BYTECODE)
    if dry_run:
        print(f"Dry run passed: seed {item['seed']} attempt {item['attempt']}: {' '.join(item['command'])}")
        print(f"Record: {write_record(record, 'dry-run')}")
        return "dry-run"
    record["problems"] = []
    base = start_attempt(record)
    child_running, started = False, time.time()
    try:
        with open(f"{base}.stdout.txt", "x", encoding="utf-8") as out, \
                open(f"{base}.stderr.txt", "x", encoding="utf-8") as err:
            child_running = True
            try:
                done = subprocess.run(command, cwd=WORKTREE, env=env, stdout=out, stderr=err, text=True,
                                      timeout=LIMIT_MINUTES * 60)
                record["exit_code"] = done.returncode
            except subprocess.TimeoutExpired:
                record.update(exit_code=None, timed_out=True, game_cleanup=rc.cleanup_own_game(GAME))
            child_running = False
        record["seconds"] = round(time.time() - started)
        postprocess(record, item)
    except BaseException as error:  # any failure or interruption still leaves the attempt's record and logs
        record["problems"].append(f"{type(error).__name__}: {error}")
        record.setdefault("seconds", round(time.time() - started))
        if child_running:
            record["interrupted_child"] = True
            record["game_cleanup"] = rc.cleanup_own_game(GAME)
        finish_attempt(base, record)
        if not isinstance(error, Exception):
            raise  # stopped by hand: the record is written, then the interruption goes on
        return "failed"
    return finish_attempt(base, record)


def run_action(dry_run: bool, limit: int | None) -> int:
    try:
        problems = identity_problems([DECLARATION, PLAN])
        if problems:
            raise Refused("; ".join(problems))
        if not PLAN.exists():
            raise Refused(f"{_relative(PLAN)} does not exist; run plan first")
        declaration, plan = load_declaration(), read_json(PLAN)
        if dry_run:
            statuses = attempt_statuses(RECORDS)
            pending = [item for item in plan["entries"] if (item["seed"], item["attempt"]) not in statuses]
            if not pending:
                print("Nothing to run: every planned attempt has a record.")
                return 0
            results = [run_entry(plan, item, declaration, dry_run=True) for item in pending]
            print(f"Preflight: {results.count('dry-run')} of {len(results)} pending entries passed.")
            return 0 if all(result == "dry-run" for result in results) else 1
        count = 0
        while limit is None or count < limit:
            item = next_entry(plan, RECORDS)  # the records are read again before every launch
            if item is None:
                print("Nothing to run: every planned attempt has a record.")
                return 0
            count += 1
            if run_entry(plan, item, declaration, dry_run=False) != "ok":
                return 1
        return 0
    except (Refused, OSError, ValueError) as error:
        print(f"Refused: {error}")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--check", action="store_true")
    plan.add_argument("--rerun", type=int)
    plan.add_argument("--replace", type=int)
    plan.add_argument("--reason")
    plan.add_argument("--rebind-amendment-3", action="store_true",
                      help="record amendment 3's declaration hash as a plan revision (one time)")
    sub.add_parser("prepare")
    sub.add_parser("check")
    run = sub.add_parser("run")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--max", type=int, help="run at most this many attempts")
    args = parser.parse_args()
    if args.action == "plan" and args.rebind_amendment_3:
        return rebind_action()
    if args.action == "plan":
        return plan_action(args.check, args.rerun, args.replace, args.reason)
    if args.action == "prepare":
        return prepare_action()
    if args.action == "check":
        return check_action()
    return run_action(args.dry_run, args.max)


if __name__ == "__main__":
    raise SystemExit(main())
