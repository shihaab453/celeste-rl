"""The retention confirmation's stage 1 plans after the donors (config/retention-confirmation.json, amendment 3).

Each round runs four committed runner campaigns, then an inclusion record; every file written here is new and immutable
(an identical rerun reports what exists; changed inputs give a linked new version):

    .venv-rl/Scripts/python.exe scripts/make_confirmation_stage1_plans.py advance donor-v1 --round 0 [--stop-time T]
    .venv-rl/Scripts/python.exe scripts/make_confirmation_stage1_plans.py advance record --round 0 [--stop-time T]
    .venv-rl/Scripts/python.exe scripts/make_confirmation_stage1_plans.py advance copy --round 0 [--stop-time T]
    .venv-rl/Scripts/python.exe scripts/make_confirmation_stage1_plans.py advance copy-v1 --round 0 [--stop-time T]
    .venv-rl/Scripts/python.exe scripts/make_confirmation_stage1_plans.py inclusion --round 0
    .venv-rl/Scripts/python.exe scripts/make_confirmation_stage1_plans.py preflight <plan> --approved <commit>

advance PHASE --round N does the next step of that phase: it writes plan version 1 for every item the phase needs
(config/campaign-confirmation-stage1-<phase>-r<N>-v1.json); once that plan has run, it audits the campaign and either
writes the phase completion record (runs/confirmation/stage1/<phase>-r<N>-complete.json: every item mapped to exactly
one valid outcome, every failed attempt kept) or a recovery version v<m+1> with the same commands for exactly the
items still without a valid outcome, pinning the earlier plans, their audited summaries and logs and the outcomes
already accepted. It never repeats a valid result. --stop-time (an owner-chosen session cutoff, ISO date and time) is
written into a new plan version; the runner skips an entry whose full limit would pass it, and a skipped entry is
recovered in the next version, which may carry a new cutoff. Phases and their items:
  donor-v1  the declared v1 inclusion evaluation of each donor of the round (round 0: all eight original donors,
            seeds 7 to 14; round N > 0: the replacements proposed by inclusion-r<N-1>, once trained);
  record    the declared recording, for donors whose donor margin passes;
  copy      the declared copy command (seed 10 + j), for donors with an accepted recording, one at a time;
  copy-v1   the declared v1 evaluation of each accepted copy.
The margin is route_macro_success_rate - 0.142 (the literal floor) rounded to 6 decimals, passing at >= 0.20.

Nothing stored is trusted as it is: an existing completion or inclusion record is used only if a reconstruction from
the pinned campaigns (re-validating and re-hashing every artifact) gives exactly the same record, and every executed
plan version must still be what this generator derives. Historical donors come only from the executor's validated
evidence (scripts/confirmation_historical_donors.py historical_evidence: the committed plan chain, attempt
membership and sequence, one verified ok attempt per seed).

inclusion --round N (after every phase of round N is complete) writes runs/confirmation/stage1/inclusion-r<N>.json:
per j the active donor, its donor and copy aggregates and margins, qualified or excluded, and the replacements
amendment 3 allocates (donor-line exclusions, then copy-line exclusions, each by ascending j; seeds 15 then 16 from one
budget; a proposed seed is reserved). Proposals go to the owner and the historical donor executor (plan --replace);
nothing is trained or authorized here. Only aggregates are printed.

A campaign is accepted only if it has provably ended (its log says "campaign finished" and its summary holds every
planned entry once); its summary carries the plan's text hash, 10 threads and the plan's game copies; the plan and
declaration at the summary's commit are the pinned ones; and a preflight receipt for this plan, written before the
campaign at that same commit, names an approved commit whose control code equals the campaign commit's. A campaign
that is still running or was interrupted, an entry started without an outcome, stale starts, two executions of one
version or an evidence conflict stop for review. Technical failures (a status other than ok, including a cutoff skip;
missing, malformed or invalid artifacts; wrong counts; invalid provenance) are recovered; they never allocate a
replacement. Recordings are bound by the single path their own successful runner attempt announced and validated by
play.json and the full recorder array schema; copies by the runner's captured Results line and results.json, whose
consumed-input audit must equal the pinned inputs.

preflight PLAN --approved COMMIT (before every real session): the plan is committed, unchanged and on the remote
branch; COMMIT (the implementation commit the review approved) is an ancestor of HEAD and on the remote branch; the
control code (the import closures of the runner, this generator, the executor, the recipe-check helpers and the
code-check utility) and the declaration at HEAD equal COMMIT's, with no working change; the plan equals what this
generator derives now from its predecessors (which re-validates the whole predecessor chain and every leaf pin); no
campaign of the plan has run; the stop time, if any, still leaves room for one entry; the three declared stage 1 code
checks pass; the exact runner command (game copies, ports, 10 threads) passes --dry-run; torch's threads are probed in
the runner's child environment. The receipt goes to runs/confirmation/stage1/preflight-<plan>-<stamp>.json.

Fresh sets: every file read or hashed here is identity-checked first (the Room 2 task and its source route before the
task loader runs); commands must pass the runner's own fresh-set check for stage "stage1" (the Room 2 v2 file only in
the copy's --heldout slot); in parsed records only the declared roles may name it (the copy results' args.heldout and
manifests.heldout, the copy plans' and summaries' commands, the copy plan's room2_v2 input), as the declared path,
never followed; fresh_sets.verify_identity is not called.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from celeste_rl import fresh_sets  # noqa: E402
from celeste_rl.heldout import validate_manifest  # noqa: E402
from celeste_rl.tasks import resolve_task_definition, task_identity  # noqa: E402
from scripts import confirmation_historical_donors as hx  # noqa: E402
from scripts import confirmation_recipe_check as rc  # noqa: E402
from scripts import run_overnight as ro  # noqa: E402

Refused = hx.Refused
DECLARATION = hx.DECLARATION
CODE_ROOT = Path(__file__).resolve().parents[1]  # code-side configuration (task definitions), never run data
STAGE1 = REPO / "runs" / "confirmation" / "stage1"
CAMPAIGNS = REPO / "runs" / "campaign"
CONFIG = REPO / "config"
PHASES = ("donor-v1", "record", "copy", "copy-v1")
EVALUATION_PHASES = ("donor-v1", "copy-v1")
FLOOR, LINE, DECIMALS = 0.142, 0.20, 6
REPLACEMENT_SEEDS = (15, 16)
ORIGINAL_SEEDS = tuple(range(7, 15))
V1 = "config/heldout_starts.json"
V1_SEED, RECORD_SEED, EPISODES = 20260920, 20260926, 25
STARTS = "canonical only; no held-out state is read"  # the recorder's marker
# The recorder's arrays: trailing shape and dtype (runs/policy-play/chapter-1-room-1/20260926-164255/dataset.npz).
RECORDING_ARRAYS = {"actions": ((24,), "|i1"), "trajectory": ((), "<i8"), "obs_player": ((4, 75), "<f4"),
                    "obs_actions": ((4, 24), "|i1"), "obs_history_valid": ((4,), "|i1"),
                    "obs_grid": ((11, 32, 32), "|u1"), "obs_context": ((2,), "<f4")}
LIMITS = {"donor-v1": 30, "record": 60, "copy": 60, "copy-v1": 30}
THREADS = 10
INTERPRETER = REPO / ".venv-rl" / "Scripts" / "python.exe"
GAME_COPIES = (("C:/Projects/celeste-research-scratch/game-probe", 32279, 32280),
               ("C:/Projects/celeste-research-scratch/game-copy-2", 32289, 32290),
               ("C:/Projects/celeste-research-scratch/game-copy-3", 32299, 32300))
ROOM2_V2 = fresh_sets.FRESH["room2"].path  # a declared constant: never opened here
ROOM2_V2_MANIFEST_SHA256 = "51ea319af3b65f50f516da33f09cf6b02aca89ef0c10778ca0080c4f895909eb"  # the declaration's pin
# The file hash the checked clone audits; the trusted audit pin from the pilots' copies on this machine
# (runs/clone/chapter-1-room-2/20260927-205018/results.json manifests.heldout_file_sha256). Never computed here.
ROOM2_V2_FILE_SHA256 = "297d7d6cd0ae84c55f3efb28aa841f6c13551a271f11d5ebc81e899bb077a0e7"
ROOM2_TASK = "config/room2.json"
COPY_DATASET = "runs/clone/chapter-1-room-2/20260924-014351/dataset.npz"
COPY_DEMONSTRATIONS = "config/demonstrations-room2-v2.json"
CODE_CHECKS = {
    "recording": ["scripts/check_evaluation_code.py", "--entry", "scripts/record_policy_play.py", "--baseline",
                  "d1e5794", "--approved", "44ffab5"],
    "copying": ["scripts/check_evaluation_code.py", "--entry", "scripts/clone_room1.py", "--baseline", "5981e37",
                "--approved", "44ffab5"],
    "evaluation": ["scripts/check_evaluation_code.py", "--baseline", "f26914d", "--approved", "44ffab5"]}
CONTROL_ENTRIES = ["scripts/run_overnight.py", "scripts/make_confirmation_stage1_plans.py",
                   "scripts/confirmation_historical_donors.py", "scripts/confirmation_recipe_check.py",
                   "scripts/check_evaluation_code.py"]
EVALUATION_FOLDER = "runs/heldout-evaluation/"
RECORDING_FOLDER = "runs/policy-play/chapter-1-room-1/"
CLONE_FOLDER = "runs/clone/chapter-1-room-2/"
ARTIFACT_ERRORS = (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, EOFError,
                   zipfile.BadZipFile)  # ordinary artifact faults: a technical failure, recovered


# ------------------------------------------------------------------------------- one action's cache, readers, hashes

_CACHE: dict | None = None  # within one action (advance, inclusion, preflight) each check runs once


class session:
    """One action's cache: files do not change during an action, so each derivation and hash is computed once."""

    def __enter__(self):
        global _CACHE
        self.outer, _CACHE = _CACHE, ({} if _CACHE is None else _CACHE)
        return self

    def __exit__(self, *exc):
        global _CACHE
        _CACHE = self.outer


def cached(key: tuple, compute):
    if _CACHE is None:
        return compute()
    if key not in _CACHE:
        _CACHE[key] = compute()
    return copy.deepcopy(_CACHE[key])


def rel(path) -> str:
    """A repository-relative path with forward slashes (as recorded), or the absolute path if outside."""
    path = Path(path)
    try:
        return (path if not path.is_absolute() else path.resolve().relative_to(REPO.resolve())).as_posix()
    except ValueError:
        return path.as_posix()


def norm(text) -> str:
    return str(text).replace("\\", "/")


def _guard(path: Path) -> None:
    problems = hx.identity_problems([path])
    if problems:
        raise Refused("; ".join(problems))


def guarded_hash(path: Path) -> str:
    """The sha256 of a file, after the fresh-set identity check."""
    def compute():
        _guard(path)
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return cached(("hash", os.path.normcase(str(Path(path).resolve()))), compute)


def text_hash(path: Path) -> str:
    """The line-ending-safe text hash of a file, after the fresh-set identity check."""
    _guard(path)
    return fresh_sets.text_sha256(path)


def pin(path: Path) -> dict:
    return {"path": rel(path), "text_sha256": text_hash(path)}


def fresh_mentions(value, allowed: dict[tuple, str], path: tuple = ()) -> list[str]:
    """Strings in parsed metadata that name a fresh set outside their allowed role. allowed maps a key path
    (list positions as '*') to the only value it may hold (compared with forward slashes); nothing is followed."""
    problems = []
    if isinstance(value, dict):
        for key, item in value.items():
            problems += fresh_mentions(item, allowed, path + (key,))
    elif isinstance(value, list):
        for item in value:
            problems += fresh_mentions(item, allowed, path + ("*",))
    elif isinstance(value, str) and fresh_sets.mentions_fresh_set(value):
        if allowed.get(path) != norm(value):
            problems.append(f"{'.'.join(map(str, path))} names a fresh set outside its allowed role")
    return problems


def read_meta(path: Path, allowed: dict[tuple, str] | None = None):
    """A control record (plan, summary, completion, inclusion, receipt): identity-checked, read, and its strings
    checked by role. A violation stops for review."""
    value = hx.read_json(path)
    problems = fresh_mentions(value, allowed or {})
    if problems:
        raise Refused(f"{rel(path)}: " + "; ".join(problems))
    return value


def plan_roles(phase: str) -> dict:
    return ({("runs", "*", "command", "*"): ROOM2_V2, ("inputs", "room2_v2", "path"): ROOM2_V2}
            if phase == "copy" else {})


def summary_roles(phase: str) -> dict:
    return ({("results", "*", "command", "*"): ROOM2_V2, ("results", "*", "attempts", "*", "command", "*"): ROOM2_V2}
            if phase == "copy" else {})


def room1_identity() -> dict:
    return task_identity(resolve_task_definition(None))


def room2_identity() -> dict:
    """The Room 2 task identity; the task file and its source route are identity-checked before the loader."""
    task = CODE_ROOT / ROOM2_TASK
    definition = read_meta(task)
    _guard(CODE_ROOT / definition["source_route"])
    return task_identity(resolve_task_definition(task))


def hard_stop(problems: list[str], where: str) -> None:
    """A fresh set named outside its role is never a technical failure to rerun: stop for review."""
    if problems:
        raise Refused(f"{where}: " + "; ".join(problems) + ": stop for review")


def margin(value: float) -> float:
    """route-macro minus the literal floor, from the unrounded aggregate, rounded to 6 decimals (amendment 3)."""
    return round(value - FLOOR, DECIMALS)


def passes(value: float) -> bool:
    return margin(value) >= LINE


def write_new(path: Path, value: dict) -> str:
    """Create a JSON file exclusively. An identical existing file is the same result; a different one refuses."""
    text = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        if hx.same_json(hx.read_json(path), json.loads(text)):
            return "exists"
        raise Refused(f"{rel(path)} exists with different content; versions are never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as handle:
        handle.write(text)
    return "written"


# ----------------------------------------------------------------------------------------- declaration and roster

def load_declaration() -> dict:
    declaration = hx.read_json(DECLARATION)
    numbered = [a for a in declaration.get("amendments", []) if isinstance(a, dict) and a.get("number") == 3]
    if len(numbered) != 1 or hx.canonical_sha256(numbered[0]) != hx.AMENDMENT_3_SHA256:
        raise Refused("the declaration does not carry the reviewed amendment 3")
    return declaration


def templates(declaration: dict) -> dict[str, str]:
    """The declared commands, read from the declaration's own text."""
    v1 = re.search(r"scripts/evaluate_heldout\.py --checkpoint <donor> --starts config/heldout_starts\.json "
                   r"--repeats 1 --seed 20260920", declaration["donors"]["inclusion"]["stage1_v1"])
    record = declaration["pipeline"]["record"].split(" [--game-dir")[0]
    copy = declaration["pipeline"]["copy"].split(" (the SD-k")[0]
    if not v1 or not record.startswith("scripts/record_policy_play.py") or not copy.startswith("scripts/clone_room1.py"):
        raise Refused("the declaration's stage 1 commands are not in their expected form")
    return {"v1": v1.group(0), "record": record, "copy": copy}


def build(template: str, values: dict[str, str]) -> list[str]:
    text = template
    for key, value in values.items():
        text = text.replace(key, str(value))
    if "<" in text or ">" in text:
        raise Refused(f"unfilled placeholder in {text}")
    return text.split()


def donor_pins(declaration: dict) -> dict[int, dict]:
    """Every trained donor: 7 and 8 by their declared pins; 9 to 16 only from the executor's validated historical
    evidence (one verified ok attempt per seed, in the committed plan chain). Each checkpoint is rechecked on disk."""
    pins = {}
    for seed in (7, 8):
        pinned = declaration["donors"]["existing"][str(seed)]
        pins[seed] = {"seed": seed, "checkpoint": pinned["checkpoint"], "sha256": pinned["sha256"],
                      "source": {"declared pin": f"donors.existing.{seed}"}}
    for seed, evidence in hx.historical_evidence().items():
        pins[seed] = {"seed": seed, "checkpoint": evidence["checkpoint"], "sha256": evidence["sha256"],
                      "source": {"attempt_record": evidence["record"],
                                 "plan_version_sha256": evidence["plan_version_sha256"]}}
    for seed, item in pins.items():
        if guarded_hash(REPO / item["checkpoint"]) != item["sha256"]:
            raise Refused(f"seed {seed}'s {item['checkpoint']} is not its pinned sha256")
    return pins


def round_donors(declaration: dict, round_: int) -> list[dict]:
    """The donors a round evaluates: round 0 the eight original donors; later rounds the trained replacements
    proposed by the previous round's (verified) inclusion record, planned in the executor with the same j."""
    pins = donor_pins(declaration)
    if round_ == 0:
        wanted = list(enumerate(ORIGINAL_SEEDS))
    else:
        proposals = verified_inclusion(round_ - 1)["proposals"]
        plan = hx.read_json(hx.PLAN)
        wanted = []
        for proposal in proposals:
            planned = [e for e in plan["entries"] if e["seed"] == proposal["seed"]
                       and f"replaces seed {proposal['replaces']}:" in str(e.get("reason", ""))]
            if not planned or planned[0]["j"] != proposal["j"]:
                raise Refused(f"seed {proposal['seed']} is not planned in the executor as the replacement of seed "
                              f"{proposal['replaces']} at j {proposal['j']}")
            wanted.append((proposal["j"], proposal["seed"]))
    missing = [seed for _, seed in wanted if seed not in pins]
    if missing:
        raise Refused(f"donors not trained ok yet: {missing}")
    return [{"j": j, "seed": seed, "checkpoint": pins[seed]["checkpoint"], "sha256": pins[seed]["sha256"],
             "donor_source": pins[seed]["source"]} for j, seed in wanted]


# ---------------------------------------------------------------------------------------------------- names

def plan_path(phase: str, round_: int, version: int) -> Path:
    return CONFIG / f"campaign-confirmation-stage1-{phase}-r{round_}-v{version}.json"


def plan_name(phase: str, round_: int, version: int) -> str:
    return f"confirmation-stage1-{phase}-r{round_}-v{version}"


def completion_path(phase: str, round_: int) -> Path:
    return STAGE1 / f"{phase}-r{round_}-complete.json"


def inclusion_path(round_: int) -> Path:
    return STAGE1 / f"inclusion-r{round_}.json"


def versions(phase: str, round_: int) -> list[int]:
    found = []
    for path in CONFIG.glob(f"campaign-confirmation-stage1-{phase}-r{round_}-v*.json"):
        match = re.fullmatch(rf"campaign-confirmation-stage1-{re.escape(phase)}-r{round_}-v(\d+)\.json", path.name)
        if match:
            found.append(int(match.group(1)))
    found.sort()
    if found != list(range(1, len(found) + 1)):
        raise Refused(f"{phase} round {round_} plan versions are not 1..n: {found}")
    return found


# ------------------------------------------------------------------------------------------------ the plans

def v1_pins() -> dict:
    """The development set the v1 evaluations consume: file hash, manifest hash and state count."""
    path = REPO / V1
    file_sha = guarded_hash(path)
    manifest = read_meta(path)
    states = len(validate_manifest(manifest, room1_identity()))
    return {"path": V1, "file_sha256": file_sha, "manifest_sha256": manifest["sha256"], "states": states}


def copy_input_pins() -> dict:
    """The declared non-fresh inputs every copy consumes, and the Room 2 v2 identities its audit must record."""
    task = read_meta(CODE_ROOT / ROOM2_TASK)
    dataset = REPO / COPY_DATASET
    sidecar = dataset.with_name("dataset.manifest.json")
    demonstrations = REPO / COPY_DEMONSTRATIONS
    demonstrations_manifest = read_meta(demonstrations)
    return {"room2_task": {"path": ROOM2_TASK, "text_sha256": text_hash(CODE_ROOT / ROOM2_TASK),
                           "identity": room2_identity()},
            "source_route": {"path": task["source_route"],
                             "sha256": guarded_hash(CODE_ROOT / task["source_route"])},
            "dataset": {"path": COPY_DATASET, "sha256": guarded_hash(dataset)},
            "dataset_sidecar": {"path": rel(sidecar), "sha256": guarded_hash(sidecar), "content": read_meta(sidecar)},
            "demonstrations": {"path": COPY_DEMONSTRATIONS, "file_sha256": guarded_hash(demonstrations),
                               "manifest_sha256": demonstrations_manifest["sha256"]},
            "room2_v2": {"path": ROOM2_V2, "manifest_sha256": ROOM2_V2_MANIFEST_SHA256,
                         "file_sha256": ROOM2_V2_FILE_SHA256,
                         "role": "the copy's refuse-only overlap check and hash audit; never opened here"}}


def donor_manifest(item: dict) -> dict:
    """The donor's training manifest, which the copy's init_from provenance names."""
    path = REPO / Path(item["checkpoint"]).parent.parent / "manifest.json"
    manifest = read_meta(path)
    return {"path": rel(path), "sha256": guarded_hash(path), "status": manifest["status"],
            "accepted_steps": manifest["accepted_steps"],
            "commits": sorted({s["provenance"]["commit"] for s in manifest["sessions"]}),
            "schema_fingerprint": manifest["sessions"][0]["provenance"]["runtime"]["schema"]["fingerprint"],
            "disabled_inputs": manifest["config"]["disabled_inputs"]}


def entry_for(phase: str, round_: int, item: dict, commands: dict[str, str]) -> dict:
    j = item["j"]
    if phase == "donor-v1":
        command = build(commands["v1"], {"<donor>": item["checkpoint"]})
    elif phase == "record":
        command = build(commands["record"], {"<donor>": item["checkpoint"], "<sha>": item["sha256"]})
    elif phase == "copy":
        command = build(commands["copy"], {"<recording>": item["recording"]["folder"], "<10+j>": 10 + j,
                                           "<donor>": item["checkpoint"], "<sha>": item["sha256"]})
    else:
        command = build(commands["v1"], {"<donor>": item["clone"]["checkpoint"]})
    ident = f"s1-r{round_}-{phase}-j{j}-seed{item['seed']}"
    return {"id": ident, "stage": f"confirmation-stage1-{phase}", "j": j, "seed": item["seed"],
            "run_dir": f"runs/evaluation/_planned-{ident}", "resumable": False, "limit_minutes": LIMITS[phase],
            "command": command}


def stop_time_problems(stop_time: str | None, phase: str, now: datetime | None = None) -> list[str]:
    """An owner cutoff must be a valid local date and time leaving room for at least one full entry limit."""
    if stop_time is None:
        return []
    try:
        at = datetime.fromisoformat(stop_time)
    except (TypeError, ValueError):
        return [f"stop time {stop_time!r} is not an ISO date and time"]
    if at.tzinfo is not None:
        return ["stop time must be local time without a zone (the runner compares it with local time)"]
    if at < (now or datetime.now()) + timedelta(minutes=LIMITS[phase]):
        return [f"stop time {stop_time} leaves no room for one {LIMITS[phase]}-minute entry"]
    return []


def build_plan(phase: str, round_: int, version: int, items: list[dict], predecessors: list[dict],
               accepted: list[dict], declaration: dict, stop_time: str | None) -> dict:
    copies = GAME_COPIES[:1] if phase == "copy" else GAME_COPIES
    commands = templates(declaration)
    plan = {"name": plan_name(phase, round_, version), "phase": phase, "round": round_, "version": version,
            "rules": "config/retention-confirmation.json amendment 3 (stage 1)",
            "declaration": pin(DECLARATION), "fresh_set_stage": "stage1",
            "runner": {"game_copies": [{"game_dir": d, "ports": [a, b]} for d, a, b in copies],
                       "threads_per_job": THREADS, "one_at_a_time": phase == "copy"},
            "predecessors": predecessors, "accepted": accepted, "items": items,
            "runs": [entry_for(phase, round_, item, commands) for item in items]}
    if stop_time is not None:
        plan["stop_time"] = datetime.fromisoformat(stop_time).isoformat(timespec="seconds")
    if phase in EVALUATION_PHASES:
        plan["inputs"] = {"v1": v1_pins()}
    if phase == "copy":
        plan["inputs"] = copy_input_pins()
    problems = ro.fresh_set_problems(plan, [Path(d) for d, _, _ in copies], fresh=hx.FRESH)
    if problems:
        raise Refused("; ".join(problems))
    return plan


# -------------------------------------------------------------------------------------------- campaign audit

def campaign_folders(name: str) -> list[Path]:
    return sorted(p for p in CAMPAIGNS.glob(f"*-{name}")
                  if p.is_dir() and re.fullmatch(rf"\d{{8}}-\d{{6}}-{re.escape(name)}", p.name))


RECEIPT_KEYS = {"plan", "approved_commit", "remote", "head", "control_files", "code_checks", "invocation", "stop_time",
                "dry_run", "threads", "audits"}


def _is_ancestor(commit: str, ref: str) -> bool:
    return subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", str(commit), str(ref)],
                          capture_output=True).returncode == 0


def receipt_problems(receipt: dict, plan: dict, plan_hash: str, commit: str) -> list[str]:
    """A complete, successful preflight receipt for this plan at this commit: its control-file set equals the
    independently derived manifest and every blob equals the campaign commit's and the approved commit's; the three
    declared code checks passed; the dry run passed; torch ran 10 threads; the full runner invocation and stop time
    are the plan's; the plan commit, the approved commit and the campaign commit are bound on the named remote."""
    if set(receipt) != RECEIPT_KEYS:
        return [f"receipt fields {sorted(receipt)} are not the preflight schema"]
    problems = []
    approved = receipt.get("approved_commit")
    plan_pin = receipt.get("plan") or {}
    files = receipt.get("control_files") or {}
    required = control_files()
    if set(files) != set(required):
        problems.append(f"control files {sorted(files)} are not the required manifest {required}")
    for name in required:
        try:
            at_commit = hx._git(REPO, "rev-parse", f"{commit}:{name}").strip()
            at_approved = hx._git(REPO, "rev-parse", f"{approved}:{name}").strip()
        except Refused:
            problems.append(f"{name} is missing at the campaign or approved commit")
            continue
        if not (at_commit == at_approved == files.get(name)):
            problems.append(f"{name} at the campaign commit is not the approved version the receipt names")
    checks = receipt.get("code_checks") or {}
    if set(checks) != set(CODE_CHECKS) or any(
            (checks.get(name) or {}).get("command") != command or (checks.get(name) or {}).get("exit_code") != 0
            for name, command in CODE_CHECKS.items()):
        problems.append("the receipt's code checks are not the three declared checks, all passing")
    if (receipt.get("dry_run") or {}).get("exit_code") != 0:
        problems.append("the receipt's dry run did not pass")
    if (receipt.get("threads") or {}).get("torch_intra_op_threads") != THREADS:
        problems.append(f"the receipt's thread probe is not {THREADS}")
    if receipt.get("invocation") != runner_invocation(REPO / plan_rel(plan), plan["phase"]):
        problems.append("the receipt's runner invocation is not the plan's")
    if receipt.get("stop_time") != plan.get("stop_time"):
        problems.append("the receipt's stop time is not the plan's")
    if plan_pin.get("path") != plan_rel(plan) or plan_pin.get("text_sha256") != plan_hash:
        problems.append("the receipt names another plan")
    plan_commit, remote = plan_pin.get("commit"), receipt.get("remote")
    try:
        committed_plan = hx._git_bytes("show", f"{plan_commit}:{plan_rel(plan)}")
        if hx._text_sha256_of(committed_plan) != plan_hash:
            problems.append("the receipt's plan commit does not hold this plan")
    except Refused:
        problems.append("the receipt's plan commit does not hold this plan")
    for what, older, newer in (("plan commit", plan_commit, commit), ("approved commit", approved, commit),
                               ("approved commit", approved, remote), ("plan commit", plan_commit, remote)):
        if not _is_ancestor(older, newer):
            problems.append(f"the receipt's {what} {str(older)[:8]} is not on {str(newer)[:40]}")
    return problems


def preflight_receipt(plan: dict, plan_hash: str, commit: str, folder: Path) -> dict:
    """The preflight receipt that authorized this campaign: for this plan text at this commit, written before the
    campaign started, complete and successful (receipt_problems). An invalid matching receipt stops for review."""
    started = folder.name[:15]
    for path in sorted(STAGE1.glob(f"preflight-{plan['name']}-*.json")):
        stamp = path.name[len(f"preflight-{plan['name']}-"):][:15]
        receipt = read_meta(path)
        if (receipt.get("plan") or {}).get("text_sha256") != plan_hash or receipt.get("head") != commit \
                or stamp >= started:
            continue
        problems = receipt_problems(receipt, plan, plan_hash, commit)
        if problems:
            raise Refused(f"{rel(path)}: " + "; ".join(problems))
        return {"path": rel(path), "sha256": guarded_hash(path), "approved_commit": receipt["approved_commit"]}
    raise Refused(f"no preflight receipt before {rel(folder)} for this plan at {str(commit)[:8]}")


def plan_rel(plan: dict) -> str:
    return rel(plan_path(plan["phase"], plan["round"], plan["version"]))


def audit(folder: Path, plan: dict, plan_file: Path) -> dict:
    """What one campaign folder of this plan shows. 'executed' only for a campaign that provably ended, with
    reviewed code, its preflight receipt and the declared scheduling."""
    log_path = folder / "campaign.log"
    if not log_path.exists():
        return {"kind": "not started", "folder": rel(folder)}
    _guard(log_path)
    log = log_path.read_text(encoding="utf-8")
    if f"campaign {plan['name']}," not in log:
        return {"kind": "not started", "folder": rel(folder)}
    if "dry run: nothing executed" in log:
        return {"kind": "dry run", "folder": rel(folder)}
    if "campaign finished:" not in log:
        raise Refused(f"{rel(folder)} has not provably ended (still running or interrupted): stop for review")
    summary_path = folder / "summary.json"
    summary = read_meta(summary_path, summary_roles(plan["phase"]))
    plan_hash = text_hash(plan_file)
    if summary.get("plan") != plan["name"] or summary.get("plan_sha256") != plan_hash:
        raise Refused(f"{rel(summary_path)} is not a summary of {rel(plan_file)} as committed")
    commit = summary.get("commit")
    committed_plan = hx._git_bytes("show", f"{commit}:{rel(plan_file)}")
    committed_declaration = hx._git_bytes("show", f"{commit}:config/retention-confirmation.json")
    if (hx._text_sha256_of(committed_plan) != plan_hash
            or not hx.same_json(hx.strict_json(committed_plan.decode("utf-8")), plan)
            or hx._text_sha256_of(committed_declaration) != plan["declaration"]["text_sha256"]):
        raise Refused(f"the plan or declaration at the campaign's commit {str(commit)[:8]} is not the pinned one")
    if summary.get("threads_per_job") != THREADS:
        raise Refused(f"{rel(summary_path)} ran at {summary.get('threads_per_job')} threads, not {THREADS}")
    ran_on = [(os.path.normcase(str(Path(c["game_dir"]).resolve())), list(c["ports"]))
              for c in (summary.get("parallel") or {}).get("copies", [])]
    planned_on = [(os.path.normcase(str(Path(c["game_dir"]).resolve())), list(c["ports"]))
                  for c in plan["runner"]["game_copies"]]
    if ran_on != planned_on:
        raise Refused(f"{rel(summary_path)} ran on {ran_on}, not the planned game copies")
    records = {}
    for record in summary.get("results", []):
        if record.get("id") in records:
            raise Refused(f"{rel(summary_path)} holds {record.get('id')} twice")
        records[record.get("id")] = record
    planned = [entry["id"] for entry in plan["runs"]]
    if set(records) != set(planned):
        raise Refused(f"{rel(summary_path)} does not hold exactly the planned entries: stop for review")
    started = set(re.findall(r"\bstart (\S+?)(?: \(resume\))?:", log))
    unresolved = sorted(started - set(records))
    if unresolved:
        raise Refused(f"entries started without an outcome: {unresolved}: stop for review")
    return {"kind": "executed", "folder": rel(folder), "commit": commit, "records": records,
            "summary": {"path": rel(summary_path), "sha256": guarded_hash(summary_path)},
            "log": {"path": rel(log_path), "sha256": guarded_hash(log_path)},
            "receipt": preflight_receipt(plan, plan_hash, commit, folder)}


# ------------------------------------------------------------------------------------ outcome validation

def _command_problems(record: dict, entry: dict, plan: dict) -> list[str]:
    command = record.get("command") or []
    copies = {os.path.normcase(str(Path(c["game_dir"]).resolve())) for c in plan["runner"]["game_copies"]}
    if (command[:-2] != entry["command"] or command[-2:-1] != ["--game-dir"]
            or os.path.normcase(command[-1]) not in copies):
        return [f"{entry['id']} ran {command}, not its planned command on a planned game copy"]
    return []


def _clean(result: dict, commit: str, what: str, attributable: bool = True) -> list[str]:
    problems = []
    if result.get("commit") != commit or result.get("uncommitted_changes") is not False or result.get("git_error"):
        problems.append(f"{what} is not from a clean tree at the campaign commit")
    if attributable and (result.get("attributable") is not True or result.get("runtime_problems") != []):
        problems.append(f"{what} is not attributable or has runtime problems")
    return problems


def _mismatches(checks: dict) -> list[str]:
    return [f"{key} is {got!r}, expected {want!r}" for key, (got, want) in checks.items() if not hx.same_json(got, want)]


def evaluation_outcome(record: dict, expected_sha: str, commit: str, pins: dict) -> tuple[dict | None, list[str]]:
    artifact = record.get("artifact") or {}
    if not artifact.get("result_file") or not norm(artifact["result_file"]).startswith(EVALUATION_FOLDER):
        return None, [f"no result in {EVALUATION_FOLDER}: {artifact.get('problem')}"]
    result_file = REPO / artifact["result_file"]
    result = hx.read_json(result_file)
    # Hard stops first, from the readable result itself, before any companion file or artifact fault is weighed.
    hard_stop(fresh_mentions(result, {}), artifact["result_file"])
    if result.get("stale_starts") != 0:
        raise Refused(f"{artifact['result_file']}: {result.get('stale_starts')!r} stale starts: stop for review")
    if artifact.get("problem"):
        return None, [f"the runner found the artifact faulty: {artifact['problem']}"]
    problems = []
    if guarded_hash(result_file) != artifact.get("result_sha256"):
        problems.append("result file changed since the runner pinned it")
    episodes = artifact.get("episodes_file")
    if not episodes or guarded_hash(REPO / episodes) != artifact.get("episodes_sha256"):
        problems.append("episode file missing or changed since the runner pinned it")
    problems += _mismatches({key: (result.get(key), value) for key, value in {
        "checkpoint_sha256": expected_sha, "heldout_file_sha256": pins["file_sha256"],
        "heldout_sha256": pins["manifest_sha256"], "repeats": 1, "deterministic": False,
        "evaluation_seed": V1_SEED, "task": room1_identity(), "heldout_states": pins["states"],
        "attempts": pins["states"], "episodes": pins["states"]}.items()})
    if norm(result.get("heldout_set")) != V1:
        problems.append(f"heldout_set is {result.get('heldout_set')!r}")
    problems += _clean(result, commit, "the evaluation")
    value = result.get("route_macro_success_rate")
    if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        problems.append(f"route_macro_success_rate is {value!r}")
    if problems:
        return None, problems
    return {"result_file": norm(artifact["result_file"]), "result_sha256": artifact["result_sha256"],
            "episodes_file": norm(episodes), "episodes_sha256": artifact["episodes_sha256"],
            "route_macro_success_rate": value, "margin": margin(value), "passes": passes(value)}, []


def recording_outcome(record: dict, item: dict, commit: str) -> tuple[dict | None, list[str]]:
    import numpy as np
    announced = record.get("announced")
    if not isinstance(announced, list) or len(announced) != 1 or not announced[0].strip():
        return None, [f"the recording entry announced {announced!r}, not exactly one folder"]
    folder = Path(announced[0].strip())
    folder = folder if folder.is_absolute() else REPO / folder
    relative = rel(folder)
    if not re.fullmatch(re.escape(RECORDING_FOLDER) + r"\d{8}-\d{6}", relative):
        return None, [f"announced folder {announced[0]!r} is not a recording folder"]
    guard = fresh_sets.paths_problems([str(folder)], fresh=hx.FRESH)
    if guard:
        raise Refused("; ".join(guard))
    play = hx.read_json(folder / "play.json")
    hard_stop(fresh_mentions(play, {}), rel(folder / "play.json"))
    problems = []
    problems += _mismatches({key: (play.get(key), value) for key, value in {
        "task": room1_identity(), "checkpoint_sha256": item["sha256"], "seed": RECORD_SEED, "episodes": EPISODES,
        "starts": STARTS, "dataset": "dataset.npz"}.items()})
    if norm(play.get("checkpoint")) != item["checkpoint"]:
        problems.append(f"checkpoint is {play.get('checkpoint')!r}")
    problems += _clean(play, commit, "the recording")
    dataset_sha = guarded_hash(folder / "dataset.npz")
    if play.get("dataset_sha256") != dataset_sha:
        problems.append("dataset.npz is not the recorded dataset_sha256")
    summaries = play.get("summaries") or []
    endings = play.get("endings") or {}
    if (sum(endings.values()) != EPISODES or len(summaries) != EPISODES
            or [s.get("episode") for s in summaries] != list(range(EPISODES))):
        problems.append("the recording does not hold 25 complete episodes")
    with np.load(folder / "dataset.npz", allow_pickle=False) as data:
        arrays = {name: (data[name].dtype.str, data[name].shape) for name in data.files}
        trajectory = data["trajectory"] if "trajectory" in data.files else np.zeros(0, int)
        ids, counts = np.unique(trajectory, return_counts=True)
    if set(arrays) != set(RECORDING_ARRAYS):
        problems.append(f"the dataset arrays are {sorted(arrays)}, not the recorder's {sorted(RECORDING_ARRAYS)}")
    else:
        for name, (tail, dtype) in RECORDING_ARRAYS.items():
            got_dtype, shape = arrays[name]
            if got_dtype != dtype or shape[1:] != tail or shape[0] != play.get("frames"):
                problems.append(f"array {name} is {got_dtype}{shape}, expected {dtype}(frames, {tail})")
    if ids.tolist() != list(range(EPISODES)) or counts.tolist() != [s.get("frames") for s in summaries]:
        problems.append("the trajectory ids are not exactly episodes 0 to 24 with their recorded frame counts")
    if problems:
        return None, problems
    return {"folder": relative, "play_sha256": guarded_hash(folder / "play.json"), "dataset_sha256": dataset_sha,
            "frames": play["frames"], "commit": play["commit"]}, []


def _mix_play_paths(text: str) -> list[str]:
    return [norm(p) for p in re.findall(r"'([^']*)'", str(text))]


def copy_outcome(record: dict, item: dict, commit: str, plan_entry: dict, inputs: dict) -> tuple[dict | None, list[str]]:
    artifact = record.get("artifact") or {}
    if not artifact or artifact.get("problem"):
        return None, [f"no valid result artifact: {artifact.get('problem')}"]
    if not re.fullmatch(re.escape(CLONE_FOLDER) + r"\d{8}-\d{6}/results\.json", norm(artifact["result_file"])):
        return None, [f"result {artifact['result_file']!r} is not a copy folder"]
    result_file = REPO / artifact["result_file"]
    result = hx.read_json(result_file)
    hard_stop(fresh_mentions(result, {("args", "heldout"): ROOM2_V2, ("manifests", "heldout"): ROOM2_V2}),
              artifact["result_file"])
    problems = []
    if guarded_hash(result_file) != artifact.get("result_sha256"):
        problems.append("result file changed since the runner pinned it")
    problems += _clean(result, commit, "the copy", attributable=False)
    command = plan_entry["command"]
    declared = {command[i][2:].replace("-", "_"): command[i + 1] for i in range(1, len(command) - 1)
                if command[i].startswith("--") and not command[i + 1].startswith("--")}
    args = result.get("args") or {}
    expected_args = {**{k: v for k, v in declared.items() if k != "mix_play"}, "routes_only": "True",
                     "no_play": "True", "mix_room": "None", "allow_dirty": "False", "allow_runtime_mismatch": "False"}
    problems += [f"args.{key} is {args.get(key)!r}, expected {value!r}" for key, value in expected_args.items()
                 if norm(args.get(key)) != norm(value)]
    recording = item["recording"]
    if _mix_play_paths(args.get("mix_play")) != ["default", recording["folder"]]:
        problems.append(f"args.mix_play is {args.get('mix_play')!r}")
    play = {"kind": "donor play", "play": recording["folder"], "dataset_sha256": recording["dataset_sha256"],
            "episodes": EPISODES, "seed": RECORD_SEED, "recorded_at": recording["commit"]}
    mix, manifests = result.get("mix") or {}, result.get("manifests") or {}
    provenance = result.get("provenance") or []
    init_from = result.get("init_from") or {}
    donor = item["donor_manifest"]
    sidecar = inputs["dataset_sidecar"]["content"]
    dataset_audit = provenance[0] if provenance and isinstance(provenance[0], dict) else {}
    problems += _mismatches({
        "init_from.path": (norm(init_from.get("path")), item["checkpoint"]),
        "init_from.sha256": (init_from.get("sha256"), item["sha256"]),
        "init_from.provenance": (init_from.get("provenance"), {
            "kind": "training run", "manifest": donor["path"], "status": donor["status"], "commits": donor["commits"],
            "schema_fingerprint": donor["schema_fingerprint"], "disabled_inputs": donor["disabled_inputs"],
            "accepted_steps": donor["accepted_steps"]}),
        "mix.room_weighting": (mix.get("room_weighting"), "equal"), "mix.targets": (mix.get("targets"), "donor"),
        "mix.states": (mix.get("states"), "donor play"), "mix.task": (mix.get("task"), room1_identity()),
        "mix.play": ({**(mix.get("play") or {}), "play": norm((mix.get("play") or {}).get("play"))}, play),
        "task": (result.get("task"), inputs["room2_task"]["identity"]),
        "cloning.epochs": ((result.get("cloning") or {}).get("epochs"), 300),
        "manifests": ({k: norm(v) if k in ("demonstrations", "heldout") else v for k, v in manifests.items()}, {
            "demonstrations": inputs["demonstrations"]["path"],
            "demonstrations_sha256": inputs["demonstrations"]["manifest_sha256"],
            "demonstrations_file_sha256": inputs["demonstrations"]["file_sha256"],
            "heldout": ROOM2_V2, "heldout_sha256": inputs["room2_v2"]["manifest_sha256"],
            "heldout_file_sha256": inputs["room2_v2"]["file_sha256"]}),
        "provenance length": (len(provenance), 2),
        "provenance[0] (the dataset audit, as its pinned sidecar)": (
            {k: v for k, v in dataset_audit.items() if k != "dataset"}, sidecar),
        "provenance[0].dataset": (norm(dataset_audit.get("dataset")), inputs["dataset"]["path"]),
        "provenance[0].dataset_sha256": (dataset_audit.get("dataset_sha256"), inputs["dataset"]["sha256"]),
        "provenance[1]": ({**(provenance[1] if len(provenance) > 1 else {}),
                           "play": norm((provenance[1] if len(provenance) > 1 else {}).get("play"))}, play)})
    history = (result.get("cloning") or {}).get("history") or []
    if not history or history[-1].get("epoch") != 299:
        problems.append("the fit did not complete 300 epochs")
    clone = result_file.parent / "cloned.zip"
    if not clone.is_file():
        problems.append("cloned.zip is missing")
    if problems:
        return None, problems
    return {"folder": rel(result_file.parent), "results_sha256": artifact["result_sha256"],
            "checkpoint": rel(clone), "sha256": guarded_hash(clone)}, []


def outcome_for(phase: str, record: dict, entry: dict, item: dict, commit: str, plan: dict) -> tuple[dict | None, list]:
    """One entry's outcome, or the technical failure. Ordinary artifact faults are failures to recover; fresh-set
    refusals, stale starts and evidence conflicts (Refused) stop for review."""
    statuses = [record.get("status")] + [a.get("status") for a in record.get("attempts") or []]
    if "refused_fresh_set" in statuses:
        raise Refused(f"{entry['id']} was refused by the runner's fresh-set guard: stop for review")
    if record.get("status") != "ok":
        return None, [f"status {record.get('status')}"]
    problems = _command_problems(record, entry, plan)
    if problems:
        return None, problems
    try:
        if phase == "donor-v1":
            return evaluation_outcome(record, item["sha256"], commit, plan["inputs"]["v1"])
        if phase == "copy-v1":
            return evaluation_outcome(record, item["clone"]["sha256"], commit, plan["inputs"]["v1"])
        if phase == "record":
            return recording_outcome(record, item, commit)
        return copy_outcome(record, item, commit, entry, plan["inputs"])
    except Refused:
        raise
    except ARTIFACT_ERRORS as error:
        return None, [f"artifact could not be validated: {type(error).__name__}: {str(error)[:200]}"]


# ------------------------------------------------------------------------------------------- phase state

def phase_items(phase: str, round_: int, declaration: dict) -> tuple[list[dict], list[dict]]:
    """The items a phase needs in a round and the verified records they come from."""
    if phase == "donor-v1":
        return round_donors(declaration, round_), ([pin(inclusion_path(round_ - 1))] if round_ else [])
    before = PHASES[PHASES.index(phase) - 1]
    completion = verified_completion(before, round_, declaration)
    items = []
    for done in completion["items"]:
        item = {k: v for k, v in done.items() if k not in ("outcome", "evidence")}
        if before == "donor-v1":
            if not done["outcome"]["passes"]:
                continue
            item["donor_v1"] = done["outcome"]
        elif before == "record":
            item["recording"] = done["outcome"]
        elif before == "copy":
            item["clone"] = done["outcome"]
        if phase == "copy":
            item["donor_manifest"] = donor_manifest(item)
        items.append(item)
    return items, [pin(completion_path(before, round_))]


def phase_state(phase: str, round_: int, declaration: dict, upto: int | None = None) -> dict:
    """Valid outcomes and failures from the executed versions of a phase (all, or those below `upto`), every
    version checked against this generator's derivation, and the version still waiting to run, if any."""
    return cached(("state", phase, round_, upto), lambda: _phase_state(phase, round_, declaration, upto))


def _phase_state(phase: str, round_: int, declaration: dict, upto: int | None) -> dict:
    items, sources = phase_items(phase, round_, declaration)
    by_j = {item["j"]: item for item in items}
    outcomes, failures, plans, campaigns, pending = {}, [], [], [], None
    for version in versions(phase, round_):
        if upto is not None and version >= upto:
            break
        if pending is not None:
            raise Refused(f"{phase} r{round_} v{version} exists although v{pending} has not run")
        path = plan_path(phase, round_, version)
        plan = read_meta(path, plan_roles(phase))
        derived = expected_plan(phase, round_, version, declaration, plan.get("stop_time"))
        if not hx.same_json(plan, derived):
            raise Refused(f"{rel(path)} is not what this generator derives: stop for review")
        plans.append(pin(path))
        audits = [audit(folder, plan, path) for folder in campaign_folders(plan["name"])]
        executed = [a for a in audits if a["kind"] == "executed"]
        if len(executed) > 1:
            raise Refused(f"{rel(path)} ran more than once: stop for review")
        if not executed:
            pending = version
            continue
        run = executed[0]
        campaigns.append({"plan": rel(path), "campaign": run["folder"], "summary": run["summary"], "log": run["log"],
                          "receipt": run["receipt"], "commit": run["commit"]})
        for entry in plan["runs"]:
            item = by_j[entry["j"]]
            outcome, problems = outcome_for(phase, run["records"][entry["id"]], entry, item, run["commit"], plan)
            evidence = {"plan": rel(path), "campaign": run["folder"], "entry": entry["id"]}
            if outcome is None:
                failures.append({"j": entry["j"], "seed": entry["seed"], "problems": problems, **evidence})
            elif entry["j"] in outcomes:
                raise Refused(f"j {entry['j']} has two valid {phase} outcomes: stop for review")
            else:
                outcomes[entry["j"]] = {"outcome": outcome, "evidence": evidence}
    return {"items": items, "sources": sources, "outcomes": outcomes, "failures": failures, "plans": plans,
            "campaigns": campaigns, "pending": pending}


def expected_plan(phase: str, round_: int, version: int, declaration: dict, stop_time: str | None) -> dict:
    """The plan this generator writes for that version, from the state of the versions before it."""
    state = phase_state(phase, round_, declaration, upto=version)
    if state["pending"] is not None:
        raise Refused(f"{phase} r{round_} v{state['pending']} has not run")
    missing = [item for item in state["items"] if item["j"] not in state["outcomes"]]
    if version > 1 and not missing:
        raise Refused(f"{phase} r{round_} needs no version {version}")
    predecessors = state["sources"] + state["plans"] + [
        {"campaign": c["campaign"], "summary": c["summary"], "log": c["log"], "receipt": c["receipt"]}
        for c in state["campaigns"]]
    accepted = [{"j": j, **state["outcomes"][j]} for j in sorted(state["outcomes"])]
    return build_plan(phase, round_, version, missing if version > 1 else state["items"], predecessors, accepted,
                      declaration, stop_time)


def completion_record(phase: str, round_: int, declaration: dict) -> dict | None:
    """The completion record the evidence supports now, or None while items still lack a valid outcome."""
    state = phase_state(phase, round_, declaration)
    if state["pending"] is not None or any(item["j"] not in state["outcomes"] for item in state["items"]):
        return None
    return {"phase": phase, "round": round_, "declaration": pin(DECLARATION), "no_work": not state["items"],
            "sources": state["sources"], "plans": state["plans"], "campaigns": state["campaigns"],
            "failures": state["failures"],
            "items": [{**item, **state["outcomes"][item["j"]]} for item in state["items"]]}


def verified_completion(phase: str, round_: int, declaration: dict) -> dict:
    """A stored completion record, used only if the evidence reconstructs exactly it (every pin rechecked)."""
    path = completion_path(phase, round_)
    if not path.exists():
        raise Refused(f"{phase} round {round_} is not complete ({rel(path)})")

    def compute():
        stored = read_meta(path)
        if not hx.same_json(stored, completion_record(phase, round_, declaration)):
            raise Refused(f"{rel(path)} is not what its evidence reconstructs: stop for review")
        return stored
    return cached(("completion", phase, round_), compute)


def advance(phase: str, round_: int, stop_time: str | None = None) -> str:
    """The next step of a phase: its first plan, a recovery version, or its completion record."""
    with session():
        return _advance(phase, round_, stop_time)


def _advance(phase: str, round_: int, stop_time: str | None) -> str:
    declaration = load_declaration()
    run_code_checks()
    problems = stop_time_problems(stop_time, phase)
    if problems:
        raise Refused("; ".join(problems))
    completion = completion_path(phase, round_)
    if completion.exists():
        verified_completion(phase, round_, declaration)
        return f"{rel(completion)} exists and its evidence reconstructs it: the phase is complete"
    state = phase_state(phase, round_, declaration)
    if state["pending"] is not None:
        return f"{rel(plan_path(phase, round_, state['pending']))} is waiting to run"
    record = completion_record(phase, round_, declaration)
    if record is not None:
        write_new(completion, record)
        return f"{rel(completion)} written ({'no work' if record['no_work'] else len(record['items'])} items)"
    version = len(state["plans"]) + 1
    plan = expected_plan(phase, round_, version, declaration, stop_time)
    result = write_new(plan_path(phase, round_, version), plan)
    return f"{rel(plan_path(phase, round_, version))} {result} ({len(plan['runs'])} entries); commit and push it"


# ----------------------------------------------------------------------------------------------- inclusion

def inclusion_record(round_: int) -> dict:
    """The round's inclusion record and the replacements amendment 3 allocates, from verified completions."""
    declaration = load_declaration()
    completions = {phase: verified_completion(phase, round_, declaration) for phase in PHASES}
    previous = verified_inclusion(round_ - 1) if round_ else None
    slots = {int(j): copy.deepcopy(slot) for j, slot in previous["slots"].items()} if previous else {}
    reserved = list(previous["reserved_seeds"]) if previous else []
    copies = {item["j"]: item for item in completions["copy-v1"]["items"]}
    donor_excluded, copy_excluded = [], []

    def aggregate(outcome):
        return {k: outcome[k] for k in ("route_macro_success_rate", "margin", "passes")}

    for donor in completions["donor-v1"]["items"]:
        j = donor["j"]
        entry = {"seed": donor["seed"], "sha256": donor["sha256"], "donor_v1": aggregate(donor["outcome"])}
        if not donor["outcome"]["passes"]:
            entry["status"] = "excluded: donor line"
            donor_excluded.append(j)
        else:
            entry["copy_v1"] = aggregate(copies[j]["outcome"])
            entry["status"] = "qualified" if copies[j]["outcome"]["passes"] else "excluded: copy line"
            if entry["status"] != "qualified":
                copy_excluded.append(j)
        history = (slots.get(j) or {}).get("history", []) + ([slots[j]["active"]] if j in slots else [])
        slots[j] = {"active": entry, "history": history}
    proposals = []
    for j in sorted(donor_excluded) + sorted(copy_excluded):
        free = [s for s in REPLACEMENT_SEEDS if s not in reserved]
        if not free:
            slots[j]["active"]["status"] += "; no replacement left"
            continue
        reserved.append(free[0])
        proposals.append({"seed": free[0], "j": j, "replaces": slots[j]["active"]["seed"],
                          "reason": slots[j]["active"]["status"]})
    qualified = sorted(j for j, slot in slots.items() if slot["active"]["status"] == "qualified")
    if proposals:
        status = f"replacements proposed for round {round_ + 1}: train them, then advance donor-v1 --round {round_ + 1}"
    elif len(qualified) == len(ORIGINAL_SEEDS):
        status = "stage 1 complete: eight qualified donors"
    elif len(qualified) >= 6:
        status = f"stage 1 complete with {len(qualified)} qualified donors (replacements used up)"
    else:
        status = f"stop for review: {len(qualified)} qualified donors, fewer than six (donors.inclusion.end_state)"
    return {"round": round_, "declaration": pin(DECLARATION),
            "rule": "amendment 3 replacement_allocation; margin = route_macro_success_rate - 0.142 rounded to 6 "
                    "decimals, >= 0.20",
            "completions": [pin(completion_path(p, round_)) for p in PHASES],
            "previous": pin(inclusion_path(round_ - 1)) if round_ else None,
            "slots": {str(j): slots[j] for j in sorted(slots)}, "proposals": proposals, "reserved_seeds": reserved,
            "qualified": qualified, "status": status}


def verified_inclusion(round_: int) -> dict:
    path = inclusion_path(round_)
    if not path.exists():
        raise Refused(f"round {round_ + 1} needs {rel(path)}")
    def compute():
        stored = read_meta(path)
        if not hx.same_json(stored, inclusion_record(round_)):
            raise Refused(f"{rel(path)} is not what its evidence reconstructs: stop for review")
        return stored
    return cached(("inclusion", round_), compute)


def inclusion(round_: int) -> dict:
    with session():
        record = inclusion_record(round_)
        write_new(inclusion_path(round_), record)
    return record


# ------------------------------------------------------------------------------------------- checks and preflight

def run_code_checks() -> dict:
    results = {}
    for name, command in CODE_CHECKS.items():
        done = subprocess.run([sys.executable, *command], cwd=REPO, capture_output=True, text=True)
        results[name] = {"command": command, "exit_code": done.returncode,
                         "stdout_tail": done.stdout.strip().splitlines()[-3:]}
        if done.returncode != 0:
            raise Refused(f"the declared {name} code check exited {done.returncode}")
    return results


def control_files() -> list[str]:
    from scripts import check_evaluation_code
    return sorted(set(check_evaluation_code.closure(CONTROL_ENTRIES)) | {"config/retention-confirmation.json"})


def parse_plan_name(path: Path) -> tuple[str, int, int]:
    match = re.fullmatch(r"campaign-confirmation-stage1-(donor-v1|record|copy|copy-v1)-r(\d+)-v(\d+)\.json", path.name)
    if not match:
        raise Refused(f"{path.name} is not a stage 1 plan")
    return match.group(1), int(match.group(2)), int(match.group(3))


def runner_invocation(plan_file: Path, phase: str) -> list[str]:
    copies = GAME_COPIES[:1] if phase == "copy" else GAME_COPIES
    copy_args = [arg for d, a, b in copies for arg in ("--copy", f"{d}:{a}:{b}")]
    return [str(INTERPRETER), "scripts/run_overnight.py", "--plan", rel(plan_file), *copy_args,
            "--threads-per-job", str(THREADS)]


def preflight(plan_file: Path, approved: str, remote: str) -> dict:
    with session():
        return _preflight(plan_file, approved, remote)


def _preflight(plan_file: Path, approved: str, remote: str) -> dict:
    plan_file = plan_file if plan_file.is_absolute() else REPO / plan_file
    problems = hx.identity_problems([DECLARATION, plan_file]) + hx.committed(plan_file)
    if problems:
        raise Refused("; ".join(problems))
    phase, round_, version = parse_plan_name(plan_file)
    plan = read_meta(plan_file, plan_roles(phase))
    approved_commit = hx._git(REPO, "rev-parse", "--verify", f"{approved}^{{commit}}").strip()
    plan_commit = hx._git(REPO, "log", "-1", "--format=%H", "--", rel(plan_file)).strip()
    for what, commit, ref in (("approved commit", approved_commit, "HEAD"), ("approved commit", approved_commit, remote),
                              ("plan commit", plan_commit, remote)):
        if subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", commit, ref]).returncode != 0:
            raise Refused(f"the {what} {commit[:8]} is not on {ref}")
    manifest = control_files()
    blobs = {f: hx._git(REPO, "rev-parse", f"HEAD:{f}").strip() for f in manifest}
    changed = [f for f in manifest if blobs[f] != hx._git(REPO, "rev-parse", f"{approved_commit}:{f}").strip()]
    dirty = hx._git(REPO, "status", "--porcelain", "--", *manifest).strip()
    if changed or dirty:
        raise Refused(f"control code or declaration differs from the approved commit: {changed} {dirty}")
    declaration = load_declaration()
    if not hx.same_json(plan, expected_plan(phase, round_, version, declaration, plan.get("stop_time"))):
        raise Refused(f"{rel(plan_file)} is not what the approved generator derives from its predecessors")
    audits = [audit(folder, plan, plan_file) for folder in campaign_folders(plan["name"])]
    if any(a["kind"] == "executed" for a in audits):
        raise Refused(f"{rel(plan_file)} has already run; a recovery uses a new version")
    problems = stop_time_problems(plan.get("stop_time"), phase)
    if problems:
        raise Refused("; ".join(problems))
    checks = run_code_checks()
    invocation = runner_invocation(plan_file, phase)
    dry = subprocess.run([*invocation, "--dry-run"], cwd=REPO, capture_output=True, text=True)
    if dry.returncode != 0:
        raise Refused(f"the runner's dry run exited {dry.returncode}: {dry.stdout.strip()[-400:]}")
    copy = ro.parse_copy(invocation[invocation.index("--copy") + 1])
    record = {"plan": {"path": rel(plan_file), "text_sha256": text_hash(plan_file), "commit": plan_commit},
              "approved_commit": approved_commit, "remote": remote, "head": hx._git(REPO, "rev-parse", "HEAD").strip(),
              "control_files": blobs, "code_checks": checks, "invocation": invocation,
              "stop_time": plan.get("stop_time"),
              "dry_run": {"exit_code": 0, "stdout_tail": dry.stdout.strip().splitlines()[-6:]},
              "threads": rc.thread_probe(copy.env(THREADS)), "audits": audits}
    out = STAGE1 / f"preflight-{plan['name']}-{datetime.now():%Y%m%d-%H%M%S}.json"
    write_new(out, record)
    record["record"] = rel(out)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    step = sub.add_parser("advance")
    step.add_argument("phase", choices=PHASES)
    step.add_argument("--round", type=int, required=True)
    step.add_argument("--stop-time", help="owner-chosen session cutoff for a new plan version (local ISO time)")
    include = sub.add_parser("inclusion")
    include.add_argument("--round", type=int, required=True)
    check = sub.add_parser("preflight")
    check.add_argument("plan", type=Path)
    check.add_argument("--approved", required=True, help="the implementation commit the review approved")
    check.add_argument("--remote", default="origin/proposal/ppo-anchor")
    args = parser.parse_args()
    try:
        if args.action == "advance":
            print(advance(args.phase, args.round, args.stop_time))
        elif args.action == "inclusion":
            record = inclusion(args.round)
            for j, slot in record["slots"].items():
                active = slot["active"]
                donor = active["donor_v1"]
                copy = active.get("copy_v1")
                print(f"j {j} seed {active['seed']}: donor margin {donor['margin']:.6f}"
                      + (f", copy margin {copy['margin']:.6f}" if copy else "") + f" -> {active['status']}")
            print(record["status"])
            for proposal in record["proposals"]:
                print(f"proposed: seed {proposal['seed']} replaces seed {proposal['replaces']} at j {proposal['j']}")
        else:
            record = preflight(args.plan, args.approved, args.remote)
            print(f"Preflight passed. Record: {record['record']}\nLive command:\n  " + " ".join(record["invocation"]))
    except (Refused, OSError, ValueError, KeyError) as error:
        print(f"Refused: {type(error).__name__}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
