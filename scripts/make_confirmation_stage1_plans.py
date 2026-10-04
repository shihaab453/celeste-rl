"""The retention confirmation's stage 1 plans after the donors (config/retention-confirmation.json, amendment 3).

Each round runs four committed runner campaigns, then an inclusion record; every file written here is new and immutable
(an identical rerun reports what exists; changed inputs give a linked new version):

    .venv-rl/Scripts/python.exe scripts/make_confirmation_stage1_plans.py advance donor-v1 --round 0
    .venv-rl/Scripts/python.exe scripts/make_confirmation_stage1_plans.py advance record --round 0
    .venv-rl/Scripts/python.exe scripts/make_confirmation_stage1_plans.py advance copy --round 0
    .venv-rl/Scripts/python.exe scripts/make_confirmation_stage1_plans.py advance copy-v1 --round 0
    .venv-rl/Scripts/python.exe scripts/make_confirmation_stage1_plans.py inclusion --round 0
    .venv-rl/Scripts/python.exe scripts/make_confirmation_stage1_plans.py preflight <plan> --approved <commit>

advance PHASE --round N does the next step of that phase: it writes plan version 1 for every item the phase needs
(config/campaign-confirmation-stage1-<phase>-r<N>-v1.json); once that plan has run, it audits the campaign and either
writes the phase completion record (runs/confirmation/stage1/<phase>-r<N>-complete.json: every item mapped to exactly
one valid outcome, every failed attempt kept) or a recovery version v<m+1> with the same commands for exactly the
items still without a valid outcome. It never repeats a valid result. Phases and their items:
  donor-v1  the declared v1 inclusion evaluation of each donor of the round (round 0: all eight original donors,
            seeds 7 to 14; round N > 0: the replacements proposed by inclusion-r<N-1>, once trained);
  record    the declared recording, for donors whose donor margin passes;
  copy      the declared copy command (seed 10 + j), for donors with an accepted recording, one at a time;
  copy-v1   the declared v1 evaluation of each accepted copy.
The margin is route_macro_success_rate - 0.142 (the literal floor) rounded to 6 decimals, passing at >= 0.20.

inclusion --round N (after copy-v1 r<N> is complete) writes runs/confirmation/stage1/inclusion-r<N>.json: per j the
active donor, its donor and copy aggregates and margins, qualified or excluded, and the replacements amendment 3
allocates (donor-line exclusions, then copy-line exclusions, each by ascending j; seeds 15 then 16 from one budget;
a proposed seed is reserved). Replacement proposals go to the owner and the historical donor executor
(plan --replace); nothing is trained or authorized here. Only aggregates are printed.

A campaign is audited only if it has provably ended (its log says "campaign finished" and its summary holds every
planned entry); its summary must carry the plan's text hash, and the plan and declaration at the summary's commit
must be the pinned ones. A campaign that is still running or was interrupted, or an entry started without an outcome,
stops for review. Stale starts in an evaluation stop for review. Technical failures (a status other than ok, a
missing or invalid artifact, wrong counts, invalid provenance) are recovered; they never allocate a replacement.
Recordings are bound by the path their own successful runner attempt announced (scripts/run_overnight.py) and
validated by play.json and dataset.npz; copies by the runner's captured Results line and results.json.

preflight PLAN --approved COMMIT (before every real session): the plan is committed, unchanged and on the remote
branch; COMMIT (the implementation commit the review approved) is an ancestor of HEAD and on the remote branch; the
control code (the import closures of the runner, this generator, the executor and the code-check utility) and the
declaration at HEAD equal COMMIT's, with no working change; the plan equals what this generator derives now from its
predecessors; no campaign of the plan has run; the three declared stage 1 code checks pass; the exact runner command
(game copies, ports, 10 threads) passes --dry-run; torch's threads are probed in the runner's child environment. The
record goes to runs/confirmation/stage1/preflight-<plan>-<stamp>.json and the live command is printed.

Fresh sets: every file read here is identity-checked first; commands must pass the runner's own fresh-set check for
stage "stage1" (the Room 2 v2 file only in the copy's --heldout slot); in parsed metadata only the copy results'
args.heldout and manifests.heldout may name it (as the declared path, never followed); fresh_sets.verify_identity is
not called.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
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
LIMITS = {"donor-v1": 30, "record": 60, "copy": 60, "copy-v1": 30}
THREADS = 10
INTERPRETER = REPO / ".venv-rl" / "Scripts" / "python.exe"
GAME_COPIES = (("C:/Projects/celeste-research-scratch/game-probe", 32279, 32280),
               ("C:/Projects/celeste-research-scratch/game-copy-2", 32289, 32290),
               ("C:/Projects/celeste-research-scratch/game-copy-3", 32299, 32300))
ROOM2_V2 = fresh_sets.FRESH["room2"].path  # a declared constant: never opened here
ROOM2_TASK = Path(__file__).resolve().parents[1] / "config" / "room2.json"  # code-side config, not run data
ROOM2_V2_MANIFEST_SHA256 = "51ea319af3b65f50f516da33f09cf6b02aca89ef0c10778ca0080c4f895909eb"
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


# ------------------------------------------------------------------------------------------------- small helpers

def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path) -> str:
    """A repository-relative path with forward slashes (as recorded), or the absolute path if outside."""
    path = Path(path)
    try:
        return (path if not path.is_absolute() else path.resolve().relative_to(REPO.resolve())).as_posix()
    except ValueError:
        return path.as_posix()


def norm(text) -> str:
    return str(text).replace("\\", "/")


def margin(value: float) -> float:
    """route-macro minus the literal floor, from the unrounded aggregate, rounded to 6 decimals (amendment 3)."""
    return round(value - FLOOR, DECIMALS)


def passes(value: float) -> bool:
    return margin(value) >= LINE


def guarded_hash(path: Path) -> str:
    """The sha256 of a file this generator reads, after the fresh-set identity check."""
    problems = hx.identity_problems([path])
    if problems:
        raise Refused("; ".join(problems))
    return _sha256(path)


def read_json(path: Path):
    return hx.read_json(path)


def write_new(path: Path, value: dict) -> str:
    """Create a JSON file exclusively. An identical existing file is the same result; a different one refuses."""
    text = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        if hx.same_json(read_json(path), json.loads(text)):
            return "exists"
        raise Refused(f"{rel(path)} exists with different content; versions are never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as handle:
        handle.write(text)
    return "written"


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


# ----------------------------------------------------------------------------------------- declaration and roster

def load_declaration() -> dict:
    declaration = read_json(DECLARATION)
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
    """Every trained donor the experiment may use: 7 and 8 by their declared pins; 9 to 16 by their unique ok
    historical attempt (the hash its verified copy recorded), each rechecked on disk."""
    pins = {}
    for seed in (7, 8):
        pinned = declaration["donors"]["existing"][str(seed)]
        pins[seed] = {"seed": seed, "checkpoint": pinned["checkpoint"], "sha256": pinned["sha256"],
                      "source": "declared pin"}
    statuses = hx.attempt_statuses(hx.RECORDS)
    for path in sorted(hx.RECORDS.glob("attempt-*.json")):
        if path.name.endswith(hx.STARTED):
            continue
        record = read_json(path)
        entry = record["entry"]
        if record.get("outcome") != "ok" or statuses.get((entry["seed"], entry["attempt"])) != "ok":
            continue
        checkpoint = f"{entry['run_dir']}/checkpoints/latest.zip"
        pins[int(entry["seed"])] = {"seed": int(entry["seed"]), "checkpoint": checkpoint,
                                    "sha256": record["copy"]["files"]["checkpoints/latest.zip"],
                                    "source": rel(path), "attempt": entry["attempt"]}
    for seed, pin in pins.items():
        if guarded_hash(REPO / pin["checkpoint"]) != pin["sha256"]:
            raise Refused(f"seed {seed}'s {pin['checkpoint']} is not its pinned sha256")
    return pins


def round_donors(declaration: dict, round_: int) -> list[dict]:
    """The donors a round evaluates: round 0 the eight original donors; later rounds the trained replacements
    proposed by the previous round's inclusion record (each must be in the executor plan with the same j)."""
    pins = donor_pins(declaration)
    if round_ == 0:
        wanted = [(j, seed) for j, seed in enumerate(ORIGINAL_SEEDS)]
    else:
        previous = inclusion_path(round_ - 1)
        if not previous.exists():
            raise Refused(f"round {round_} needs {rel(previous)}")
        proposals = read_json(previous)["proposals"]
        plan = read_json(hx.PLAN)
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
    return [{"j": j, "seed": seed, "checkpoint": pins[seed]["checkpoint"], "sha256": pins[seed]["sha256"]}
            for j, seed in wanted]


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
    manifest = read_json(path)
    states = len(validate_manifest(manifest, task_identity(resolve_task_definition(None))))
    return {"path": V1, "file_sha256": file_sha, "manifest_sha256": manifest["sha256"], "states": states}


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


def build_plan(phase: str, round_: int, version: int, items: list[dict], predecessors: list[Path],
               declaration: dict) -> dict:
    copies = GAME_COPIES[:1] if phase == "copy" else GAME_COPIES
    commands = templates(declaration)
    plan = {"name": plan_name(phase, round_, version), "phase": phase, "round": round_, "version": version,
            "rules": "config/retention-confirmation.json amendment 3 (stage 1)",
            "declaration": {"path": "config/retention-confirmation.json",
                            "text_sha256": fresh_sets.text_sha256(DECLARATION)},
            "fresh_set_stage": "stage1",
            "runner": {"game_copies": [{"game_dir": d, "ports": [a, b]} for d, a, b in copies],
                       "threads_per_job": THREADS, "one_at_a_time": phase == "copy"},
            "predecessors": [{"path": rel(p), "text_sha256": fresh_sets.text_sha256(p)} for p in predecessors],
            "items": items,
            "runs": [entry_for(phase, round_, item, commands) for item in items]}
    if phase in EVALUATION_PHASES:
        plan["inputs"] = {"v1": v1_pins()}
    if phase == "copy":
        plan["inputs"] = {"room2_v2": {"path": ROOM2_V2, "manifest_sha256": ROOM2_V2_MANIFEST_SHA256,
                                       "role": "the copy's refuse-only overlap check and hash audit; never opened"}}
    problems = ro.fresh_set_problems(plan, [Path(d) for d, _, _ in copies], fresh=hx.FRESH)
    if problems:
        raise Refused("; ".join(problems))
    return plan


# -------------------------------------------------------------------------------------------- campaign audit

def campaign_folders(name: str) -> list[Path]:
    return sorted(p for p in CAMPAIGNS.glob(f"*-{name}") if p.is_dir() and p.name.endswith(f"-{name}")
                  and re.fullmatch(rf"\d{{8}}-\d{{6}}-{re.escape(name)}", p.name))


def audit(folder: Path, plan: dict, plan_file: Path) -> dict:
    """What one campaign folder of this plan shows. 'executed' only for a campaign that provably ended."""
    log_path = folder / "campaign.log"
    if not log_path.exists():
        return {"kind": "not started", "folder": rel(folder)}
    if hx.identity_problems([log_path]):
        raise Refused(f"{rel(log_path)} refers to a fresh set")
    log = log_path.read_text(encoding="utf-8")
    if f"campaign {plan['name']}," not in log:
        return {"kind": "not started", "folder": rel(folder)}
    if "dry run: nothing executed" in log:
        return {"kind": "dry run", "folder": rel(folder)}
    if "campaign finished:" not in log:
        raise Refused(f"{rel(folder)} has not provably ended (still running or interrupted): stop for review")
    summary_path = folder / "summary.json"
    summary = read_json(summary_path)
    plan_hash = fresh_sets.text_sha256(plan_file)
    if summary.get("plan") != plan["name"] or summary.get("plan_sha256") != plan_hash:
        raise Refused(f"{rel(summary_path)} is not a summary of {rel(plan_file)} as committed")
    commit = summary.get("commit")
    committed_plan = hx._git_bytes("show", f"{commit}:{rel(plan_file)}")
    committed_declaration = hx._git_bytes("show", f"{commit}:config/retention-confirmation.json")
    if (hx._text_sha256_of(committed_plan) != plan_hash
            or not hx.same_json(hx.strict_json(committed_plan.decode("utf-8")), plan)
            or hx._text_sha256_of(committed_declaration) != plan["declaration"]["text_sha256"]):
        raise Refused(f"the plan or declaration at the campaign's commit {str(commit)[:8]} is not the pinned one")
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
            "summary_sha256": guarded_hash(summary_path), "log_sha256": guarded_hash(log_path)}


# ------------------------------------------------------------------------------------ outcome validation

def _command_problems(record: dict, entry: dict) -> list[str]:
    command = record.get("command") or []
    copies = {os.path.normcase(str(Path(d).resolve())) for d, _, _ in GAME_COPIES}
    if (command[:-2] != entry["command"] or command[-2:-1] != ["--game-dir"]
            or os.path.normcase(command[-1]) not in copies):
        return [f"{entry['id']} ran {command}, not its planned command"]
    return []


def _clean(result: dict, commit: str, what: str, attributable: bool = True) -> list[str]:
    problems = []
    if result.get("commit") != commit or result.get("uncommitted_changes") is not False or result.get("git_error"):
        problems.append(f"{what} is not from a clean tree at the campaign commit")
    if attributable and (result.get("attributable") is not True or result.get("runtime_problems") != []):
        problems.append(f"{what} is not attributable or has runtime problems")
    return problems


def evaluation_outcome(record: dict, expected_sha: str, commit: str, pins: dict) -> tuple[dict | None, list[str]]:
    artifact = record.get("artifact") or {}
    if not artifact or artifact.get("problem"):
        return None, [f"no valid result artifact: {artifact.get('problem')}"]
    result_file = REPO / artifact["result_file"]
    if not norm(artifact["result_file"]).startswith(EVALUATION_FOLDER):
        return None, [f"result outside {EVALUATION_FOLDER}"]
    result = read_json(result_file)
    problems = fresh_mentions(result, {})
    if guarded_hash(result_file) != artifact.get("result_sha256"):
        problems.append("result file changed since the runner pinned it")
    episodes = artifact.get("episodes_file")
    if not episodes or guarded_hash(REPO / episodes) != artifact.get("episodes_sha256"):
        problems.append("episode file missing or changed since the runner pinned it")
    room1 = task_identity(resolve_task_definition(None))
    expected = {"checkpoint_sha256": expected_sha, "heldout_file_sha256": pins["file_sha256"],
                "heldout_sha256": pins["manifest_sha256"], "repeats": 1, "deterministic": False,
                "evaluation_seed": V1_SEED, "task": room1, "heldout_states": pins["states"],
                "attempts": pins["states"], "episodes": pins["states"]}
    problems += [f"{key} is {result.get(key)!r}, expected {value!r}" for key, value in expected.items()
                 if not hx.same_json(result.get(key), value)]
    if norm(result.get("heldout_set")) != V1:
        problems.append(f"heldout_set is {result.get('heldout_set')!r}")
    if result.get("stale_starts") != 0:
        raise Refused(f"{artifact['result_file']}: {result.get('stale_starts')} stale starts: stop for review")
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
    problems = fresh_sets.paths_problems([str(folder)], fresh=hx.FRESH)
    if problems:
        raise Refused("; ".join(problems))
    play = read_json(folder / "play.json")
    problems = fresh_mentions(play, {})
    room1 = task_identity(resolve_task_definition(None))
    expected = {"task": room1, "checkpoint_sha256": item["sha256"], "seed": RECORD_SEED, "episodes": EPISODES,
                "dataset": "dataset.npz"}
    problems += [f"{key} is {play.get(key)!r}, expected {value!r}" for key, value in expected.items()
                 if not hx.same_json(play.get(key), value)]
    if norm(play.get("checkpoint")) != item["checkpoint"]:
        problems.append(f"checkpoint is {play.get('checkpoint')!r}")
    if not str(play.get("starts", "")).startswith("canonical"):
        problems.append("not canonical starts")
    problems += _clean(play, commit, "the recording")
    dataset_sha = guarded_hash(folder / "dataset.npz")
    if play.get("dataset_sha256") != dataset_sha:
        problems.append("dataset.npz is not the recorded dataset_sha256")
    summaries = play.get("summaries") or []
    endings = play.get("endings") or {}
    if (sum(endings.values()) != EPISODES or len(summaries) != EPISODES
            or [s.get("episode") for s in summaries] != list(range(EPISODES))):
        problems.append("the recording does not hold 25 complete episodes")
    with np.load(folder / "dataset.npz") as data:
        trajectory = data["trajectory"]
        lengths = {name: len(data[name]) for name in data.files}
        counts = np.bincount(trajectory, minlength=EPISODES) if len(trajectory) else np.zeros(EPISODES, int)
    if (set(lengths.values()) != {play.get("frames")} or len(counts) != EPISODES
            or [int(c) for c in counts] != [s.get("frames") for s in summaries]):
        problems.append("the dataset arrays do not match the recorded frames and episodes")
    if problems:
        return None, problems
    return {"folder": relative, "play_sha256": guarded_hash(folder / "play.json"), "dataset_sha256": dataset_sha,
            "frames": play["frames"], "commit": play["commit"]}, []


def _mix_play_paths(text: str) -> list[str]:
    return [norm(p) for p in re.findall(r"'([^']*)'", str(text))]


def copy_outcome(record: dict, item: dict, commit: str, plan_entry: dict) -> tuple[dict | None, list[str]]:
    artifact = record.get("artifact") or {}
    if not artifact or artifact.get("problem"):
        return None, [f"no valid result artifact: {artifact.get('problem')}"]
    result_file = REPO / artifact["result_file"]
    if not re.fullmatch(re.escape(CLONE_FOLDER) + r"\d{8}-\d{6}/results\.json", norm(artifact["result_file"])):
        return None, [f"result {artifact['result_file']!r} is not a copy folder"]
    result = read_json(result_file)
    problems = fresh_mentions(result, {("args", "heldout"): ROOM2_V2, ("manifests", "heldout"): ROOM2_V2})
    if guarded_hash(result_file) != artifact.get("result_sha256"):
        problems.append("result file changed since the runner pinned it")
    problems += _clean(result, commit, "the copy", attributable=False)
    command = plan_entry["command"]
    declared = {command[i][2:].replace("-", "_"): command[i + 1] for i in range(1, len(command) - 1)
                if command[i].startswith("--") and not command[i + 1].startswith("--")}
    args = result.get("args") or {}
    expected_args = {**{k: v for k, v in declared.items() if k != "mix_play"},
                     "routes_only": "True", "no_play": "True", "mix_room": "None", "allow_dirty": "False",
                     "allow_runtime_mismatch": "False"}
    for key, value in expected_args.items():
        if norm(args.get(key)) != norm(value):
            problems.append(f"args.{key} is {args.get(key)!r}, expected {value!r}")
    recording = item["recording"]
    if _mix_play_paths(args.get("mix_play")) != ["default", recording["folder"]]:
        problems.append(f"args.mix_play is {args.get('mix_play')!r}")
    mix = result.get("mix") or {}
    play = mix.get("play") or {}
    room1 = task_identity(resolve_task_definition(None))
    room2 = task_identity(resolve_task_definition(ROOM2_TASK))
    checks = {"init_from.sha256": ((result.get("init_from") or {}).get("sha256"), item["sha256"]),
              "mix.room_weighting": (mix.get("room_weighting"), "equal"), "mix.targets": (mix.get("targets"), "donor"),
              "mix.states": (mix.get("states"), "donor play"), "mix.task": (mix.get("task"), room1),
              "mix.play.play": (norm(play.get("play")), recording["folder"]),
              "mix.play.dataset_sha256": (play.get("dataset_sha256"), recording["dataset_sha256"]),
              "mix.play.episodes": (play.get("episodes"), EPISODES), "mix.play.seed": (play.get("seed"), RECORD_SEED),
              "mix.play.recorded_at": (play.get("recorded_at"), recording["commit"]),
              "task": (result.get("task"), room2),
              "cloning.epochs": ((result.get("cloning") or {}).get("epochs"), 300),
              "manifests.heldout": (norm((result.get("manifests") or {}).get("heldout")), ROOM2_V2),
              "manifests.heldout_sha256": ((result.get("manifests") or {}).get("heldout_sha256"),
                                           ROOM2_V2_MANIFEST_SHA256)}
    problems += [f"{key} is {got!r}, expected {want!r}" for key, (got, want) in checks.items()
                 if not hx.same_json(got, want)]
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
    if record.get("status") != "ok":
        return None, [f"status {record.get('status')}"]
    problems = _command_problems(record, entry)
    if problems:
        return None, problems
    if phase == "donor-v1":
        return evaluation_outcome(record, item["sha256"], commit, plan["inputs"]["v1"])
    if phase == "copy-v1":
        return evaluation_outcome(record, item["clone"]["sha256"], commit, plan["inputs"]["v1"])
    if phase == "record":
        return recording_outcome(record, item, commit)
    return copy_outcome(record, item, commit, entry)


# ------------------------------------------------------------------------------------------- phase state

def phase_items(phase: str, round_: int, declaration: dict) -> tuple[list[dict], list[Path]]:
    """The items a phase needs in a round and the records they come from."""
    if phase == "donor-v1":
        donors = round_donors(declaration, round_)
        return donors, ([inclusion_path(round_ - 1)] if round_ else [])
    before = PHASES[PHASES.index(phase) - 1]
    source = completion_path(before, round_)
    if not source.exists():
        raise Refused(f"{phase} round {round_} needs {rel(source)}")
    completion = read_json(source)
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
        items.append(item)
    return items, [source]


def phase_state(phase: str, round_: int, declaration: dict, upto: int | None = None) -> dict:
    """Valid outcomes and failures from the executed versions of a phase (all, or those below `upto`), and the
    version still waiting to run, if any."""
    items, sources = phase_items(phase, round_, declaration)
    by_j = {item["j"]: item for item in items}
    outcomes, failures, plans, pending = {}, [], [], None
    for version in versions(phase, round_):
        if upto is not None and version >= upto:
            break
        path = plan_path(phase, round_, version)
        plan = read_json(path)
        plans.append({"path": rel(path), "text_sha256": fresh_sets.text_sha256(path)})
        audits = [audit(folder, plan, path) for folder in campaign_folders(plan["name"])]
        executed = [a for a in audits if a["kind"] == "executed"]
        if len(executed) > 1:
            raise Refused(f"{rel(path)} ran more than once: stop for review")
        if not executed:
            pending = version
            continue
        run = executed[0]
        for entry in plan["runs"]:
            item = by_j[entry["j"]]
            outcome, problems = outcome_for(phase, run["records"][entry["id"]], entry, item, run["commit"], plan)
            evidence = {"plan": rel(path), "campaign": run["folder"], "summary_sha256": run["summary_sha256"],
                        "log_sha256": run["log_sha256"], "entry": entry["id"]}
            if outcome is None:
                failures.append({"j": entry["j"], "seed": entry["seed"], "problems": problems, **evidence})
            elif entry["j"] in outcomes:
                raise Refused(f"j {entry['j']} has two valid {phase} outcomes: stop for review")
            else:
                outcomes[entry["j"]] = {"outcome": outcome, "evidence": evidence}
    return {"items": items, "sources": sources, "outcomes": outcomes, "failures": failures, "plans": plans,
            "pending": pending}


def expected_plan(phase: str, round_: int, version: int, declaration: dict) -> dict:
    """The plan this generator writes for that version, from the state of the versions before it."""
    state = phase_state(phase, round_, declaration, upto=version)
    if state["pending"] is not None:
        raise Refused(f"{phase} r{round_} v{state['pending']} has not run")
    missing = [item for item in state["items"] if item["j"] not in state["outcomes"]]
    if version > 1 and not missing:
        raise Refused(f"{phase} r{round_} needs no version {version}")
    predecessors = [*state["sources"], *(REPO / p["path"] for p in state["plans"])]
    return build_plan(phase, round_, version, missing if version > 1 else state["items"], predecessors, declaration)


def advance(phase: str, round_: int) -> str:
    """The next step of a phase: its first plan, a recovery version, or its completion record."""
    declaration = load_declaration()
    run_code_checks()
    completion = completion_path(phase, round_)
    if completion.exists():
        return f"{rel(completion)} exists: the phase is complete"
    state = phase_state(phase, round_, declaration)
    if state["pending"] is not None:
        path = plan_path(phase, round_, state["pending"])
        if not hx.same_json(read_json(path), expected_plan(phase, round_, state["pending"], declaration)):
            raise Refused(f"{rel(path)} is not what this generator derives: stop for review")
        return f"{rel(path)} is waiting to run"
    missing = [item for item in state["items"] if item["j"] not in state["outcomes"]]
    if not missing:
        record = {"phase": phase, "round": round_, "declaration_text_sha256": fresh_sets.text_sha256(DECLARATION),
                  "no_work": not state["items"], "sources": [{"path": rel(p), "text_sha256": fresh_sets.text_sha256(p)}
                                                             for p in state["sources"]],
                  "plans": state["plans"], "failures": state["failures"],
                  "items": [{**item, **state["outcomes"][item["j"]]} for item in state["items"]]}
        write_new(completion, record)
        return f"{rel(completion)} written ({'no work' if record['no_work'] else len(record['items'])} items)"
    version = len(state["plans"]) + 1
    plan = expected_plan(phase, round_, version, declaration)
    result = write_new(plan_path(phase, round_, version), plan)
    return f"{rel(plan_path(phase, round_, version))} {result} ({len(plan['runs'])} entries); commit and push it"


# ----------------------------------------------------------------------------------------------- inclusion

def inclusion(round_: int) -> dict:
    """The round's inclusion record and the replacements amendment 3 allocates."""
    declaration = load_declaration()
    path = inclusion_path(round_)
    for phase in PHASES:
        if not completion_path(phase, round_).exists():
            raise Refused(f"inclusion r{round_} needs {rel(completion_path(phase, round_))}")
    previous = read_json(inclusion_path(round_ - 1)) if round_ else None
    slots = {int(j): slot for j, slot in previous["slots"].items()} if previous else {}
    reserved = list(previous["reserved_seeds"]) if previous else []
    donors = read_json(completion_path("donor-v1", round_))["items"]
    copies = {item["j"]: item for item in read_json(completion_path("copy-v1", round_))["items"]}
    donor_excluded, copy_excluded = [], []
    for donor in donors:
        j = donor["j"]
        entry = {"seed": donor["seed"], "sha256": donor["sha256"],
                 "donor_v1": {k: donor["outcome"][k] for k in ("route_macro_success_rate", "margin", "passes")}}
        if not donor["outcome"]["passes"]:
            entry["status"] = "excluded: donor line"
            donor_excluded.append(j)
        elif not copies[j]["outcome"]["passes"]:
            entry["copy_v1"] = {k: copies[j]["outcome"][k] for k in ("route_macro_success_rate", "margin", "passes")}
            entry["status"] = "excluded: copy line"
            copy_excluded.append(j)
        else:
            entry["copy_v1"] = {k: copies[j]["outcome"][k] for k in ("route_macro_success_rate", "margin", "passes")}
            entry["status"] = "qualified"
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
    record = {"round": round_, "declaration_text_sha256": fresh_sets.text_sha256(DECLARATION),
              "rule": "amendment 3 replacement_allocation; margin = route_macro_success_rate - 0.142 rounded to 6 "
                      "decimals, >= 0.20",
              "completions": [{"path": rel(completion_path(p, round_)),
                               "text_sha256": fresh_sets.text_sha256(completion_path(p, round_))} for p in PHASES],
              "previous": ({"path": rel(inclusion_path(round_ - 1)),
                            "text_sha256": fresh_sets.text_sha256(inclusion_path(round_ - 1))} if round_ else None),
              "slots": {str(j): slots[j] for j in sorted(slots)}, "proposals": proposals, "reserved_seeds": reserved,
              "qualified": qualified, "status": status}
    write_new(path, record)
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
    plan_file = plan_file if plan_file.is_absolute() else REPO / plan_file
    problems = hx.identity_problems([DECLARATION, plan_file]) + hx.committed(plan_file)
    if problems:
        raise Refused("; ".join(problems))
    phase, round_, version = parse_plan_name(plan_file)
    plan = read_json(plan_file)
    approved_commit = hx._git(REPO, "rev-parse", "--verify", f"{approved}^{{commit}}").strip()
    plan_commit = hx._git(REPO, "log", "-1", "--format=%H", "--", rel(plan_file)).strip()
    for what, commit, ref in (("approved commit", approved_commit, "HEAD"), ("approved commit", approved_commit, remote),
                              ("plan commit", plan_commit, remote)):
        if subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", commit, ref]).returncode != 0:
            raise Refused(f"the {what} {commit[:8]} is not on {ref}")
    manifest = control_files()
    changed = [f for f in manifest
               if hx._git(REPO, "rev-parse", f"HEAD:{f}").strip() != hx._git(REPO, "rev-parse", f"{approved_commit}:{f}").strip()]
    dirty = hx._git(REPO, "status", "--porcelain", "--", *manifest).strip()
    if changed or dirty:
        raise Refused(f"control code or declaration differs from the approved commit: {changed} {dirty}")
    declaration = load_declaration()
    if not hx.same_json(plan, expected_plan(phase, round_, version, declaration)):
        raise Refused(f"{rel(plan_file)} is not what the approved generator derives from its predecessors")
    audits = [audit(folder, plan, plan_file) for folder in campaign_folders(plan["name"])]
    if any(a["kind"] == "executed" for a in audits):
        raise Refused(f"{rel(plan_file)} has already run; a recovery uses a new version")
    checks = run_code_checks()
    invocation = runner_invocation(plan_file, phase)
    dry = subprocess.run([*invocation, "--dry-run"], cwd=REPO, capture_output=True, text=True)
    if dry.returncode != 0:
        raise Refused(f"the runner's dry run exited {dry.returncode}: {dry.stdout.strip()[-400:]}")
    copy = ro.parse_copy(invocation[invocation.index("--copy") + 1])
    record = {"plan": {"path": rel(plan_file), "text_sha256": fresh_sets.text_sha256(plan_file),
                       "commit": plan_commit},
              "approved_commit": approved_commit, "remote": remote,
              "head": hx._git(REPO, "rev-parse", "HEAD").strip(),
              "control_files": {f: hx._git(REPO, "rev-parse", f"HEAD:{f}").strip() for f in manifest},
              "code_checks": checks, "invocation": invocation,
              "dry_run": {"exit_code": 0, "stdout_tail": dry.stdout.strip().splitlines()[-6:]},
              "threads": rc.thread_probe(copy.env(THREADS)), "audits": audits}
    STAGE1.mkdir(parents=True, exist_ok=True)
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
    include = sub.add_parser("inclusion")
    include.add_argument("--round", type=int, required=True)
    check = sub.add_parser("preflight")
    check.add_argument("plan", type=Path)
    check.add_argument("--approved", required=True, help="the implementation commit the review approved")
    check.add_argument("--remote", default="origin/proposal/ppo-anchor")
    args = parser.parse_args()
    try:
        if args.action == "advance":
            print(advance(args.phase, args.round))
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
