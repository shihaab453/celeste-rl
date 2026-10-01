"""The two fresh held-out sets and the only ways the retention confirmation may use them.

Source of truth: config/retention-confirmation.json (running.v2_guard, measurements, pipeline.copy). Where the guard
runs: the confirmation plan generators (each planned command), scripts/run_overnight.py before any plan starts
(`plan_problems`, on every command exactly as the runner will run it, `--game-dir` included), and the confirmation
analyzer (summary commands and results).

- `classify(command)`: None if the command refers to neither fresh set; the declared shape it uses a fresh set in;
  otherwise `FreshSetRefused`. A token refers to a fresh set if it names one, if it is an existing path whose real
  long path names one (Windows short 8.3 names, symlinks and junctions), or if it is an existing file whose content
  is a fresh set's committed blob (renamed copies and hard links, LF or CRLF).
- `refuse_unless_declared(commands, allowed)` and `STAGE_ALLOWED`: a stage 1 plan may only copy, a training night
  may not touch a fresh set at all, and only the evaluation plan may evaluate on them. A plan that declares no stage
  may not touch them either.
- `verify_identity(name)` checks a set's committed identity (git blob, task identity, for Room 2 the pinned manifest
  hash) and returns hashes and a state count, never states. Call it for Room 2 from stage 1 on (the copy step reads
  that file anyway) and for Room 1 only when the evaluation plan is generated, after night B.

Older tools (the pilots' and the tie-break's generators and analyzers) keep refusing both sets by name and do not
use this module.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from celeste_rl.heldout import validate_manifest
from celeste_rl.tasks import resolve_task_definition, task_identity
from celeste_rl.texthash import text_sha256

REPO = Path(__file__).resolve().parents[1]
EVAL_SEED = "20261001"
MAX_HASH_BYTES = 256 * 1024 * 1024  # larger files cannot be either fresh set (both are under 2 MB)


@dataclass(frozen=True)
class FreshSet:
    path: str
    git_blob: str
    manifest_sha256: str | None  # None: computed at plan time and pinned in the plan (Room 1, never read before)
    task_definition: str | None  # None: the original Room 1 task


FRESH = {
    "room1": FreshSet("config/heldout_starts-room1-v2.json", "40fc0eb573b51d702932598d8c994e4056012a16", None, None),
    "room2": FreshSet("config/heldout_starts-room2-v2.json", "5386bb86d25b39c1a8cc45a04e42c6c939265a0f",
                      "51ea319af3b65f50f516da33f09cf6b02aca89ef0c10778ca0080c4f895909eb", "config/room2.json"),
}
MARKERS = tuple(Path(spec.path).stem.lower() for spec in FRESH.values())


class FreshSetRefused(ValueError):
    pass


@dataclass(frozen=True)
class Shape:
    script: str
    fixed: dict          # option -> exact list of values
    choices: dict        # option -> allowed single values
    required: tuple      # options that take exactly one free value
    optional: tuple = ()  # options that may appear once with one free value
    flags: tuple = ()    # options that take no value and must appear


SHAPES = {
    "room1_eval": Shape(
        "scripts/evaluate_heldout.py",
        fixed={"--starts": [FRESH["room1"].path], "--seed": [EVAL_SEED]},
        choices={"--repeats": ("1", "3")},
        required=("--checkpoint",), optional=("--game-dir",)),
    "room2_eval": Shape(
        "scripts/evaluate_heldout.py",
        fixed={"--task-definition": ["config/room2.json"], "--starts": [FRESH["room2"].path], "--repeats": ["1"],
               "--seed": [EVAL_SEED], "--reward-version": ["rew-v2"], "--shaping-scale": ["2.0"]},
        choices={}, required=("--checkpoint",), optional=("--game-dir",)),
    "copy_overlap_guard": Shape(
        "scripts/clone_room1.py",
        fixed={"--task-definition": ["config/room2.json"],
               "--dataset": ["runs/clone/chapter-1-room-2/20260924-014351/dataset.npz"],
               "--demonstrations": ["config/demonstrations-room2-v2.json"], "--heldout": [FRESH["room2"].path],
               "--mix-targets": ["donor"], "--room-weighting": ["equal"], "--holdout": ["0.25"], "--epochs": ["300"],
               "--batch-size": ["64"], "--learning-rate": ["0.001"]},
        choices={"--seed": tuple(str(10 + j) for j in range(8))},  # clone seed 10 + j, j = 0 to 7
        required=("--init-from", "--init-from-sha256"),
        optional=("--game-dir",),  # run_overnight.py appends it to every command
        flags=("--routes-only", "--no-play")),
}
MIX_PLAY = "--mix-play"  # two values: the literal "default" and the donor's recording folder
STAGE_ALLOWED = {"stage1": ("copy_overlap_guard",), "night": (), "evaluation": ("room1_eval", "room2_eval")}


def _blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


@lru_cache(maxsize=4096)
def _file_blobs(path: str, size: int, mtime_ns: int) -> frozenset[str]:
    """Blob ids of a file as it is and with CRLF turned into LF (size and mtime only key the cache)."""
    data = Path(path).read_bytes()
    return frozenset({_blob(data), _blob(data.replace(b"\r\n", b"\n"))})


def refers_to_fresh(token: str, fresh: dict = FRESH, repo: Path = REPO) -> bool:
    markers = tuple(Path(spec.path).stem.lower() for spec in fresh.values())
    blobs = {spec.git_blob for spec in fresh.values()}
    for candidate in (token, token.split("=", 1)[1] if "=" in token else None):
        if not candidate:
            continue
        if any(marker in candidate.lower() for marker in markers):
            return True
        path = Path(candidate) if Path(candidate).is_absolute() else repo / candidate
        try:
            if not path.exists():
                continue
            if any(marker in os.path.realpath(path).lower() for marker in markers):
                return True
            if path.is_file():
                stat = path.stat()
                if stat.st_size <= MAX_HASH_BYTES and _file_blobs(str(path), stat.st_size, stat.st_mtime_ns) & blobs:
                    return True
        except (OSError, ValueError):
            continue
    return False


def mentions_fresh_set(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in MARKERS)


def _script(token: str) -> str:
    return token.replace("\\", "/")


def _options(args: list[str]) -> dict[str, list[str]]:
    if args and not args[0].startswith("--"):
        raise FreshSetRefused(f"unexpected leading value {args[0]!r}")
    options: dict[str, list[str]] = {}
    current = None
    for token in args:
        if token.startswith("--"):
            if token in options:
                raise FreshSetRefused(f"option {token} given twice")
            options[token], current = [], token
        else:
            options[current].append(token)
    return options


def _match(shape: Shape, command: list[str], fresh: dict, repo: Path) -> str | None:
    """None if the command has this shape, else the first reason it does not."""
    if not command or _script(command[0]) != shape.script:
        return f"not {shape.script}"
    options = _options(command[1:])
    expected = set(shape.fixed) | set(shape.choices) | set(shape.required) | set(shape.flags)
    if shape.script == "scripts/clone_room1.py":
        expected.add(MIX_PLAY)
    missing = expected - set(options)
    extra = set(options) - expected - set(shape.optional)
    if missing or extra:
        return f"options differ from the declared shape (missing {sorted(missing)}, extra {sorted(extra)})"
    for option, values in shape.fixed.items():
        if options[option] != values:
            return f"{option} {options[option]} is not the declared {values}"
    for option, allowed in shape.choices.items():
        if len(options[option]) != 1 or options[option][0] not in allowed:
            return f"{option} {options[option]} is not one of {list(allowed)}"
    for option in (*shape.required, *[o for o in shape.optional if o in options]):
        if len(options[option]) != 1 or refers_to_fresh(options[option][0], fresh, repo):
            return f"{option} must have one value that is not a fresh set"
    for option in shape.flags:
        if options[option]:
            return f"{option} takes no value"
    if shape.script == "scripts/clone_room1.py":
        mix = options[MIX_PLAY]
        if len(mix) != 2 or mix[0] != "default" or refers_to_fresh(mix[1], fresh, repo):
            return f"{MIX_PLAY} must be 'default <recording>'"
    return None


def classify(command: list[str], fresh: dict = FRESH, repo: Path = REPO) -> str | None:
    """The declared shape a command uses a fresh set in, None if it refers to none; refuses anything else."""
    if not any(refers_to_fresh(token, fresh, repo) for token in command):
        return None
    reasons = {}
    for name, shape in SHAPES.items():
        reason = _match(shape, command, fresh, repo)
        if reason is None:
            return name
        reasons[name] = reason
    raise FreshSetRefused(f"command refers to a fresh held-out set outside the declared uses: {command}; {reasons}")


def refuse_unless_declared(commands: list[list[str]], allowed: tuple[str, ...], fresh: dict = FRESH,
                           repo: Path = REPO) -> dict[str, int]:
    """Classify every command; refuse any fresh-set use whose shape is not in `allowed`. Returns counts per shape."""
    counts: dict[str, int] = {}
    for command in commands:
        shape = classify(command, fresh, repo)
        if shape is None:
            continue
        if shape not in allowed:
            raise FreshSetRefused(f"fresh-set use {shape} is not allowed here: {command}")
        counts[shape] = counts.get(shape, 0) + 1
    return counts


def plan_problems(plan: dict, commands: list[list[str]], fresh: dict = FRESH, repo: Path = REPO) -> list[str]:
    """Why a plan may not start: a fresh-set use its declared stage does not allow (no stage: none allowed).
    `commands` are the commands as they will run, without the interpreter."""
    stage = plan.get("fresh_set_stage")
    if stage is not None and stage not in STAGE_ALLOWED:
        return [f"unknown fresh_set_stage {stage!r} (known: {sorted(STAGE_ALLOWED)})"]
    try:
        refuse_unless_declared(commands, STAGE_ALLOWED.get(stage, ()), fresh, repo)
    except FreshSetRefused as error:
        return [f"fresh-set guard ({stage or 'no stage declared'}): {error}"]
    return []


def git_blob(path: Path, repo: Path = REPO) -> str:
    """The blob id git would store for the working file, with the repository's line-ending filters applied."""
    relative = Path(path).resolve().relative_to(repo.resolve()).as_posix()
    return subprocess.run(["git", "-C", str(repo), "hash-object", f"--path={relative}", str(path)],
                          capture_output=True, text=True, check=True).stdout.strip()


def verify_identity(name: str, repo: Path = REPO, fresh: dict = FRESH) -> dict:
    """Check a fresh set's committed identity; return hashes and the state count, never states."""
    spec = fresh[name]
    path = repo / spec.path
    blob = git_blob(path, repo)
    if blob != spec.git_blob:
        raise FreshSetRefused(f"{spec.path} has git blob {blob}, declared {spec.git_blob}")
    definition = resolve_task_definition(repo / spec.task_definition if spec.task_definition else None)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entries = validate_manifest(manifest, task_identity(definition))
    if spec.manifest_sha256 is not None and manifest.get("sha256") != spec.manifest_sha256:
        raise FreshSetRefused(f"{spec.path} manifest sha256 {manifest.get('sha256')}, declared {spec.manifest_sha256}")
    return {"path": spec.path, "git_blob": blob, "manifest_sha256": manifest["sha256"],
            "text_sha256": text_sha256(path), "states": len(entries)}
