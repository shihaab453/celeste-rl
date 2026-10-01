"""The two fresh held-out sets and the only ways the retention confirmation may use them.

Source of truth: config/retention-confirmation.json (running.v2_guard, measurements, pipeline.copy). Where the guard
runs: the confirmation plan generators (each planned command); scripts/run_overnight.py before anything else reads a
plan's inputs (every command as declared, as built for every game copy, and in its automatic-resume form) and again
on each built command immediately before it launches; and the confirmation analyzer (summary commands and results).

How a fresh set is recognised, cheapest first, so a file that IS a fresh set is known without reading it:
1. its name, in the token or in the value of an `--opt=value` token;
2. the real long path of an existing path (`os.path.realpath`: Windows short 8.3 names, symlinks, junctions);
3. file identity with a fresh set's canonical path (`os.path.samefile`, from metadata: hard links);
4. only then content: an existing file small enough to be a copy whose git blob, as is or with CRLF turned into LF,
   equals a fresh blob (renamed byte copies). Re-formatted copies (for example re-indented JSON) are NOT detected;
   the protection against those is trusted, pinned input provenance, not this guard.
No answer is cached between checks.

Inputs a command makes its child read are checked too, before anything reads them (known formats only; this is a
guard for the confirmation's own commands, not a sandbox for arbitrary scripts): the files inside a folder passed in
(a recording, a run folder, and its checkpoints/), the provenance files tools read beside a file (manifest.json,
results.json and play.json in its folder and the folder above, and a `<name>.manifest.json` sidecar such as
dataset.manifest.json), and the paths named inside a JSON input (for example
a task definition's source route). A token that refers to a fresh set is never listed or parsed.

A fresh set may appear only in its shape's designated slot, for its own room; every other input of an allowed
command, fixed or free, direct or indirect, must not refer to either set.

- `classify(command)`: None, the declared shape, or `FreshSetRefused`.
- `refuse_unless_declared(commands, allowed)`, `STAGE_ALLOWED`, `plan_problems(plan, commands)`: a stage 1 plan may
  only copy, a training night may not touch a fresh set, only the evaluation plan may evaluate on them, and a plan
  that declares no stage may not touch them.
- `verify_identity(name)`: the committed identity (git blob, task identity, for Room 2 the pinned manifest hash);
  returns hashes and a state count, never states. Call it for Room 2 from stage 1 on (the copy step reads that file
  anyway) and for Room 1 only when the evaluation plan is generated, after night B.

Older tools (the pilots' and the tie-break's generators and analyzers) keep refusing both sets by name and do not
use this module.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from celeste_rl.heldout import validate_manifest
from celeste_rl.tasks import resolve_task_definition, task_identity
from celeste_rl.texthash import text_sha256

REPO = Path(__file__).resolve().parents[1]
EVAL_SEED = "20261001"
PROVENANCE_FILES = ("manifest.json", "results.json", "play.json")
MAX_FOLDER_FILES = 2000
MAX_JSON_BYTES = 64 * 1024 * 1024


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
    fresh_slot: tuple    # (option, fresh set name): the only place a fresh set may appear
    fixed: dict          # option -> exact list of values
    choices: dict        # option -> allowed single values
    required: tuple      # options that take exactly one free value
    optional: tuple = ()  # options that may appear once with one free value
    flags: tuple = ()    # options that take no value and must appear


SHAPES = {
    "room1_eval": Shape(
        "scripts/evaluate_heldout.py", ("--starts", "room1"),
        fixed={"--starts": [FRESH["room1"].path], "--seed": [EVAL_SEED]},
        choices={"--repeats": ("1", "3")},
        required=("--checkpoint",), optional=("--game-dir",)),
    "room2_eval": Shape(
        "scripts/evaluate_heldout.py", ("--starts", "room2"),
        fixed={"--task-definition": ["config/room2.json"], "--starts": [FRESH["room2"].path], "--repeats": ["1"],
               "--seed": [EVAL_SEED], "--reward-version": ["rew-v2"], "--shaping-scale": ["2.0"]},
        choices={}, required=("--checkpoint",), optional=("--game-dir",)),
    "copy_overlap_guard": Shape(
        "scripts/clone_room1.py", ("--heldout", "room2"),
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


def _inside(path: Path, root: Path) -> bool:
    try:
        Path(os.path.realpath(path)).relative_to(Path(os.path.realpath(root)))
        return True
    except ValueError:
        return False


class _Scan:
    """One check. Its memo lives only as long as the check, so no answer survives into a later check."""

    def __init__(self, fresh: dict, repo: Path, read_content=None):
        self.fresh, self.repo = fresh, repo
        self.markers = {name: Path(spec.path).stem.lower() for name, spec in fresh.items()}
        self.blobs = {spec.git_blob: name for name, spec in fresh.items()}
        self.canonical = {name: repo / spec.path for name, spec in fresh.items()}
        sizes = [path.stat().st_size for path in self.canonical.values() if path.exists()]  # metadata only
        self.hash_limit = 4 * max(sizes) if sizes else 8 * 1024 * 1024
        self.read_content = read_content or (lambda path: path.read_bytes())
        self.memo: dict[str, frozenset] = {}

    def _path(self, text: str) -> Path:
        return Path(text) if Path(text).is_absolute() else self.repo / text

    def names(self, token: str) -> frozenset:
        """The fresh sets a token refers to (empty if none)."""
        if token in self.memo:
            return self.memo[token]
        found: set[str] = set()
        for candidate in (token, token.split("=", 1)[1] if "=" in token else None):
            if candidate:
                found |= self._candidate(candidate)
        self.memo[token] = frozenset(found)
        return self.memo[token]

    def _candidate(self, candidate: str) -> set[str]:
        lowered = candidate.lower()
        hits = {name for name, marker in self.markers.items() if marker in lowered}
        if hits:
            return hits
        path = self._path(candidate)
        try:
            if not path.exists():
                return set()
            real = os.path.realpath(path).lower()
            hits = {name for name, marker in self.markers.items() if marker in real}
            if hits:
                return hits
            hits = {name for name, canonical in self.canonical.items()
                    if canonical.exists() and os.path.samefile(path, canonical)}
            if hits:
                return hits
            if path.is_file() and path.stat().st_size <= self.hash_limit:
                data = self.read_content(path)
                return {self.blobs[b] for b in {_blob(data), _blob(data.replace(b"\r\n", b"\n"))} if b in self.blobs}
        except (OSError, ValueError):
            return set()
        return set()

    def indirect(self, token: str) -> list[str]:
        """Inputs the child would read because of this token (never for a token that refers to a fresh set)."""
        if self.names(token):
            return []
        text = token.split("=", 1)[1] if token.startswith("--") and "=" in token else token
        path = self._path(text)
        try:
            if not path.exists() or not _inside(path, self.repo):
                return []
            found: list[Path] = []
            if path.is_dir():
                for pattern in ("*", "checkpoints/*"):
                    found += sorted(p for p in path.glob(pattern) if p.is_file())
                if len(found) > MAX_FOLDER_FILES:
                    raise FreshSetRefused(f"{text}: more than {MAX_FOLDER_FILES} files to check")
                return [str(p) for p in found]
            for folder in (path.parent, path.parent.parent):
                found += [folder / name for name in PROVENANCE_FILES if (folder / name).is_file()]
            sidecar = path.with_name(f"{path.stem}.manifest.json")  # e.g. dataset.npz -> dataset.manifest.json
            if sidecar.is_file() and sidecar != path:
                found.append(sidecar)
            if path.suffix.lower() == ".json" and path.stat().st_size <= MAX_JSON_BYTES:
                data = json.loads(path.read_text(encoding="utf-8"))
                found += [self._path(s) for s in _strings(data) if _looks_like_path(s) and self._path(s).exists()]
            return [str(p) for p in found]
        except (OSError, ValueError, json.JSONDecodeError):
            return []


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _looks_like_path(text: str) -> bool:
    return 0 < len(text) < 400 and ("/" in text or "\\" in text or text.endswith((".json", ".npz", ".zip", ".tas")))


def refers_to_fresh(token: str, fresh: dict = FRESH, repo: Path = REPO) -> bool:
    return bool(_Scan(fresh, repo).names(token))


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


def _match(shape: Shape, command: list[str], scan: _Scan) -> str | None:
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
        if len(options[option]) != 1:
            return f"{option} must have one value"
    for option in shape.flags:
        if options[option]:
            return f"{option} takes no value"
    if shape.script == "scripts/clone_room1.py":
        mix = options[MIX_PLAY]
        if len(mix) != 2 or mix[0] != "default":
            return f"{MIX_PLAY} must be 'default <recording>'"
    slot, room = shape.fresh_slot
    if scan.names(command[0]):
        return "the script refers to a fresh set"
    for option, values in options.items():
        for value in values:
            names = scan.names(value)
            if option == slot and names != {room}:
                return f"{option} must be the {room} fresh set and nothing else"
            if option != slot and names:
                return f"{option} {value} refers to a fresh set outside the designated slot {slot}"
    return None


def classify(command: list[str], fresh: dict = FRESH, repo: Path = REPO, read_content=None,
             scan: _Scan | None = None) -> str | None:
    """The declared shape a command uses a fresh set in, None if it refers to none; refuses anything else.
    `scan` lets one check (a whole plan) share its memo; a new check always starts a new scan."""
    scan = scan or _Scan(fresh, repo, read_content)
    direct = any(scan.names(token) for token in command)
    indirect = [(token, child) for token in command for child in scan.indirect(token) if scan.names(child)]
    if indirect:
        raise FreshSetRefused(f"command makes its child read a fresh set through another input: {indirect}")
    if not direct:
        return None
    reasons = {}
    for name, shape in SHAPES.items():
        reason = _match(shape, command, scan)
        if reason is None:
            return name
        reasons[name] = reason
    raise FreshSetRefused(f"command refers to a fresh held-out set outside the declared uses: {command}; {reasons}")


def refuse_unless_declared(commands: list[list[str]], allowed: tuple[str, ...], fresh: dict = FRESH,
                           repo: Path = REPO) -> dict[str, int]:
    """Classify every command; refuse any fresh-set use whose shape is not in `allowed`. Returns counts per shape."""
    counts: dict[str, int] = {}
    scan = _Scan(fresh, repo)  # this check only
    for command in commands:
        shape = classify(command, fresh, repo, scan=scan)
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
