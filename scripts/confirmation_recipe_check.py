"""The retention confirmation's recipe check (config/retention-confirmation.json, donors.recipe_check).

Before any new donor is trained, rerun the seed 7 donor command at the current commit, alone and with no thread
variables set, and compare it with the original run trained on 2026-09-19 at 8423081. Only an IDENTICAL verdict on
a validated, complete, successful replay means the current code reproduces the arm B recipe.

Run from the repo root with the RL interpreter (Steam running, no other game or experiment running):
    .venv-rl/Scripts/python.exe scripts/confirmation_recipe_check.py run [--dry-run]
    .venv-rl/Scripts/python.exe scripts/confirmation_recipe_check.py compare
    .venv-rl/Scripts/python.exe scripts/confirmation_recipe_check.py compare-diagnostic

run: refuses unless the tracked tree is clean; the declared training code check exits 0; the command passes the
fresh-set guard with no fresh use; the run folder does not exist; no Celeste process runs in ANY folder and no other
experiment script (campaign, training, evaluation, recording, copying) is running. It removes every thread-related
environment variable (OMP_, KMP_, MKL_, OPENBLAS_, GOTO_, BLIS_, NUMEXPR_, VECLIB_, TORCH_NUM...), records what it
removed and torch's default thread counts in that environment, then trains for up to 90 minutes. Complete stdout and
stderr are saved beside the JSON record, also after a timeout, when this check's own game (in the chosen folder) is
stopped and the result recorded.

compare: four verdicts. INVALID: the evidence is incomplete (no successful run record matching the declared
command, declaration and commit; either run unfinished, unattributable, with a different config, missing any of the
original's 13 checkpoints or a required member, or without the final 500k evaluation of 50 episodes; the original's
latest.zip not the declared pin; anything unreadable). DIFFERENT: a declared comparison differs. IDENTICAL: every
declared comparison matches. Declared comparisons: every checkpoint (all saved objects compared structurally: key
presence and key types, container types and lengths, tensors by dtype, shape and value; every data field except the
wall-clock start_time, whose presence and type are still checked; abort_checkpoint_path must name each run's own
<run folder>/checkpoints/aborted.zip, decoded by a reader that accepts only path objects; system_info.txt ignored),
every progress.csv row except speed and memory columns, every evaluations.jsonl row with each checkpoint reference
normalized to its file name and each recorded checkpoint_sha256 verified against that run's own file. Supplementary
diagnostics (episodes.jsonl, archive.json) are reported separately: a supplementary difference gives DIAGNOSE, which
does not unlock new donors either. Records are created exclusively and never overwritten. Exit 0 only for IDENTICAL
(or a passing dry run).

Comparator version 2 (after the review of the replay and the 8423081 diagnostic): three saved data fields differ
between any two training processes, so version 1 called two runs of the same code DIFFERENT. policy_class and
rollout_buffer_class: Stable-Baselines3 saves, beside the serialized class, the text description of each attribute,
which includes its memory address in that process; only the address in the two recognized forms ("<function NAME at
0x...>", "<_abc._abc_data object at 0x...>") is replaced, by a tuple marker no saved JSON text can equal; names,
structure and the serialized payload are compared exactly, and no other field is touched. ep_info_buffer: decoded by a
reader that accepts only a deque (and refuses the pickle extension opcodes, which skip that check); capacity, order,
length and every entry's reward r and length l (values and types) are compared exactly; each entry's wall-clock time t
must be present, a float, finite and not negative, and only its value is excluded. Any other buffer structure is
INVALID. Every record names the comparator version, this script's hash and the commit it ran at.

compare-diagnostic: the same comparison between the original and the copy of the 2026-10-01 diagnostic rerun at the
historical commit 8423081, bound to that diagnostic's live record. It supersedes that record's automatic reading,
concerns the historical code only and says nothing about the current code (whose replay stays DIFFERENT). It cites
the launcher source note and the source saved after the run, by path and hash; they cannot show the exact bytes that ran.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import collections
import csv
import hashlib
import io
import json
import math
import os
import pathlib
import pickle
import pickletools
import re
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from celeste_rl import fresh_sets, game_process, runtime  # noqa: E402

DECLARATION = REPO / "config" / "retention-confirmation.json"
ORIGINAL = Path("runs/train/overnight-B-seed7")
CHECK_RUN = Path("runs/train/confirm-recipe-check-B-seed7")
RECORDS = REPO / "runs" / "confirmation" / "recipe-check"
THREAD_PREFIXES = ("OMP_", "KMP_", "MKL_", "OPENBLAS_", "GOTO_", "BLIS_", "NUMEXPR_", "VECLIB_", "TORCH_NUM")
CODE_CHECK = ["scripts/check_evaluation_code.py", "--entry", "scripts/train_room1.py", "--baseline", "331e6c9",
              "--approved", "44ffab5"]
# Every script of this repository counts as another experiment (campaigns, training, evaluation, recording, copying,
# fixtures), so the list cannot go stale as scripts are added.
EXPERIMENT_SCRIPTS = tuple(sorted(p.name for p in (REPO / "scripts").glob("*.py") if p.name != Path(__file__).name))
QUERY_SECONDS = 60


class ProcessEvidenceUnavailable(Exception):
    """A process query failed, so the alone condition cannot be established."""
LIMIT_MINUTES = 90  # the original campaign's limit for this run
FINAL_STEPS = 501760
FINAL_EPISODES = 50
EXPECTED_CHECKPOINTS = tuple(f"step_{s:09d}.zip" for s in (51200, 100352, 151552, 200704, 251904, 301056, 350208,
                                                           401408, 450560, 501760)) + ("best.zip", "latest.zip",
                                                                                         "previous.zip")
REQUIRED_MEMBERS = ("data", "policy.pth", "policy.optimizer.pth", "pytorch_variables.pth", "_stable_baselines3_version")
TENSOR_MEMBERS = ("policy.pth", "policy.optimizer.pth", "pytorch_variables.pth")
VOLATILE_DATA = ("start_time",)                     # wall clock at training start (presence and type still checked)
RUN_FOLDER_DATA = ("abort_checkpoint_path",)        # <run folder>/checkpoints/aborted.zip, checked per run
IGNORED_MEMBERS = ("system_info.txt",)              # OS and library description text
VOLATILE_PROGRESS = ("env_steps_per_second", "game_private_mb", "game_working_set_mb")
# Fields added to the training config after 8423081; a new run may carry them only at their unchanged defaults.
COMPATIBILITY_DEFAULTS = {"task_definition": "", "stall_frames": 0}
COMPARATOR_VERSION = 2
CLASS_FIELDS = ("policy_class", "rollout_buffer_class")  # descriptive text: process addresses replaced
# The two descriptive forms that carry a process address; the name before " at " is kept and compared.
ADDRESS_REPR = re.compile(r"<(function [A-Za-z_][\w.]*|_abc\._abc_data object) at 0x[0-9A-Fa-f]+>")
ADDRESS_MARKER = "process address removed"
# Pickle opcodes that load an object from the extension registry cache without calling find_class.
EXTENSION_OPCODES = frozenset({"EXT1", "EXT2", "EXT4"})
EPISODE_FIELD = "ep_info_buffer"                         # decoded; each entry's wall-clock t excluded after checks
DEQUE_TYPE = "<class 'collections.deque'>"
HISTORICAL_COMMIT = "842308100d8ee32e135e82d295860dab2acb11d2"  # the original donors' training commit
DIAGNOSTIC = RECORDS / "diagnostic-20261001-202114.json"
DIAGNOSTIC_RUN = RECORDS / "diagnostic-20261001-202114-run"


class Invalid(Exception):
    """The evidence is incomplete, so no verdict about the recipe can be given."""


# ---------------------------------------------------------------------------------------------------- environment

def stripped_env(env: dict) -> dict:
    """The environment with every thread-related variable removed, so libraries use their machine defaults."""
    return {key: value for key, value in env.items() if not key.upper().startswith(THREAD_PREFIXES)}


def thread_variables(env: dict) -> dict:
    return {key: value for key, value in env.items() if key.upper().startswith(THREAD_PREFIXES)}


def declared_command(declaration: dict, game_dir: Path) -> list[str]:
    """The declared donor command for seed 7 with the recipe-check run folder (only --seed and --run-dir change)."""
    template = declaration["donors"]["command"].replace("<s>", "7").split()
    index = template.index("--run-dir")
    template[index + 1] = CHECK_RUN.as_posix()
    return [*template, "--game-dir", str(game_dir)]


def _query(args: list[str], what: str) -> str:
    """A bounded process query; a failure or timeout raises ProcessEvidenceUnavailable (never an empty answer)."""
    try:
        done = subprocess.run(args, capture_output=True, text=True, timeout=QUERY_SECONDS)
    except (OSError, subprocess.SubprocessError) as error:
        raise ProcessEvidenceUnavailable(f"{what}: {type(error).__name__}: {error}") from error
    if done.returncode != 0:
        raise ProcessEvidenceUnavailable(f"{what} exited {done.returncode}: {(done.stderr or '').strip()[:300]}")
    return done.stdout


def other_experiments(own_pid: int) -> list[str]:
    """Command lines of other running python processes that run any script of this repository."""
    marker = "__QUERY_OK__"  # proves the query ran to the end, so an empty list really means none
    query = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" -ErrorAction Stop | "
             "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }; '" + marker + "'")
    output = _query(["powershell", "-NoProfile", "-Command", query], "python process query")
    if marker not in output:
        raise ProcessEvidenceUnavailable("python process query did not complete")
    found = []
    for line in output.splitlines():
        pid, _, command = line.partition("\t")
        if pid.strip().isdigit() and int(pid) != own_pid and any(name in command for name in EXPERIMENT_SCRIPTS):
            found.append(command.strip()[:300])
    return found


def all_game_processes() -> list[str]:
    """Every running Celeste process in any folder, found by image name (an unreadable path cannot hide one)."""
    output = _query(["tasklist", "/FI", "IMAGENAME eq Celeste.exe", "/FO", "CSV", "/NH"], "Celeste process query")
    return [line.strip() for line in output.splitlines() if line.strip().lower().startswith('"celeste.exe"')]


# ------------------------------------------------------------------------------------------------ structural equality

def structural_differences(a, b, path: str = "") -> list[str]:
    """Every difference between two loaded objects, keeping key presence, key types, container types, lengths and
    empty containers; tensors by dtype, shape and value."""
    import torch

    here = path or "(root)"
    if type(a) is not type(b):
        return [f"{here}: type {type(a).__name__} vs {type(b).__name__}"]
    if isinstance(a, torch.Tensor):
        if a.dtype != b.dtype or a.shape != b.shape:
            return [f"{here}: tensor {a.dtype}{tuple(a.shape)} vs {b.dtype}{tuple(b.shape)}"]
        return [] if torch.equal(a, b) else [f"{here}: tensor values differ"]
    if isinstance(a, dict):
        keys_a = {(type(k).__name__, k) for k in a}
        keys_b = {(type(k).__name__, k) for k in b}
        problems = [f"{here}: keys only in the original {sorted(map(str, keys_a - keys_b))}, "
                    f"only in the new run {sorted(map(str, keys_b - keys_a))}"] if keys_a != keys_b else []
        for key in sorted((k for k in a if (type(k).__name__, k) in keys_b), key=repr):
            problems += structural_differences(a[key], b[key], f"{path}/{key!r}")
        return problems
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return [f"{here}: length {len(a)} vs {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in structural_differences(x, y, f"{path}[{i}]")]
    return [] if a == b else [f"{here}: {a!r} vs {b!r}"[:300]]


class _PathOnly(pickle.Unpickler):
    """Unpickles a stored path and nothing else."""

    def find_class(self, module, name):
        if module == "pathlib" and name in ("WindowsPath", "PosixPath", "Path", "PureWindowsPath", "PurePosixPath"):
            return getattr(pathlib, name)
        raise pickle.UnpicklingError(f"refusing {module}.{name}")


def restricted_load(raw: bytes, unpickler: type[pickle.Unpickler]):
    """Unpickle with a restricted reader, refusing first any opcode that could load an object without find_class."""
    found = {op.name for op, _, _ in pickletools.genops(raw)} & EXTENSION_OPCODES
    if found:
        raise pickle.UnpicklingError(f"refusing extension opcodes {sorted(found)}")
    return unpickler(io.BytesIO(raw)).load()


def stored_path_parts(field) -> tuple[str, ...] | None:
    """The parts of an SB3-serialized path field, or None if it is not one."""
    try:
        value = restricted_load(base64.b64decode(field[":serialized:"]), _PathOnly)
        return tuple(pathlib.PurePath(value).parts)
    except (KeyError, TypeError, ValueError, pickle.UnpicklingError, EOFError):
        return None


def without_addresses(field):
    """A saved class field with the process address removed from each recognized descriptive leaf. The serialized
    payload, the type, every name and every other text are kept exactly; a field that is not a record is unchanged.
    A recognized leaf becomes a tuple marker: JSON has no tuples, so no saved text can compare equal to one."""
    if not isinstance(field, dict):
        return field

    def leaf(key, value):
        if key in (":type:", ":serialized:") or not isinstance(value, str):
            return value
        match = ADDRESS_REPR.fullmatch(value)
        return (ADDRESS_MARKER, match.group(1)) if match else value

    return {key: leaf(key, value) for key, value in field.items()}


class _DequeOnly(pickle.Unpickler):
    """Unpickles a deque of plain values and nothing else."""

    def find_class(self, module, name):
        if (module, name) == ("collections", "deque"):
            return collections.deque
        raise pickle.UnpicklingError(f"refusing {module}.{name}")


def episode_buffer(field, capacity, what: str) -> dict:
    """The saved episode buffer with each entry's wall-clock time checked and then left out. Raises Invalid unless
    it is a deque of the run's window size whose entries are exactly {r: finite float, l: int >= 0, t: finite float
    >= 0}."""
    if not isinstance(field, dict) or set(field) != {":type:", ":serialized:"} or field[":type:"] != DEQUE_TYPE:
        raise Invalid(f"{what}: {EPISODE_FIELD} is not a saved deque")
    try:
        value = restricted_load(base64.b64decode(field[":serialized:"], validate=True), _DequeOnly)
    except (TypeError, ValueError, KeyError, IndexError, AttributeError, OverflowError, binascii.Error,
            pickle.UnpicklingError, EOFError) as error:
        raise Invalid(f"{what}: {EPISODE_FIELD} cannot be read: {type(error).__name__}: {str(error)[:200]}") from error
    if type(value) is not collections.deque or type(capacity) is not int or value.maxlen != capacity:
        raise Invalid(f"{what}: {EPISODE_FIELD} is not a deque of capacity {capacity!r}")
    entries = []
    for index, entry in enumerate(value):
        if type(entry) is not dict or set(entry) != {"r", "l", "t"}:
            raise Invalid(f"{what}: {EPISODE_FIELD} entry {index} is not exactly {{r, l, t}}")
        r, length, t = entry["r"], entry["l"], entry["t"]
        if type(r) is not float or not math.isfinite(r) or type(length) is not int or length < 0 \
                or type(t) is not float or not math.isfinite(t) or t < 0:
            raise Invalid(f"{what}: {EPISODE_FIELD} entry {index} has r {r!r}, l {length!r}, t {t!r}")
        entries.append({"r": r, "l": length})
    return {":type:": field[":type:"], "maxlen": value.maxlen, "entries": entries}


def comparable_data(data: dict, what: str) -> dict:
    """A checkpoint's data record as compared: the two class fields without process addresses and the episode
    buffer decoded without wall-clock times. All three must be present."""
    missing = [key for key in (*CLASS_FIELDS, EPISODE_FIELD) if key not in data]
    if missing:
        raise Invalid(f"{what} checkpoint data lacks {missing}")
    return {**data, **{key: without_addresses(data[key]) for key in CLASS_FIELDS},
            EPISODE_FIELD: episode_buffer(data[EPISODE_FIELD], data.get("_stats_window_size"), what)}


# ----------------------------------------------------------------------------------------------------- checkpoints

def _load_tensors(raw: bytes, what: str):
    """A saved torch object, loaded in torch's restricted weights-only mode; a load failure is INVALID evidence."""
    import torch

    try:
        return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except (EOFError, pickle.UnpicklingError, RuntimeError, ValueError, AttributeError, TypeError) as error:
        raise Invalid(f"{what} cannot be loaded: {type(error).__name__}: {str(error)[:200]}") from error


def compare_checkpoint(original: Path, new: Path, run_original: Path | None = None, run_new: Path | None = None) -> list[str]:
    """Differences between two checkpoint zips. Missing required members raise Invalid. A run-folder field must name
    each run's own folder (when the run folders are given) and otherwise be equal."""
    differences = []
    with zipfile.ZipFile(original) as a, zipfile.ZipFile(new) as b:
        for archive, label in ((a, "original"), (b, "new")):
            missing = [m for m in REQUIRED_MEMBERS if m not in archive.namelist()]
            if missing:
                raise Invalid(f"{label} checkpoint {archive.filename} lacks {missing}")
        names_a = set(a.namelist()) - set(IGNORED_MEMBERS)
        names_b = set(b.namelist()) - set(IGNORED_MEMBERS)
        if names_a != names_b:
            differences.append(f"members differ: only original {sorted(names_a - names_b)}, "
                               f"only new {sorted(names_b - names_a)}")
        for name in sorted(names_a & names_b):
            raw_a, raw_b = a.read(name), b.read(name)
            if name == "data":
                data_a, data_b = json.loads(raw_a), json.loads(raw_b)
                for data, label in ((data_a, "original"), (data_b, "new")):
                    if not isinstance(data, dict):
                        raise Invalid(f"{label} checkpoint data is not a record")
                    if not isinstance(data.get("start_time"), int):
                        raise Invalid(f"{label} checkpoint data has no integer start_time")
                skipped = set(VOLATILE_DATA)
                if run_original is not None and run_new is not None:
                    skipped |= set(RUN_FOLDER_DATA)
                    for key in RUN_FOLDER_DATA:
                        for data, run in ((data_a, run_original), (data_b, run_new)):
                            parts = stored_path_parts(data.get(key, {}))
                            expected = (*pathlib.PurePath(run).parts, "checkpoints", "aborted.zip")
                            if parts != expected:
                                differences.append(f"data field {key} is {parts}, expected {expected}")
                kept_a = {k: v for k, v in comparable_data(data_a, "original").items() if k not in skipped}
                kept_b = {k: v for k, v in comparable_data(data_b, "new").items() if k not in skipped}
                differences += [f"data {d}" for d in structural_differences(kept_a, kept_b)]
            elif name in TENSOR_MEMBERS:
                differences += [f"{name} {d}" for d in structural_differences(
                    _load_tensors(raw_a, f"original {Path(original).name}:{name}"),
                    _load_tensors(raw_b, f"new {Path(new).name}:{name}"))]
            elif raw_a != raw_b:
                differences.append(f"member {name} differs")
    return differences


# ------------------------------------------------------------------------------------------------------- records

def _rows_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _rows_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compare_rows(original: list[dict], new: list[dict], volatile: tuple, what: str) -> list[str]:
    """Row-by-row differences; key presence counts (a missing key is not equal to a null value)."""
    differences = []
    if len(original) != len(new):
        differences.append(f"{what}: {len(original)} rows in the original, {len(new)} in the new run")
    for index, (a, b) in enumerate(zip(original, new)):
        kept_a = {k: v for k, v in a.items() if k not in volatile}
        kept_b = {k: v for k, v in b.items() if k not in volatile}
        problems = structural_differences(kept_a, kept_b)
        if problems:
            differences.append(f"{what} row {index}: {problems[:3]}")
    return differences


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def own_checkpoint_name(run: Path, reference) -> str | None:
    """The file name an evaluation reference names in this run's own checkpoints folder: a bare file name, or a path
    whose folder is this run's checkpoints folder (relative to the repository or absolute). None for anything else,
    which is never followed or read."""
    if not isinstance(reference, str) or not reference:
        return None
    path = Path(reference)
    if len(path.parts) == 1:
        return path.name
    folder = (path if path.is_absolute() else REPO / path).parent
    try:
        same = os.path.normcase(os.path.abspath(folder)) == os.path.normcase(os.path.abspath(Path(run) / "checkpoints"))
    except (OSError, ValueError):
        return None
    return path.name if same else None


def evaluation_problems(run: Path, rows: list[dict], label: str) -> list[str]:
    """Each evaluation must name a checkpoint file of this run whose sha256 it records truthfully."""
    problems = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            problems.append(f"{label} evaluation {index} is not a record")
            continue
        name = own_checkpoint_name(run, row.get("checkpoint"))
        if name is None:
            problems.append(f"{label} evaluation {index} names {row.get('checkpoint')!r}, not this run's checkpoints")
            continue
        target = run / "checkpoints" / name
        if not target.is_file():
            problems.append(f"{label} evaluation {index} names {name}, which this run does not have")
        elif _sha256(target) != row.get("checkpoint_sha256"):
            problems.append(f"{label} evaluation {index}: recorded checkpoint_sha256 is not that of {name}")
    return problems


def normalized_evaluations(run: Path, rows: list[dict]) -> list[dict]:
    """Evaluation rows with an own-run checkpoint reference reduced to its file name (any other reference is kept as
    it is, so it differs) and the per-run file hash removed (each is verified against its own file separately)."""
    normalized = []
    for row in rows:
        name = own_checkpoint_name(run, row.get("checkpoint"))
        normalized.append({**{k: v for k, v in row.items() if k != "checkpoint_sha256"},
                           "checkpoint": name if name is not None else row.get("checkpoint")})
    return normalized


# --------------------------------------------------------------------------------------------------- validation

def _manifest(run: Path, label: str) -> dict:
    path = run / "manifest.json"
    if not path.is_file():
        raise Invalid(f"{label} run has no manifest.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("config"), dict):
        raise Invalid(f"{label} manifest is not a record with a config")
    sessions = manifest.get("sessions")
    if not isinstance(sessions, list) or not all(isinstance(s, dict) and isinstance(s.get("provenance"), dict)
                                                 for s in sessions):
        raise Invalid(f"{label} manifest sessions are not records with provenance")
    if manifest.get("status") != "finished" or manifest.get("accepted_steps") != FINAL_STEPS:
        raise Invalid(f"{label} run is not finished at {FINAL_STEPS} accepted steps "
                      f"({manifest.get('status')}, {manifest.get('accepted_steps')})")
    sessions = manifest.get("sessions") or []
    if not sessions or not all(s.get("provenance", {}).get("attributable") is True for s in sessions):
        raise Invalid(f"{label} run has an unattributable session")
    return manifest


def validate_run(run: Path, label: str) -> dict:
    """A complete run: finished, attributable, every expected checkpoint, nonempty records, the final evaluation."""
    manifest = _manifest(run, label)
    missing = [name for name in EXPECTED_CHECKPOINTS if not (run / "checkpoints" / name).is_file()]
    if missing:
        raise Invalid(f"{label} run lacks checkpoints {missing}")
    for name in ("progress.csv", "evaluations.jsonl", "episodes.jsonl", "archive.json"):
        if not (run / name).is_file() or (run / name).stat().st_size == 0:
            raise Invalid(f"{label} run lacks a nonempty {name}")
    final = _rows_jsonl(run / "evaluations.jsonl")[-1]
    if not isinstance(final, dict) or final.get("accepted_steps") != FINAL_STEPS \
            or final.get("stochastic_episodes") != FINAL_EPISODES:
        raise Invalid(f"{label} run's last evaluation is not the {FINAL_STEPS}-step one with {FINAL_EPISODES} episodes")
    return manifest


def config_problems(original: dict, new: dict) -> list[str]:
    """The new run's training config must equal the original's, allowing only the documented later defaults."""
    extra = {k: v for k, v in new.items() if k not in original}
    bad_extra = {k: v for k, v in extra.items() if COMPATIBILITY_DEFAULTS.get(k, object()) != v}
    problems = [f"new config fields not at their documented defaults: {bad_extra}"] if bad_extra else []
    missing = [k for k in original if k not in new]
    if missing:
        problems.append(f"config fields missing from the new run: {missing}")
    problems += [f"config {k}: {original[k]!r} vs {new[k]!r}" for k in original if k in new and original[k] != new[k]]
    return problems


def latest_successful_run(records: Path, command: list[str], declaration_sha: str, commit: str) -> dict:
    """The newest run record that succeeded with the declared command, declaration and commit."""
    candidates = sorted(records.glob("run-*.json"), reverse=True)
    for path in candidates:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue  # an unreadable record cannot be the successful one
        if not isinstance(record, dict) or not isinstance(record.get("git"), dict):
            continue
        if (record.get("exit_code") == 0 and record.get("command") == command
                and record.get("declaration_text_sha256") == declaration_sha
                and record.get("git", {}).get("commit") == commit and not record.get("problems")):
            return {"path": path.as_posix(), **record}
    raise Invalid("no successful run record with the declared command, declaration and the new run's commit")


# ------------------------------------------------------------------------------------------------------- compare

def compare_runs(original: Path, new: Path, new_recorded: Path | None = None) -> dict:
    """The declared and supplementary comparisons between two runs already validated as complete. new_recorded is
    the folder the new run's training wrote to, when its files were copied elsewhere afterwards."""
    run_a, run_b = _relative(original), new_recorded or _relative(new)
    primary: dict[str, list[str]] = {}
    for name in EXPECTED_CHECKPOINTS:
        primary[f"checkpoints/{name}"] = compare_checkpoint(original / "checkpoints" / name,
                                                            new / "checkpoints" / name, run_a, run_b)
    primary["progress.csv"] = compare_rows(_rows_csv(original / "progress.csv"), _rows_csv(new / "progress.csv"),
                                           VOLATILE_PROGRESS, "progress.csv")
    rows_a, rows_b = _rows_jsonl(original / "evaluations.jsonl"), _rows_jsonl(new / "evaluations.jsonl")
    primary["evaluations.jsonl"] = (evaluation_problems(original, rows_a, "original")
                                    + evaluation_problems(new, rows_b, "new")
                                    + compare_rows(normalized_evaluations(original, rows_a),
                                                   normalized_evaluations(new, rows_b), (), "evaluations.jsonl"))
    supplementary = {
        "episodes.jsonl": compare_rows(_rows_jsonl(original / "episodes.jsonl"), _rows_jsonl(new / "episodes.jsonl"),
                                       (), "episodes.jsonl"),
        "archive.json": structural_differences(json.loads((original / "archive.json").read_text(encoding="utf-8")),
                                               json.loads((new / "archive.json").read_text(encoding="utf-8"))),
    }
    return {"primary": {k: v for k, v in primary.items() if v},
            "supplementary": {k: v for k, v in supplementary.items() if v},
            "checkpoints_compared": list(EXPECTED_CHECKPOINTS),
            "artifacts": {label: {name: _sha256(run / "checkpoints" / name) for name in EXPECTED_CHECKPOINTS}
                          for label, run in (("original", original), ("new", new))}}


def acceptance(original: Path, new: Path, declaration: dict, records: Path, game_dir: Path) -> dict:
    """The acceptance record: INVALID, DIFFERENT, DIAGNOSE or IDENTICAL, with everything it rests on."""
    return _accepted(report_header(original, new), original, new, declaration,
                     lambda manifest: latest_successful_run(records, declared_command(declaration, game_dir),
                                                            fresh_sets.text_sha256(DECLARATION),
                                                            manifest["sessions"][-1]["provenance"]["commit"]))


def report_header(original: Path, new: Path) -> dict:
    return {"original": original.as_posix(), "new": new.as_posix(),
            "ignored": {"data": VOLATILE_DATA, "members": IGNORED_MEMBERS, "progress": VOLATILE_PROGRESS,
                        "evaluations": "checkpoint path reduced to its file name; each hash verified per run",
                        "class_fields": f"{CLASS_FIELDS}: the process address in '<function NAME at 0x...>' and "
                                        "'<_abc._abc_data object at 0x...>' leaves only",
                        "episode_buffer": f"{EPISODE_FIELD}: each entry's wall-clock t value, after checking it"}}


def _accepted(report: dict, original: Path, new: Path, declaration: dict, bind, new_recorded: Path | None = None) -> dict:
    """Validate both runs, bind the new one to its live record (bind(new manifest) returns the record or raises
    Invalid), compare, and give the verdict. Any malformed or unreadable evidence gives INVALID."""
    try:
        manifest_a = validate_run(original, "original")
        pin = declaration["donors"]["existing"]["7"]["sha256"]
        if _sha256(original / "checkpoints" / "latest.zip") != pin:
            raise Invalid("the original's latest.zip is not the declared seed 7 pin")
        manifest_b = validate_run(new, "new")
        problems = config_problems(manifest_a["config"], manifest_b["config"])
        if problems:
            raise Invalid(f"the new run's config differs from the original: {problems}")
        report["run_record"] = bind(manifest_b)
        report.update(compare_runs(original, new, new_recorded))
    except (Invalid, OSError, ValueError, KeyError, IndexError, TypeError, AttributeError, EOFError,
            pickle.UnpicklingError, RuntimeError, zipfile.BadZipFile) as error:
        # Malformed or unreadable evidence of any expected kind: recorded as INVALID, never a crash or a pass.
        report.update(verdict="INVALID", reason=f"{type(error).__name__}: {str(error)[:500]}")
        return report
    report["verdict"] = ("DIFFERENT" if report["primary"] else "DIAGNOSE" if report["supplementary"] else "IDENTICAL")
    return report


def diagnostic_binding(path: Path, declaration: dict, game_dir: Path, manifest: dict) -> dict:
    """The diagnostic's live record, checked: exit 0, no problems, alone with no thread variables, run at the
    historical commit (record and every manifest session), with the declared seed 7 command in its own run folder."""
    record = json.loads(path.read_text(encoding="utf-8"))
    run_dir = recorded_run_dir(record, path.name)
    checks = {"exit_code": 0, "problems": [], "game_processes": [], "other_experiments": [],
              "thread_variables_removed": {}, "worktree_commit": HISTORICAL_COMMIT}
    wrong = {key: record.get(key) for key, expected in checks.items() if record.get(key) != expected}
    if wrong or record.get("timed_out"):
        raise Invalid(f"{path.name} is not a successful historical diagnostic: {wrong or 'timed out'}")
    commits = {session["provenance"].get("commit") for session in manifest["sessions"]}
    if commits != {HISTORICAL_COMMIT}:
        raise Invalid(f"the diagnostic run's sessions ran at {sorted(map(str, commits))}, not {HISTORICAL_COMMIT}")
    command = record["command"]
    expected = declared_command(declaration, game_dir)
    expected[expected.index("--run-dir") + 1] = run_dir
    if command != expected:
        raise Invalid(f"{path.name} command is not the declared seed 7 command: {command}")
    kept = ("purpose", "started", "worktree_commit", "init_clone_sha256", "command", "threads", "exit_code", "seconds",
            "run_copy", "verdict", "reading")
    return {"path": _relative(path).as_posix(), "sha256": _sha256(path), "run_dir": run_dir,
            "launcher_source": launcher_source(path), **{key: record.get(key) for key in kept}}


def recorded_run_dir(record, what: str) -> str:
    """The run folder a diagnostic record's command names; Invalid unless the record and command are well formed."""
    command = record.get("command") if isinstance(record, dict) else None
    if not isinstance(command, list) or not all(isinstance(part, str) for part in command) \
            or command.count("--run-dir") != 1 or command[-1] == "--run-dir":
        raise Invalid(f"{what} is not a record whose command names one run folder")
    return command[command.index("--run-dir") + 1]


def launcher_source(record: Path) -> dict:
    """The diagnostic launcher's source as preserved after the run: the note and the saved file, by path and hash.
    The note must name that file and its hash. This cannot show the bytes that ran; the record says so."""
    stem = record.name.removesuffix(".json")
    note_path, source_path = record.with_name(f"{stem}.source-note.json"), record.with_name(f"{stem}.source-saved-afterwards.py")
    note = json.loads(note_path.read_text(encoding="utf-8"))
    if not isinstance(note, dict) or note.get("file") != source_path.name or note.get("sha256") != _sha256(source_path):
        raise Invalid(f"{note_path.name} does not name {source_path.name} with its hash")
    return {"note": {"path": _relative(note_path).as_posix(), "sha256": _sha256(note_path)},
            "saved_source": {"path": _relative(source_path).as_posix(), "sha256": _sha256(source_path)},
            "saved_afterwards": True,
            "limit": "saved after the diagnostic ran; its hash was not recorded in the live record, so it cannot "
                     "establish the exact launcher bytes that executed"}


def comparator_identity() -> dict:
    """What produced a comparison record: the comparator version, this script's hash and the commit it ran at."""
    return {"version": COMPARATOR_VERSION, "script_sha256": _sha256(Path(__file__)), "git": runtime.git_state()}


def _relative(run: Path) -> Path:
    """The run folder as training recorded it (relative to the repository when inside it)."""
    try:
        return Path(run).resolve().relative_to(REPO.resolve())
    except ValueError:
        return Path(run)


# ----------------------------------------------------------------------------------------------------- the actions

def _write(record: dict, stem: str, extra: dict[str, str] | None = None) -> Path:
    """Create the record (and any text files beside it) exclusively; never overwrite."""
    RECORDS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for attempt in range(1, 100):  # a second record in the same second gets -2, -3, ...; nothing is overwritten
        base = f"{stem}-{stamp}" + (f"-{attempt}" if attempt > 1 else "")
        path = RECORDS / f"{base}.json"
        try:
            handle = open(path, "x", encoding="utf-8")
        except FileExistsError:
            continue
        sides = {suffix: (RECORDS / f"{base}.{suffix}").as_posix() for suffix in (extra or {})}
        with handle:
            handle.write(json.dumps({**record, "log_files": sides} if sides else record, indent=2, default=str) + "\n")
        for suffix, text in (extra or {}).items():
            with open(RECORDS / f"{base}.{suffix}", "x", encoding="utf-8") as side:
                side.write(text)
        return path
    raise FileExistsError(f"no free record name for {stem}-{stamp}")


def thread_probe(env: dict) -> dict:
    probe = subprocess.run([sys.executable, "-c", "import torch, os; print(torch.get_num_threads(), "
                            "torch.get_num_interop_threads(), os.cpu_count())"],
                           cwd=REPO, env=env, capture_output=True, text=True, check=True)
    intra, interop, cpus = (int(x) for x in probe.stdout.split())
    return {"torch_intra_op_threads": intra, "torch_inter_op_threads": interop, "os_cpu_count": cpus}


def launch_problems(command: list[str], game_dir: Path) -> tuple[list[str], dict]:
    """Why the check must not start, and the recorded session conditions."""
    problems = []
    refusal = runtime.refusal(runtime.git_state(), allow_dirty=False)
    if refusal:
        problems.append(refusal)
    check = subprocess.run([sys.executable, *CODE_CHECK], cwd=REPO, capture_output=True, text=True)
    conditions = {"code_check": {"command": CODE_CHECK, "exit_code": check.returncode,
                                 "stdout_tail": check.stdout.strip().splitlines()[-3:],
                                 "stderr_tail": (check.stderr or "").strip().splitlines()[-5:]}}
    if check.returncode != 0:
        problems.append(f"training code check exited {check.returncode}")
    try:
        fresh_sets.refuse_unless_declared([command], ())
    except fresh_sets.FreshSetRefused as error:
        problems.append(str(error))
    if (REPO / CHECK_RUN).exists():
        problems.append(f"{CHECK_RUN} already exists")
    for key, query in (("game_processes", all_game_processes), ("other_experiments", lambda: other_experiments(os.getpid()))):
        try:
            found = query()
        except ProcessEvidenceUnavailable as error:
            conditions[key] = f"unavailable: {error}"
            problems.append(f"cannot establish that the check runs alone: {error}")
            continue
        conditions[key] = found
        if found:
            problems.append(f"{key.replace('_', ' ')} found ({found}); the check must run alone")
    conditions["alone"] = "no Celeste process in any folder and no other script of this repository, at launch"
    return problems, conditions


def run(game_dir: Path, dry_run: bool) -> int:
    declaration = json.loads(DECLARATION.read_text(encoding="utf-8"))
    command = declared_command(declaration, game_dir)
    problems, conditions = launch_problems(command, game_dir)
    env = stripped_env(dict(os.environ))
    record = {"declaration_text_sha256": fresh_sets.text_sha256(DECLARATION), "command": command,
              "thread_variables_removed": thread_variables(dict(os.environ)), "threads": thread_probe(env),
              "threads_note": "probed in the training process's environment; training itself does not log threads",
              "limit_minutes": LIMIT_MINUTES, "problems": problems, "dry_run": dry_run, "git": runtime.git_state(),
              "conditions": conditions, "run_folder": CHECK_RUN.as_posix()}
    if problems or dry_run:
        print(("Refused:\n  " + "\n  ".join(problems)) if problems else f"Dry run passed: {' '.join(command)}")
        print(f"Record: {_write(record, 'dry-run' if dry_run else 'refused')}")
        return 1 if problems else 0
    started = time.time()
    output = {"stdout": "", "stderr": ""}
    try:
        finished = subprocess.run([sys.executable, *command], cwd=REPO, env=env, capture_output=True, text=True,
                                  timeout=LIMIT_MINUTES * 60)
        output.update(stdout=finished.stdout or "", stderr=finished.stderr or "")
        record.update(exit_code=finished.returncode)
    except subprocess.TimeoutExpired as error:
        for key, value in (("stdout", error.stdout), ("stderr", error.stderr)):
            output[key] = value.decode(errors="replace") if isinstance(value, bytes) else (value or "")
        record.update(exit_code=None, timed_out=True)
        record["game_cleanup"] = cleanup_own_game(game_dir)
    except BaseException as error:  # anything else: still keep whatever was captured
        record.update(exit_code=None, launch_error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record.update(seconds=round(time.time() - started), stdout_tail=output["stdout"].strip().splitlines()[-5:],
                      stderr_tail=output["stderr"].strip().splitlines()[-8:])
        print(f"Record: {_write(record, 'run', {'stdout.txt': output['stdout'], 'stderr.txt': output['stderr']})}")
    return 0 if record.get("exit_code") == 0 else 1


def cleanup_own_game(game_dir: Path) -> dict:
    """Stop the game processes in this check's own folder after a timeout (never another folder). Every failure is
    recorded, never raised, so the run's record and logs are always saved."""
    result: dict = {"stopped": [], "errors": []}
    try:
        pids = game_process.running_game_pids(game_dir)
    except Exception as error:  # the record must still be written
        result["errors"].append(f"listing: {type(error).__name__}: {error}")
        pids = []
    for pid in pids:
        try:
            done = subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, text=True, timeout=30)
            result["stopped"].append({"pid": pid, "exit_code": done.returncode})
        except (OSError, subprocess.SubprocessError) as error:
            result["errors"].append(f"pid {pid}: {type(error).__name__}: {error}")
    try:
        result["still_running"] = game_process.running_game_pids(game_dir)
    except Exception as error:
        result["still_running"] = f"unknown: {type(error).__name__}: {error}"
    return result


NEXT_IF_DIFFERENT = ("DIFFERENT: stop. Declared next step (donors.recipe_check.different): rerun the same command at "
                     "8423081 in a separate worktree under the same conditions, to tell 'the code changed behaviour' "
                     "from 'the conditions changed'; then stop for review and an owner choice before any new donor. "
                     "That rerun needs its own authorization; this tool does not start it. (Done on 2026-10-01: see "
                     "compare-diagnostic and the declaration's amendments.)")
DIAGNOSTIC_SCOPE = ("Concerns the historical code 8423081 only. It says nothing about the current code, whose replay "
                    "stays DIFFERENT, and does not by itself authorize any donor: that needs the owner's recorded "
                    "choice in a reviewed, pushed declaration amendment.")


def compare(game_dir: Path) -> int:
    refusal = runtime.refusal(runtime.git_state(), allow_dirty=False)
    if refusal:
        print(f"Refused: a comparison record must name a committed comparator.\n{refusal}")
        return 1
    declaration = json.loads(DECLARATION.read_text(encoding="utf-8"))
    report = {"comparator": comparator_identity(),
              **acceptance(REPO / ORIGINAL, REPO / CHECK_RUN, declaration, RECORDS, game_dir)}
    path = _write(report, "compare")
    messages = {"IDENTICAL": "IDENTICAL: the current code reproduces the arm B recipe for seed 7 under these conditions",
                "DIAGNOSE": "DIAGNOSE: the declared comparisons match but a supplementary one differs; stop and review",
                "INVALID": f"INVALID: {report.get('reason')}", "DIFFERENT": NEXT_IF_DIFFERENT}
    print(messages[report["verdict"]])
    print(f"Record: {path}")
    return 0 if report["verdict"] == "IDENTICAL" else 1


def diagnostic_acceptance(original: Path, run_copy: Path, record: Path, declaration: dict, game_dir: Path) -> dict:
    """The comparison of the original with the historical diagnostic's run copy, bound to its live record."""
    bound: dict = {}

    def bind(manifest):
        bound.update(diagnostic_binding(record, declaration, game_dir, manifest))
        return bound

    report = {**report_header(original, run_copy), "scope": DIAGNOSTIC_SCOPE}
    # The run folder is read first so the abort path can be checked against the folder training actually wrote to.
    try:
        recorded = Path(recorded_run_dir(json.loads(record.read_text(encoding="utf-8")), record.name))
    except (Invalid, OSError, ValueError) as error:
        return {**report, "verdict": "INVALID", "reason": f"{type(error).__name__}: {str(error)[:300]}"}
    report = _accepted(report, original, run_copy, declaration, bind, recorded)
    if bound:
        report["launcher_source"] = bound["launcher_source"]
        report["supersedes"] = {"record": bound["path"], "sha256": bound["sha256"], "field": "reading",
                                "was": bound.get("reading"),
                                "why": f"comparator version 1 counted {CLASS_FIELDS} process addresses and "
                                       f"{EPISODE_FIELD} wall-clock times, which differ between any two processes"}
    return report


def compare_diagnostic(game_dir: Path) -> int:
    refusal = runtime.refusal(runtime.git_state(), allow_dirty=False)
    if refusal:
        print(f"Refused: a comparison record must name a committed comparator.\n{refusal}")
        return 1
    declaration = json.loads(DECLARATION.read_text(encoding="utf-8"))
    report = {"comparator": comparator_identity(),
              **diagnostic_acceptance(REPO / ORIGINAL, DIAGNOSTIC_RUN, DIAGNOSTIC, declaration, game_dir)}
    path = _write(report, "compare-diagnostic")
    messages = {"IDENTICAL": "IDENTICAL: the historical code 8423081 reproduces the original seed 7 run under the "
                             "diagnostic's recorded conditions.",
                "DIAGNOSE": "DIAGNOSE: the declared comparisons match but a supplementary one differs; stop and review.",
                "INVALID": f"INVALID: {report.get('reason')}",
                "DIFFERENT": "DIFFERENT: the historical code does not reproduce the original; stop and review."}
    print(messages[report["verdict"]])
    print(DIAGNOSTIC_SCOPE)
    print(f"Record: {path}")
    return 0 if report["verdict"] == "IDENTICAL" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    game = Path("C:/Projects/celeste-research-scratch/game-probe")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--game-dir", type=Path, default=game)
    run_parser.add_argument("--dry-run", action="store_true")
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("--game-dir", type=Path, default=game, help="the folder the run used (its command)")
    diagnostic_parser = sub.add_parser("compare-diagnostic")
    diagnostic_parser.add_argument("--game-dir", type=Path, default=game, help="the folder the diagnostic used")
    args = parser.parse_args()
    actions = {"run": lambda: run(args.game_dir, args.dry_run), "compare": lambda: compare(args.game_dir),
               "compare-diagnostic": lambda: compare_diagnostic(args.game_dir)}
    return actions[args.action]()


if __name__ == "__main__":
    raise SystemExit(main())
