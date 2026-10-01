"""The retention confirmation's recipe check (config/retention-confirmation.json, donors.recipe_check).

Before any new donor is trained, rerun the seed 7 donor command at the current commit, alone and with no thread
variables set, and compare it with the original run trained on 2026-09-19 at 8423081. Only an IDENTICAL verdict on
a validated, complete, successful replay means the current code reproduces the arm B recipe.

Run from the repo root with the RL interpreter (Steam running, no other game or experiment running):
    .venv-rl/Scripts/python.exe scripts/confirmation_recipe_check.py run [--dry-run]
    .venv-rl/Scripts/python.exe scripts/confirmation_recipe_check.py compare

run: refuses unless the tracked tree is clean; the declared training code check exits 0; the command passes the
fresh-set guard with no fresh use; the run folder does not exist; no Celeste process runs in ANY folder and no other
experiment script (campaign, training, evaluation, recording, copying) is running. It removes every thread-related
environment variable (OMP_, KMP_, MKL_, OPENBLAS_, GOTO_, BLIS_, NUMEXPR_, VECLIB_, TORCH_NUM...), records what it
removed and torch's default thread counts in that environment, then trains for up to 90 minutes. Complete stdout and
stderr are saved beside the JSON record, also after a timeout, when this check's own game (in the chosen folder) is
stopped and the result recorded.

compare: three verdicts. INVALID: the evidence is incomplete (no successful run record matching the declared
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
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import pathlib
import pickle
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
EXPERIMENT_SCRIPTS = ("run_overnight.py", "train_room1.py", "train_anchored.py", "evaluate_heldout.py",
                      "evaluate_checkpoint.py", "record_policy_play.py", "clone_room1.py")
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


def other_experiments(own_pid: int) -> list[str]:
    """Command lines of other running experiment scripts (Windows process list)."""
    query = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }")
    listed = subprocess.run(["powershell", "-NoProfile", "-Command", query], capture_output=True, text=True)
    found = []
    for line in listed.stdout.splitlines():
        pid, _, command = line.partition("\t")
        if pid.strip().isdigit() and int(pid) != own_pid and any(name in command for name in EXPERIMENT_SCRIPTS):
            found.append(command.strip()[:300])
    return found


def all_game_processes() -> list[str]:
    return [f"{pid} {exe}" for pid, exe in game_process._running_celeste_processes()]


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


def stored_path_parts(field) -> tuple[str, ...] | None:
    """The parts of an SB3-serialized path field, or None if it is not one."""
    try:
        value = _PathOnly(io.BytesIO(base64.b64decode(field[":serialized:"]))).load()
        return tuple(pathlib.PurePath(value).parts)
    except (KeyError, TypeError, ValueError, pickle.UnpicklingError, EOFError):
        return None


# ----------------------------------------------------------------------------------------------------- checkpoints

def _load_tensors(raw: bytes):
    import torch

    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)


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
                kept_a = {k: v for k, v in data_a.items() if k not in skipped}
                kept_b = {k: v for k, v in data_b.items() if k not in skipped}
                differences += [f"data {d}" for d in structural_differences(kept_a, kept_b)]
            elif name in TENSOR_MEMBERS:
                differences += [f"{name} {d}" for d in structural_differences(_load_tensors(raw_a), _load_tensors(raw_b))]
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


def evaluation_problems(run: Path, rows: list[dict], label: str) -> list[str]:
    """Each evaluation must name a checkpoint file of this run whose sha256 it records truthfully."""
    problems = []
    for index, row in enumerate(rows):
        name = Path(str(row.get("checkpoint", ""))).name
        target = run / "checkpoints" / name
        if not name or not target.is_file():
            problems.append(f"{label} evaluation {index} names {row.get('checkpoint')!r}, not a file of this run")
        elif _sha256(target) != row.get("checkpoint_sha256"):
            problems.append(f"{label} evaluation {index}: recorded checkpoint_sha256 is not that of {target.name}")
    return problems


def normalized_evaluations(rows: list[dict]) -> list[dict]:
    """Evaluation rows with the checkpoint reference reduced to its file name and the per-run file hash removed
    (each hash is verified against its own run's file separately)."""
    return [{**{k: v for k, v in row.items() if k != "checkpoint_sha256"},
             "checkpoint": Path(str(row.get("checkpoint", ""))).name} for row in rows]


# --------------------------------------------------------------------------------------------------- validation

def _manifest(run: Path, label: str) -> dict:
    path = run / "manifest.json"
    if not path.is_file():
        raise Invalid(f"{label} run has no manifest.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
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
    if final.get("accepted_steps") != FINAL_STEPS or final.get("stochastic_episodes") != FINAL_EPISODES:
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
        record = json.loads(path.read_text(encoding="utf-8"))
        if (record.get("exit_code") == 0 and record.get("command") == command
                and record.get("declaration_text_sha256") == declaration_sha
                and record.get("git", {}).get("commit") == commit and not record.get("problems")):
            return {"path": path.as_posix(), **record}
    raise Invalid("no successful run record with the declared command, declaration and the new run's commit")


# ------------------------------------------------------------------------------------------------------- compare

def compare_runs(original: Path, new: Path) -> dict:
    """The declared and supplementary comparisons between two runs already validated as complete."""
    run_a, run_b = _relative(original), _relative(new)
    primary: dict[str, list[str]] = {}
    for name in EXPECTED_CHECKPOINTS:
        primary[f"checkpoints/{name}"] = compare_checkpoint(original / "checkpoints" / name,
                                                            new / "checkpoints" / name, run_a, run_b)
    primary["progress.csv"] = compare_rows(_rows_csv(original / "progress.csv"), _rows_csv(new / "progress.csv"),
                                           VOLATILE_PROGRESS, "progress.csv")
    rows_a, rows_b = _rows_jsonl(original / "evaluations.jsonl"), _rows_jsonl(new / "evaluations.jsonl")
    primary["evaluations.jsonl"] = (evaluation_problems(original, rows_a, "original")
                                    + evaluation_problems(new, rows_b, "new")
                                    + compare_rows(normalized_evaluations(rows_a), normalized_evaluations(rows_b), (),
                                                   "evaluations.jsonl"))
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
    report: dict = {"original": original.as_posix(), "new": new.as_posix(),
                    "ignored": {"data": VOLATILE_DATA, "members": IGNORED_MEMBERS, "progress": VOLATILE_PROGRESS,
                                "evaluations": "checkpoint path reduced to its file name; each hash verified per run"}}
    try:
        manifest_a = validate_run(original, "original")
        pin = declaration["donors"]["existing"]["7"]["sha256"]
        if _sha256(original / "checkpoints" / "latest.zip") != pin:
            raise Invalid("the original's latest.zip is not the declared seed 7 pin")
        manifest_b = validate_run(new, "new")
        problems = config_problems(manifest_a["config"], manifest_b["config"])
        if problems:
            raise Invalid(f"the new run's config differs from the original: {problems}")
        commit = manifest_b["sessions"][-1]["provenance"]["commit"]
        report["run_record"] = latest_successful_run(
            records, declared_command(declaration, game_dir), fresh_sets.text_sha256(DECLARATION), commit)
        report.update(compare_runs(original, new))
    except (Invalid, OSError, ValueError, KeyError, IndexError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        report.update(verdict="INVALID", reason=f"{type(error).__name__}: {error}")
        return report
    report["verdict"] = ("DIFFERENT" if report["primary"] else "DIAGNOSE" if report["supplementary"] else "IDENTICAL")
    return report


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
        with handle:
            handle.write(json.dumps(record, indent=2, default=str) + "\n")
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
    games = all_game_processes()
    others = other_experiments(os.getpid())
    conditions.update(game_processes=games, other_experiments=others,
                      alone="no Celeste process in any folder and no other experiment script, at launch")
    if games:
        problems.append(f"Celeste is running ({games}); the check must run alone")
    if others:
        problems.append(f"other experiment scripts are running ({others}); the check must run alone")
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
    stdout = stderr = ""
    try:
        finished = subprocess.run([sys.executable, *command], cwd=REPO, env=env, capture_output=True, text=True,
                                  timeout=LIMIT_MINUTES * 60)
        stdout, stderr = finished.stdout or "", finished.stderr or ""
        record.update(exit_code=finished.returncode)
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
        stderr = error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else (error.stderr or "")
        stopped = []
        for pid in game_process.running_game_pids(game_dir):  # only this check's game folder
            done = subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, text=True, timeout=30)
            stopped.append({"pid": pid, "exit_code": done.returncode})
        record.update(exit_code=None, timed_out=True, game_cleanup=stopped,
                      game_still_running=game_process.running_game_pids(game_dir))
    record.update(seconds=round(time.time() - started), stdout_tail=stdout.strip().splitlines()[-5:],
                  stderr_tail=stderr.strip().splitlines()[-8:], logs="stdout.txt and stderr.txt beside this record")
    print(f"Record: {_write(record, 'run', {'stdout.txt': stdout, 'stderr.txt': stderr})}")
    return 0 if record.get("exit_code") == 0 else 1


NEXT_IF_DIFFERENT = ("DIFFERENT: stop. Declared next step (donors.recipe_check.different): rerun the same command at "
                     "8423081 in a separate worktree under the same conditions, to tell 'the code changed behaviour' "
                     "from 'the conditions changed'; then stop for review and an owner choice before any new donor. "
                     "That rerun needs its own authorization; this tool does not start it.")


def compare(game_dir: Path) -> int:
    declaration = json.loads(DECLARATION.read_text(encoding="utf-8"))
    report = acceptance(REPO / ORIGINAL, REPO / CHECK_RUN, declaration, RECORDS, game_dir)
    path = _write(report, "compare")
    messages = {"IDENTICAL": "IDENTICAL: the current code reproduces the arm B recipe for seed 7 under these conditions",
                "DIAGNOSE": "DIAGNOSE: the declared comparisons match but a supplementary one differs; stop and review",
                "INVALID": f"INVALID: {report.get('reason')}", "DIFFERENT": NEXT_IF_DIFFERENT}
    print(messages[report["verdict"]])
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
    args = parser.parse_args()
    return run(args.game_dir, args.dry_run) if args.action == "run" else compare(args.game_dir)


if __name__ == "__main__":
    raise SystemExit(main())
