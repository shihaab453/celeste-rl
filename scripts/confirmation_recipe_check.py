"""The retention confirmation's recipe check (config/retention-confirmation.json, donors.recipe_check).

Before any new donor is trained, rerun the seed 7 donor command at the current commit, alone and with no thread
variables set, and compare it with the original run trained on 2026-09-19 at 8423081. Identical results mean the
current code reproduces the arm B recipe, so new donors are that recipe.

Run from the repo root with the RL interpreter (Steam running, no other game script running):
    .venv-rl/Scripts/python.exe scripts/confirmation_recipe_check.py run [--dry-run]
    .venv-rl/Scripts/python.exe scripts/confirmation_recipe_check.py compare

`run` refuses unless the tracked tree is clean, the declared training code check passes, the command passes the
fresh-set guard and the run folder does not exist; it records the thread environment it removed and torch's default
thread counts, then trains for up to 90 minutes. `compare` checks every checkpoint file the original has (every
tensor of the policy, optimizer and saved variables, and every saved field except the wall-clock start time; the
saved abort-checkpoint path must name each run's own <run folder>/checkpoints/aborted.zip, decoded with a reader
that accepts only path objects; the system-info text is ignored), every progress.csv row (except speed and memory columns) and every evaluations.jsonl
row (except the checkpoint's path and file hash, which differ by folder name and start time). Both write a JSON
record under runs/confirmation/recipe-check/ and never overwrite one. Exit 0: identical (or a dry run passed);
1: different or refused.
"""
from __future__ import annotations

import argparse
import base64
import csv
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
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "TORCH_NUM_THREADS")
CODE_CHECK = ["scripts/check_evaluation_code.py", "--entry", "scripts/train_room1.py", "--baseline", "331e6c9",
              "--approved", "44ffab5"]
LIMIT_MINUTES = 90  # the original campaign's limit for this run
VOLATILE_DATA = ("start_time",)                     # wall clock at training start
RUN_FOLDER_DATA = ("abort_checkpoint_path",)        # <run folder>/checkpoints/aborted.zip, checked per run
IGNORED_MEMBERS = ("system_info.txt",)              # OS and library description text
VOLATILE_PROGRESS = ("env_steps_per_second", "game_private_mb", "game_working_set_mb")
VOLATILE_EVALUATION = ("checkpoint", "checkpoint_sha256")
TENSOR_MEMBERS = ("policy.pth", "policy.optimizer.pth", "pytorch_variables.pth")


def stripped_env(env: dict) -> dict:
    """The environment with every thread-count variable removed, so libraries use their machine defaults."""
    return {key: value for key, value in env.items() if key.upper() not in THREAD_VARIABLES}


def declared_command(declaration: dict, game_dir: Path) -> list[str]:
    """The declared donor command for seed 7 with the recipe-check run folder (only --seed and --run-dir change)."""
    template = declaration["donors"]["command"].replace("<s>", "7").split()
    index = template.index("--run-dir")
    template[index + 1] = CHECK_RUN.as_posix()
    return [*template, "--game-dir", str(game_dir)]


def _tensors(value):
    """Flatten a loaded torch object into (path, leaf) pairs for exact comparison."""
    import torch

    if isinstance(value, torch.Tensor):
        yield "", value
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            for path, leaf in _tensors(value[key]):
                yield f"{key}/{path}" if path else str(key), leaf
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            for path, leaf in _tensors(item):
                yield f"{index}/{path}" if path else str(index), leaf
    else:
        yield "", value


def _equal(a, b) -> bool:
    import torch

    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype
                and a.shape == b.shape and torch.equal(a, b))
    return a == b


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


def compare_checkpoint(original: Path, new: Path, run_original: Path | None = None, run_new: Path | None = None) -> list[str]:
    """Differences between two checkpoint zips, ignoring only the declared volatile parts. A run-folder field must
    name each run's own folder (when the run folders are given) and otherwise be equal."""
    import torch

    differences = []
    with zipfile.ZipFile(original) as a, zipfile.ZipFile(new) as b:
        names_a = set(a.namelist()) - set(IGNORED_MEMBERS)
        names_b = set(b.namelist()) - set(IGNORED_MEMBERS)
        if names_a != names_b:
            differences.append(f"members differ: only original {sorted(names_a - names_b)}, "
                               f"only new {sorted(names_b - names_a)}")
        for name in sorted(names_a & names_b):
            raw_a, raw_b = a.read(name), b.read(name)
            if name == "data":
                data_a, data_b = json.loads(raw_a), json.loads(raw_b)
                keys = (set(data_a) | set(data_b)) - set(VOLATILE_DATA)
                if run_original is not None and run_new is not None:
                    keys -= set(RUN_FOLDER_DATA)
                    for key in RUN_FOLDER_DATA:
                        for data, run in ((data_a, run_original), (data_b, run_new)):
                            parts = stored_path_parts(data.get(key, {}))
                            expected = (*pathlib.PurePath(run).parts, "checkpoints", "aborted.zip")
                            if parts != expected:
                                differences.append(f"data field {key} is {parts}, expected {expected}")
                differences += [f"data field {key} differs" for key in sorted(keys) if data_a.get(key) != data_b.get(key)]
            elif name in TENSOR_MEMBERS:
                leaves_a = dict(_tensors(torch.load(io.BytesIO(raw_a), map_location="cpu", weights_only=True)))
                leaves_b = dict(_tensors(torch.load(io.BytesIO(raw_b), map_location="cpu", weights_only=True)))
                if set(leaves_a) != set(leaves_b):
                    differences.append(f"{name}: entries differ")
                differences += [f"{name}: {key or '(value)'} differs"
                                for key in sorted(set(leaves_a) & set(leaves_b)) if not _equal(leaves_a[key], leaves_b[key])]
            elif raw_a != raw_b:
                differences.append(f"member {name} differs")
    return differences


def _rows_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _rows_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compare_rows(original: list[dict], new: list[dict], volatile: tuple, what: str) -> list[str]:
    differences = []
    if len(original) != len(new):
        differences.append(f"{what}: {len(original)} rows in the original, {len(new)} in the new run")
    for index, (a, b) in enumerate(zip(original, new)):
        keys = (set(a) | set(b)) - set(volatile)
        changed = sorted(key for key in keys if a.get(key) != b.get(key))
        if changed:
            differences.append(f"{what} row {index}: {changed}")
    return differences


def _relative(run: Path) -> Path:
    """The run folder as training recorded it (relative to the repository when inside it)."""
    try:
        return Path(run).resolve().relative_to(REPO.resolve())
    except ValueError:
        return Path(run)


def compare_runs(original: Path, new: Path) -> dict:
    """Every declared comparison between the original run and the recipe-check run."""
    checkpoints = sorted(p.name for p in (original / "checkpoints").glob("*.zip"))
    differences: dict[str, list[str]] = {}
    for name in checkpoints:
        target = new / "checkpoints" / name
        differences[f"checkpoints/{name}"] = (
            compare_checkpoint(original / "checkpoints" / name, target, _relative(original), _relative(new))
            if target.exists() else ["missing in the new run"])
    differences["progress.csv"] = compare_rows(_rows_csv(original / "progress.csv"), _rows_csv(new / "progress.csv"),
                                               VOLATILE_PROGRESS, "progress.csv")
    differences["evaluations.jsonl"] = compare_rows(_rows_jsonl(original / "evaluations.jsonl"),
                                                    _rows_jsonl(new / "evaluations.jsonl"), VOLATILE_EVALUATION,
                                                    "evaluations.jsonl")
    differing = {key: value for key, value in differences.items() if value}
    return {"original": original.as_posix(), "new": new.as_posix(), "checkpoints_compared": checkpoints,
            "ignored": {"data": VOLATILE_DATA, "members": IGNORED_MEMBERS, "progress": VOLATILE_PROGRESS,
                        "evaluations": VOLATILE_EVALUATION},
            "identical": not differing, "differences": differing}


def _write(record: dict, stem: str) -> Path:
    RECORDS.mkdir(parents=True, exist_ok=True)
    path = RECORDS / f"{stem}-{datetime.now():%Y%m%d-%H%M%S}.json"
    if path.exists():
        raise SystemExit(f"{path} already exists")
    path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def thread_probe(env: dict) -> dict:
    probe = subprocess.run([sys.executable, "-c", "import torch, os; print(torch.get_num_threads(), "
                            "torch.get_num_interop_threads(), os.cpu_count())"],
                           cwd=REPO, env=env, capture_output=True, text=True, check=True)
    intra, interop, cpus = (int(x) for x in probe.stdout.split())
    return {"torch_intra_op_threads": intra, "torch_inter_op_threads": interop, "os_cpu_count": cpus}


def run(game_dir: Path, dry_run: bool) -> int:
    declaration = json.loads(DECLARATION.read_text(encoding="utf-8"))
    command = declared_command(declaration, game_dir)
    problems = []
    refusal = runtime.refusal(runtime.git_state(), allow_dirty=False)
    if refusal:
        problems.append(refusal)
    check = subprocess.run([sys.executable, *CODE_CHECK], cwd=REPO, capture_output=True, text=True)
    if "PASS" not in check.stdout:
        problems.append(f"training code check did not pass: {check.stdout.strip()[-300:]}")
    try:
        fresh_sets.refuse_unless_declared([command], ())
    except fresh_sets.FreshSetRefused as error:
        problems.append(str(error))
    if (REPO / CHECK_RUN).exists():
        problems.append(f"{CHECK_RUN} already exists")
    if game_process.running_game_pids(game_dir):
        problems.append(f"a game is already running in {game_dir}")
    env = stripped_env(dict(os.environ))
    record = {"declaration_text_sha256": fresh_sets.text_sha256(DECLARATION), "command": command,
              "thread_variables_removed": sorted(k for k in os.environ if k.upper() in THREAD_VARIABLES),
              "threads": thread_probe(env), "limit_minutes": LIMIT_MINUTES, "problems": problems,
              "dry_run": dry_run, "git": runtime.git_state()}
    if problems or dry_run:
        print(("Refused:\n  " + "\n  ".join(problems)) if problems else f"Dry run passed: {' '.join(command)}")
        print(f"Record: {_write(record, 'dry-run' if dry_run else 'refused')}")
        return 1 if problems else 0
    started = time.time()
    try:
        finished = subprocess.run([sys.executable, *command], cwd=REPO, env=env, capture_output=True, text=True,
                                  timeout=LIMIT_MINUTES * 60)
        record.update(exit_code=finished.returncode, stdout_tail=finished.stdout.strip().splitlines()[-5:],
                      stderr_tail=(finished.stderr or "").strip().splitlines()[-8:])
    except subprocess.TimeoutExpired:
        record.update(exit_code=None, timed_out=True)
    record["seconds"] = round(time.time() - started)
    print(f"Record: {_write(record, 'run')}")
    return 0 if record.get("exit_code") == 0 else 1


def compare() -> int:
    report = compare_runs(REPO / ORIGINAL, REPO / CHECK_RUN)
    path = _write(report, "compare")
    print("IDENTICAL: the current code reproduces the arm B recipe" if report["identical"]
          else f"DIFFERENT in {len(report['differences'])} places; stop for review (declaration: recipe_check.different)")
    print(f"Record: {path}")
    return 0 if report["identical"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--game-dir", type=Path, default=Path("C:/Projects/celeste-research-scratch/game-probe"))
    run_parser.add_argument("--dry-run", action="store_true")
    sub.add_parser("compare")
    args = parser.parse_args()
    return run(args.game_dir, args.dry_run) if args.action == "run" else compare()


if __name__ == "__main__":
    raise SystemExit(main())
