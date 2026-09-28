"""Run a declared campaign of runs unattended, and leave usable results if it dies halfway.

Run from the repo root with the RL interpreter (Steam running, sleep disabled):
    .venv-rl/Scripts/python.exe scripts/run_overnight.py --plan config/campaign-finetune-variance.json
    .venv-rl/Scripts/python.exe scripts/run_overnight.py --plan <plan> --dry-run

The plan is a committed file listing every run in order with its exact command, written before the campaign
starts. That is the guard against the failure mode where a run crashes at 3am, a seed is quietly swapped, and
the morning's table describes an experiment nobody designed. The summary records the plan's own sha256, so the
campaign is auditable the same way a single run is.

What this does that a shell loop does not:

- **Refuses to start** on a dirty tree, on a run directory that already holds a manifest, or while a game
  process is holding the ports. An untracked file is enough to make the tree dirty, which is why the held-out
  set has to be committed before the night rather than generated during it.
- **Bounds every run in wall-clock time.** `train_room1.py` recovers from bridge faults and relaunches a dead
  game, but a game that is alive and not answering only surfaces when a deadline fires. On expiry the child is
  killed and any game still holding the ports is stopped before the next run starts.
- **Retries an abort once** (`--resume`), then records it and moves on. Never retries forever, and never drops
  a declared run from the summary: every entry appears with its status, including the ones that failed.
- **Writes after every run.** `summary.json` and a timestamped `campaign.log`, so a crash at hour five still
  leaves the first four answers.
- **Stops rather than starts** a run that cannot finish before the declared stop time.
- **Optionally runs 2 or 3 at once** (`--copy DIR:DEBUGRC_PORT:LOCKSTEP_PORT`, once per game copy): each job gets
  its own game copy and ports through CELESTE_RL_DEBUGRC_PORT and CELESTE_RL_LOCKSTEP_PORT, and
  `--threads-per-job N` sets OMP_NUM_THREADS for each child. Each copy's Everest settings must name its DebugRC
  port. The 2026-09-28 feasibility test found episodes identical in 1, 2 and 3 copies and 1.8x / 2.4x throughput
  with one torch thread per job. Without --copy, everything runs as before: one copy, default ports, the child
  environment unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from celeste_rl import game_process, runtime  # noqa: E402


RESULT_SUMMARY_FIELDS = (
    "checkpoint",
    "checkpoint_sha256",
    "heldout_set",
    "heldout_sha256",
    "heldout_format_version",
    "heldout_states",
    "repeats",
    "deterministic",
    "evaluation_seed",
    "attempts",
    "episodes",
    "stale_starts",
    "successes",
    "success_rate",
    "uncertainty",
    "route_macro_success_rate",
)


_LOG_LOCK = threading.Lock()


def log(path: Path, message: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {message}"
    with _LOG_LOCK:  # parallel jobs log from several threads
        print(line, flush=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


# Children name their output folders after the current second (runs/<kind>/%Y%m%d-%H%M%S), so two started in the same
# second collide and the second crashes. Side by side, starts are therefore spaced at least this far apart.
START_SPACING_SECONDS = 1.5
_START_LOCK = threading.Lock()
_LAST_START = [float("-inf")]


def wait_for_start_slot() -> None:
    with _START_LOCK:
        wait = _LAST_START[0] + START_SPACING_SECONDS - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _LAST_START[0] = time.monotonic()


@dataclass(frozen=True)
class GameCopy:
    """A game copy and the ports its child process uses. No ports: the defaults, exactly as a single copy always ran."""

    game_dir: Path
    debug_port: int | None = None
    lockstep_port: int | None = None

    def env(self, threads: int | None) -> dict | None:
        """The child's environment; None (inherit unchanged) for a default copy without a thread setting."""
        if self.debug_port is None and threads is None:
            return None
        env = dict(os.environ)
        if self.debug_port is not None:
            env.update(CELESTE_RL_DEBUGRC_PORT=str(self.debug_port), CELESTE_RL_LOCKSTEP_PORT=str(self.lockstep_port))
        if threads is not None:
            env.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
        return env


def parse_copy(text: str) -> GameCopy:
    """DIR:DEBUGRC_PORT:LOCKSTEP_PORT; the directory may itself contain colons (C:/...)."""
    try:
        directory, debug, lockstep = text.rsplit(":", 2)
        return GameCopy(Path(directory), int(debug), int(lockstep))
    except ValueError as error:
        raise ValueError(f"--copy {text!r} is not DIR:DEBUGRC_PORT:LOCKSTEP_PORT") from error


def settings_debugrc_port(game_dir: Path) -> int | None:
    settings = Path(game_dir) / "probe-profile" / "Saves" / "modsettings-Everest.celeste"
    if not settings.exists():
        return None
    match = re.search(r"^DebugRCPort:\s*(\d+)\s*$", settings.read_text(encoding="utf-8-sig"), re.MULTILINE)
    return int(match.group(1)) if match else None


def copy_problems(copies: list[GameCopy], port_open=game_process._port_accepts_connections) -> list[str]:
    """Reasons these copies cannot run side by side (empty when they can)."""
    problems = []
    ports = [port for copy in copies for port in (copy.debug_port, copy.lockstep_port)]
    if len(set(ports)) != len(ports):
        problems.append(f"the copies' ports are not all different: {ports}")
    if len({Path(copy.game_dir).resolve() for copy in copies}) != len(copies):
        problems.append("two copies name the same game directory")
    for copy in copies:
        configured = settings_debugrc_port(copy.game_dir)
        if configured != copy.debug_port:
            problems.append(f"{copy.game_dir}: its Everest settings name DebugRC port {configured}, not {copy.debug_port}")
        for port in (copy.debug_port, copy.lockstep_port):
            if port_open(port):
                problems.append(f"port {port} is already in use")
    return problems


def clear_game(game_dir: Path, logfile: Path) -> None:
    """Leave no game process holding the ports before the next run launches into them."""
    for pid in game_process.running_game_pids(game_dir):
        log(logfile, f"  stopping stray game process {pid}")
        try:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as error:
            log(logfile, f"  could not stop {pid}: {error}")
    time.sleep(3)


def _without_option(arguments: list[str], option: str) -> list[str]:
    """Remove both ``--option value`` and ``--option=value`` forms."""
    cleaned = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == option:
            if index + 1 >= len(arguments):
                raise ValueError(f"{option} requires a value")
            index += 2
            continue
        if argument.startswith(f"{option}="):
            index += 1
            continue
        cleaned.append(argument)
        index += 1
    return cleaned


def build_command(entry: dict, game_dir: Path, resume: bool = False) -> list[str]:
    """Build one child command with the same game directory used by the process guard."""
    declared = list(entry.get("command", []))
    if not declared:
        raise ValueError(f"campaign entry {entry.get('id', '<unknown>')} has no command")
    if resume:
        if Path(declared[0]).name.lower() != "train_room1.py":
            raise ValueError(f"campaign entry {entry.get('id', '<unknown>')} does not support --resume")
        child_arguments = [declared[0], "--resume", str(entry["run_dir"])]
    else:
        child_arguments = _without_option(declared, "--game-dir")
    child_arguments.extend(["--game-dir", str(game_dir.resolve())])
    return [str(REPO / ".venv-rl" / "Scripts" / "python.exe"), *child_arguments]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_path(path: Path, repo: Path) -> str:
    try:
        return path.relative_to(repo.resolve()).as_posix()
    except ValueError:
        return str(path)


def result_artifact(stdout: str, repo: Path = REPO) -> dict | None:
    """Capture the result and episode files announced by a held-out evaluation."""
    announced = [line.split("Results:", 1)[1].strip()
                 for line in stdout.splitlines() if line.strip().startswith("Results:")]
    if not announced:
        return None

    repository = repo.resolve()
    result_path = Path(announced[-1])
    if not result_path.is_absolute():
        result_path = repository / result_path
    result_path = result_path.resolve()
    if result_path.suffix.lower() != ".json":
        result_path /= "results.json"

    artifact = {"result_file": _repo_path(result_path, repository)}
    try:
        result_path.relative_to(repository)
    except ValueError:
        artifact["problem"] = "announced result is outside the repository"
        return artifact
    if not result_path.is_file():
        artifact["problem"] = "announced result file does not exist"
        return artifact

    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        artifact["problem"] = f"could not read announced result: {error}"
        return artifact

    artifact["result_sha256"] = _sha256(result_path)
    artifact["summary"] = {field: result[field] for field in RESULT_SUMMARY_FIELDS if field in result}
    episodes_file = result.get("episodes_file")
    if episodes_file:
        episodes_path = (result_path.parent / episodes_file).resolve()
        artifact["episodes_file"] = _repo_path(episodes_path, repository)
        try:
            episodes_path.relative_to(repository)
        except ValueError:
            artifact["problem"] = "declared episode file is outside the repository"
        else:
            if episodes_path.is_file():
                artifact["episodes_sha256"] = _sha256(episodes_path)
            else:
                artifact["problem"] = "declared episode file does not exist"
    return artifact


def execute(entry: dict, logfile: Path, game_dir: Path, resume: bool = False, env: dict | None = None) -> dict:
    """One run, bounded in time. Returns its outcome. `env` is the child's environment (None: inherit)."""
    command = build_command(entry, game_dir, resume=resume)
    limit = entry.get("limit_minutes", 75) * 60
    started = time.time()
    wait_for_start_slot()
    log(logfile, f"start {entry['id']}{' (resume)' if resume else ''}: {' '.join(command[1:])}")
    try:
        finished = subprocess.run(command, cwd=REPO, capture_output=True, text=True, timeout=limit, env=env)
        code = finished.returncode
        stdout = finished.stdout or ""
        tail = stdout.strip().splitlines()[-3:]
    except subprocess.TimeoutExpired:
        log(logfile, f"  {entry['id']} exceeded {entry.get('limit_minutes', 75)} minutes and was killed")
        clear_game(game_dir, logfile)
        return {"status": "timed_out", "seconds": round(time.time() - started), "command": command[1:]}
    for line in tail:
        log(logfile, f"  | {line}")
    if code != 0:
        # A child that crashes before printing anything leaves only its traceback, on stderr.
        for line in (finished.stderr or "").strip().splitlines()[-6:]:
            log(logfile, f"  ! {line}")
    outcome = {"status": "ok" if code == 0 else f"exit_{code}",
               "seconds": round(time.time() - started), "command": command[1:]}
    artifact = result_artifact(stdout)
    if artifact is not None:
        outcome["artifact"] = artifact
    return outcome


def merge_attempts(first: dict, retry: dict) -> dict:
    """Use the retry as the final outcome while retaining both attempts for audit."""
    return {**retry, "seconds": first["seconds"] + retry["seconds"], "attempts": [first, retry]}


def campaign_exit_code(results: list[dict]) -> int:
    return 0 if all(result.get("status") == "ok" for result in results) else 1


def reported_success_rate(record: dict):
    training_rate = record.get("final_success_rate")
    if training_rate is not None:
        return training_rate
    return record.get("artifact", {}).get("summary", {}).get("success_rate")


def write_summary(path: Path, plan: dict, plan_hash: str, commit: str, results: list[dict],
                  parallel: dict | None = None) -> None:
    path.write_text(json.dumps(
        {"plan": plan["name"], "plan_sha256": plan_hash, "commit": commit,
         "definitions": plan.get("definitions"), "results": results,
         **({"parallel": parallel} if parallel else {})}, indent=2, default=str), encoding="utf-8")


def outcome_of(run_dir: Path) -> dict:
    """What a finished run says about itself, read from its own records."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return {"manifest": None}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evaluations = run_dir / "evaluations.jsonl"
    final = None
    if evaluations.exists():
        lines = [json.loads(line) for line in evaluations.open(encoding="utf-8")]
        final = lines[-1] if lines else None
    return {"manifest": manifest["status"], "accepted_steps": manifest["accepted_steps"],
            "discarded_rollouts": manifest["fault_stats"]["discarded_rollouts"],
            "final_success_rate": final and final["stochastic_success_rate"],
            "final_median_max_x": final and final.get("median_max_x"),
            "final_deterministic": final and final["deterministic"]["ending"]}


def side_by_side(entries: list[dict], copies: list[GameCopy], run_one, publish) -> list[dict]:
    """Run every entry, at most one per copy at a time: each job takes a free copy and returns it when done.
    `publish` gets the finished records in plan order after every job; the result is all records in plan order."""
    free: queue.Queue[GameCopy] = queue.Queue()
    for copy in copies:
        free.put(copy)
    slots: list[dict | None] = [None] * len(entries)
    lock = threading.Lock()

    def job(index: int, entry: dict) -> None:
        copy = free.get()
        try:
            record = run_one(entry, copy)
        finally:
            free.put(copy)
        with lock:
            slots[index] = record
            publish([r for r in slots if r is not None])

    with ThreadPoolExecutor(max_workers=len(copies)) as pool:
        list(pool.map(job, range(len(entries)), entries))
    return [r for r in slots if r is not None]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--game-dir", type=Path, default=Path("C:/Projects/celeste-research-scratch/game-probe"))
    parser.add_argument("--dry-run", action="store_true", help="check everything and print the plan, run nothing")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--copy", action="append", default=[], metavar="DIR:DEBUGRC_PORT:LOCKSTEP_PORT",
                        help="run jobs side by side, one per game copy (repeat for each copy; replaces --game-dir)")
    parser.add_argument("--threads-per-job", type=int, help="OMP_NUM_THREADS for each child (1 is best side by side)")
    args = parser.parse_args()
    try:
        copies = [parse_copy(text) for text in args.copy] or [GameCopy(args.game_dir)]
    except ValueError as error:
        parser.error(str(error))
    parallel = ({"copies": [{"game_dir": str(c.game_dir), "ports": [c.debug_port, c.lockstep_port]} for c in copies],
                 "threads_per_job": args.threads_per_job} if args.copy or args.threads_per_job else None)

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    plan_hash = hashlib.sha256(args.plan.read_bytes()).hexdigest()
    output_dir = REPO / "runs" / "campaign" / f"{datetime.now():%Y%m%d-%H%M%S}-{plan['name']}"
    output_dir.mkdir(parents=True)
    logfile = output_dir / "campaign.log"

    git = runtime.git_state()
    refusal = runtime.refusal(git, args.allow_dirty)
    problems = [refusal] if refusal else []
    for copy in copies:
        for pid in game_process.running_game_pids(copy.game_dir):
            problems.append(f"a game process is already running in {copy.game_dir} (pid {pid}); it would hold the ports")
    if args.copy:
        problems.extend(copy_problems(copies))
    for entry in plan["runs"]:
        if (REPO / entry["run_dir"] / "manifest.json").exists():
            problems.append(f"{entry['run_dir']} already holds a run")
    if problems:
        print("Not starting:\n  " + "\n  ".join(problems))
        return 2

    stop_at = datetime.fromisoformat(plan["stop_time"]) if plan.get("stop_time") else None
    log(logfile, f"campaign {plan['name']}, {len(plan['runs'])} runs, commit {git['commit'][:8]}, "
                 f"plan {plan_hash[:16]}")
    if stop_at:
        log(logfile, f"declared stop time {stop_at:%Y-%m-%d %H:%M}, "
                     f"{(stop_at - datetime.now()).total_seconds() / 3600:.1f} hours from now")
    for entry in plan["runs"]:
        log(logfile, f"  planned {entry['id']}: {entry.get('condition', '')} "
                     f"limit {entry.get('limit_minutes', 75)} min")
    if parallel:
        log(logfile, f"side by side: {len(copies)} game copies {parallel['copies']}, "
                     f"threads per job {args.threads_per_job}")
    if args.dry_run:
        log(logfile, "dry run: nothing executed")
        return 0

    def run_entry(entry: dict, copy: GameCopy) -> dict:
        if stop_at and datetime.now() + timedelta(minutes=entry.get("limit_minutes", 75)) > stop_at:
            log(logfile, f"skip {entry['id']}: cannot finish before the declared stop time")
            return {**entry, "status": "skipped_out_of_time", "seconds": 0, "manifest": None}
        env = copy.env(args.threads_per_job)
        clear_game(copy.game_dir, logfile)
        result = execute(entry, logfile, copy.game_dir, env=env)
        # An aborted run exhausted its fault budget; give it exactly one resume, then move on.
        if result["status"] != "ok" and (REPO / entry["run_dir"] / "manifest.json").exists() and entry.get("resumable", True):
            log(logfile, f"  {entry['id']} ended {result['status']}, retrying once with --resume")
            clear_game(copy.game_dir, logfile)
            retry = execute(entry, logfile, copy.game_dir, resume=True, env=env)
            result = merge_attempts(result, retry)
        record = {**entry, **result, **outcome_of(REPO / entry["run_dir"])}
        if parallel:
            record["game_copy"] = str(copy.game_dir)
        log(logfile, f"done {entry['id']}: {record['status']} in {record['seconds'] // 60} min, "
                     f"final success {reported_success_rate(record)}")
        return record

    if len(copies) == 1:
        results = []
        for entry in plan["runs"]:
            results.append(run_entry(entry, copies[0]))
            write_summary(output_dir / "summary.json", plan, plan_hash, git["commit"], results, parallel)
    else:
        results = side_by_side(plan["runs"], copies, run_entry, lambda done: write_summary(
            output_dir / "summary.json", plan, plan_hash, git["commit"], done, parallel))

    finished = [r for r in results if r.get("status") == "ok"]
    log(logfile, f"campaign finished: {len(finished)} of {len(plan['runs'])} runs completed")
    for record in results:
        log(logfile, f"  {record['id']:<24} {record.get('status'):<22} "
                     f"success {reported_success_rate(record)}")
    return campaign_exit_code(results)


if __name__ == "__main__":
    sys.exit(main())
