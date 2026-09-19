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
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from celeste_rl import game_process, runtime  # noqa: E402


def log(path: Path, message: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def clear_game(game_dir: Path, logfile: Path) -> None:
    """Leave no game process holding the ports before the next run launches into them."""
    for pid in game_process.running_game_pids(game_dir):
        log(logfile, f"  stopping stray game process {pid}")
        try:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as error:
            log(logfile, f"  could not stop {pid}: {error}")
    time.sleep(3)


def execute(entry: dict, logfile: Path, game_dir: Path, resume: bool = False) -> dict:
    """One run, bounded in time. Returns its outcome."""
    command = [str(REPO / ".venv-rl" / "Scripts" / "python.exe"), *entry["command"]]
    if resume:
        command += ["--resume", entry["run_dir"]]
    limit = entry.get("limit_minutes", 75) * 60
    started = time.time()
    log(logfile, f"start {entry['id']}{' (resume)' if resume else ''}: {' '.join(entry['command'])}")
    try:
        finished = subprocess.run(command, cwd=REPO, capture_output=True, text=True, timeout=limit)
        code, tail = finished.returncode, (finished.stdout or "").strip().splitlines()[-3:]
    except subprocess.TimeoutExpired:
        log(logfile, f"  {entry['id']} exceeded {entry.get('limit_minutes', 75)} minutes and was killed")
        clear_game(game_dir, logfile)
        return {"status": "timed_out", "seconds": round(time.time() - started)}
    for line in tail:
        log(logfile, f"  | {line}")
    return {"status": "ok" if code == 0 else f"exit_{code}", "seconds": round(time.time() - started)}


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--game-dir", type=Path, default=Path("C:/Projects/celeste-research-scratch/game-probe"))
    parser.add_argument("--dry-run", action="store_true", help="check everything and print the plan, run nothing")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    plan_hash = hashlib.sha256(args.plan.read_bytes()).hexdigest()
    output_dir = REPO / "runs" / "campaign" / f"{datetime.now():%Y%m%d-%H%M%S}-{plan['name']}"
    output_dir.mkdir(parents=True)
    logfile = output_dir / "campaign.log"

    git = runtime.git_state()
    refusal = runtime.refusal(git, args.allow_dirty)
    problems = [refusal] if refusal else []
    for pid in game_process.running_game_pids(args.game_dir):
        problems.append(f"a game process is already running (pid {pid}); it would hold the ports")
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
    if args.dry_run:
        log(logfile, "dry run: nothing executed")
        return 0

    results = []
    for entry in plan["runs"]:
        if stop_at and datetime.now() + timedelta(minutes=entry.get("limit_minutes", 75)) > stop_at:
            log(logfile, f"skip {entry['id']}: cannot finish before the declared stop time")
            results.append({**entry, "status": "skipped_out_of_time"})
            continue
        clear_game(args.game_dir, logfile)
        result = execute(entry, logfile, args.game_dir)
        # An aborted run exhausted its fault budget; give it exactly one resume, then move on.
        if result["status"] != "ok" and (REPO / entry["run_dir"] / "manifest.json").exists() and entry.get("resumable", True):
            log(logfile, f"  {entry['id']} ended {result['status']}, retrying once with --resume")
            clear_game(args.game_dir, logfile)
            retry = execute(entry, logfile, args.game_dir, resume=True)
            result = {"status": f"{result['status']}_then_{retry['status']}",
                      "seconds": result["seconds"] + retry["seconds"]}
        record = {**entry, **result, **outcome_of(REPO / entry["run_dir"])}
        results.append(record)
        log(logfile, f"done {entry['id']}: {record['status']} in {record['seconds'] // 60} min, "
                     f"final success {record.get('final_success_rate')}")
        (output_dir / "summary.json").write_text(json.dumps(
            {"plan": plan["name"], "plan_sha256": plan_hash, "commit": git["commit"],
             "definitions": plan.get("definitions"), "results": results}, indent=2, default=str), encoding="utf-8")

    finished = [r for r in results if r.get("status") == "ok"]
    log(logfile, f"campaign finished: {len(finished)} of {len(plan['runs'])} runs completed")
    for record in results:
        log(logfile, f"  {record['id']:<24} {record.get('status'):<22} "
                     f"success {record.get('final_success_rate')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
