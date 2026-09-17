"""Run the declared campaign seeds one after another, unattended.

Run from the repo root with the RL interpreter (Steam running, mod installed):
    .venv-rl/Scripts/python.exe scripts/run_campaign.py --seeds 1,2
    .venv-rl/Scripts/python.exe scripts/run_campaign.py --seeds 1,2 --dry-run

Each seed is a separate scripts/train_room1.py process with its own run directory (runs/train/<label>-seed<N>) and
log, so a seed that aborts does not take the others with it. A seed whose run directory already exists is resumed
from its latest checkpoint, which makes this safe to start again after an interruption.

The campaign summary is written to runs/campaign/<timestamp>/results.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from celeste_rl import game_process, runtime  # noqa: E402
from celeste_rl.bridge import CelesteBridge  # noqa: E402

PYTHON = REPO / ".venv-rl" / "Scripts" / "python.exe"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", default="1,2")
    parser.add_argument("--label", default="unshaped", help="run directory prefix")
    parser.add_argument("--game-dir", type=Path, default=Path("C:/Projects/celeste-research-scratch/game-probe"))
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--checkpoint-every", type=int, default=250_000)
    parser.add_argument("--dry-run", action="store_true", help="print the commands and the preflight checks only")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    git = runtime.git_state()
    running = game_process.running_game_pids(args.game_dir)
    problems = []
    if git["uncommitted_changes"]:
        problems.append("uncommitted changes: a campaign result must name its commit")
    if running:
        problems.append(f"a game copy is already running (PID {', '.join(map(str, running))})")
    # The game is not running yet, so the versions it logs come from its last launch; everything else is checked now.
    manifest = runtime.collect(args.game_dir, CelesteBridge(REPO / "runs" / "scratch" / "episode.tas")._prefix_lines())
    problems += [p for p in runtime.check(manifest, runtime.load_pins()) if not p.startswith("running.")]

    output_dir = REPO / "runs" / "campaign" / datetime.now().strftime("%Y%m%d-%H%M%S")
    plan = []
    for seed in seeds:
        run_dir = REPO / "runs" / "train" / f"{args.label}-seed{seed}"
        command = [str(PYTHON), "scripts/train_room1.py"]
        command += ["--resume", str(run_dir)] if (run_dir / "manifest.json").exists() else [
            "--seed", str(seed), "--run-dir", str(run_dir),
            "--total-timesteps", str(args.total_timesteps), "--checkpoint-every", str(args.checkpoint_every)]
        plan.append({"seed": seed, "run_dir": str(run_dir), "command": command,
                     "log": str(output_dir / f"seed{seed}.log")})

    print(f"Commit {git['commit'][:7]}, seeds {seeds}, {args.total_timesteps:,} steps each")
    for step in plan:
        print(f"  seed {step['seed']}: {' '.join(step['command'][1:])}")
    if problems:
        print("Problems:\n  " + "\n  ".join(problems))
    if args.dry_run or problems:
        return 0 if args.dry_run and not problems else 2

    output_dir.mkdir(parents=True)
    results = {**git, "seeds": seeds, "runtime": manifest, "runs": []}
    for step in plan:
        started = time.perf_counter()
        print(f"[{time.strftime('%H:%M:%S')}] seed {step['seed']} starting, log {step['log']}")
        with open(step["log"], "w", encoding="utf-8") as log:
            code = subprocess.run(step["command"], cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode
        record = {**step, "exit_code": code, "minutes": round((time.perf_counter() - started) / 60, 1)}
        manifest_path = Path(step["run_dir"]) / "manifest.json"
        if manifest_path.exists():
            run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            record.update({"status": run_manifest["status"], "accepted_steps": run_manifest["accepted_steps"],
                           "fault_stats": {k: v for k, v in run_manifest["fault_stats"].items() if k != "faults"}})
        results["runs"].append(record)
        (output_dir / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(f"[{time.strftime('%H:%M:%S')}] seed {step['seed']} {record.get('status', 'no manifest')} "
              f"after {record['minutes']} min (exit {code})")

    print(f"Campaign summary: {output_dir / 'results.json'}")
    return 0 if all(r["exit_code"] == 0 for r in results["runs"]) else 1


if __name__ == "__main__":
    sys.exit(main())
