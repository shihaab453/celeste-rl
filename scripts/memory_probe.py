"""Isolate what makes the game's memory grow during long lockstep runs.

Run from the repo root with the RL interpreter, one condition per run:
    .venv-rl/Scripts/python.exe scripts/memory_probe.py --mode resets --window minimized --minutes 4

Modes:
  steps   long episodes: stepping with a reset only every 100,000 steps
  resets  back-to-back resets with one step each (savestate restores dominate)
  mixed   the soak test's pattern: a reset every 300 steps

Every --sample-seconds it records the game's working set and private bytes (GetProcessMemoryInfo)
alongside cumulative steps and resets. Writes runs/memory-probe/<timestamp>-<mode>-<window>/results.json.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from celeste_rl import game_process  # noqa: E402
from celeste_rl.bridge import CelesteBridge  # noqa: E402
from celeste_rl.lockstep import LockstepBridge  # noqa: E402

EPISODE_LENGTH = {"steps": 100_000, "resets": 1, "mixed": 300}
RANDOM_BUTTONS = "LRUDJXZG"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game-dir", type=Path, default=Path("C:/Projects/celeste-research-scratch/game-probe"))
    parser.add_argument("--mode", choices=list(EPISODE_LENGTH), required=True)
    parser.add_argument("--window", choices=["focused", "background", "minimized"], default="minimized")
    parser.add_argument("--minutes", type=float, default=4.0)
    parser.add_argument("--sample-seconds", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{args.mode}-{args.window}"
    output_dir = REPO / "runs" / "memory-probe" / name
    output_dir.mkdir(parents=True)
    git = lambda *a: subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()
    results = {"commit": git("rev-parse", "HEAD"), "uncommitted_changes": bool(git("status", "--porcelain")),
               "args": {**vars(args), "game_dir": str(args.game_dir)}, "samples": []}
    rng = random.Random(args.seed)
    episode_length = EPISODE_LENGTH[args.mode]

    process = game_process.launch(args.game_dir, focus=False)
    bridge = LockstepBridge(CelesteBridge(output_dir / "episode.tas"))
    try:
        bridge.reset()
        game_process.set_window_mode(process.pid, args.window)
        start = time.perf_counter()
        next_sample = start
        steps = resets = 0
        in_episode = 0

        while True:
            now = time.perf_counter()
            if now >= next_sample:
                memory = game_process.process_memory(process.pid)
                sample = {"seconds": round(now - start, 1), "steps": steps, "resets": resets,
                          "working_set_mb": round(memory["working_set_mb"]), "private_mb": round(memory["private_mb"]),
                          "focused": game_process.window_is_focused(process.pid),
                          "minimized": game_process.window_is_minimized(process.pid)}
                results["samples"].append(sample)
                print(f"  {sample['seconds']:6.0f} s  steps {steps:>9,}  resets {resets:>7,}  "
                      f"working set {sample['working_set_mb']:>5} MB  private {sample['private_mb']:>5} MB  "
                      f"focused {sample['focused']}  minimized {sample['minimized']}")
                next_sample += args.sample_seconds
                if now - start >= args.minutes * 60:
                    break

            if in_episode >= episode_length:
                bridge.reset()
                resets += 1
                in_episode = 0
            bridge.step("".join(b for b in RANDOM_BUTTONS if rng.random() < 0.3))
            steps += 1
            in_episode += 1
    finally:
        bridge.close()
        (output_dir / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
        game_process.stop(process)

    first, last = results["samples"][1], results["samples"][-1]
    print(f"{args.mode}/{args.window}: private {first['private_mb']} -> {last['private_mb']} MB, "
          f"working set {first['working_set_mb']} -> {last['working_set_mb']} MB, "
          f"{last['steps']:,} steps, {last['resets']:,} resets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
