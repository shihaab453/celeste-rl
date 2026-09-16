"""Check the lockstep bridge against the HTTP bridge, then measure its speed.

Run from the repo root with the RL interpreter (the CelesteRLLockstep mod must be installed):
    .venv-rl/Scripts/python.exe scripts/lockstep_check.py

1. Equivalence: the same seeded inputs are played through the HTTP bridge (the reference) and the
   lockstep bridge. The complete game state must match on every frame.
2. Restore: the lockstep trace is replayed after an unrelated episode and must match again.
3. Throughput: random legal inputs with regular resets over the lockstep bridge.

Results are written to runs/lockstep-check/<timestamp>/results.json.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
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

RANDOM_BUTTONS = "LRUDJXZG"


def random_actions(rng: random.Random, count: int) -> list[str]:
    return ["".join(b for b in RANDOM_BUTTONS if rng.random() < 0.3) for _ in range(count)]


def trace(bridge, actions: list[str]) -> list[dict]:
    """Reset, play the actions, and record the frame number and complete state after each one."""
    start = bridge.reset()
    frames = [{"frame": start.tas_frame, "state": start.state}]
    for buttons in actions:
        observation = bridge.step(buttons)
        frames.append({"frame": observation.tas_frame, "state": observation.state})
    return frames


def first_difference(expected: list[dict], actual: list[dict]) -> dict | None:
    for index, (a, b) in enumerate(zip(expected, actual)):
        if a != b:
            fields = sorted(
                key for key in set(a["state"] or {}) | set(b["state"] or {})
                if (a["state"] or {}).get(key) != (b["state"] or {}).get(key)
            )
            return {"step": index, "frame_expected": a["frame"], "frame_actual": b["frame"], "differing_fields": fields}
    if len(expected) != len(actual):
        return {"step": min(len(expected), len(actual)), "reason": "different lengths"}
    return None


def summarize(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": ordered[int(0.5 * (len(ordered) - 1))],
        "p99_ms": ordered[int(0.99 * (len(ordered) - 1))],
        "max_ms": ordered[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game-dir", type=Path, default=Path("C:/Projects/celeste-research-scratch/game-probe"))
    parser.add_argument("--attach", action="store_true")
    parser.add_argument("--trace-length", type=int, default=300)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--episode-length", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = REPO / "runs" / "lockstep-check" / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True)
    git = lambda *a: subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()
    results = {"commit": git("rev-parse", "HEAD"), "uncommitted_changes": bool(git("status", "--porcelain")),
               "args": {**vars(args), "game_dir": str(args.game_dir)}}

    process = None if args.attach else game_process.launch(args.game_dir)
    http = CelesteBridge(output_dir / "episode.tas")
    lockstep = LockstepBridge(http)
    passed = False
    try:
        rng = random.Random(args.seed)
        actions = random_actions(rng, args.trace_length)
        other = random_actions(rng, args.trace_length)

        print(f"1. Reference trace over HTTP ({args.trace_length} steps)")
        reference = trace(http, actions)

        print("2. Same inputs over lockstep")
        fast = trace(lockstep, actions)
        trace(lockstep, other)
        fast_after_other = trace(lockstep, actions)

        # Keep the actual inputs and per-frame states, not just a summary of the first difference.
        (output_dir / "traces.json").write_text(json.dumps({
            "actions": actions,
            "other_actions": other,
            "http": reference,
            "lockstep": fast,
            "lockstep_after_other_episode": fast_after_other,
        }), encoding="utf-8")

        differences = {
            "lockstep_vs_http": first_difference(reference, fast),
            "lockstep_replay_after_other_episode": first_difference(reference, fast_after_other),
        }
        results["equivalence"] = differences
        rooms = sorted({entry["state"]["RoomName"] for entry in reference if entry["state"]})
        results["frames_without_player"] = sum(entry["state"] is None for entry in reference)
        for name, difference in differences.items():
            print(f"   {name}: {'identical' if difference is None else difference}")
        print(f"   rooms visited: {rooms}, frames with no player (deaths): {results['frames_without_player']}")
        equivalent = all(difference is None for difference in differences.values())

        passed = equivalent
        if args.steps <= 0:
            return 0 if passed else 1

        print(f"3. Throughput over lockstep ({args.steps:,} steps)")
        step_ms, reset_ms = [], []
        wall_start = time.perf_counter()
        for done in range(args.steps):
            if done % args.episode_length == 0:
                start = time.perf_counter()
                lockstep.reset()
                reset_ms.append((time.perf_counter() - start) * 1000)
            lockstep.step("".join(b for b in RANDOM_BUTTONS if rng.random() < 0.3))
            step_ms.append(lockstep.last_timing.total_ms)
        wall = time.perf_counter() - wall_start
        results["throughput"] = {
            "steps": args.steps, "wall_seconds": wall, "steps_per_second_including_resets": args.steps / wall,
            "step": summarize(step_ms), "reset": summarize(reset_ms), "step_ms": step_ms, "reset_ms": reset_ms,
        }
        s, r = results["throughput"]["step"], results["throughput"]["reset"]
        print(f"   {args.steps / wall:,.0f} steps/s overall ({wall:.1f} s)")
        print(f"   step  mean {s['mean_ms']:.2f} / p50 {s['p50_ms']:.2f} / p99 {s['p99_ms']:.2f} / max {s['max_ms']:.1f} ms")
        print(f"   reset mean {r['mean_ms']:.2f} / p50 {r['p50_ms']:.2f} / p99 {r['p99_ms']:.2f} / max {r['max_ms']:.1f} ms")
        passed = equivalent
    finally:
        lockstep.close()
        (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Results: {output_dir / 'results.json'}")
        if process is not None:
            game_process.stop(process)

    print("Lockstep matches the HTTP bridge." if passed else "LOCKSTEP CHECK FAILED.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
