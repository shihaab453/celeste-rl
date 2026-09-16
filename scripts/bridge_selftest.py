"""Check the bridge against the real game.

Run from the repo root with the RL interpreter:
    .venv-rl/Scripts/python.exe scripts/bridge_selftest.py --game-dir C:/Projects/celeste-research-scratch/game-probe

Checks, in order:
1. Input timing: a button pressed on step k changes the player on exactly that step, not before.
2. Restore: the same inputs replayed from reset produce identical player state on every frame,
   including after an unrelated episode in between.
3. Throughput: random legal inputs with regular resets, recording step and reset latency and how
   often CelesteTAS had not re-read the file yet (advance requests > 1).

Results are written to runs/bridge-selftest/<timestamp>/results.json.
"""
from __future__ import annotations

import argparse
import json
import platform
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
from celeste_rl.bridge import CelesteBridge, DebugRcClient  # noqa: E402

# Gameplay buttons only. Pause, journal, confirm and quick restart open menus, which would make
# the timing and restore checks meaningless; menus get their own tests later.
RANDOM_BUTTONS = "LRUDJXZG"


def player(observation) -> dict:
    return (observation.state or {}).get("Player") or {}


def summarize(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": ordered[int(0.5 * (len(ordered) - 1))],
        "p99_ms": ordered[int(0.99 * (len(ordered) - 1))],
        "max_ms": ordered[-1],
    }


def check_input_timing(bridge: CelesteBridge) -> dict:
    """Stand still, press right for one frame, and find the first frame the player moved."""
    bridge.reset()
    speeds = []
    plan = [""] * 10 + ["R"] + [""] * 5
    for buttons in plan:
        speeds.append(player(bridge.step(buttons)).get("Speed", {}).get("X"))

    press_step = plan.index("R")
    first_moving = next((i for i, speed in enumerate(speeds) if speed), None)
    passed = first_moving == press_step and not any(speeds[:press_step])
    print(f"  speed X per step: {speeds}")
    print(f"  right pressed on step {press_step}, first nonzero speed on step {first_moving}: "
          f"{'OK' if passed else 'FAIL'}")
    return {"passed": passed, "press_step": press_step, "first_moving_step": first_moving, "speed_x": speeds}


def run_trace(bridge: CelesteBridge, actions: list[str]) -> list[dict]:
    """Reset, play the actions, and record everything that should be identical on a replay."""
    start = bridge.reset()
    trace = [{"room": start.room, "player": player(start)}]
    for buttons in actions:
        observation = bridge.step(buttons)
        trace.append({
            "room": observation.room,
            "player": player(observation),
            "state_name": (observation.state or {}).get("PlayerStateName"),
            "chapter_time": (observation.state or {}).get("ChapterTime"),
        })
    return trace


def check_restore(bridge: CelesteBridge, length: int, seed: int) -> dict:
    rng = random.Random(seed)
    actions = ["".join(b for b in RANDOM_BUTTONS if rng.random() < 0.3) for _ in range(length)]
    other = ["".join(b for b in RANDOM_BUTTONS if rng.random() < 0.3) for _ in range(length)]

    first = run_trace(bridge, actions)
    second = run_trace(bridge, actions)
    run_trace(bridge, other)  # an unrelated episode, to catch state leaking across resets
    third = run_trace(bridge, actions)

    mismatches = []
    for name, trace in (("replay", second), ("after other episode", third)):
        for frame, (expected, actual) in enumerate(zip(first, trace)):
            if expected != actual:
                mismatches.append({"run": name, "step": frame, "expected": expected, "actual": actual})
                break

    rooms = sorted({entry["room"] for entry in first})
    passed = not mismatches
    print(f"  {length} steps x 3 replays, rooms visited: {rooms}: {'OK' if passed else 'FAIL'}")
    for mismatch in mismatches:
        print(f"  first difference in {mismatch['run']} at step {mismatch['step']}")
    return {"passed": passed, "length": length, "seed": seed, "rooms": rooms, "mismatches": mismatches}


def check_throughput(bridge: CelesteBridge, steps: int, episode_length: int, seed: int) -> dict:
    rng = random.Random(seed)
    step_ms, reset_ms, requests = [], [], []
    for done in range(steps):
        if done % episode_length == 0:
            start = time.perf_counter()
            bridge.reset()
            reset_ms.append((time.perf_counter() - start) * 1000)
        bridge.step("".join(b for b in RANDOM_BUTTONS if rng.random() < 0.3))
        step_ms.append(bridge.last_timing.total_ms)
        requests.append(bridge.last_timing.advance_requests)
        if (done + 1) % 1000 == 0:
            print(f"  {done + 1}/{steps} steps")

    result = {
        "steps": steps,
        "episode_length": episode_length,
        "step": summarize(step_ms),
        "reset": summarize(reset_ms),
        "steps_needing_resend": sum(r > 1 for r in requests),
        "max_advance_requests": max(requests),
        "step_ms": step_ms,
        "reset_ms": reset_ms,
    }
    s, r = result["step"], result["reset"]
    print(f"  step  mean {s['mean_ms']:.1f} / p50 {s['p50_ms']:.1f} / p99 {s['p99_ms']:.1f} / max {s['max_ms']:.1f} ms")
    print(f"  reset mean {r['mean_ms']:.1f} / p50 {r['p50_ms']:.1f} / p99 {r['p99_ms']:.1f} / max {r['max_ms']:.1f} ms")
    print(f"  steps where the first advance was refused: {result['steps_needing_resend']}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game-dir", type=Path, default=Path("C:/Projects/celeste-research-scratch/game-probe"))
    parser.add_argument("--attach", action="store_true", help="use an already running game instead of launching one")
    parser.add_argument("--keep-open", action="store_true", help="leave the game running afterwards")
    parser.add_argument("--restore-length", type=int, default=120)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--episode-length", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = REPO / "runs" / "bridge-selftest" / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout.strip())

    process = None
    if not args.attach:
        print("Launching game...")
        process = game_process.launch(args.game_dir)

    bridge = CelesteBridge(output_dir / "episode.tas")
    results = {"commit": commit, "uncommitted_changes": dirty, "python": platform.python_version(), "args": vars(args)}
    try:
        results["raw_info_sample"] = DebugRcClient().get("/tas/info")
        print("1. Input timing")
        results["input_timing"] = check_input_timing(bridge)
        print("2. Restore")
        results["restore"] = check_restore(bridge, args.restore_length, args.seed)
        print("3. Throughput")
        results["throughput"] = check_throughput(bridge, args.steps, args.episode_length, args.seed)
    finally:
        bridge.close()
        results["args"]["game_dir"] = str(args.game_dir)
        (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Results: {output_dir / 'results.json'}")
        if process is not None and not args.keep_open:
            game_process.stop(process)

    passed = results["input_timing"]["passed"] and results["restore"]["passed"]
    print("All checks passed." if passed else "SOME CHECKS FAILED.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
