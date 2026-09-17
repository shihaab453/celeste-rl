"""Run the lockstep bridge for a long time and watch for slowdowns, leaks, crashes and drift.

Run from the repo root with the RL interpreter:
    .venv-rl/Scripts/python.exe scripts/soak_test.py --minutes 30 --window background

Every minute it records throughput, step latency, resets, deaths, game memory (private bytes and
working set) and whether the window is still in the requested state. Every --check-every episodes it replays one fixed input
sequence and compares it with the first replay, to catch state corruption that builds up over time.

Results are written to runs/soak/<timestamp>/results.json, including after a failure.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from celeste_rl import game_process  # noqa: E402
from celeste_rl.bridge import CelesteBridge  # noqa: E402
from celeste_rl.lockstep import LockstepBridge  # noqa: E402

RANDOM_BUTTONS = "LRUDJXZG"


def random_buttons(rng: random.Random) -> str:
    return "".join(b for b in RANDOM_BUTTONS if rng.random() < 0.3)


def play_trace(bridge: LockstepBridge, actions: list[str]) -> list:
    frames = [bridge.reset().state]
    frames += [bridge.step(buttons).state for buttons in actions]
    return frames


def percentile(ordered: list[float], q: float) -> float:
    return ordered[int(q * (len(ordered) - 1))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game-dir", type=Path, default=Path("C:/Projects/celeste-research-scratch/game-probe"))
    parser.add_argument("--minutes", type=float, default=30.0)
    parser.add_argument("--window", choices=["focused", "background", "minimized"], default="background")
    parser.add_argument("--no-launch-focus", action="store_true", help="do not focus the window while the game starts")
    parser.add_argument("--episode-length", type=int, default=300)
    parser.add_argument("--check-every", type=int, default=20, help="episodes between drift checks")
    parser.add_argument("--trace-length", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = REPO / "runs" / "soak" / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True)
    git = lambda *a: subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()
    results = {
        "commit": git("rev-parse", "HEAD"),
        "uncommitted_changes": bool(git("status", "--porcelain")),
        "args": {**vars(args), "game_dir": str(args.game_dir)},
        "minutes": [],
        "drift_checks": [],
    }

    def save():
        (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    rng = random.Random(args.seed)
    trace_actions = [random_buttons(rng) for _ in range(args.trace_length)]

    launch_start = time.perf_counter()
    try:
        process = game_process.launch(args.game_dir, focus=not args.no_launch_focus)
    except Exception as error:
        results["launch"] = {"ok": False, "error": str(error), "seconds": time.perf_counter() - launch_start}
        save()
        print(f"Launch failed: {error}")
        return 1
    results["launch"] = {"ok": True, "seconds": time.perf_counter() - launch_start, "focused_during_launch": not args.no_launch_focus}
    print(f"Game up in {results['launch']['seconds']:.1f} s (focus during launch: {not args.no_launch_focus})")

    bridge = LockstepBridge(CelesteBridge(output_dir / "episode.tas"))
    exit_code = 1
    try:
        reference = play_trace(bridge, trace_actions)
        game_process.set_window_mode(process.pid, args.window)
        time.sleep(0.5)
        print(f"Window mode: {args.window}, game focused: {game_process.window_is_focused(process.pid)}")
        results["memory_start"] = game_process.process_memory(process.pid)
        print(f"Private bytes at start: {results['memory_start']['private_mb']:.0f} MB")

        deadline = time.perf_counter() + args.minutes * 60

        def new_window():
            return {"started": time.perf_counter(), "random_steps": 0, "drift_trace_steps": 0, "resets": 0,
                    "no_player_frames": 0, "step_ms": []}

        window = new_window()
        episodes = total_steps = 0

        while time.perf_counter() < deadline:
            bridge.reset()
            episodes += 1
            window["resets"] += 1
            for _ in range(args.episode_length):
                observation = bridge.step(random_buttons(rng))
                window["step_ms"].append(bridge.last_timing.total_ms)
                window["random_steps"] += 1
                window["no_player_frames"] += observation.state is None
            total_steps += args.episode_length

            if episodes % args.check_every == 0:
                matches = play_trace(bridge, trace_actions) == reference
                window["resets"] += 1
                window["drift_trace_steps"] += args.trace_length
                results["drift_checks"].append({"episode": episodes, "matches_first_replay": matches})
                if not matches:
                    raise RuntimeError(f"Fixed trace diverged from its first replay at episode {episodes}")

            elapsed = time.perf_counter() - window["started"]
            if elapsed >= 60:
                ordered = sorted(window.pop("step_ms"))
                window.pop("started")
                all_steps = window["random_steps"] + window["drift_trace_steps"]
                summary = {
                    "window": len(results["minutes"]) + 1,
                    "seconds": elapsed,
                    **window,
                    # All steps, including drift-check replays, over the measured window length.
                    "steps_per_second": all_steps / elapsed,
                    "random_step_p50_ms": percentile(ordered, 0.5),
                    "random_step_p99_ms": percentile(ordered, 0.99),
                    "random_step_max_ms": ordered[-1],
                    # Private bytes include native and GPU driver allocations; working set alone hid the
                    # render target leak.
                    **{f"game_{key}": value for key, value in game_process.process_memory(process.pid).items()},
                    "game_focused": game_process.window_is_focused(process.pid),
                    "game_minimized": game_process.window_is_minimized(process.pid),
                }
                results["minutes"].append(summary)
                save()
                print(f"#{summary['window']:3d} ({elapsed:.0f} s): {summary['steps_per_second']:6.0f} steps/s  "
                      f"p50 {summary['random_step_p50_ms']:.2f} p99 {summary['random_step_p99_ms']:.2f} "
                      f"max {summary['random_step_max_ms']:.1f} ms  private {summary['game_private_mb']:.0f} MB  "
                      f"focused {summary['game_focused']} minimized {summary['game_minimized']}  "
                      f"drift checks OK {sum(c['matches_first_replay'] for c in results['drift_checks'])}")
                window = new_window()

        results["total"] = {"episodes": episodes, "steps": total_steps}
        if results["minutes"]:
            growth = results["minutes"][-1]["game_private_mb"] - results["memory_start"]["private_mb"]
            results["total"]["private_mb_growth"] = growth
            print(f"Private bytes change since start: {growth:+.0f} MB ({growth * 1024 / episodes:+.1f} KB per episode)")
        exit_code = 0
        print(f"Completed {total_steps:,} steps in {episodes} episodes without errors.")
    except Exception as error:
        results["failure"] = {"error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(),
                              "after_minutes": len(results["minutes"])}
        print(f"FAILED: {type(error).__name__}: {error}")
    finally:
        bridge.close()
        save()
        print(f"Results: {output_dir / 'results.json'}")
        game_process.stop(process)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
