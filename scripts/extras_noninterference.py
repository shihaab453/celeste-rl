"""Check that the mod's extras and events do not change the game (Phase 2 spec, probe P11).

Run from the repo root with the RL interpreter (the CelesteRLLockstep mod must be installed):
    .venv-rl/Scripts/python.exe scripts/extras_noninterference.py

The game is launched twice: once as normal, once with CELESTE_RL_LOCKSTEP_EXTRAS=0, which makes the mod
skip reading extras and subscribing to events. The same input traces are played in both (the room 1
exit, death and pause-menu restart fixtures, plus seeded random inputs using every gameplay button and
dash-only / move-only directions), and the exported game state must match on every frame. The enabled
run must also deliver extras and events on every reply, and the disabled run none.

Results are written to runs/extras-noninterference/<timestamp>/results.json.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from celeste_rl import game_process  # noqa: E402
from celeste_rl.bridge import CelesteBridge  # noqa: E402
from celeste_rl.lockstep import LockstepBridge  # noqa: E402

FIXTURES = ["room1_exit_dash_route", "room1_spike_death_route", "room1_pause_levelexit_loading_route"]
# Gameplay buttons only: pause, quick restart and journal can make CelesteTAS stop the TAS.
BUTTONS = "LRUDJKXCZVGH"
DIRECTIONS = "LRUD"


def random_trace(rng: random.Random, frames: int) -> list[tuple[str, str, str]]:
    trace = []
    for _ in range(frames):
        buttons = "".join(b for b in BUTTONS if rng.random() < 0.15)
        dash_only = "".join(d for d in DIRECTIONS if rng.random() < 0.05)
        move_only = "".join(d for d in DIRECTIONS if rng.random() < 0.05)
        trace.append((buttons, dash_only, move_only))
    return trace


def play(bridge: LockstepBridge, trace: list[tuple[str, str, str]]) -> dict:
    start = bridge.reset()
    frames = [{"frame": start.tas_frame, "state": start.state}]
    extras_replies = int(start.events is not None)
    for buttons, dash_only, move_only in trace:
        observation = bridge.step(buttons, dash_only, move_only)
        frames.append({"frame": observation.tas_frame, "state": observation.state})
        extras_replies += observation.events is not None
    return {"frames": frames, "replies_with_extras": extras_replies, "replies": len(trace) + 1}


def run_game(game_dir: Path, output_dir: Path, traces: dict, extras_enabled: bool) -> dict:
    if extras_enabled:
        os.environ.pop("CELESTE_RL_LOCKSTEP_EXTRAS", None)
    else:
        os.environ["CELESTE_RL_LOCKSTEP_EXTRAS"] = "0"
    process = game_process.launch(game_dir, focus=False)
    bridge = LockstepBridge(CelesteBridge(output_dir / f"episode-{'on' if extras_enabled else 'off'}.tas"))
    try:
        return {name: play(bridge, trace) for name, trace in traces.items()}
    finally:
        bridge.close()
        game_process.stop(process)
        os.environ.pop("CELESTE_RL_LOCKSTEP_EXTRAS", None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game-dir", type=Path, default=Path("C:/Projects/celeste-research-scratch/game-probe"))
    parser.add_argument("--random-frames", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = REPO / "runs" / "extras-noninterference" / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True)
    git = lambda *a: subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()

    traces = {}
    for name in FIXTURES:
        route = json.loads((REPO / "tests" / "fixtures" / f"{name}.json").read_text(encoding="utf-8"))
        traces[name] = [(buttons, "", "") for buttons in route["actions"]]
    traces["random"] = random_trace(random.Random(args.seed), args.random_frames)

    enabled = run_game(args.game_dir, output_dir, traces, extras_enabled=True)
    disabled = run_game(args.game_dir, output_dir, traces, extras_enabled=False)

    comparisons = {}
    for name in traces:
        on, off = enabled[name], disabled[name]
        mismatch = next((i for i, (a, b) in enumerate(zip(on["frames"], off["frames"])) if a != b), None)
        comparisons[name] = {
            "frames": len(on["frames"]),
            "identical": mismatch is None and len(on["frames"]) == len(off["frames"]),
            "first_mismatch_index": mismatch,
            "enabled_replies_with_extras": f"{on['replies_with_extras']}/{on['replies']}",
            "disabled_replies_with_extras": f"{off['replies_with_extras']}/{off['replies']}",
            "extras_delivered_correctly": on["replies_with_extras"] == on["replies"] and off["replies_with_extras"] == 0,
        }
        print(f"   {name}: {comparisons[name]['frames']} frames, identical {comparisons[name]['identical']}, "
              f"extras on {comparisons[name]['enabled_replies_with_extras']}, off {comparisons[name]['disabled_replies_with_extras']}")

    passed = all(c["identical"] and c["extras_delivered_correctly"] for c in comparisons.values())
    results = {
        "commit": git("rev-parse", "HEAD"),
        "uncommitted_changes": bool(git("status", "--porcelain")),
        "args": {**vars(args), "game_dir": str(args.game_dir)},
        "comparisons": comparisons,
        "passed": passed,
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Extras and events do not change the game state." if passed else "FAILED: see results.json")
    print(f"Results: {output_dir / 'results.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
