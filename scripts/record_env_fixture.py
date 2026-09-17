"""Record lockstep replies for the environment's offline tests (tests/fixtures/env_replies.json).

Run from the repo root with the RL interpreter (the CelesteRLLockstep mod must be installed):
    .venv-rl/Scripts/python.exe scripts/record_env_fixture.py

Plays the room 1 exit, death and pause-menu restart fixture routes plus a seeded random trace. For every
step it keeps a compact record (frame, room, whether a player exists, events). It also keeps complete
replies (state, extras, events) at a few chosen steps: each start, a mid-dash frame, a ducking frame, a
frame with a buffered jump, a paused frame, each death event, the first frame without a player and the
first frame in room 2. Test fixture generator only; never used for training.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from celeste_rl import game_process  # noqa: E402
from celeste_rl.bridge import CelesteBridge  # noqa: E402
from celeste_rl.lockstep import LockstepBridge  # noqa: E402

ROUTES = ["room1_exit_dash_route", "room1_spike_death_route", "room1_pause_levelexit_loading_route"]


def random_trace(rng: random.Random, frames: int) -> list[tuple[str, str, str]]:
    trace = []
    for _ in range(frames):
        buttons = "".join(b for b in "LRUDJKXCZVGH" if rng.random() < 0.15)
        dash_only = "".join(d for d in "LRUD" if rng.random() < 0.05)
        move_only = "".join(d for d in "LRUD" if rng.random() < 0.05)
        trace.append((buttons, dash_only, move_only))
    return trace


def sample_reasons(step: int, observation, seen: set[str]) -> list[str]:
    extras = observation.extras
    player = (extras or {}).get("player", {})
    candidates = {
        "start": step == 0,
        "dash": player.get("State") == 2,
        "ducking": player.get("Ducking") is True,
        "jump_buffered": (extras or {}).get("input_buffers", {}).get("Jump", 0) > 0,
        "paused": (extras or {}).get("level", {}).get("Paused") is True,
        "death_event": any(e["type"] == "death" for e in observation.events or []),
        "no_player": observation.state is None,
        "room_2": observation.room == "2",
    }
    # Starts and death events are kept every time; the rest once per trace.
    return [name for name, hit in candidates.items() if hit and (name in ("start", "death_event") or name not in seen)]


def record(bridge: LockstepBridge, actions: list[tuple[str, str, str]]) -> dict:
    steps, samples, seen = [], [], set()
    observation = bridge.reset()
    for step in range(len(actions) + 1):
        if step > 0:
            observation = bridge.step(*actions[step - 1])
        steps.append({"step": step, "frame": observation.tas_frame, "room": observation.room,
                      "player": observation.state is not None, "events": observation.events})
        reasons = sample_reasons(step, observation, seen)
        if reasons:
            seen.update(reasons)
            samples.append({"step": step, "reasons": reasons, "frame": observation.tas_frame,
                            "state": observation.state, "extras": observation.extras, "events": observation.events,
                            "action": list(actions[step - 1]) if step > 0 else None})
    return {"actions": [list(a) for a in actions], "steps": steps, "samples": samples}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game-dir", type=Path, default=Path("C:/Projects/celeste-research-scratch/game-probe"))
    parser.add_argument("--random-frames", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=REPO / "tests" / "fixtures" / "env_replies.json")
    args = parser.parse_args()

    traces = {}
    for name in ROUTES:
        route = json.loads((REPO / "tests" / "fixtures" / f"{name}.json").read_text(encoding="utf-8"))
        traces[name] = [(buttons, "", "") for buttons in route["actions"]]
    traces["random_seed0"] = random_trace(random.Random(args.seed), args.random_frames)

    scratch = REPO / "runs" / "env-fixture"
    scratch.mkdir(parents=True, exist_ok=True)
    process = game_process.launch(args.game_dir, focus=False)
    bridge = LockstepBridge(CelesteBridge(scratch / "episode.tas"))
    try:
        recorded = {name: record(bridge, actions) for name, actions in traces.items()}
    finally:
        bridge.close()
        game_process.stop(process)

    args.output.write_text(json.dumps(recorded, separators=(",", ":")), encoding="utf-8")
    for name, data in recorded.items():
        print(f"{name}: {len(data['steps'])} steps, samples {[(s['step'], s['reasons']) for s in data['samples']]}")
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
