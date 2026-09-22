"""Search for a legal input sequence that reaches the target of one room task.

This is a test-fixture generator, not a learning method: it uses exact savestate replays and the room's
whole tile map at once. A policy does see local geometry (obs-v1 carries a 32 by 32 cell grid around the
player), so the map is not privileged information in itself; what it never gets is the whole room, the
breadth-first distances computed over it, or the ability to jump back to a saved state. It is a small
Go-Explore-style search:

1. Compute each open tile's distance to the room's exits by breadth-first search over the tile map
   from the episode start (spike tiles count as walls). This ignores Celeste's movement physics; it
   only ranks how promising a position is.
2. Keep an archive of every tile the player has reached, with the shortest input sequence that got there.
3. Repeatedly pick a promising tile, replay its inputs, then try random bursts of legal buttons,
   recording newly reached tiles. Stop when the room changes.
4. Continue from the transition with inputs that survive at least --after-frames frames in the new room,
   including a dash so the trace contains freeze frames.

Run from the repo root with the RL interpreter:
    .venv-rl/Scripts/python.exe scripts/find_room_exit.py
    .venv-rl/Scripts/python.exe scripts/find_room_exit.py --task-definition config/room2.json

Writes `<output-root>/<timestamp>/route.json` with the per-frame inputs. The default output root is
`runs/routes`; held-out generation uses `runs/heldout-routes` so evaluation sources never share the
demonstration namespace.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from celeste_rl import game_process  # noqa: E402
from celeste_rl.actions import parse_line, to_parts  # noqa: E402
from celeste_rl.bridge import CelesteBridge  # noqa: E402
from celeste_rl.lockstep import LockstepBridge  # noqa: E402
from celeste_rl.tasks import (  # noqa: E402
    TaskDefinition,
    TaskDefinitionError,
    base_task_definition,
    load_task_definition,
    task_identity,
)

TILE = 8
# Button combinations for random bursts: move, jump, dash in eight directions, grab/climb.
MACROS = ["R", "L", "", "J", "RJ", "LJ", "RX", "LX", "UX", "URX", "ULX", "DRX", "DLX",
          "RG", "LG", "UG", "URG", "ULG", "RJG", "LJG", "UJG"]


def tile_map(state: dict) -> tuple[list[str], set[tuple[int, int]]]:
    rows = state["SolidsData"].split("\n")
    bounds = state["Level"]["Bounds"]
    blocked = set()
    for spike in state["Spikes"]:
        b = spike["Bounds"]
        for tx in range(int((b["X"] - bounds["X"]) // TILE),
                        int((b["X"] + b["W"] - 1 - bounds["X"]) // TILE) + 1):
            for ty in range(int((b["Y"] - bounds["Y"]) // TILE),
                            int((b["Y"] + b["H"] - 1 - bounds["Y"]) // TILE) + 1):
                blocked.add((tx, ty))
    return rows, blocked


def exit_distances(rows: list[str], blocked: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    """Tile distance from each open tile to an open tile on the top, left or right edge."""
    height, width = len(rows), len(rows[0])
    open_tile = lambda x, y: 0 <= x < width and 0 <= y < height and rows[y][x] == "0" and (x, y) not in blocked
    queue = deque()
    distance = {}
    for x in range(width):
        for y in range(height):
            on_edge = y == 0 or x in (0, width - 1)
            if on_edge and open_tile(x, y):
                distance[(x, y)] = 0
                queue.append((x, y))
    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if open_tile(nx, ny) and (nx, ny) not in distance:
                distance[(nx, ny)] = distance[(x, y)] + 1
                queue.append((nx, ny))
    return distance


def player_tile(state: dict) -> tuple[int, int]:
    position = state["Player"]["Position"]
    bounds = state["Level"]["Bounds"]
    # Position is the bottom centre of the hitbox; use a point inside the body.
    return int((position["X"] - bounds["X"]) // TILE), int((position["Y"] - 4 - bounds["Y"]) // TILE)


def live_in_room(observation, room: str) -> bool:
    """Whether a continuation frame still has a player in the room it is meant to validate."""
    return observation.state is not None and observation.room == room


def random_burst(rng: random.Random, frames: int) -> list[str]:
    actions = []
    while len(actions) < frames:
        actions += [rng.choice(MACROS)] * rng.randint(1, 16)
    return actions[:frames]


def reset_to_task(bridge: LockstepBridge, definition: TaskDefinition | None):
    """Reset to the base savestate, replay a later room's start recipe, and verify the arrival."""
    observation = bridge.reset()
    if definition is None or definition.start is None:
        return observation
    start = definition.start
    for line in start.lines:
        observation = bridge.step(*to_parts(parse_line(line, canonical_only=True)))
    state = observation.state
    position = None if state is None else state["Player"]["Position"]
    arrived = None if position is None else (position["X"], position["Y"])
    dashes = None if observation.extras is None else observation.extras["player"]["Dashes"]
    problems = []
    if observation.room != start.room:
        problems.append(f"room {observation.room!r}, expected {start.room!r}")
    if arrived != start.position:
        problems.append(f"position {arrived!r}, expected {start.position!r}")
    if start.dashes is not None and dashes != start.dashes:
        problems.append(f"dashes {dashes!r}, expected {start.dashes!r}")
    if problems:
        raise RuntimeError("task start no longer replays: " + "; ".join(problems))
    return observation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game-dir", type=Path, default=Path("C:/Projects/celeste-research-scratch/game-probe"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--burst-frames", type=int, default=45)
    parser.add_argument("--max-minutes", type=float, default=15)
    parser.add_argument("--after-frames", type=int, default=90)
    parser.add_argument("--output-root", type=Path, default=REPO / "runs" / "routes")
    parser.add_argument("--task-definition", type=Path,
                        help="hash-pinned later-room task whose canonical start is replayed before searching")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    try:
        definition = load_task_definition(args.task_definition) if args.task_definition else None
    except TaskDefinitionError as error:
        parser.error(str(error))
    identity = task_identity(definition or base_task_definition())

    output_dir = args.output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True)
    process = game_process.launch(args.game_dir, focus=False)
    bridge = LockstepBridge(CelesteBridge(output_dir / "episode.tas"))
    try:
        start = reset_to_task(bridge, definition)
        start_room = start.room
        target_room = definition.task.target_room if definition is not None else None
        rows, blocked = tile_map(start.state)
        distance = exit_distances(rows, blocked)
        start_tile = player_tile(start.state)
        print(f"Start room {start_room}, tile {start_tile}, tile distance to exit {distance.get(start_tile)}")

        archive = {start_tile: []}  # tile -> shortest input sequence reaching it
        chosen = {start_tile: 0}
        deadline = time.perf_counter() + args.max_minutes * 60
        rollouts = 0
        route = None
        next_room = None

        while route is None and time.perf_counter() < deadline:
            # Prefer tiles close to the exit that have not been tried much.
            ranked = sorted(archive, key=lambda t: (distance.get(t, 999), chosen.get(t, 0)))
            tile = ranked[0] if rng.random() < 0.5 else rng.choice(ranked[: max(1, len(ranked) // 4)])
            chosen[tile] = chosen.get(tile, 0) + 1
            prefix = archive[tile]

            reset_to_task(bridge, definition)
            for buttons in prefix:
                bridge.step(buttons)
            actions = list(prefix)
            for buttons in random_burst(rng, args.burst_frames):
                observation = bridge.step(buttons)
                actions.append(buttons)
                if observation.state is None:
                    break  # died
                if observation.room != start_room:
                    if target_room is None or observation.room == target_room:
                        route = actions
                        next_room = observation.room
                    break
                reached = player_tile(observation.state)
                if reached in distance and (reached not in archive or len(actions) < len(archive[reached])):
                    archive[reached] = list(actions)
            rollouts += 1
            if rollouts % 50 == 0:
                best = min(archive, key=lambda t: distance.get(t, 999))
                print(f"  {rollouts} rollouts, {len(archive)} tiles, closest tile {best} "
                      f"at distance {distance.get(best, 'outside map')}")

        if route is None:
            print("No exit found within the time limit.")
            return 1
        transition_index = len(route) - 1
        print(f"Left room {start_room} after {len(route)} frames ({rollouts} rollouts)")

        # Continue in the new room: a dash first, so the trace has freeze frames, then survive.
        for attempt in range(500):
            extension = ["", "", "URX"] + random_burst(rng, args.after_frames - 3)
            reset_to_task(bridge, definition)
            for buttons in route:
                bridge.step(buttons)
            freeze_frames = 0
            survived = True
            for buttons in extension:
                observation = bridge.step(buttons)
                freeze_frames += (observation.diagnostics or {}).get("freeze_timer", 0) > 0
                if not live_in_room(observation, next_room):
                    survived = False
                    break
            if survived and freeze_frames > 0:
                break
        else:
            print("Could not find a surviving continuation with a dash in the new room.")
            return 1

        full = route + extension
        (output_dir / "route.json").write_text(json.dumps({
            "start_room": start_room,
            "next_room": next_room,
            "transition_step": transition_index,
            "frames": len(full),
            "freeze_frames_after_transition": freeze_frames,
            "actions": full,
            "seed": args.seed,
            "task_definition": str(args.task_definition) if args.task_definition else None,
            "task": identity,
        }, indent=1), encoding="utf-8")
        print(f"Route: {len(full)} frames, room {start_room} -> {next_room} at step {transition_index}, "
              f"{freeze_frames} freeze frames after the transition")
        print(f"Saved {output_dir / 'route.json'}")
        return 0
    finally:
        bridge.close()
        game_process.stop(process)


if __name__ == "__main__":
    sys.exit(main())
