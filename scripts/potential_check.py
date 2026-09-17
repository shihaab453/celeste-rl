"""Check candidate shaping potentials against the recorded legal clear of room 1, offline (no game).

Run from the repo root with the RL interpreter:
    .venv-rl/Scripts/python.exe scripts/potential_check.py

A potential-based shaping term never rewards pacing back and forth, but it can still teach the wrong thing: if the
potential falls while the player follows a real solution, the agent is being told to avoid the solution. This
replays the recorded room 1 clear (a trace saved by the transition check) and measures, for each candidate:

  rises / falls        frames where the potential increased or decreased
  longest fall         the longest run of consecutive falling frames
  total fall           the sum of all decreases (how much progress the route appears to give up)
  start, end, minimum  the potential at the first and last frame, and its lowest point

A candidate whose potential falls for long stretches of a real solution is rejected before any training. The second
check is the opposite trap: the potential must not rise on the way into a hazard, or shaping pays the agent for the
move that kills it. That is measured along a recorded death route and at a few named cells of room 1.

Candidates:
  tiles       breadth-first distance to the exit gap over open tiles, 4-connected, ignoring physics. This is the
              heuristic scripts/find_room_exit.py already uses to rank savestates.
  tiles_fall  the same, with cells made expensive in proportion to how far the player would fall from them.
              Rejected by the trap check: it rates the pit floor next to the spikes higher than the start ledge,
              because the fall from the floor is short.
  tiles_hazard the same, with cells near spikes and cells over a gap made expensive. Passes both checks.
  route       progress along the recorded route by nearest point, minus a small penalty for distance from it. Uses
              the recorded solution, so it is a demonstration-derived potential and belongs to the demonstration
              branch, measured here only for comparison.

The candidates themselves live in `celeste_rl/potential.py`, which is what rew-v2 imports: a potential
validated here and reimplemented there would not be validated at all.

Results are written to runs/potential-check/<timestamp>/results.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from celeste_rl import runtime  # noqa: E402
from celeste_rl.potential import (  # noqa: E402
    fall_cost,
    hazard_cost,
    room,
    route_potential,
    tile_distances,
    tile_potential,
)

DEFAULT_TRACE = REPO / "runs" / "transition-check" / "20260916-124511-exit-regression" / "traces.json"


def evaluate(potential, positions: list[tuple[float, float]]) -> dict:
    values = [potential(x, y) for x, y in positions]
    rises = sum(1 for a, b in zip(values, values[1:]) if b > a)
    falls = sum(1 for a, b in zip(values, values[1:]) if b < a)
    total_fall = sum(a - b for a, b in zip(values, values[1:]) if b < a)
    longest, run = 0, 0
    for a, b in zip(values, values[1:]):
        run = run + 1 if b < a else 0
        longest = max(longest, run)
    return {"frames": len(values), "rises": rises, "falls": falls, "flat": len(values) - 1 - rises - falls,
            "longest_fall_frames": longest, "total_fall": round(total_fall, 4),
            "start": round(values[0], 4), "end": round(values[-1], 4), "minimum": round(min(values), 4),
            "values_every_20_frames": [round(v, 3) for v in values[::20]]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE, help="traces.json from the exit route check")
    parser.add_argument("--death-trace", type=Path, default=REPO / "runs" / "transition-check" / "20260916-121411-death" / "traces.json")
    parser.add_argument("--spike-radius", type=int, default=2)
    parser.add_argument("--near-spike", type=float, default=8.0)
    parser.add_argument("--over-gap", type=float, default=1.0)
    args = parser.parse_args()

    frames = json.loads(args.trace.read_text(encoding="utf-8"))["lockstep"]
    inside = [f for f in frames if f["state"] and f["state"]["RoomName"] == "1"]
    positions = [(f["state"]["Player"]["Position"]["X"], f["state"]["Player"]["Position"]["Y"]) for f in inside]
    geometry = room(inside[0]["state"])

    candidates = {
        "tiles": tile_potential(geometry, tile_distances(geometry)),
        "tiles_fall": tile_potential(geometry, tile_distances(
            geometry, lambda g, r, c: fall_cost(g, r, c, 1.0, 6.0))),
        "tiles_hazard": tile_potential(geometry, tile_distances(
            geometry, lambda g, r, c: hazard_cost(g, r, c, args.spike_radius, args.near_spike, args.over_gap))),
        "route": route_potential(positions),
    }
    results = {**runtime.git_state(), "trace": str(args.trace), "route_frames": len(positions),
               "exit_cells": geometry["exits"], "args": vars(args), "candidates": {}}
    print(f"Room 1 route: {len(positions)} frames in the room, exit cells (row, col) {geometry['exits']}")
    print(f"{'candidate':12s} {'rises':>6s} {'falls':>6s} {'longest fall':>13s} {'total fall':>11s} "
          f"{'start':>6s} {'end':>6s} {'min':>6s}")
    for name, potential in candidates.items():
        summary = evaluate(potential, positions)
        results["candidates"][name] = summary
        print(f"{name:12s} {summary['rises']:6d} {summary['falls']:6d} {summary['longest_fall_frames']:13d} "
              f"{summary['total_fall']:11.3f} {summary['start']:6.3f} {summary['end']:6.3f} {summary['minimum']:6.3f}")

    # The trap check: a death route, and the cells a walker meets going right from the start ledge into the pit.
    death_frames = json.loads(args.death_trace.read_text(encoding="utf-8"))["lockstep"]
    death_positions = [(f["state"]["Player"]["Position"]["X"], f["state"]["Player"]["Position"]["Y"])
                       for f in death_frames if f["state"] and f["state"]["RoomName"] == "1"]
    named_cells = {"start (19,144)": (19, 144), "ledge edge (39,144)": (39, 144), "above pit (48,144)": (48, 144),
                   "falling in pit (60,152)": (60, 152), "pit floor by spikes (60,164)": (60, 164),
                   "on platform (84,128)": (84, 128), "under exit (261,60)": (261, 60), "exit gap (261,4)": (261, 4)}
    print()
    print(f"{'candidate':12s} {'death route: rises':>19s} {'rise before death':>18s} | potential at named cells")
    for name, potential in candidates.items():
        values = [potential(x, y) for x, y in death_positions]
        rises = sum(1 for a, b in zip(values, values[1:]) if b > a)
        before_death = round(values[-1] - values[0], 4)
        cells = {label: round(potential(x, y), 3) for label, (x, y) in named_cells.items()}
        results["candidates"][name]["death_route"] = {"frames": len(values), "rises": rises,
                                                      "change_to_death": before_death, "named_cells": cells}
        print(f"{name:12s} {rises:19d} {before_death:18.3f} | " +
              " ".join(f"{label.split(' (')[0]}={value}" for label, value in cells.items()))

    output_dir = REPO / "runs" / "potential-check" / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"Results: {output_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
