"""Progress potentials over a room's tiles, for potential-based reward shaping (rew-v2).

The candidates here were validated offline against the recorded room 1 clear and a recorded death route by
`scripts/potential_check.py`, which is why they live in a module the reward and the check both import: a
potential that was validated in one place and reimplemented in another has not been validated.

Only `tiles_hazard` (breadth-first tile distance to the exit with spike and gap costs) passed both checks and
is the one rew-v2 uses. The plain tile distance pays the agent to walk into the first spike pit, and the
fall-weighted variant rates the pit floor above the start ledge. `route_potential` uses the recorded
solution, so it is a demonstration-derived method and belongs to Phase 3B; it is kept here for comparison
only and must not be used for training.

A potential is a function of the state alone. It never enters the observation, and shaping built from it
leaves the optimal policy unchanged (Ng, Harada and Russell) as long as the terminal potential is 0.
"""
from __future__ import annotations

import math
from collections import deque

from celeste_rl.schema import CELL_SIZE

# The spike-weighted parameters validated in runs/potential-check/20260917-230728.
SPIKE_RADIUS = 2
NEAR_SPIKE = 8.0
OVER_GAP = 1.0


def room(state: dict) -> dict:
    """Tile grid, exit cells and spike cells from one game state."""
    rows = state["SolidsData"].replace("\r", "").split("\n")
    solid = [[c != "0" for c in row] for row in rows]
    bounds = state["Level"]["Bounds"]
    # The exit is the open stretch of the room's top row: the player leaves the room upwards through it.
    exits = [(0, col) for col, filled in enumerate(solid[0]) if not filled]
    spikes = set()
    for spike in state.get("Spikes") or []:
        b, direction = spike["Bounds"], spike["Direction"]
        # Up spikes sit 3 px above their exported position; they occupy the tile row above their surface.
        top = b["Y"] - 3 if direction == 0 else b["Y"]
        for x in range(int(b["X"]), int(b["X"] + b["W"]), CELL_SIZE):
            spikes.add((int((top - bounds["Y"]) // CELL_SIZE), int((x - bounds["X"]) // CELL_SIZE)))
    return {"solid": solid, "exits": exits, "spikes": spikes, "bounds": bounds,
            "rows": len(solid), "cols": max(len(r) for r in solid)}


def hazard_cost(geometry: dict, row: int, col: int, radius: int = SPIKE_RADIUS, near_spike: float = NEAR_SPIKE,
                over_gap: float = OVER_GAP) -> float:
    """Extra cost for being near spikes, or airborne over a gap: the two ways a cell kills a Celeste player."""
    cost = 0.0
    if any(abs(r - row) <= radius and abs(c - col) <= radius for r, c in geometry["spikes"]):
        cost += near_spike
    depth, below = 0, row + 1
    while below < geometry["rows"] and not geometry["solid"][below][col]:
        depth, below = depth + 1, below + 1
    return cost + over_gap * min(depth, 6)


def fall_cost(geometry: dict, row: int, col: int, per_tile: float, onto_hazard: float) -> float:
    """How costly it is to be in this cell, based on what is below it: falling far, onto spikes, or out of the room."""
    depth = 0
    r = row + 1
    while r < geometry["rows"] and not geometry["solid"][r][col]:
        if (r, col) in geometry["spikes"]:
            return depth * per_tile + onto_hazard
        depth += 1
        r += 1
    if r >= geometry["rows"]:
        return depth * per_tile + onto_hazard  # nothing below: the player leaves the room and dies
    return depth * per_tile


def tile_distances(geometry: dict, cost_of=None) -> dict:
    """Cost to reach the exit from every open cell, 4-connected, ignoring physics."""
    costs: dict[tuple[int, int], float] = {}
    queue: deque = deque()
    for cell in geometry["exits"]:
        costs[cell] = 0.0
        queue.append(cell)
    # Dijkstra-like when weighted, plain breadth-first when not; the weighted graph has small integer-ish costs.
    while queue:
        row, col = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            r, c = row + dr, col + dc
            if not (0 <= r < geometry["rows"] and 0 <= c < len(geometry["solid"][r])) or geometry["solid"][r][c]:
                continue
            step = 1.0 + (cost_of(geometry, r, c) if cost_of else 0.0)
            if costs.get((r, c), math.inf) > costs[(row, col)] + step:
                costs[(r, c)] = costs[(row, col)] + step
                queue.append((r, c))
    return costs


def tile_potential(geometry: dict, costs: dict):
    largest = max(costs.values()) or 1.0

    def potential(x: float, y: float) -> float:
        row = int((y - geometry["bounds"]["Y"]) // CELL_SIZE)
        col = int((x - geometry["bounds"]["X"]) // CELL_SIZE)
        best = costs.get((row, col))
        if best is None:  # inside a wall or outside the map: use the nearest open cell
            best = min((cost for (r, c), cost in costs.items() if abs(r - row) + abs(c - col) <= 2), default=largest)
        return max(0.0, 1.0 - best / largest)

    return potential


def route_potential(positions: list[tuple[float, float]], distance_weight: float = 0.01):
    """Progress along a recorded route. Demonstration-derived: comparison only, never training (Codex K3)."""

    def potential(x: float, y: float) -> float:
        best, nearest = None, 0
        for index, (px, py) in enumerate(positions):
            gap = math.hypot(px - x, py - y)
            if best is None or gap < best:
                best, nearest = gap, index
        progress = nearest / (len(positions) - 1)
        return max(0.0, progress - distance_weight * best)

    return potential


def hazard_potential(geometry: dict, radius: int = SPIKE_RADIUS, near_spike: float = NEAR_SPIKE,
                     over_gap: float = OVER_GAP):
    """The `tiles_hazard` candidate: the one rew-v2 uses."""
    return tile_potential(geometry, tile_distances(
        geometry, lambda g, r, c: hazard_cost(g, r, c, radius, near_spike, over_gap)))


class RoomPotential:
    """The `tiles_hazard` potential of one room, in the form the environment needs during an episode.

    Building the distances costs a breadth-first pass over the room, so an episode keeps its RoomPotential and
    `matches()` reports whether a new episode's room is the same one (room 1 never changes, so the pass runs
    once per training run rather than once per episode).

    `value()` is 0 whenever there is no player or the player is not in this room. Every such step is an ending
    (death, success, wrong room), and potential-based shaping needs the terminal potential to be 0 for an
    episode's shaping to sum to exactly `-scale * potential(start)`.
    """

    def __init__(self, state: dict, radius: int = SPIKE_RADIUS, near_spike: float = NEAR_SPIKE,
                 over_gap: float = OVER_GAP):
        self.room = state["RoomName"]
        self.parameters = (radius, near_spike, over_gap)
        self._signature = self._signature_of(state)
        self.geometry = room(state)
        self._potential = hazard_potential(self.geometry, radius, near_spike, over_gap)

    @staticmethod
    def _signature_of(state: dict):
        level = state["Level"]
        return (state["RoomName"], state["SolidsData"], repr(level["Bounds"]), repr(state.get("Spikes")))

    def matches(self, state: dict | None) -> bool:
        """Whether this potential was built for the room in `state`: the reset check, not the per-step one."""
        return state is not None and self._signature_of(state) == self._signature

    def value(self, state: dict | None) -> float:
        """Per step, so it compares the room name only; a room's geometry cannot change inside an episode."""
        if state is None or state.get("RoomName") != self.room:
            return 0.0
        position = state["Player"]["Position"]
        return self._potential(position["X"], position["Y"])
