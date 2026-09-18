"""starts-v1: an archive of entry states the agent has actually reached, and how to sample from it.

Every episode so far has begun at the canonical start, so the agent has spent nine hours of training re-running
the same first 76 pixels of room 1. An archive lets an episode begin somewhere the agent has already got to, so
the frontier can extend itself instead of being re-discovered from scratch each time.

**A start is a recipe, not a savestate.** An entry stores the sequence of input lines that reached a cell from
the canonical start. To use it, the environment resets canonically and replays those inputs, which is
reproducible because the same inputs from the same savestate give the same state (verified frame by frame by
`scripts/transition_check.py`). Nothing here stores game memory, and nothing here can invent a state the agent
did not reach.

**Agent-reached only** (Codex K3). Entries come from the agent's own episodes. A prefix taken from the recorded
clear would put the agent next to the exit and would produce successes quickly, but it is derived from a
demonstration and belongs to Phase 3B, not to the no-demonstration branch. Nothing in this module reads a route
file, and `offer()` is the only way in.

**Coverage weighting** (Codex K10). Cells are chosen with weight `1 / (1 + times used)`, so rarely used cells
come up more often. There is no exit-distance term: preferring cells closer to the exit would be a second,
unvalidated progress heuristic on top of the shaping potential, and would concentrate starts exactly where the
agent already is rather than spreading them. A fixed fraction of episodes (25% by default) still start
canonically, so the task the agent is measured on is never trained away from.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from celeste_rl.schema import CELL_SIZE

CANONICAL_FRACTION = 0.25


@dataclass(frozen=True)
class Start:
    """How to reach one entry state from the canonical start, and what should be true when you get there.

    `lines` are canonical one-frame input lines (`celeste_rl.actions.to_line`), replayed in order. `position`,
    `room` and `dashes` are what the agent observed on arrival, and the environment checks them after replaying
    so a start that no longer reproduces is caught rather than trained on.
    """

    lines: tuple[str, ...]
    position: tuple[float, float]
    room: str
    dashes: int | None = None

    @property
    def frames(self) -> int:
        return len(self.lines)

    def to_json(self) -> dict:
        return {**asdict(self), "lines": list(self.lines), "position": list(self.position)}

    @staticmethod
    def from_json(data: dict) -> "Start":
        return Start(tuple(data["lines"]), tuple(data["position"]), data["room"], data.get("dashes"))


def cell_of(position, cell_size: int = CELL_SIZE) -> tuple[int, int]:
    """The archive cell a position falls in. Tile sized, so it matches the observation grid's resolution."""
    x, y = position
    return int(x // cell_size), int(y // cell_size)


class StartArchive:
    """Entry states the agent reached, one kept per cell, chosen by inverse use count.

    Keeping the *shortest* prefix per cell is deliberate: it is the cheapest way back to that cell, it leaves
    the most of the 30 second deadline for the episode itself, and it stops prefixes growing without bound as
    episodes start from other starts.
    """

    def __init__(self, cell_size: int = CELL_SIZE, canonical_fraction: float = CANONICAL_FRACTION,
                 max_frames: int | None = None, seed: int | None = None):
        if not 0.0 <= canonical_fraction <= 1.0:
            raise ValueError(f"canonical_fraction must be between 0 and 1, got {canonical_fraction}")
        self.cell_size = cell_size
        self.canonical_fraction = canonical_fraction
        self.max_frames = max_frames
        self.starts: dict[tuple[int, int], Start] = {}
        self.uses: dict[tuple[int, int], int] = {}
        self.rng = random.Random(seed)
        self.offered = 0
        self.replaced = 0

    def __len__(self) -> int:
        return len(self.starts)

    def would_keep(self, position, frames: int) -> bool:
        """Whether offering a start from here with this prefix length would be kept.

        The environment asks this on every step, where building the Start itself would mean copying the whole
        input history each frame. Cheap enough to call at 400 steps per second.
        """
        if self.max_frames is not None and frames > self.max_frames:
            return False
        held = self.starts.get(cell_of(position, self.cell_size))
        return held is None or held.frames > frames

    def offer(self, start: Start) -> bool:
        """Record a reached state. Kept if its cell is new or its prefix is shorter than the one held."""
        self.offered += 1
        if self.max_frames is not None and start.frames > self.max_frames:
            return False
        cell = cell_of(start.position, self.cell_size)
        held = self.starts.get(cell)
        if held is not None and held.frames <= start.frames:
            return False
        self.starts[cell] = start
        self.replaced += held is not None
        return True

    def sample(self) -> Start | None:
        """A start for the next episode, or None meaning the canonical start.

        None is returned for the canonical fraction of calls and whenever the archive is empty, so a run can
        always begin before anything has been archived.
        """
        if not self.starts or self.rng.random() < self.canonical_fraction:
            return None
        cells = sorted(self.starts)
        weights = [1.0 / (1 + self.uses.get(cell, 0)) for cell in cells]
        cell = self.rng.choices(cells, weights=weights, k=1)[0]
        self.uses[cell] = self.uses.get(cell, 0) + 1
        return self.starts[cell]

    def coverage(self) -> dict:
        """What the archive holds, for a run's records."""
        frames = [s.frames for s in self.starts.values()]
        return {
            "cells": len(self.starts),
            "offered": self.offered,
            "replaced": self.replaced,
            "prefix_frames": {"min": min(frames), "median": sorted(frames)[len(frames) // 2], "max": max(frames)}
            if frames else None,
            "furthest_x": max((s.position[0] for s in self.starts.values()), default=None),
            "highest_y": min((s.position[1] for s in self.starts.values()), default=None),
            "cells_used": len(self.uses),
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps({
            "cell_size": self.cell_size,
            "canonical_fraction": self.canonical_fraction,
            "max_frames": self.max_frames,
            "starts": [{"cell": list(cell), "uses": self.uses.get(cell, 0), **start.to_json()}
                       for cell, start in sorted(self.starts.items())],
        }, indent=1), encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def load(path: Path, seed: int | None = None) -> "StartArchive":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        archive = StartArchive(data["cell_size"], data["canonical_fraction"], data["max_frames"], seed)
        for entry in data["starts"]:
            cell = (entry["cell"][0], entry["cell"][1])
            archive.starts[cell] = Start.from_json(entry)
            if entry["uses"]:
                archive.uses[cell] = entry["uses"]
        return archive
