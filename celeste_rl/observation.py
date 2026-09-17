"""obs-v1: encode one lockstep reply (state and extras) into the policy's observation arrays.

Pure encoding: these functions receive only the game state, the mod's extras, the applied action and the
elapsed decision count. Events, reward, ending causes and bridge diagnostics never reach them. The schema
(features, scales, channels, constants) lives in celeste_rl/schema.py.

Malformed input (a missing field, a non-numeric or non-finite value, state without extras) raises
SchemaViolation. The environment treats that as a bridge fault: no transition, the episode is abandoned.
"""
from __future__ import annotations

import math

import numpy as np
from gymnasium import spaces

from celeste_rl.schema import (
    ACTION_INPUTS,
    CELL_SIZE,
    CONTEXT_NAMES,
    DEADLINE_FRAMES,
    GRID_ANCHOR,
    GRID_CHANNELS,
    GRID_SIZE,
    HISTORY,
    PLAYER_FEATURE_COUNT,
    PLAYER_FEATURES,
    PLAYER_STATES,
    LIGHTNING_OFFSET,
    SPIKE_OFFSETS,
    SPINNER_BOX,
    STATE_SOURCE,
)

CHANNEL = {name: i for i, name in enumerate(GRID_CHANNELS)}
DIRECTION_NAMES = ("up", "down", "left", "right")  # CelesteTAS GameState.Direction order


class SchemaViolation(ValueError):
    """A reply that cannot be encoded under the schema."""


def observation_space() -> spaces.Dict:
    return spaces.Dict({
        "player": spaces.Box(-np.inf, np.inf, (HISTORY, PLAYER_FEATURE_COUNT), np.float32),
        "actions": spaces.MultiBinary((HISTORY, len(ACTION_INPUTS))),
        "history_valid": spaces.MultiBinary(HISTORY),
        "grid": spaces.Box(0, 1, (len(GRID_CHANNELS), GRID_SIZE, GRID_SIZE), np.uint8),
        "context": spaces.Box(0, 1, (len(CONTEXT_NAMES),), np.float32),
    })


def _lookup(root: dict, path: tuple[str, ...]):
    value = root
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise SchemaViolation(f"missing {'.'.join(path)}")
        value = value[key]
    return value


def _number(value, path: tuple[str, ...]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaViolation(f"{'.'.join(path)} is not a number: {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise SchemaViolation(f"{'.'.join(path)} is not finite: {value!r}")
    return number


def _compile_features():
    """Per-feature (index, path, kind, scale, bounds_key) tuples, resolved once instead of on every step."""
    compiled = []
    for i, feature in enumerate(PLAYER_FEATURES):
        bounds_key = {"room_x": ("X", "W"), "room_y": ("Y", "H")}.get(feature.kind)
        compiled.append((i, feature.source, feature.kind, feature.scale, bounds_key))
    return tuple(compiled)


_COMPILED_FEATURES = _compile_features()
_STATE_OFFSET = len(PLAYER_FEATURES)
_OTHER_STATE = len(PLAYER_STATES) - 1


def encode_player(state: dict, extras: dict) -> np.ndarray:
    """The player feature vector (float32, PLAYER_FEATURE_COUNT) for a frame with a player."""
    root = {"state": state, "extras": extras}
    out = np.zeros(PLAYER_FEATURE_COUNT, dtype=np.float32)
    values = [0.0] * _STATE_OFFSET
    try:
        bounds = root["state"]["Level"]["Bounds"]
        for i, path, kind, scale, bounds_key in _COMPILED_FEATURES:
            raw = root
            for key in path:
                raw = raw[key]
            if kind == "flag":
                if raw is not True and raw is not False:
                    raise SchemaViolation(f"{'.'.join(path)} is not a boolean: {raw!r}")
                values[i] = 1.0 if raw else 0.0
                continue
            if raw.__class__ is not float and raw.__class__ is not int:
                raise SchemaViolation(f"{'.'.join(path)} is not a number: {raw!r}")
            if kind == "value":
                values[i] = raw / scale
            elif kind == "nonzero":
                values[i] = 1.0 if raw != 0 else 0.0
            elif kind == "buffer":
                values[i] = (raw if raw > 0 else 0.0) / scale
            else:
                origin, size = bounds[bounds_key[0]], bounds[bounds_key[1]]
                if origin.__class__ not in (int, float) or size.__class__ not in (int, float) or not size > 0:
                    raise SchemaViolation(f"Level.Bounds is not numeric with a positive size: {bounds!r}")
                values[i] = (raw - origin) / size
        state_index = root["extras"]["player"]["State"]
    except (KeyError, TypeError) as error:
        raise SchemaViolation(f"missing or malformed field: {error!r}") from None
    if state_index.__class__ is not int:
        raise SchemaViolation(f"player state is not an integer: {state_index!r}")
    out[:_STATE_OFFSET] = values
    if not np.all(np.isfinite(out[:_STATE_OFFSET])):
        bad = [PLAYER_FEATURES[i].name for i in np.flatnonzero(~np.isfinite(out[:_STATE_OFFSET]))]
        raise SchemaViolation(f"not finite: {bad}")
    out[_STATE_OFFSET + (state_index if 0 <= state_index < _OTHER_STATE else _OTHER_STATE)] = 1.0
    return out


class GeometryCache:
    """Room tile solidity parsed from SolidsData, cached by room name, bounds and the tile text."""

    def __init__(self):
        self._key = None
        self._solid: np.ndarray | None = None

    def solid_tiles(self, state: dict) -> np.ndarray:
        text = state.get("SolidsData")
        if not isinstance(text, str):
            raise SchemaViolation("SolidsData is not a string")
        bounds = state["Level"]["Bounds"]
        # Comparing the tile text itself is cheaper than hashing it on every step.
        key = (state.get("RoomName"), bounds["X"], bounds["Y"], bounds["W"], bounds["H"], text)
        if key != self._key:
            rows = text.replace("\r", "").split("\n")
            width = max(len(row) for row in rows)
            self._solid = np.array([[c != "0" for c in row.ljust(width, "0")] for row in rows], dtype=bool)
            self._key = key
        return self._solid


def _cell_span(start: float, length: float, origin: float, anchor_offset: int) -> tuple[int, int]:
    """Crop indices [lo, hi) of the cells a half-open world interval [start, start + length) overlaps."""
    lo = math.floor((start - origin) / CELL_SIZE) - anchor_offset
    hi = math.ceil((start + length - origin) / CELL_SIZE) - anchor_offset
    return max(lo, 0), min(hi, GRID_SIZE)


def _rect(value, what: str) -> tuple[float, float, float, float]:
    if not isinstance(value, dict):
        raise SchemaViolation(f"{what} is not a rectangle: {value!r}")
    return tuple(_number(value.get(k), (what, k)) for k in ("X", "Y", "W", "H"))


def _direction(item, what: str) -> str:
    direction = item.get("Direction") if isinstance(item, dict) else None
    if isinstance(direction, bool) or not isinstance(direction, int) or not 0 <= direction < len(DIRECTION_NAMES):
        raise SchemaViolation(f"{what} has an invalid direction: {item!r}")
    return DIRECTION_NAMES[direction]


def encode_grid(state: dict, extras: dict, cache: GeometryCache) -> np.ndarray:
    """The local geometry grid (uint8, channels x 32 x 32) around the player's collider centre."""
    grid = np.zeros((len(GRID_CHANNELS), GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    bounds = _rect(_lookup({"state": state}, ("state", "Level", "Bounds")), "Level.Bounds")
    bx, by, bw, bh = bounds
    if not (bw > 0 and bh > 0):
        raise SchemaViolation(f"Level.Bounds has no positive size: {bounds!r}")
    position = _lookup({"state": state}, ("state", "Player", "Position"))
    collider = _rect(_lookup({"extras": extras}, ("extras", "player", "Collider")), "Collider")
    centre_x = _number(position.get("X"), ("Position", "X")) + collider[0] + collider[2] / 2
    centre_y = _number(position.get("Y"), ("Position", "Y")) + collider[1] + collider[3] / 2
    # Room-tile coordinates of the crop's top-left cell.
    col0 = math.floor((centre_x - bx) / CELL_SIZE) - GRID_ANCHOR
    row0 = math.floor((centre_y - by) / CELL_SIZE) - GRID_ANCHOR

    solid = cache.solid_tiles(state)
    rows, cols = solid.shape
    r_lo, r_hi = max(row0, 0), min(row0 + GRID_SIZE, rows)
    c_lo, c_hi = max(col0, 0), min(col0 + GRID_SIZE, cols)
    if r_lo < r_hi and c_lo < c_hi:
        grid[CHANNEL["solid"], r_lo - row0:r_hi - row0, c_lo - col0:c_hi - col0] = solid[r_lo:r_hi, c_lo:c_hi]

    def stamp(channel: int, x: float, y: float, w: float, h: float) -> None:
        if w <= 0 or h <= 0:
            return
        c0, c1 = _cell_span(x, w, bx, col0)
        r0, r1 = _cell_span(y, h, by, row0)
        if r0 < r1 and c0 < c1:
            grid[channel, r0:r1, c0:c1] = 1

    for rect in state.get("StaticSolids") or []:
        stamp(CHANNEL["solid"], *_rect(rect, "StaticSolids"))
    for item in state.get("JumpThrus") or []:
        stamp(CHANNEL[f"jumpthru_{_direction(item, 'JumpThrus')}"], *_rect(item.get("Bounds"), "JumpThrus"))
    for item in state.get("Spikes") or []:
        direction = _direction(item, "Spikes")
        x, y, w, h = _rect(item.get("Bounds"), "Spikes")
        dx, dy = SPIKE_OFFSETS[direction]
        stamp(CHANNEL[f"spikes_{direction}"], x + dx, y + dy, w, h)
    for rect in state.get("Lightning") or []:
        x, y, w, h = _rect(rect, "Lightning")
        stamp(CHANNEL["other_hazards"], x + LIGHTNING_OFFSET[0], y + LIGHTNING_OFFSET[1], w, h)
    for spinner in state.get("Spinners") or []:
        x, y = _number(spinner.get("X"), ("Spinners", "X")), _number(spinner.get("Y"), ("Spinners", "Y"))
        stamp(CHANNEL["other_hazards"], x - SPINNER_BOX / 2, y - SPINNER_BOX / 2, SPINNER_BOX, SPINNER_BOX)

    # Outside the room: any part of the cell beyond Level.Bounds.
    room_rows = np.arange(row0, row0 + GRID_SIZE)
    room_cols = np.arange(col0, col0 + GRID_SIZE)
    rows_out = (room_rows < 0) | ((room_rows + 1) * CELL_SIZE > bh)
    cols_out = (room_cols < 0) | ((room_cols + 1) * CELL_SIZE > bw)
    grid[CHANNEL["outside_room"]] = rows_out[:, None] | cols_out[None, :]
    return grid


class ObservationBuilder:
    """Keeps the four-decision history and produces observations that match observation_space().

    Row 0 is the most recent decision. actions[i] is the applied action that led to player[i]; at reset it is
    all zeros, which is exact for the canonical start (its last prefix frame is neutral). Every returned
    array is a fresh copy.
    """

    def __init__(self):
        self._geometry = GeometryCache()
        self._player = np.zeros((HISTORY, PLAYER_FEATURE_COUNT), dtype=np.float32)
        self._actions = np.zeros((HISTORY, len(ACTION_INPUTS)), dtype=np.int8)
        self._valid = np.zeros(HISTORY, dtype=np.int8)
        self._grid = np.zeros((len(GRID_CHANNELS), GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        self._context = np.zeros(len(CONTEXT_NAMES), dtype=np.float32)

    def _encode_current(self, state: dict | None, extras: dict | None) -> tuple[np.ndarray, np.ndarray, bool]:
        if state is None:
            return (np.zeros(PLAYER_FEATURE_COUNT, dtype=np.float32),
                    np.zeros((len(GRID_CHANNELS), GRID_SIZE, GRID_SIZE), dtype=np.uint8), False)
        if not isinstance(state, dict) or not isinstance(state.get("Player"), dict):
            raise SchemaViolation("state has no Player object")
        if not isinstance(extras, dict):
            raise SchemaViolation("a frame with a player has no extras")
        return encode_player(state, extras), encode_grid(state, extras, self._geometry), True

    def reset(self, state: dict | None, extras: dict | None) -> dict[str, np.ndarray]:
        player, grid, present = self._encode_current(state, extras)
        self._player[:] = 0
        self._actions[:] = 0
        self._valid[:] = 0
        self._player[0], self._valid[0], self._grid = player, 1, grid
        self._context[:] = (0.0, float(present))
        return self.observation()

    def step(self, state: dict | None, extras: dict | None, applied_action: np.ndarray, elapsed: int) -> dict[str, np.ndarray]:
        # Encode first, so a schema violation leaves the history untouched.
        player, grid, present = self._encode_current(state, extras)
        self._player[1:] = self._player[:-1].copy()
        self._actions[1:] = self._actions[:-1].copy()
        self._valid[1:] = self._valid[:-1].copy()
        self._player[0], self._actions[0], self._valid[0], self._grid = player, applied_action, 1, grid
        self._context[:] = (elapsed / DEADLINE_FRAMES, float(present))
        return self.observation()

    def observation(self) -> dict[str, np.ndarray]:
        return {
            "player": self._player.copy(),
            "actions": self._actions.copy(),
            "history_valid": self._valid.copy(),
            "grid": self._grid.copy(),
            "context": self._context.copy(),
        }
