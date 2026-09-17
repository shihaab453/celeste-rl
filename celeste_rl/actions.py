"""act-v1: the environment's 24 on/off inputs to and from canonical input lines.

Every action vector maps to exactly one canonical line (letters in the bridge's fixed order, each at most
once, dash-only and move-only directions omitted when none are set), and different vectors give different
canonical lines. The mod's regex also accepts non-canonical text; parse_line canonicalises it or rejects
it, so recorded lines always round-trip to one vector.

Disabled inputs (decision D1: pause, quick restart and journal for initial training) are forced to 0 before
the line is built. The applied vector, not the requested one, is what reaches the game and the observation.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from celeste_rl.bridge import BUTTON_LETTERS, DIRECTIONS, format_input_line
from celeste_rl.schema import ACTION_INPUTS

INDEX = {name: i for i, name in enumerate(ACTION_INPUTS)}


def disabled_mask(disabled: Iterable[str]) -> np.ndarray:
    """A boolean mask over ACTION_INPUTS, True where the input is disabled."""
    mask = np.zeros(len(ACTION_INPUTS), dtype=bool)
    for name in disabled:
        if name not in INDEX:
            raise ValueError(f"Unknown input {name!r}; expected one of {ACTION_INPUTS}")
        mask[INDEX[name]] = True
    return mask


def apply_disabled(action, mask: np.ndarray) -> np.ndarray:
    """Validate an action and return the applied vector (int8, disabled inputs forced to 0)."""
    vector = np.asarray(action)
    if vector.shape != (len(ACTION_INPUTS),):
        raise ValueError(f"Action must have shape ({len(ACTION_INPUTS)},), got {vector.shape}")
    if not np.all((vector == 0) | (vector == 1)):
        raise ValueError(f"Action values must be 0 or 1, got {vector.tolist()}")
    applied = vector.astype(np.int8)
    applied[mask] = 0
    return applied


def to_parts(applied: np.ndarray) -> tuple[str, str, str]:
    """(buttons, dash_only, move_only) letters for an applied action vector, as the bridge's step() takes them."""
    buttons = "".join(name for name, on in zip(ACTION_INPUTS, applied) if on and len(name) == 1)
    dash_only = "".join(name[1] for name, on in zip(ACTION_INPUTS, applied) if on and name[0] == "A" and len(name) == 2)
    move_only = "".join(name[1] for name, on in zip(ACTION_INPUTS, applied) if on and name[0] == "M" and len(name) == 2)
    return buttons, dash_only, move_only


def to_line(applied: np.ndarray) -> str:
    """The canonical input line for an applied action vector."""
    return format_input_line(*to_parts(applied))


def parse_line(line: str, canonical_only: bool = False) -> np.ndarray:
    """The action vector for an input line.

    Accepts any line the mod would accept for a single frame of held inputs. Repeated or reordered letters are
    canonicalised; with canonical_only=True they are rejected instead. Anything else raises ValueError.
    """
    tokens = line.split(",")
    if tokens[0] != "1":
        raise ValueError(f"Not a one-frame input line: {line!r}")
    vector = np.zeros(len(ACTION_INPUTS), dtype=np.int8)
    for token in tokens[1:]:
        if len(token) == 1 and token in BUTTON_LETTERS:
            vector[INDEX[token]] = 1
        elif len(token) >= 2 and token[0] in "AM" and all(d in DIRECTIONS for d in token[1:]):
            for direction in token[1:]:
                vector[INDEX[token[0] + direction]] = 1
        else:
            raise ValueError(f"Unknown token {token!r} in {line!r}")
    if canonical_only and to_line(vector) != line:
        raise ValueError(f"Not canonical: {line!r} (canonical form {to_line(vector)!r})")
    return vector
