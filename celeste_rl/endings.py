"""Episode ending classification from the mod's game events (Phase 2 spec, section 6.3).

Each step's reply carries the events the game raised since the previous reply, including during automatic
loading. The ending is decided from those events first and the state second, in a fixed precedence:

  2. death (a death event), restart (a level load that is not a room transition: chapter restart, reload,
     respawn), left_level (a level exit). Always beat success in the same reply.
  3. success: a transition from the task's start room to its target room.
  4. wrong_room: any other transition.
  5. timeout: the decision-frame counter reached the deadline.
  6. stalled (training option only, set by the environment, never by `classify`): no new best progress
     potential for the configured number of frames. It is an ordinary failure and never beats the others.

Anything the rules cannot explain raises EndingFault, which the environment reports as a bridge fault: no
player without a death, restart or level exit in the same reply; an unknown event type; a malformed event.
Absence of a player is never relabelled as death.
"""
from __future__ import annotations

from dataclasses import dataclass

from celeste_rl.schema import DEADLINE_FRAMES

SUCCESS = "success"
DEATH = "death"
RESTART = "restart"
LEFT_LEVEL = "left_level"
WRONG_ROOM = "wrong_room"
TIMEOUT = "timeout"
STALLED = "stalled"
FAILURES = (DEATH, RESTART, LEFT_LEVEL, WRONG_ROOM, TIMEOUT, STALLED)
# Every ending, in the order records list them.
ENDINGS = (SUCCESS, *FAILURES)

KNOWN_EVENTS = {"death", "transition", "load_level", "level_exit", "pause", "unpause"}


class EndingFault(ValueError):
    """A reply whose events and state do not fit the ending rules."""


@dataclass(frozen=True)
class RoomTask:
    start_room: str = "1"
    target_room: str = "2"
    deadline_frames: int = DEADLINE_FRAMES


def _string(event: dict, key: str) -> str:
    value = event.get(key)
    if not isinstance(value, str):
        raise EndingFault(f"{event.get('type')} event has no string {key!r}: {event!r}")
    return value


def classify(events: list[dict] | None, state: dict | None, elapsed: int, task: RoomTask) -> str | None:
    """The ending cause for one step's reply, or None if the episode continues."""
    if not isinstance(events, list):
        raise EndingFault("the reply has no events list (are the mod's extras disabled?)")
    types = []
    for event in events:
        if not isinstance(event, dict) or event.get("type") not in KNOWN_EVENTS:
            raise EndingFault(f"unknown event {event!r}")
        types.append(event["type"])

    if "death" in types:
        return DEATH
    if any(event["type"] == "load_level" and _string(event, "intro") != "Transition" for event in events):
        return RESTART
    if "level_exit" in types:
        return LEFT_LEVEL

    if state is None:
        raise EndingFault("no player in the reply, and no death, restart or level exit explains it")

    transitions = [event for event in events if event["type"] == "transition"]
    if transitions:
        if len(transitions) == 1 and _string(transitions[0], "from") == task.start_room \
                and _string(transitions[0], "to") == task.target_room:
            return SUCCESS
        return WRONG_ROOM

    if elapsed >= task.deadline_frames:
        return TIMEOUT
    return None
