"""Versioned room-task definitions and reproducible starts reached from the base Room 1 savestate."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from celeste_rl.bridge import format_input_line
from celeste_rl.endings import RoomTask
from celeste_rl.starts import Start

REPO = Path(__file__).resolve().parents[1]
FORMAT_VERSION = 1
BASE_ROOM = "1"


class TaskDefinitionError(ValueError):
    """A task definition or its source route is incomplete, altered, or inconsistent."""


@dataclass(frozen=True)
class TaskDefinition:
    name: str
    task: RoomTask
    start: Start | None = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO / path
    path = path.resolve()
    try:
        path.relative_to(REPO.resolve())
    except ValueError as error:
        raise TaskDefinitionError(f"task source is outside the repository: {value}") from error
    return path


def load_task_definition(path: str | Path) -> TaskDefinition:
    """Load a room task and derive its canonical start from a hash-pinned route prefix."""
    definition_path = _repo_path(str(path))
    try:
        data = json.loads(definition_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TaskDefinitionError(f"cannot read task definition {definition_path}: {error}") from error
    if data.get("format_version") != FORMAT_VERSION:
        raise TaskDefinitionError(f"unsupported task format {data.get('format_version')!r}")
    for field in ("name", "start_room", "target_room"):
        if not isinstance(data.get(field), str) or not data[field]:
            raise TaskDefinitionError(f"task definition needs a non-empty {field}")
    if data["start_room"] == data["target_room"]:
        raise TaskDefinitionError("task start_room and target_room must differ")

    source = data.get("source_route")
    if source is None:
        if data["start_room"] != BASE_ROOM:
            raise TaskDefinitionError("a later room task needs a source route from the base savestate")
        return TaskDefinition(data["name"], RoomTask(data["start_room"], data["target_room"]))
    source_path = _repo_path(source)
    if file_sha256(source_path) != data.get("source_route_sha256"):
        raise TaskDefinitionError("task source route does not match its declared SHA-256")
    try:
        route = json.loads(source_path.read_text(encoding="utf-8"))
        prefix_frames = data["prefix_frames"]
        position = data["position"]
        dashes = data["dashes"]
    except (OSError, json.JSONDecodeError, KeyError) as error:
        raise TaskDefinitionError(f"task start metadata is incomplete: {error}") from error
    if (not isinstance(prefix_frames, int) or isinstance(prefix_frames, bool) or prefix_frames <= 0
            or prefix_frames > len(route.get("actions", []))):
        raise TaskDefinitionError("task prefix_frames is outside the source route")
    if route.get("start_room") != BASE_ROOM or route.get("next_room") != data["start_room"]:
        raise TaskDefinitionError("task source route does not enter the declared start room")
    transition_step = route.get("transition_step")
    if not isinstance(transition_step, int) or isinstance(transition_step, bool) \
            or prefix_frames != transition_step + 1:
        raise TaskDefinitionError("task prefix must end on the first frame in the declared start room")
    if (not isinstance(position, list) or len(position) != 2
            or any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in position)):
        raise TaskDefinitionError("task start position must contain two numbers")
    if dashes is not None and (not isinstance(dashes, int) or isinstance(dashes, bool)):
        raise TaskDefinitionError("task start dashes must be an integer or null")
    try:
        lines = tuple(format_input_line(action) for action in route["actions"][:prefix_frames])
    except ValueError as error:
        raise TaskDefinitionError(f"task source route has an invalid action: {error}") from error
    start = Start(lines, tuple(position), data["start_room"], dashes)
    return TaskDefinition(data["name"], RoomTask(data["start_room"], data["target_room"]), start)
