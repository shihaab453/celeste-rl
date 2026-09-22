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
BASE_TASK_NAME = "chapter-1-room-1"


class TaskDefinitionError(ValueError):
    """A task definition or its source route is incomplete, altered, or inconsistent."""


@dataclass(frozen=True)
class TaskDefinition:
    name: str
    task: RoomTask
    start: Start | None = None
    definition_path: str | None = None
    definition_sha256: str | None = None
    source_route_sha256: str | None = None


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


def base_task_definition() -> TaskDefinition:
    """The original Room 1 task, which needs no replay recipe beyond the base savestate."""
    return TaskDefinition(BASE_TASK_NAME, RoomTask())


def resolve_task_definition(path: str | Path | None) -> TaskDefinition:
    return load_task_definition(path) if path else base_task_definition()


def task_identity(definition: TaskDefinition) -> dict:
    """Stable task provenance carried by every later-room dataset and result."""
    return {
        "name": definition.name,
        "task_definition": definition.definition_path,
        "task_definition_sha256": definition.definition_sha256,
        "source_route_sha256": definition.source_route_sha256,
        "start_room": definition.task.start_room,
        "target_room": definition.task.target_room,
    }


def canonical_task_identity(value: dict) -> dict:
    if not isinstance(value, dict):
        raise TaskDefinitionError("task identity must be an object")
    required = ("name", "task_definition", "task_definition_sha256", "source_route_sha256",
                "start_room", "target_room")
    missing = [field for field in required if field not in value]
    if missing:
        raise TaskDefinitionError(f"task identity is missing {missing}")
    for field in ("name", "start_room", "target_room"):
        if not isinstance(value[field], str) or not value[field]:
            raise TaskDefinitionError(f"task identity {field} must be a non-empty string")
    path, definition_hash = value["task_definition"], value["task_definition_sha256"]
    if (path is None) != (definition_hash is None):
        raise TaskDefinitionError("task definition path and SHA-256 must either both be set or both be null")
    if path is not None and (not isinstance(path, str) or not path):
        raise TaskDefinitionError("task identity task_definition must be a non-empty string or null")
    for field in ("task_definition_sha256", "source_route_sha256"):
        digest = value[field]
        if digest is not None and (not isinstance(digest, str) or len(digest) != 64
                                   or any(character not in "0123456789abcdef" for character in digest)):
            raise TaskDefinitionError(f"task identity {field} must be a lowercase SHA-256 or null")
    if value["start_room"] == value["target_room"]:
        raise TaskDefinitionError("task identity start_room and target_room must differ")
    if value["start_room"] != BASE_ROOM:
        if path is None:
            raise TaskDefinitionError("a later-room task identity must name its task definition")
        if value["source_route_sha256"] is None:
            raise TaskDefinitionError("a later-room task identity must include its source-route SHA-256")
    return {field: value[field] for field in required}


def manifest_task_identity(manifest: dict) -> dict:
    """Read explicit task provenance, treating legacy manifests as the original Room 1 task."""
    if "task" not in manifest:
        return task_identity(base_task_definition())
    return canonical_task_identity(manifest["task"])


def require_task_identity(manifest: dict, expected: dict, label: str) -> dict:
    actual = manifest_task_identity(manifest)
    canonical_expected = canonical_task_identity(expected)
    if actual != canonical_expected:
        differences = [field for field in canonical_expected if actual[field] != canonical_expected[field]]
        raise TaskDefinitionError(f"{label} task identity does not match ({', '.join(differences)})")
    return actual


def load_task_definition(path: str | Path) -> TaskDefinition:
    """Load a room task and derive its canonical start from a hash-pinned route prefix."""
    definition_path = _repo_path(str(path))
    try:
        relative_definition = definition_path.relative_to(REPO.resolve()).as_posix()
        definition_hash = file_sha256(definition_path)
        data = json.loads(definition_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
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
        return TaskDefinition(data["name"], RoomTask(data["start_room"], data["target_room"]),
                              definition_path=relative_definition, definition_sha256=definition_hash)
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
    return TaskDefinition(data["name"], RoomTask(data["start_room"], data["target_room"]), start,
                          relative_definition, definition_hash, data["source_route_sha256"])
