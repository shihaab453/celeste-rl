"""Versioned, auditable held-out state manifests."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from celeste_rl.tasks import TaskDefinitionError, require_task_identity

FORMAT_VERSION = 3


class HeldoutManifestError(ValueError):
    """The frozen held-out set is incomplete, altered, or from an unsupported format."""


def _canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def canonical_entry(entry: dict) -> dict:
    """Only the complete state definition and provenance, with stable JSON-compatible types."""
    required = ("route", "route_sha256", "search_seed", "frames", "room", "position", "dashes", "lines")
    missing = [field for field in required if field not in entry]
    if missing:
        raise HeldoutManifestError(f"held-out entry is missing {missing}")
    if not isinstance(entry["route"], str) or not entry["route"]:
        raise HeldoutManifestError("held-out entry route must be a non-empty string")
    if (not isinstance(entry["route_sha256"], str) or len(entry["route_sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in entry["route_sha256"])):
        raise HeldoutManifestError("held-out entry route_sha256 must be a lowercase SHA-256")
    if not isinstance(entry["search_seed"], int) or isinstance(entry["search_seed"], bool):
        raise HeldoutManifestError("held-out entry search_seed must be an integer")
    if not isinstance(entry["frames"], int) or isinstance(entry["frames"], bool) or entry["frames"] < 0:
        raise HeldoutManifestError("held-out entry frames must be a non-negative integer")
    if not isinstance(entry["room"], str) or not entry["room"]:
        raise HeldoutManifestError("held-out entry room must be a non-empty string")
    if (not isinstance(entry["position"], (list, tuple)) or len(entry["position"]) != 2
            or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                   for value in entry["position"])):
        raise HeldoutManifestError("held-out entry position must contain two numbers")
    if entry["dashes"] is not None and (not isinstance(entry["dashes"], int)
                                        or isinstance(entry["dashes"], bool)):
        raise HeldoutManifestError("held-out entry dashes must be an integer or null")
    if not isinstance(entry["lines"], (list, tuple)) or any(not isinstance(line, str) for line in entry["lines"]):
        raise HeldoutManifestError("held-out entry lines must be a sequence of strings")
    if entry["frames"] != len(entry["lines"]):
        raise HeldoutManifestError(
            f"held-out entry says {entry['frames']} frames but contains {len(entry['lines'])} input lines")
    canonical = {
        "route": entry["route"],
        "route_sha256": entry["route_sha256"],
        "search_seed": entry["search_seed"],
        "frames": entry["frames"],
        "room": entry["room"],
        "position": list(entry["position"]),
        "dashes": entry["dashes"],
        "lines": list(entry["lines"]),
    }
    if "task_frames" in entry:
        task_frames = entry["task_frames"]
        if (not isinstance(task_frames, int) or isinstance(task_frames, bool) or task_frames < 0
                or task_frames > entry["frames"]):
            raise HeldoutManifestError("held-out entry task_frames must be between zero and frames")
        canonical["task_frames"] = task_frames
    if "room_position" in entry:
        room_position = entry["room_position"]
        if (not isinstance(room_position, (list, tuple)) or len(room_position) != 2
                or any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in room_position)):
            raise HeldoutManifestError("held-out entry room_position must contain two numbers")
        canonical["room_position"] = list(room_position)
    return canonical


def state_id(entry: dict) -> str:
    """A state identity independent of the route directory that supplied it."""
    canonical = canonical_entry(entry)
    state = {field: canonical[field] for field in ("frames", "room", "position", "dashes", "lines")}
    return hashlib.sha256(_canonical_json(state).encode("utf-8")).hexdigest()


def manifest_sha256(entries: Iterable[dict]) -> str:
    """Hash every state field and provenance field, in evaluation order."""
    canonical = [canonical_entry(entry) for entry in entries]
    return hashlib.sha256(_canonical_json(canonical).encode("utf-8")).hexdigest()


def freeze_entries(entries: Iterable[dict]) -> list[dict]:
    frozen = []
    for entry in entries:
        canonical = canonical_entry(entry)
        frozen.append({"state_id": state_id(canonical), **canonical})
    return frozen


def validate_manifest(manifest: dict, expected_task: dict | None = None) -> list[dict]:
    """Validate the version, complete-set hash, metadata, IDs, and uniqueness."""
    if expected_task is not None:
        try:
            require_task_identity(manifest, expected_task, "held-out manifest")
        except TaskDefinitionError as error:
            raise HeldoutManifestError(str(error)) from error
    if manifest.get("format_version") != FORMAT_VERSION:
        raise HeldoutManifestError(
            f"held-out format {manifest.get('format_version')!r} is unsupported; regenerate format {FORMAT_VERSION}")
    raw_entries = manifest.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise HeldoutManifestError("held-out manifest must contain a non-empty entries list")
    entries, seen = [], set()
    for index, raw in enumerate(raw_entries):
        canonical = canonical_entry(raw)
        expected_id = state_id(canonical)
        if raw.get("state_id") != expected_id:
            raise HeldoutManifestError(f"held-out entry {index} does not match its state_id")
        if expected_id in seen:
            raise HeldoutManifestError(f"held-out entry {index} duplicates state {expected_id}")
        seen.add(expected_id)
        entries.append({"state_id": expected_id, **canonical})
    digest = manifest_sha256(entries)
    if manifest.get("sha256") != digest:
        raise HeldoutManifestError("held-out entries do not match the manifest sha256")
    if manifest.get("states") != len(entries):
        raise HeldoutManifestError(f"held-out states metadata says {manifest.get('states')}; found {len(entries)}")
    routes = sorted({entry["route"] for entry in entries})
    if manifest.get("routes") != routes:
        raise HeldoutManifestError("held-out routes metadata does not match the entries")
    frames = {"min": min(entry["frames"] for entry in entries),
              "max": max(entry["frames"] for entry in entries)}
    if manifest.get("frames") != frames:
        raise HeldoutManifestError("held-out frame metadata does not match the entries")
    task_frames = [entry.get("task_frames") for entry in entries]
    if any(value is not None for value in task_frames):
        if any(value is None for value in task_frames):
            raise HeldoutManifestError("held-out task_frames must be present on every entry or none")
        expected_task_frames = {"min": min(task_frames), "max": max(task_frames)}
        if manifest.get("task_frames") != expected_task_frames:
            raise HeldoutManifestError("held-out task-frame metadata does not match the entries")
    return entries
