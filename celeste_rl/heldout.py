"""Versioned, auditable held-out state manifests."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

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
    return {
        "route": entry["route"],
        "route_sha256": entry["route_sha256"],
        "search_seed": entry["search_seed"],
        "frames": entry["frames"],
        "room": entry["room"],
        "position": list(entry["position"]),
        "dashes": entry["dashes"],
        "lines": list(entry["lines"]),
    }


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


def validate_manifest(manifest: dict) -> list[dict]:
    """Validate the version, complete-set hash, metadata, IDs, and uniqueness."""
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
    return entries
