"""Auditable demonstration sources and cloning-dataset provenance."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from celeste_rl.bridge import format_input_line

DEMONSTRATION_FORMAT_VERSION = 1
DATASET_FORMAT_VERSION = 1


class DemonstrationManifestError(ValueError):
    """A demonstration source, manifest, or dataset audit record is invalid."""


def _canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _valid_sha256(value) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def route_sha256(lines: Iterable[str]) -> str:
    """Identify one complete route by its canonical per-frame TAS input lines."""
    canonical = list(lines)
    if any(not isinstance(line, str) for line in canonical):
        raise DemonstrationManifestError("route lines must all be strings")
    return hashlib.sha256(_canonical_json(canonical).encode("utf-8")).hexdigest()


def canonical_demonstration(entry: dict) -> dict:
    required = ("kind", "name", "source", "route_sha256")
    missing = [field for field in required if field not in entry]
    if missing:
        raise DemonstrationManifestError(f"demonstration entry is missing {missing}")
    if entry["kind"] not in ("route", "archive"):
        raise DemonstrationManifestError("demonstration kind must be 'route' or 'archive'")
    for field in ("name", "source"):
        if not isinstance(entry[field], str) or not entry[field]:
            raise DemonstrationManifestError(f"demonstration {field} must be a non-empty string")
    if not _valid_sha256(entry["route_sha256"]):
        raise DemonstrationManifestError("demonstration route_sha256 must be a lowercase SHA-256")
    canonical = {field: entry[field] for field in required}
    if entry["kind"] == "archive":
        index = entry.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise DemonstrationManifestError("archive demonstration index must be a non-negative integer")
        canonical["index"] = index
    elif "index" in entry:
        raise DemonstrationManifestError("route demonstration must not declare an index")
    return canonical


def demonstration_manifest_sha256(entries: Iterable[dict]) -> str:
    canonical = [canonical_demonstration(entry) for entry in entries]
    return hashlib.sha256(_canonical_json(canonical).encode("utf-8")).hexdigest()


def validate_demonstration_manifest(manifest: dict) -> list[dict]:
    if manifest.get("format_version") != DEMONSTRATION_FORMAT_VERSION:
        raise DemonstrationManifestError(
            f"demonstration format {manifest.get('format_version')!r} is unsupported; "
            f"expected {DEMONSTRATION_FORMAT_VERSION}")
    raw = manifest.get("demonstrations")
    if not isinstance(raw, list) or len(raw) < 2:
        raise DemonstrationManifestError("demonstration manifest must contain at least two routes")
    entries = [canonical_demonstration(entry) for entry in raw]
    route_hashes = [entry["route_sha256"] for entry in entries]
    if len(route_hashes) != len(set(route_hashes)):
        raise DemonstrationManifestError("demonstration manifest contains a duplicate route hash")
    if manifest.get("sha256") != demonstration_manifest_sha256(entries):
        raise DemonstrationManifestError("demonstrations do not match the manifest sha256")
    return entries


def _source_path(repo: Path, source: str) -> Path:
    repository = repo.resolve()
    path = (repository / source).resolve()
    try:
        path.relative_to(repository)
    except ValueError as error:
        raise DemonstrationManifestError(f"demonstration source is outside the repository: {source}") from error
    if not path.is_file():
        raise DemonstrationManifestError(f"demonstration source does not exist: {source}")
    return path


def materialize_demonstrations(entries: Iterable[dict], repo: Path) -> list[dict]:
    """Read only the explicitly declared sources and verify every complete-route hash."""
    demonstrations = []
    for entry in entries:
        source = _source_path(repo, entry["source"])
        try:
            document = json.loads(source.read_text(encoding="utf-8"))
            if entry["kind"] == "route":
                transition_step = document["transition_step"]
                actions = document["actions"]
                if (not isinstance(transition_step, int) or isinstance(transition_step, bool)
                        or transition_step <= 0 or len(actions) < transition_step):
                    raise DemonstrationManifestError(f"route source has no complete transition: {entry['source']}")
                lines = [format_input_line(buttons) for buttons in actions[:transition_step]]
            else:
                lines = list(document["demonstrations"][entry["index"]]["lines"])
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise DemonstrationManifestError(f"could not read demonstration source {entry['source']}: {error}") \
                from error
        actual = route_sha256(lines)
        if actual != entry["route_sha256"]:
            raise DemonstrationManifestError(
                f"demonstration source {entry['source']} does not match route_sha256")
        demonstrations.append({**entry, "lines": lines})
    return demonstrations


def reject_heldout_overlap(demonstrations: Iterable[dict], heldout_entries: Iterable[dict]) -> None:
    demonstration_hashes = {entry["route_sha256"] for entry in demonstrations}
    heldout_hashes = {entry["route_sha256"] for entry in heldout_entries}
    overlap = sorted(demonstration_hashes & heldout_hashes)
    if overlap:
        raise DemonstrationManifestError(
            "demonstration routes overlap the held-out manifest: " + ", ".join(overlap))


def dataset_manifest_path(dataset: Path) -> Path:
    return dataset.with_suffix(".manifest.json")


def write_dataset_manifest(dataset: Path, demonstrations_sha256: str, heldout_sha256: str,
                           route_hashes: Iterable[str]) -> dict:
    if not _valid_sha256(demonstrations_sha256) or not _valid_sha256(heldout_sha256):
        raise DemonstrationManifestError("dataset source manifest hashes must be lowercase SHA-256 values")
    canonical_route_hashes = sorted(set(route_hashes))
    if not canonical_route_hashes or any(not _valid_sha256(value) for value in canonical_route_hashes):
        raise DemonstrationManifestError("dataset must name at least one valid demonstration route hash")
    manifest = {
        "format_version": DATASET_FORMAT_VERSION,
        "dataset_sha256": file_sha256(dataset),
        "demonstrations_manifest_sha256": demonstrations_sha256,
        "heldout_manifest_sha256": heldout_sha256,
        "demonstration_route_sha256s": canonical_route_hashes,
    }
    dataset_manifest_path(dataset).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_dataset_manifest(dataset: Path, demonstrations_sha256: str, heldout_sha256: str) -> dict:
    path = dataset_manifest_path(dataset)
    if not path.is_file():
        raise DemonstrationManifestError(f"dataset provenance manifest does not exist: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DemonstrationManifestError(f"dataset provenance manifest is invalid JSON: {path}") from error
    if manifest.get("format_version") != DATASET_FORMAT_VERSION:
        raise DemonstrationManifestError("unsupported dataset provenance manifest format")
    checks = {
        "dataset_sha256": file_sha256(dataset),
        "demonstrations_manifest_sha256": demonstrations_sha256,
        "heldout_manifest_sha256": heldout_sha256,
    }
    for field, expected in checks.items():
        if manifest.get(field) != expected:
            raise DemonstrationManifestError(f"dataset provenance {field} does not match")
    route_hashes = manifest.get("demonstration_route_sha256s")
    if (not isinstance(route_hashes, list) or not route_hashes
            or any(not _valid_sha256(value) for value in route_hashes)
            or route_hashes != sorted(set(route_hashes))):
        raise DemonstrationManifestError("dataset provenance route hashes are invalid")
    return manifest
