"""Offline checks for structural separation of demonstrations and held-out routes."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from celeste_rl.demonstrations import (
    DEMONSTRATION_FORMAT_VERSION,
    DemonstrationManifestError,
    dataset_manifest_path,
    demonstration_manifest_sha256,
    materialize_demonstrations,
    reject_heldout_overlap,
    route_sha256,
    validate_demonstration_manifest,
    verify_dataset_manifest,
    write_dataset_manifest,
)


def entry(name: str, route: list[str], source: str | None = None) -> dict:
    return {
        "kind": "route",
        "name": name,
        "source": source or f"runs/routes/{name}/route.json",
        "route_sha256": route_sha256(route),
    }


def manifest(entries: list[dict]) -> dict:
    return {
        "format_version": DEMONSTRATION_FORMAT_VERSION,
        "sha256": demonstration_manifest_sha256(entries),
        "demonstrations": entries,
    }


class DemonstrationManifestTests(unittest.TestCase):
    def test_explicit_manifest_round_trips(self) -> None:
        entries = [entry("a", ["1,R"]), entry("b", ["1,L"])]

        self.assertEqual(validate_demonstration_manifest(manifest(entries)), entries)

    def test_duplicate_route_content_is_rejected_even_under_another_name(self) -> None:
        entries = [entry("a", ["1,R"]), entry("renamed", ["1,R"])]

        with self.assertRaisesRegex(DemonstrationManifestError, "duplicate route hash"):
            validate_demonstration_manifest(manifest(entries))

    def test_overlap_is_content_based_not_seed_or_name_based(self) -> None:
        route_hash = route_sha256(["1,R", "1,J"])
        demonstrations = [{"route_sha256": route_hash, "name": "training", "search_seed": None}]
        heldout = [{"route_sha256": route_hash, "route": "different-name", "search_seed": 2}]

        with self.assertRaisesRegex(DemonstrationManifestError, "overlap"):
            reject_heldout_overlap(demonstrations, heldout)

    def test_declared_source_must_match_its_route_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            source = repo / "runs" / "routes" / "a" / "route.json"
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps({
                "transition_step": 2,
                "actions": ["R", "J"],
            }), encoding="utf-8")
            declared = entry("a", ["1,R", "1,J"])

            loaded = materialize_demonstrations([declared], repo)
            self.assertEqual(loaded[0]["lines"], ["1,R", "1,J"])

            source.write_text(json.dumps({
                "transition_step": 2,
                "actions": ["L", "J"],
            }), encoding="utf-8")
            with self.assertRaisesRegex(DemonstrationManifestError, "does not match route_sha256"):
                materialize_demonstrations([declared], repo)


class DatasetManifestTests(unittest.TestCase):
    def test_dataset_and_both_source_manifest_hashes_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "dataset.npz"
            dataset.write_bytes(b"recorded observations")
            route_hash = route_sha256(["1,R"])
            written = write_dataset_manifest(dataset, "d" * 64, "e" * 64, [route_hash])

            verified = verify_dataset_manifest(dataset, "d" * 64, "e" * 64)

            self.assertEqual(verified, written)
            self.assertTrue(dataset_manifest_path(dataset).is_file())

            with self.assertRaisesRegex(DemonstrationManifestError, "heldout_manifest_sha256"):
                verify_dataset_manifest(dataset, "d" * 64, "f" * 64)

    def test_dataset_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "dataset.npz"
            dataset.write_bytes(b"original")
            write_dataset_manifest(dataset, "d" * 64, "e" * 64, [route_sha256(["1,R"])])
            dataset.write_bytes(b"changed")

            with self.assertRaisesRegex(DemonstrationManifestError, "dataset_sha256"):
                verify_dataset_manifest(dataset, "d" * 64, "e" * 64)


if __name__ == "__main__":
    unittest.main()
