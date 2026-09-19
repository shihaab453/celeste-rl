"""Held-out manifest integrity and route-aware evaluation summaries."""
from __future__ import annotations

import copy
import unittest

from celeste_rl.demonstrations import route_sha256
from celeste_rl.heldout import (
    FORMAT_VERSION,
    HeldoutManifestError,
    freeze_entries,
    manifest_sha256,
    state_id,
    validate_manifest,
)
from scripts.evaluate_heldout import route_summaries


def entry(route: str = "route-a", seed: int = 100, line: str = "1,R") -> dict:
    return {
        "route": route,
        "route_sha256": route_sha256([line, "1,J"]),
        "search_seed": seed,
        "frames": 1,
        "room": "1",
        "position": [19.0, 152.0],
        "dashes": 1,
        "lines": [line],
    }


def manifest(entries: list[dict]) -> dict:
    frozen = freeze_entries(entries)
    return {
        "format_version": FORMAT_VERSION,
        "sha256": manifest_sha256(frozen),
        "states": len(frozen),
        "routes": sorted({item["route"] for item in frozen}),
        "frames": {"min": min(item["frames"] for item in frozen),
                   "max": max(item["frames"] for item in frozen)},
        "entries": frozen,
    }


class ManifestTests(unittest.TestCase):
    def test_complete_manifest_round_trips(self):
        source = [entry(), entry("route-b", 101, "1,L")]

        validated = validate_manifest(manifest(source))

        self.assertEqual([item["state_id"] for item in validated], [state_id(item) for item in source])

    def test_state_id_is_independent_of_route_provenance(self):
        first = entry()
        same_state = {**first, "route": "renamed-route", "search_seed": 999}

        self.assertEqual(state_id(first), state_id(same_state))

    def test_manifest_hash_covers_every_state_and_provenance_field(self):
        original = entry()
        changes = [
            {**original, "route": "route-b"},
            {**original, "route_sha256": route_sha256(["1,L", "1,J"])},
            {**original, "search_seed": 101},
            {**original, "room": "2"},
            {**original, "position": [20.0, 152.0]},
            {**original, "dashes": 0},
            {**original, "frames": 2, "lines": ["1,R", "1,J"]},
        ]

        original_hash = manifest_sha256([original])

        for changed in changes:
            with self.subTest(changed=changed):
                self.assertNotEqual(manifest_sha256([changed]), original_hash)

    def test_legacy_manifest_is_rejected(self):
        old = manifest([entry()])
        del old["format_version"]

        with self.assertRaisesRegex(HeldoutManifestError, f"regenerate format {FORMAT_VERSION}"):
            validate_manifest(old)

    def test_duplicate_state_ids_are_rejected_even_with_different_provenance(self):
        first = entry()
        duplicate = {**first, "route": "route-b", "search_seed": 101}
        frozen = freeze_entries([first, duplicate])
        data = {
            "format_version": FORMAT_VERSION,
            "sha256": manifest_sha256(frozen),
            "states": 2,
            "routes": ["route-a", "route-b"],
            "frames": {"min": 1, "max": 1},
            "entries": frozen,
        }

        with self.assertRaisesRegex(HeldoutManifestError, "duplicates state"):
            validate_manifest(data)

    def test_tampering_with_a_used_field_breaks_validation(self):
        data = manifest([entry()])
        tampered = copy.deepcopy(data)
        tampered["entries"][0]["position"][0] = 200.0

        with self.assertRaisesRegex(HeldoutManifestError, "does not match its state_id"):
            validate_manifest(tampered)


class RouteSummaryTests(unittest.TestCase):
    def test_rows_are_summarised_by_route_with_stale_attempts_visible(self):
        rows = [
            {"route": "a", "search_seed": 100, "state_id": "a1", "problem": None,
             "ending": "success", "length": 10},
            {"route": "a", "search_seed": 100, "state_id": "a2", "problem": None,
             "ending": "death", "length": 20},
            {"route": "b", "search_seed": 101, "state_id": "b1", "problem": "stale",
             "ending": None},
        ]

        result = route_summaries(rows)

        self.assertEqual(result["a"]["states"], 2)
        self.assertEqual(result["a"]["success_rate"], 0.5)
        self.assertEqual(result["a"]["endings"], {"success": 1, "death": 1})
        self.assertEqual(result["b"]["states"], 1)
        self.assertEqual(result["b"]["episodes"], 0)
        self.assertEqual(result["b"]["stale_starts"], 1)
        self.assertIsNone(result["b"]["success_rate"])


if __name__ == "__main__":
    unittest.main()
