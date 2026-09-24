"""Offline checks for the held-out route and state selection machinery."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from celeste_rl.demonstrations import DemonstrationManifestError, route_sha256
from celeste_rl.heldout import FORMAT_VERSION, HeldoutManifestError, freeze_entries, manifest_sha256
from celeste_rl.tasks import base_task_definition, load_task_definition, task_identity

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

import make_heldout_starts  # noqa: E402
from make_heldout_starts import (  # noqa: E402
    ATTEMPTS,
    attempted_seeds,
    candidates,
    FIRST_SEED,
    excluded_heldout_routes,
    existing_heldout_routes,
    reject_excluded_heldout_routes,
    require_minimum_routes,
    reject_demonstration_routes,
    search_command,
    select_states,
    unique_routes,
    unused_search_seeds,
)


def write_route(root: Path, name: str, seed: int, actions: list[str]) -> Path:
    path = root / name
    path.mkdir()
    (path / "route.json").write_text(json.dumps({
        "seed": seed,
        "transition_step": len(actions),
        "actions": actions,
    }), encoding="utf-8")
    return path


def write_room2_route(root: Path, name: str, seed: int, actions: list[str], include_identity: bool = False) -> Path:
    path = root / name
    path.mkdir()
    document = {
        "seed": seed,
        "start_room": "2",
        "next_room": "3",
        "task_definition": "config\\room2.json",
        "transition_step": len(actions),
        "actions": actions,
    }
    if include_identity:
        document["task"] = task_identity(load_task_definition("config/room2.json"))
    (path / "route.json").write_text(json.dumps(document), encoding="utf-8")
    return path


def action(buttons: str) -> str:
    return buttons


class RouteSelectionTests(unittest.TestCase):
    def test_restart_skips_every_seed_already_on_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_route(root, "first", 100, [action("R")] * 4)
            write_route(root, "duplicate-seed", 100, [action("L")] * 4)
            write_route(root, "third", 102, [action("U")] * 4)
            routes = existing_heldout_routes(root)

            self.assertEqual(unused_search_seeds(routes, 4), [101, 103, 104, 105])

    def test_dedicated_namespace_does_not_depend_on_seed_value(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            low_seed = write_route(root, "explicit-heldout", 2, [action("R")] * 4)

            self.assertEqual(existing_heldout_routes(root), [low_seed])

    def test_search_child_writes_to_the_declared_heldout_namespace(self):
        routes_dir = Path("C:/project/runs/heldout-routes")

        command = search_command(100, Path("C:/game"), 5.0, routes_dir)

        self.assertEqual(command[-2:], ["--output-root", str(routes_dir)])
        self.assertNotIn("runs/routes", " ".join(command).replace("\\", "/"))

    def test_search_child_receives_the_task_definition(self):
        command = search_command(100, Path("C:/game"), 5.0, Path("C:/routes"), Path("config/room2.json"))

        self.assertIn("--task-definition", command)
        self.assertEqual(command[command.index("--task-definition") + 1], "config\\room2.json")

    def test_identical_route_traces_are_kept_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = write_route(root, "100-first", 100, [action("R"), action("J"), action("R")])
            duplicate = write_route(root, "101-duplicate", 101, [action("R"), action("J"), action("R")])
            other = write_route(root, "102-other", 102, [action("L"), action("J"), action("R")])

            unique, duplicates = unique_routes([duplicate, other, first])

            self.assertEqual([path.name for path in unique], ["100-first", "102-other"])
            self.assertEqual([path.name for path in duplicates], ["101-duplicate"])

    def test_minimum_route_count_fails_closed(self):
        routes = [Path(f"route-{index}") for index in range(9)]

        with self.assertRaisesRegex(ValueError, "only 9 unique routes were produced; 10 required"):
            require_minimum_routes(routes, 10)

        require_minimum_routes([*routes, Path("route-9")], 10)

    def test_shared_route_prefixes_make_one_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared = [action("R"), action("R")]
            first = write_route(root, "first", 100, [*shared, action("J"), action("L")])
            second = write_route(root, "second", 101, [*shared, action("G"), action("R")])

            picked = candidates([first, second], earliest=1, spacing=1)
            keys = [tuple(candidate["lines"]) for candidate in picked]

            self.assertEqual(len(keys), len(set(keys)))
            self.assertEqual(sum(len(key) == 1 for key in keys), 1)
            self.assertEqual(sum(len(key) == 2 for key in keys), 1)
            self.assertTrue(all(len(candidate["route_sha256"]) == 64 for candidate in picked))

    def test_later_room_candidates_include_the_pinned_setup_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            route = write_room2_route(root, "room2", 100, ["R", "J", "R", "L"])
            definition = load_task_definition("config/room2.json")

            picked = candidates([route], earliest=1, spacing=1, definition=definition)

            self.assertTrue(picked)
            self.assertEqual(tuple(picked[0]["lines"][:definition.start.frames]), definition.start.lines)
            self.assertEqual(picked[0]["task_frames"], 1)
            self.assertEqual(picked[0]["frames"], definition.start.frames + 1)

    def test_later_room_route_with_another_task_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            route = write_room2_route(root, "room2", 100, ["R", "J"])
            document = json.loads((route / "route.json").read_text(encoding="utf-8"))
            document["task_definition"] = "config/other.json"
            (route / "route.json").write_text(json.dumps(document), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "task definition"):
                unique_routes([route], load_task_definition("config/room2.json"))

    def test_later_room_route_with_another_task_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            route = write_room2_route(root, "room2", 100, ["R", "J"], include_identity=True)
            document = json.loads((route / "route.json").read_text(encoding="utf-8"))
            document["task"]["task_definition_sha256"] = "0" * 64
            (route / "route.json").write_text(json.dumps(document), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "task identity"):
                unique_routes([route], load_task_definition("config/room2.json"))

    def test_demonstration_route_cannot_enter_the_evaluation_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            route = write_room2_route(root, "room2", 100, ["R", "J"], include_identity=True)
            definition = load_task_definition("config/room2.json")
            demonstrations = [{"route_sha256": route_sha256(["1,R", "1,J"])}]

            with self.assertRaisesRegex(DemonstrationManifestError, "overlap"):
                reject_demonstration_routes([route], demonstrations, definition)


class StateSelectionTests(unittest.TestCase):
    @staticmethod
    def state(name: str, frames: int, x: float, lines: list[str] | None = None) -> dict:
        recipe = lines or [f"{name}-{index}" for index in range(frames)]
        return {"route": name, "route_sha256": route_sha256(recipe),
                "search_seed": 100, "frames": len(recipe), "room": "1",
                "position": [x, 10.0], "dashes": 1, "lines": recipe}

    def test_duplicate_validated_states_are_kept_once(self):
        first = self.state("a", 20, 10.0, ["R"])
        duplicate = {**first, "route": "duplicate"}
        other = self.state("b", 30, 20.0, ["L"])

        selected = select_states([first, duplicate, other], 2)

        self.assertEqual(selected, [first, other])

    def test_exact_requested_count_is_selected_across_depths(self):
        states = [self.state("route", frame, float(frame)) for frame in range(10, 110, 10)]

        selected = select_states(states, 4)

        self.assertEqual(len(selected), 4)
        self.assertEqual([state["frames"] for state in selected], [10, 30, 60, 80])

    def test_too_few_unique_states_fails_closed(self):
        state = self.state("route", 20, 10.0)

        with self.assertRaisesRegex(ValueError, "only 1 unique valid states were produced; 2 requested"):
            select_states([state, dict(state)], 2)


class FirstSeedTests(unittest.TestCase):
    def test_the_default_first_seed_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_route(root, "old", 101, [action("R")] * 4)
            routes = existing_heldout_routes(root)

            self.assertEqual(FIRST_SEED, 100)
            self.assertEqual(unused_search_seeds(routes, 3), [100, 102, 103])
            self.assertEqual(unused_search_seeds(routes, 3), unused_search_seeds(routes, 3, FIRST_SEED))

    def test_a_fresh_namespace_starts_at_the_given_seed(self):
        self.assertEqual(unused_search_seeds([], 12, first_seed=112), list(range(112, 124)))

    def test_seeds_already_in_the_namespace_are_still_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_route(root, "a", 112, [action("R")] * 4)
            write_route(root, "b", 114, [action("L")] * 4)
            write_route(root, "below-first-seed", 100, [action("U")] * 4)
            routes = existing_heldout_routes(root)

            self.assertEqual(unused_search_seeds(routes, 3, first_seed=112), [113, 115, 116])


class SearchAttemptTests(unittest.TestCase):
    """A seed counts as tried as soon as its search starts (v2 procedure, amendment 2)."""

    def test_a_seed_whose_search_left_no_route_is_still_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_route(root, "found", 112, [action("R")] * 4)
            (root / ATTEMPTS).mkdir()
            for seed in (112, 113):
                (root / ATTEMPTS / f"seed-{seed}.json").write_text(json.dumps({"seed": seed}), encoding="utf-8")
            routes = existing_heldout_routes(root)

            self.assertEqual(routes, [root / "found"], "the attempt records are never read as a route")
            self.assertEqual(attempted_seeds(root), {112, 113})
            self.assertEqual(unused_search_seeds(routes, 3, 112, attempted_seeds(root)), [114, 115, 116])

    def _fake_run(self, root: Path, seed: int, creates_route: bool, returncode: int):
        record = root / ATTEMPTS / f"seed-{seed}.json"

        def run(command, **kwargs):
            self.assertTrue(record.exists(), "the attempt is recorded before the search starts")
            if creates_route:
                write_route(root, "20260924-120000", seed, [action("R")] * 4)
            return subprocess.CompletedProcess(command, returncode, stdout="searching\n",
                                               stderr="Steam not found\n" if returncode else "")
        return run

    def test_a_failed_search_is_recorded_with_its_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(make_heldout_starts.subprocess, "run", self._fake_run(root, 112, False, 1)):
                found = make_heldout_starts.search(112, Path("C:/game"), 1.0, root)

            self.assertIsNone(found)
            record = json.loads((root / ATTEMPTS / "seed-112.json").read_text(encoding="utf-8"))
            self.assertEqual((record["seed"], record["returncode"], record["route"]), (112, 1, None))
            self.assertIn("Steam not found", (root / ATTEMPTS / "seed-112.out").read_text(encoding="utf-8"))
            self.assertEqual(attempted_seeds(root), {112})

    def test_the_first_search_in_a_namespace_still_finds_its_route_folder(self):
        # The attempts folder is created in the same call; it must not be taken for the route folder.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(make_heldout_starts.subprocess, "run", self._fake_run(root, 116, True, 0)):
                found = make_heldout_starts.search(116, Path("C:/game"), 1.0, root)

            self.assertEqual(found, root / "20260924-120000")
            record = json.loads((root / ATTEMPTS / "seed-116.json").read_text(encoding="utf-8"))
            self.assertEqual((record["returncode"], record["route"]), (0, "20260924-120000"))


class ExcludeHeldoutTests(unittest.TestCase):
    """G1: a new set may not repeat a complete route of an earlier held-out set. These use the Room 1 base
    task, which needs no task file."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.definition = base_task_definition()
        self.identity = task_identity(self.definition)

    def write_manifest(self, name: str, route_lines: dict[str, list[str]], identity: dict | None = None) -> Path:
        """A valid held-out manifest with one state per route, as the generator would freeze it."""
        states = [{"route": route, "route_sha256": route_sha256(lines), "search_seed": 100 + index,
                   "frames": 2, "task_frames": 2, "room": self.definition.task.start_room,
                   "position": [10 + index, 20], "room_position": [10 + index, 20], "dashes": 1,
                   "lines": lines[:2]}
                  for index, (route, lines) in enumerate(sorted(route_lines.items()))]
        entries = freeze_entries(states)
        path = self.root / name
        path.write_text(json.dumps({
            "format_version": FORMAT_VERSION, "task": identity or self.identity,
            "sha256": manifest_sha256(entries), "states": len(entries),
            "routes": sorted(route_lines), "frames": {"min": 2, "max": 2}, "task_frames": {"min": 2, "max": 2},
            "entries": entries,
        }), encoding="utf-8")
        return path

    def test_a_new_route_that_repeats_an_excluded_route_is_refused(self):
        old = self.write_manifest("old.json", {"old-route": ["1,R", "1,J", "1,R"]})
        routes_dir = self.root / "routes"
        routes_dir.mkdir()
        repeat = write_route(routes_dir, "new-repeat", 112, [action("R"), action("J"), action("R")])
        fresh = write_route(routes_dir, "new-fresh", 113, [action("L"), action("J"), action("R")])

        excluded, _ = excluded_heldout_routes([old], self.identity)

        with self.assertRaisesRegex(ValueError, "repeat a route of an excluded held-out set: new-repeat$"):
            reject_excluded_heldout_routes([fresh, repeat], excluded, self.definition)

    def test_routes_absent_from_every_excluded_set_are_accepted(self):
        first = self.write_manifest("first.json", {"a": ["1,R", "1,J", "1,R"]})
        second = self.write_manifest("second.json", {"b": ["1,U", "1,J", "1,R"], "c": ["1,D", "1,J", "1,R"]})
        routes_dir = self.root / "routes"
        routes_dir.mkdir()
        fresh = write_route(routes_dir, "new-fresh", 112, [action("L"), action("J"), action("R")])

        excluded, provenance = excluded_heldout_routes([first, second], self.identity)

        reject_excluded_heldout_routes([fresh], excluded, self.definition)
        self.assertEqual(len(excluded), 3)
        self.assertEqual([record["routes"] for record in provenance], [1, 2])

    def test_no_excluded_manifest_changes_nothing(self):
        routes_dir = self.root / "routes"
        routes_dir.mkdir()
        route = write_route(routes_dir, "any", 100, [action("R"), action("J")])

        self.assertEqual(excluded_heldout_routes([], self.identity), ([], []))
        reject_excluded_heldout_routes([route], [], self.definition)

    def test_a_tampered_manifest_is_refused(self):
        path = self.write_manifest("old.json", {"old-route": ["1,R", "1,J", "1,R"]})
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["entries"][0]["route_sha256"] = "0" * 64
        path.write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertRaisesRegex(HeldoutManifestError, "state_id|sha256"):
            excluded_heldout_routes([path], self.identity)

    def test_a_manifest_for_another_task_is_refused(self):
        other = {**self.identity, "name": "another-task"}
        path = self.write_manifest("other.json", {"old-route": ["1,R", "1,J", "1,R"]}, identity=other)

        with self.assertRaises(HeldoutManifestError):
            excluded_heldout_routes([path], self.identity)

    def test_a_file_that_is_not_a_manifest_is_refused(self):
        path = self.root / "list.json"
        path.write_text("[]", encoding="utf-8")

        with self.assertRaisesRegex(HeldoutManifestError, "not a held-out manifest"):
            excluded_heldout_routes([path], self.identity)

    def test_the_frozen_room2_set_yields_its_twelve_route_hashes(self):
        """The file the v2 procedure excludes validates against the procedure's pinned task identity."""
        procedure = json.loads((REPO / "config" / "heldout-room2-v2-procedure.json").read_text(encoding="utf-8"))

        excluded, provenance = excluded_heldout_routes([REPO / "config" / "heldout_starts-room2.json"],
                                                       procedure["task"])

        self.assertEqual(len({entry["route_sha256"] for entry in excluded}), 12)
        self.assertEqual(provenance[0]["sha256"],
                         "ce57def1b3daed78a5aa3e454ab20cc12858a50b764bd0d94f80a1b2d734f8b0")


if __name__ == "__main__":
    unittest.main()
