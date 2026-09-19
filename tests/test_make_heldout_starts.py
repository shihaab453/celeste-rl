"""Offline checks for the held-out route and state selection machinery."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from celeste_rl.demonstrations import route_sha256

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from make_heldout_starts import (  # noqa: E402
    candidates,
    existing_heldout_routes,
    require_minimum_routes,
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


if __name__ == "__main__":
    unittest.main()
