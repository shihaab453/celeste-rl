"""obs-v1 encoding against replies recorded from the game (tests/fixtures/env_replies.json).

Run from the repo root:
    .venv-rl/Scripts/python.exe -W error -m unittest

The fixture is produced by scripts/record_env_fixture.py.
"""
from __future__ import annotations

import copy
import json
import math
import unittest
from pathlib import Path

import numpy as np

from celeste_rl.observation import (
    GeometryCache,
    ObservationBuilder,
    SchemaViolation,
    encode_grid,
    encode_player,
    observation_space,
)
from celeste_rl.schema import (
    ACTION_INPUTS,
    CELL_SIZE,
    FINGERPRINT,
    GRID_ANCHOR,
    GRID_CHANNELS,
    GRID_SIZE,
    HISTORY,
    PLAYER_FEATURE_NAMES,
    SPIKE_OFFSETS,
    SPINNER_BOX,
)

FIXTURE = json.loads((Path(__file__).resolve().parent / "fixtures" / "env_replies.json").read_text(encoding="utf-8"))
NAMES = {name: i for i, name in enumerate(PLAYER_FEATURE_NAMES)}
CH = {name: i for i, name in enumerate(GRID_CHANNELS)}


def sample(trace: str, reason: str) -> dict:
    return next(s for s in FIXTURE[trace]["samples"] if reason in s["reasons"])


def player_samples() -> list[dict]:
    return [s for trace in FIXTURE.values() for s in trace["samples"] if s["state"] is not None]


def reference_grid(state: dict, extras: dict) -> np.ndarray:
    """A slow, direct restatement of the grid rules, cell by cell, to check the vectorised encoder."""
    b = state["Level"]["Bounds"]
    collider = extras["player"]["Collider"]
    cx = state["Player"]["Position"]["X"] + collider["X"] + collider["W"] / 2
    cy = state["Player"]["Position"]["Y"] + collider["Y"] + collider["H"] / 2
    col0 = math.floor((cx - b["X"]) / CELL_SIZE) - GRID_ANCHOR
    row0 = math.floor((cy - b["Y"]) / CELL_SIZE) - GRID_ANCHOR
    tiles = state["SolidsData"].replace("\r", "").split("\n")

    rects: list[tuple[str, float, float, float, float]] = []
    for r in state.get("StaticSolids") or []:
        rects.append(("solid", r["X"], r["Y"], r["W"], r["H"]))
    directions = ("up", "down", "left", "right")
    for item in state.get("JumpThrus") or []:
        r = item["Bounds"]
        rects.append((f"jumpthru_{directions[item['Direction']]}", r["X"], r["Y"], r["W"], r["H"]))
    for item in state.get("Spikes") or []:
        r, d = item["Bounds"], directions[item["Direction"]]
        dx, dy = SPIKE_OFFSETS[d]
        rects.append((f"spikes_{d}", r["X"] + dx, r["Y"] + dy, r["W"], r["H"]))
    for r in state.get("Lightning") or []:
        rects.append(("other_hazards", r["X"], r["Y"], r["W"], r["H"]))
    for p in state.get("Spinners") or []:
        rects.append(("other_hazards", p["X"] - SPINNER_BOX / 2, p["Y"] - SPINNER_BOX / 2, SPINNER_BOX, SPINNER_BOX))

    grid = np.zeros((len(GRID_CHANNELS), GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            row, col = row0 + i, col0 + j
            x0, y0 = b["X"] + col * CELL_SIZE, b["Y"] + row * CELL_SIZE
            if 0 <= row < len(tiles) and 0 <= col < len(tiles[row]) and tiles[row][col] != "0":
                grid[CH["solid"], i, j] = 1
            for channel, x, y, w, h in rects:
                # Half-open overlap of [x, x + w) x [y, y + h) with the cell.
                if w > 0 and h > 0 and x < x0 + CELL_SIZE and x + w > x0 and y < y0 + CELL_SIZE and y + h > y0:
                    grid[CH[channel], i, j] = 1
            if row < 0 or col < 0 or (row + 1) * CELL_SIZE > b["H"] or (col + 1) * CELL_SIZE > b["W"]:
                grid[CH["outside_room"], i, j] = 1
    return grid


class SchemaTests(unittest.TestCase):
    def test_fingerprint_is_pinned(self):
        # A deliberate schema change must update this value and the version tags together.
        self.assertEqual(FINGERPRINT, "cb601cccc2c4252d")

    def test_space_and_names(self):
        space = observation_space()
        self.assertEqual(set(space.spaces), {"player", "actions", "history_valid", "grid", "context"})
        self.assertEqual(space["player"].shape, (HISTORY, len(PLAYER_FEATURE_NAMES)))
        self.assertEqual(space["actions"].shape, (HISTORY, len(ACTION_INPUTS)))
        self.assertEqual(space["grid"].shape, (len(GRID_CHANNELS), GRID_SIZE, GRID_SIZE))
        self.assertEqual(len(set(PLAYER_FEATURE_NAMES)), len(PLAYER_FEATURE_NAMES))
        # Nothing that identifies the room, the goal or the reward is a feature.
        for name in PLAYER_FEATURE_NAMES:
            for banned in ("room", "exit", "goal", "target", "reward", "potential", "checkpoint", "event"):
                self.assertNotIn(banned, name)


class PlayerFeatureTests(unittest.TestCase):
    def test_start_frame(self):
        s = sample("room1_exit_dash_route", "start")
        v = encode_player(s["state"], s["extras"])
        expected = {
            "position_x": 19 / 320, "position_y": 144 / 180, "on_ground": 1, "in_control": 1, "dashes": 0.5,
            "max_dashes": 0.5, "stamina": 1, "facing": 1, "collider_top": -1, "collider_height": 1,
            "hurtbox_height": 9 / 11, "state_StNormal": 1, "speed_x": 0, "paused": 0, "jump_buffer": 0,
        }
        for name, value in expected.items():
            self.assertAlmostEqual(float(v[NAMES[name]]), value, places=5, msg=name)
        self.assertEqual(v[NAMES["state_StNormal"]:].sum(), 1)

    def test_recorded_mechanics(self):
        dash = encode_player(**{k: sample("room1_exit_dash_route", "dash")[k] for k in ("state", "extras")})
        self.assertEqual(dash[NAMES["state_StDash"]], 1)
        ducking = sample("room1_spike_death_route", "ducking")
        v = encode_player(ducking["state"], ducking["extras"])
        self.assertEqual(v[NAMES["ducking"]], 1)
        self.assertAlmostEqual(float(v[NAMES["collider_height"]]), ducking["extras"]["player"]["Collider"]["H"] / 11, places=5)
        self.assertLess(v[NAMES["collider_height"]], 1)
        buffered = sample("room1_exit_dash_route", "jump_buffered")
        v = encode_player(buffered["state"], buffered["extras"])
        self.assertGreater(v[NAMES["jump_buffer"]], 0)
        self.assertLessEqual(v[NAMES["jump_buffer"]], 1)
        paused = sample("room1_pause_levelexit_loading_route", "paused")
        self.assertEqual(encode_player(paused["state"], paused["extras"])[NAMES["paused"]], 1)

    def test_values_are_not_clipped(self):
        room2 = sample("room1_exit_dash_route", "room_2")
        v = encode_player(room2["state"], room2["extras"])
        self.assertGreater(v[NAMES["position_y"]], 1, "room 2's first frame is below its bounds")
        s = copy.deepcopy(sample("room1_exit_dash_route", "start"))
        s["state"]["Player"]["Speed"]["X"] = 1000
        s["extras"]["input_buffers"]["Jump"] = -0.5
        v = encode_player(s["state"], s["extras"])
        self.assertAlmostEqual(float(v[NAMES["speed_x"]]), 2.5)
        self.assertEqual(v[NAMES["jump_buffer"]], 0)

    def test_every_recorded_frame_encodes_finitely(self):
        for s in player_samples():
            v = encode_player(s["state"], s["extras"])
            self.assertTrue(np.all(np.isfinite(v)))

    def test_schema_violations(self):
        base = sample("room1_exit_dash_route", "start")
        cases = {
            "missing extras field": lambda s: s["extras"]["player"].pop("Stamina"),
            "not finite": lambda s: s["state"]["Player"]["Speed"].__setitem__("X", float("nan")),
            "string number": lambda s: s["extras"]["player"].__setitem__("Dashes", "1"),
            "bool as number": lambda s: s["extras"]["player"].__setitem__("Stamina", True),
            "number as flag": lambda s: s["extras"]["player"].__setitem__("Ducking", 1),
            "state index": lambda s: s["extras"]["player"].__setitem__("State", 2.0),
        }
        for name, corrupt in cases.items():
            s = copy.deepcopy(base)
            corrupt(s)
            with self.subTest(name), self.assertRaises(SchemaViolation):
                encode_player(s["state"], s["extras"])

    def test_unknown_state_uses_other(self):
        s = copy.deepcopy(sample("room1_exit_dash_route", "start"))
        s["extras"]["player"]["State"] = 40
        v = encode_player(s["state"], s["extras"])
        self.assertEqual(v[NAMES["state_other"]], 1)
        self.assertEqual(v[NAMES["state_StNormal"]:].sum(), 1)


class GridTests(unittest.TestCase):
    def test_matches_cell_by_cell_reference_on_every_recorded_frame(self):
        cache = GeometryCache()
        for s in player_samples():
            with self.subTest(frame=s["frame"], reasons=s["reasons"]):
                np.testing.assert_array_equal(encode_grid(s["state"], s["extras"], cache), reference_grid(s["state"], s["extras"]))

    def test_start_frame_landmarks(self):
        s = sample("room1_exit_dash_route", "start")
        g = encode_grid(s["state"], s["extras"], GeometryCache())
        # Player collider centre (19, 138.5) is room cell (row 17, col 2), at crop (16, 16): crop column = room col + 14,
        # crop row = room row - 1.
        self.assertEqual(g[CH["solid"], 16, 16], 0)
        self.assertEqual(g[CH["solid"], 16, 14], 1, "left wall, room col 0")
        self.assertTrue(np.all(g[CH["solid"], 16, 24:27] == 1), "platform, room cols 10-12")
        self.assertTrue(np.all(g[CH["spikes_up"], 19, 19:24] == 1), "floor spikes x 40-80, hitbox y 165-168, in room row 20")
        self.assertEqual(g[CH["spikes_up"], 20, 19], 0, "not in the floor tiles below, room row 21")
        self.assertTrue(np.all(g[CH["outside_room"], :, :14] == 1), "x < 0")
        self.assertEqual(g[CH["outside_room"], 20, 14], 0)
        self.assertTrue(np.all(g[CH["outside_room"], 21:, :] == 1), "room row 22 spans y 176-184, past the 180 px bounds")

    def test_translated_room_bounds(self):
        s = sample("room1_exit_dash_route", "room_2")
        self.assertEqual(s["state"]["Level"]["Bounds"]["Y"], -184)
        np.testing.assert_array_equal(encode_grid(s["state"], s["extras"], GeometryCache()),
                                      reference_grid(s["state"], s["extras"]))

    def test_synthetic_entities_and_negative_coordinates(self):
        s = copy.deepcopy(sample("room1_exit_dash_route", "start"))
        s["state"]["StaticSolids"] = [{"X": 30.5, "Y": 120, "W": 1, "H": 1}]
        s["state"]["JumpThrus"] = [{"Bounds": {"X": -20, "Y": 100, "W": 24, "H": 5}, "Direction": 0, "PullsPlayer": True}]
        s["state"]["Spikes"] += [{"Bounds": {"X": 0, "Y": 40, "W": 3, "H": 16}, "Direction": 3},
                                 {"Bounds": {"X": 8, "Y": 8, "W": 16, "H": 3}, "Direction": 1}]
        s["state"]["Lightning"] = [{"X": 16, "Y": 16, "W": 0, "H": 8}]
        s["state"]["Spinners"] = [{"X": 40, "Y": 100}]
        cache = GeometryCache()
        g = encode_grid(s["state"], s["extras"], cache)
        np.testing.assert_array_equal(g, reference_grid(s["state"], s["extras"]))
        self.assertEqual(g[CH["solid"], 15 - 1, 3 + 14], 1, "a 1 px solid at (30.5, 120) marks room cell (15, 3)")
        self.assertEqual(g[CH["other_hazards"]].sum(), 6, "spinner box x 32-48 (2 columns, half-open) by y 92-108 (3 rows); zero-width lightning none")

    def test_player_centre_outside_the_room(self):
        # Leaving through the top edge puts the collider centre above the room, where flooring and truncating
        # toward zero pick different cells.
        cache = GeometryCache()
        for x, y in ((19, 2), (19, -20), (-3, 144), (330, 190), (261, 1)):
            s = copy.deepcopy(sample("room1_exit_dash_route", "start"))
            s["state"]["Player"]["Position"] = {"X": x, "Y": y}
            with self.subTest(x=x, y=y):
                np.testing.assert_array_equal(encode_grid(s["state"], s["extras"], cache), reference_grid(s["state"], s["extras"]))

    def test_cache_follows_tiles_and_bounds(self):
        s = copy.deepcopy(sample("room1_exit_dash_route", "start"))
        cache = GeometryCache()
        before = encode_grid(s["state"], s["extras"], cache)
        rows = s["state"]["SolidsData"].split("\n")
        rows[17] = "0" + rows[17][1:]
        s["state"]["SolidsData"] = "\n".join(rows)
        after = encode_grid(s["state"], s["extras"], cache)
        self.assertEqual(before[CH["solid"], 16, 14], 1)
        self.assertEqual(after[CH["solid"], 16, 14], 0)

    def test_bad_entities_are_schema_violations(self):
        for corrupt in (lambda st: st["Spikes"][0].__setitem__("Direction", 7),
                        lambda st: st["Spikes"][0].__setitem__("Bounds", None),
                        lambda st: st.__setitem__("SolidsData", None)):
            s = copy.deepcopy(sample("room1_exit_dash_route", "start"))
            corrupt(s["state"])
            with self.assertRaises(SchemaViolation):
                encode_grid(s["state"], s["extras"], GeometryCache())


class BuilderTests(unittest.TestCase):
    def setUp(self):
        self.space = observation_space()
        self.start = sample("room1_exit_dash_route", "start")

    def test_reset_history_and_no_player(self):
        builder = ObservationBuilder()
        obs = builder.reset(self.start["state"], self.start["extras"])
        self.assertTrue(self.space.contains(obs))
        np.testing.assert_array_equal(obs["history_valid"], [1, 0, 0, 0])
        self.assertFalse(obs["actions"].any())
        self.assertFalse(obs["player"][1:].any())
        np.testing.assert_array_equal(obs["context"], [0, 1])

        actions = []
        for step in range(1, 6):
            action = np.zeros(len(ACTION_INPUTS), dtype=np.int8)
            action[step] = 1
            actions.append(action)
            obs = builder.step(self.start["state"], self.start["extras"], action, step)
            self.assertTrue(self.space.contains(obs))
        np.testing.assert_array_equal(obs["history_valid"], [1, 1, 1, 1])
        for row in range(HISTORY):
            np.testing.assert_array_equal(obs["actions"][row], actions[-1 - row])
        self.assertAlmostEqual(float(obs["context"][0]), 5 / 1800)

        death = obs_after = builder.step(None, None, actions[0], 6)
        self.assertTrue(self.space.contains(death))
        self.assertFalse(obs_after["player"][0].any())
        self.assertFalse(obs_after["grid"].any())
        self.assertEqual(obs_after["context"][1], 0)
        np.testing.assert_array_equal(obs_after["player"][1], obs["player"][0])

        obs = builder.reset(self.start["state"], self.start["extras"])
        np.testing.assert_array_equal(obs["history_valid"], [1, 0, 0, 0])
        self.assertFalse(obs["actions"].any())

    def test_violation_leaves_history_untouched(self):
        builder = ObservationBuilder()
        builder.reset(self.start["state"], self.start["extras"])
        before = builder.observation()
        with self.assertRaises(SchemaViolation):
            builder.step(self.start["state"], None, np.zeros(len(ACTION_INPUTS), dtype=np.int8), 1)
        after = builder.observation()
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])

    def test_returned_arrays_are_not_shared(self):
        builder = ObservationBuilder()
        first = builder.reset(self.start["state"], self.start["extras"])
        second = builder.step(self.start["state"], self.start["extras"], np.ones(len(ACTION_INPUTS), dtype=np.int8), 1)
        for key in first:
            self.assertFalse(np.shares_memory(first[key], second[key]), key)
        first["grid"][:] = 1
        self.assertFalse(np.array_equal(builder.observation()["grid"], first["grid"]))


if __name__ == "__main__":
    unittest.main()
