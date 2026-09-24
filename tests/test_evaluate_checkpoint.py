"""Checkpoint evaluation: default play order, demonstration starts built in memory, and fail-closed starts."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from celeste_rl.actions import apply_disabled, disabled_mask, parse_line
from celeste_rl.demonstrations import route_sha256
from celeste_rl.endings import RoomTask
from celeste_rl.schema import MENU_INPUTS, PLAYER_FEATURE_COUNT, PLAYER_FEATURE_NAMES
from celeste_rl.starts import Start
from celeste_rl.tasks import TaskDefinition
from scripts.evaluate_checkpoint import (
    CANONICAL_ID,
    StartBuildError,
    demonstration_starts,
    episode_plan,
    play_all,
    play_from_start,
    refuse_heldout_paths,
    require_sha256,
    start_list_record,
)

FEATURE = {name: index for index, name in enumerate(PLAYER_FEATURE_NAMES)}
BOUNDS = (240.0, -184.0, 320.0, 180.0)
SETUP = ("1,R", "1,R")
DEFINITION = TaskDefinition("room-2", RoomTask("2", "3"), Start(SETUP, (261, 1), "2", 0))
MASK = disabled_mask(MENU_INPUTS)


def demo(name: str, lines: list[str]) -> dict:
    return {"kind": "route", "name": name, "source": f"runs/{name}/route.json", "route_sha256": route_sha256(lines),
            "lines": lines}


def dataset(demos_in_trajectory_order: list[dict], dashes_at: int = 3):
    """Fake cloning records: row k of a trajectory is the state after k actions; x = 261 + k, y = 1 - k."""
    trajectory, actions, player = [], [], []
    for index, item in enumerate(demos_in_trajectory_order):
        for k, line in enumerate(item["lines"]):
            trajectory.append(index)
            actions.append(apply_disabled(parse_line(line), MASK))
            row = np.zeros(PLAYER_FEATURE_COUNT, dtype=np.float32)
            row[FEATURE["position_x"]] = (261 + k - BOUNDS[0]) / BOUNDS[2]
            row[FEATURE["position_y"]] = (1 - k - BOUNDS[1]) / BOUNDS[3]
            row[FEATURE["dashes"]] = (1 if k >= dashes_at else 0) / 2
            player.append(row)
    return np.asarray(trajectory), np.stack(actions), np.stack(player)


ROUTE_A = demo("route-a", ["1,R"] * 6 + ["1,R,J"] * 6)   # 12 frames
ROUTE_B = demo("route-b", ["1,L"] * 5 + ["1,X"] * 5)     # 10 frames


class DemonstrationStartTests(unittest.TestCase):
    def test_starts_follow_the_rule_and_the_evidence_not_the_manifest_order(self):
        # The dataset records route B first, the manifest lists route A first.
        trajectory, actions, player = dataset([ROUTE_B, ROUTE_A])

        starts = demonstration_starts([ROUTE_A, ROUTE_B], trajectory, actions, player, DEFINITION, BOUNDS, 4)

        self.assertEqual([s["start_id"] for s in starts], ["route-b@4", "route-a@4", "route-a@8"])
        by_id = {s["start_id"]: s for s in starts}
        self.assertEqual(by_id["route-b@4"]["dataset_trajectory"], 0)
        self.assertEqual(by_id["route-a@8"]["position"], [269, -7])
        self.assertEqual(by_id["route-a@8"]["dashes"], 1)
        self.assertEqual(by_id["route-a@4"]["dashes"], 1)
        self.assertEqual(by_id["route-b@4"]["lines"], list(SETUP) + ["1,L"] * 4)
        self.assertEqual(by_id["route-a@8"]["room"], "2")

    def test_a_trajectory_that_matches_no_route_is_refused(self):
        altered = demo("route-a", ROUTE_A["lines"][:-1] + ["1,L"])
        trajectory, actions, player = dataset([altered, ROUTE_B])

        with self.assertRaisesRegex(StartBuildError, "matches 0 demonstrations"):
            demonstration_starts([ROUTE_A, ROUTE_B], trajectory, actions, player, DEFINITION, BOUNDS, 4)

    def test_a_trajectory_that_matches_two_routes_is_refused(self):
        twin = {**ROUTE_A, "name": "twin", "route_sha256": "f" * 64}
        trajectory, actions, player = dataset([ROUTE_A, ROUTE_B])

        with self.assertRaisesRegex(StartBuildError, "matches 2 demonstrations"):
            demonstration_starts([ROUTE_A, twin, ROUTE_B], trajectory, actions, player, DEFINITION, BOUNDS, 4)

    def test_a_route_with_no_trajectory_is_refused(self):
        trajectory, actions, player = dataset([ROUTE_A])

        with self.assertRaisesRegex(StartBuildError, "no dataset trajectory"):
            demonstration_starts([ROUTE_A, ROUTE_B], trajectory, actions, player, DEFINITION, BOUNDS, 4)

    def test_wrong_room_size_is_refused_by_the_exact_round_trip(self):
        trajectory, actions, player = dataset([ROUTE_A, ROUTE_B])

        with self.assertRaisesRegex(StartBuildError, "position y does not reconstruct exactly"):
            demonstration_starts([ROUTE_A, ROUTE_B], trajectory, actions, player, DEFINITION,
                                 (240.0, -184.0, 320.0, 184.0), 4)

    def test_wrong_room_offset_is_refused_by_the_task_start_anchor(self):
        trajectory, actions, player = dataset([ROUTE_A, ROUTE_B])

        with self.assertRaisesRegex(StartBuildError, "does not begin at the task start"):
            demonstration_starts([ROUTE_A, ROUTE_B], trajectory, actions, player, DEFINITION,
                                 (241.0, -184.0, 320.0, 180.0), 4)

    def test_start_list_hash_covers_the_listed_fields(self):
        trajectory, actions, player = dataset([ROUTE_A, ROUTE_B])
        starts = demonstration_starts([ROUTE_A, ROUTE_B], trajectory, actions, player, DEFINITION, BOUNDS, 4)

        record = start_list_record(starts)
        changed = [dict(s) for s in starts]
        changed[0]["position"] = [0, 0]

        self.assertNotIn("lines", record["starts"][0])
        self.assertNotEqual(start_list_record(changed)["sha256"], record["sha256"])


class InputRefusalTests(unittest.TestCase):
    def test_a_wrong_file_hash_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "dataset.npz"
            path.write_bytes(b"data")
            with self.assertRaisesRegex(StartBuildError, "expected"):
                require_sha256(path, "0" * 64, "dataset")

    def test_held_out_inputs_are_refused(self):
        for path in ("config/heldout_starts-room2.json", "runs/heldout-routes-x/a/route.json"):
            with self.subTest(path=path), self.assertRaisesRegex(StartBuildError, "held-out"):
                refuse_heldout_paths([Path("config/demonstrations-room2.json"), Path(path)])
        refuse_heldout_paths([Path("config/demonstrations-room2.json"), None])


class FakeModel:
    def predict(self, obs, deterministic=False):
        return np.zeros(24, dtype=np.int8), None


class FakeEnv:
    """Replays a scripted list of player x positions and ends with `ending`."""

    def __init__(self, xs, ending="success", start_kind="archive", problem=None):
        self.xs, self.ending, self.start_kind, self.problem = xs, ending, start_kind, problem
        self.resets = []

    def _info(self, x, **extra):
        return {"player": {"x": x, "y": -10}, "potential": 0.5, **extra}

    def reset(self, options=None):
        self.resets.append(options)
        self.step_index = 0
        return {}, self._info(self.xs[0], start=self.start_kind, start_problem=self.problem)

    def step(self, action):
        self.step_index += 1
        done = self.step_index == len(self.xs) - 1
        return {}, 0.0, done, False, self._info(self.xs[self.step_index], ending=self.ending if done else None)


def _canonical(model, env, deterministic):
    return {"ending": "timeout", "length": 1800, "max_x": 320, "end_x": 322, "end_y": -30}


class PlayTests(unittest.TestCase):
    START = {"start_id": "route-a@4", "route": "route-a", "route_sha256": "a" * 64, "prefix_frames": 4,
             "position": [265, -3], "dashes": 1, "room": "2", "lines": ["1,R"] * 6}

    def test_default_plan_is_stochastic_canonical_episodes_then_one_deterministic(self):
        plan = episode_plan(3, 1, [], 5)

        self.assertEqual([(p["start_id"], p["mode"], p["repeat"]) for p in plan],
                         [(CANONICAL_ID, "stochastic", 0), (CANONICAL_ID, "stochastic", 1),
                          (CANONICAL_ID, "stochastic", 2), (CANONICAL_ID, "deterministic", 0)])

    def test_default_run_calls_the_canonical_episode_in_order_and_leaves_its_records_alone(self):
        calls = []

        def canonical(model, env, deterministic):
            calls.append(deterministic)
            return {"ending": "success", "length": 5, "return": 1.0, "max_x": 300, "end_x": 290, "end_y": -20}

        rows, aborted = play_all(FakeModel(), None, episode_plan(2, 1, [], 5), lambda row: None,
                                 canonical_episode=canonical)

        self.assertIsNone(aborted)
        self.assertEqual(calls, [False, False, True])
        self.assertNotIn("problem", rows[-1]["episode"])
        self.assertEqual(rows[-1]["episode"], {"ending": "success", "length": 5, "return": 1.0, "max_x": 300,
                                               "end_x": 290, "end_y": -20})

    def test_rows_carry_the_longest_stretch_without_a_new_best(self):
        def canonical(model, env, deterministic):
            return {"ending": "success", "length": 400, "max_x": 540, "end_x": 535, "end_y": -179,
                    "longest_without_new_best": 131}

        rows, _ = play_all(FakeModel(), None, episode_plan(1, 1, [], 5), lambda row: None,
                           canonical_episode=canonical)

        self.assertEqual([row["longest_without_new_best"] for row in rows], [131, 131])

    def test_starts_get_all_stochastic_then_all_deterministic_repeats(self):
        plan = episode_plan(0, 2, [self.START], 3)

        self.assertEqual([p["mode"] for p in plan], ["deterministic", "deterministic", "stochastic", "stochastic",
                                                      "stochastic", "deterministic", "deterministic"])

    def test_rows_carry_true_max_x_end_position_and_start_provenance(self):
        written = []
        env = FakeEnv([265, 300, 350, 330])

        rows, aborted = play_all(FakeModel(), env, episode_plan(0, 1, [self.START], 1), written.append,
                                 canonical_episode=_canonical)

        self.assertIsNone(aborted)
        self.assertEqual(written[0]["start_id"], CANONICAL_ID)
        start_rows = [row for row in written if row["start_id"] == "route-a@4"]
        self.assertEqual(len(start_rows), 2)
        row = start_rows[0]
        self.assertEqual((row["ending"], row["length"], row["max_x"], row["end_x"], row["end_y"]),
                         ("success", 3, 350, 330, -10))
        self.assertEqual((row["route"], row["prefix_frames"], row["start_position"], row["start_dashes"]),
                         ("route-a", 4, [265, -3], 1))
        self.assertIsNone(row["problem"])
        self.assertEqual(env.resets[0]["start"].position, (265, -3))

    def test_a_start_that_falls_back_to_the_task_start_is_a_failure_and_stops_the_run(self):
        env = FakeEnv([261, 270], start_kind="canonical", problem="the prefix arrived at (1, 1), not (265, -3)")
        second = {**self.START, "start_id": "route-a@8"}

        rows, aborted = play_all(FakeModel(), env, episode_plan(0, 1, [self.START, second], 2), lambda row: None,
                                 canonical_episode=_canonical)

        self.assertEqual([row["start_id"] for row in rows], [CANONICAL_ID, "route-a@4"])
        self.assertIsNone(rows[-1]["ending"])
        self.assertIn("did not replay", aborted)
        self.assertIn("arrived at", rows[-1]["problem"])

    def test_a_fallback_without_a_reported_problem_is_still_a_failure(self):
        env = FakeEnv([261, 270], start_kind="canonical", problem=None)

        episode = play_from_start(FakeModel(), env, Start(("1,R",), (265, -3), "2", 1), deterministic=True)

        self.assertIsNone(episode["ending"])
        self.assertIn("canonical", episode["problem"])


if __name__ == "__main__":
    unittest.main()
