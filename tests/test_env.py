"""CelesteRoomEnv, ending rules and reward, with a fake bridge replaying recorded game replies.

Run from the repo root:
    .venv-rl/Scripts/python.exe -W error -m unittest

The recorded event sequences come from tests/fixtures/env_replies.json (scripts/record_env_fixture.py).
"""
from __future__ import annotations

import copy
import json
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from gymnasium.utils.env_checker import check_env

from celeste_rl.actions import INDEX
from celeste_rl.bridge import BridgeError, Observation
from celeste_rl.endings import (
    DEATH,
    LEFT_LEVEL,
    RESTART,
    SUCCESS,
    TIMEOUT,
    WRONG_ROOM,
    EndingFault,
    RoomTask,
    classify,
)
from celeste_rl.env import BridgeFault, CelesteRoomEnv
from celeste_rl.reward import RewardConfig, reward_components
from celeste_rl.schema import ACTION_INPUTS, DEADLINE_FRAMES

FIXTURE = json.loads((Path(__file__).resolve().parent / "fixtures" / "env_replies.json").read_text(encoding="utf-8"))
START = next(s for s in FIXTURE["room1_exit_dash_route"]["samples"] if "start" in s["reasons"])
TIME = -1 / 60_000
TASK = RoomTask()
PLAYER = START["state"]


class ReplayBridge:
    """Replays a recorded trace's events and player presence, whatever the actions. After the recording ends,
    every step has a player and no events. Records what the environment sent."""

    http = SimpleNamespace(start_frame=300)

    def __init__(self, trace: str | None = None, fail_at: int | None = None, drop_extras_at: int | None = None,
                 start_state: dict | None = None):
        self.steps = FIXTURE[trace]["steps"] if trace else []
        self.fail_at, self.drop_extras_at = fail_at, drop_extras_at
        self.start_state = start_state or START["state"]
        self.sent: list[tuple[str, str, str]] = []
        self.index = 0
        self.closed = False

    def reset(self):
        self.index = 0
        return Observation(1, 0, 300, "1", copy.deepcopy(self.start_state), None, copy.deepcopy(START["extras"]), [])

    def step(self, buttons="", dash_only="", move_only=""):
        self.index += 1
        self.sent.append((buttons, dash_only, move_only))
        if self.index == self.fail_at:
            raise BridgeError("Lockstep session ended: simulated timeout")
        record = self.steps[self.index] if self.index < len(self.steps) else {"player": True, "events": []}
        state = copy.deepcopy(START["state"]) if record["player"] else None
        extras = copy.deepcopy(START["extras"]) if record["player"] and self.index != self.drop_extras_at else None
        return Observation(1, self.index, 300 + self.index, "1", state, None, extras, record["events"])

    def close(self):
        self.closed = True


def noop() -> np.ndarray:
    return np.zeros(len(ACTION_INPUTS), dtype=np.int8)


def run(env: CelesteRoomEnv, limit: int = 5000):
    env.reset()
    for step in range(1, limit + 1):
        _, reward, terminated, truncated, info = env.step(noop())
        assert not truncated
        if terminated:
            return step, reward, info
    raise AssertionError("episode did not end")


class EndingRuleTests(unittest.TestCase):
    def test_precedence(self):
        transition = {"type": "transition", "from": "1", "to": "2", "direction": {"X": 0, "Y": -1}}
        cases = [
            ([], PLAYER, 5, None),
            ([{"type": "pause", "quick_reset": False}, {"type": "unpause"}], PLAYER, 5, None),
            ([transition], PLAYER, 5, SUCCESS),
            ([transition, {"type": "load_level", "room": "2", "intro": "Transition", "from_loader": False}], PLAYER, 5, SUCCESS),
            ([{"type": "death", "room": "1"}], PLAYER, 5, DEATH),
            ([{"type": "death", "room": "1"}], None, 5, DEATH),
            ([transition, {"type": "death", "room": "1"}], PLAYER, 5, DEATH),
            ([{"type": "death", "room": "1"}], PLAYER, DEADLINE_FRAMES, DEATH),
            ([transition], PLAYER, DEADLINE_FRAMES, SUCCESS),
            ([{"type": "level_exit", "mode": "Restart"},
              {"type": "load_level", "room": "1", "intro": "Jump", "from_loader": True}], PLAYER, 5, RESTART),
            ([transition, {"type": "load_level", "room": "1", "intro": "Respawn", "from_loader": False}], PLAYER, 5, RESTART),
            ([{"type": "level_exit", "mode": "SaveAndQuit"}], None, 5, LEFT_LEVEL),
            ([{**transition, "to": "3"}], PLAYER, 5, WRONG_ROOM),
            ([{**transition, "from": "2"}], PLAYER, 5, WRONG_ROOM),
            ([], PLAYER, DEADLINE_FRAMES, TIMEOUT),
            ([], PLAYER, DEADLINE_FRAMES - 1, None),
        ]
        for events, state, elapsed, expected in cases:
            with self.subTest(events=[e["type"] for e in events], player=state is not None, elapsed=elapsed):
                self.assertEqual(classify(events, state, elapsed, TASK), expected)

    def test_unexplained_replies_are_faults(self):
        cases = [
            (None, PLAYER),                                     # extras and events disabled
            ([], None),                                         # no player, no reason
            ([{"type": "pause", "quick_reset": False}], None),  # a pause does not remove the player
            ([{"type": "teleport"}], PLAYER),                   # unknown event
            (["death"], PLAYER),                                # malformed event
            ([{"type": "load_level", "room": "1"}], PLAYER),    # no intro
            ([{"type": "transition", "to": "2"}], PLAYER),      # no source room
        ]
        for events, state in cases:
            with self.subTest(events=events, player=state is not None), self.assertRaises(EndingFault):
                classify(events, state, 5, TASK)


class RewardTests(unittest.TestCase):
    def test_components(self):
        config = RewardConfig()
        self.assertEqual(reward_components(None, config), {"completion": 0.0, "failure": 0.0, "time": TIME, "shaping": 0.0})
        self.assertEqual(sum(reward_components(SUCCESS, config).values()), 1 + TIME)
        for ending in (DEATH, RESTART, LEFT_LEVEL, WRONG_ROOM, TIMEOUT):
            self.assertEqual(sum(reward_components(ending, config).values()), -1 + TIME)
        with self.assertRaises(ValueError):
            reward_components("won", config)


class RecordedEpisodeTests(unittest.TestCase):
    """The recorded game event sequences end where the spec says, with the right reward."""

    def check(self, trace: str, expected_step: int, expected_ending: str):
        env = CelesteRoomEnv(ReplayBridge(trace), disabled_inputs=())
        step, reward, info = run(env)
        self.assertEqual((step, info["ending"]), (expected_step, expected_ending))
        self.assertAlmostEqual(reward, (1 if expected_ending == SUCCESS else -1) + TIME)
        self.assertEqual(info["elapsed"], expected_step)
        return env

    def test_exit_route_succeeds_on_the_transition_step(self):
        self.check("room1_exit_dash_route", 285, SUCCESS)

    def test_death_route_ends_on_the_death_event(self):
        self.check("room1_spike_death_route", 74, DEATH)

    def test_pause_menu_restart_hidden_by_loading_is_a_restart(self):
        self.check("room1_pause_levelexit_loading_route", 139, RESTART)

    def test_random_trace_first_death(self):
        self.check("random_seed0", 58, DEATH)

    def test_timeout_at_the_deadline(self):
        env = CelesteRoomEnv(ReplayBridge())
        env.reset()
        total = 0.0
        for step in range(1, DEADLINE_FRAMES + 1):
            _, reward, terminated, _, info = env.step(noop())
            total += reward
            self.assertEqual(terminated, step == DEADLINE_FRAMES)
        self.assertEqual(info["ending"], TIMEOUT)
        self.assertAlmostEqual(total, -1 + DEADLINE_FRAMES * TIME)


class EnvironmentContractTests(unittest.TestCase):
    def test_gymnasium_env_checker(self):
        env = CelesteRoomEnv(ReplayBridge("room1_exit_dash_route"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            check_env(env, skip_render_check=True)
        # The player features are deliberately unbounded (never clipped, spec 5.1); any other warning fails.
        expected = ("minimum value is -infinity", "maximum value is infinity")
        unexpected = [str(w.message) for w in caught if not any(text in str(w.message) for text in expected)]
        self.assertEqual(unexpected, [])
        self.assertEqual(len(caught), 2)

    def test_step_after_ending_needs_reset(self):
        env = CelesteRoomEnv(ReplayBridge("room1_spike_death_route"))
        run(env)
        with self.assertRaisesRegex(RuntimeError, "call reset"):
            env.step(noop())
        env.reset()
        env.step(noop())

    def test_bridge_error_is_a_fault_with_no_transition(self):
        bridge = ReplayBridge("room1_exit_dash_route", fail_at=10)
        env = CelesteRoomEnv(bridge)
        env.reset()
        for _ in range(9):
            env.step(noop())
        with self.assertRaisesRegex(BridgeFault, "simulated timeout"):
            env.step(noop())
        with self.assertRaisesRegex(RuntimeError, "call reset"):
            env.step(noop())
        self.assertEqual(len(bridge.sent), 10, "nothing is sent or retried after a fault")
        env.reset()
        self.assertEqual(env.step(noop())[4]["elapsed"], 1)

    def test_uninterpretable_replies_are_faults(self):
        env = CelesteRoomEnv(ReplayBridge("room1_exit_dash_route", drop_extras_at=3))
        env.reset()
        env.step(noop())
        env.step(noop())
        with self.assertRaisesRegex(BridgeFault, "could not be interpreted"):
            env.step(noop())
        with self.assertRaisesRegex(RuntimeError, "call reset"):
            env.step(noop())

    def test_invalid_action_is_rejected_before_sending(self):
        bridge = ReplayBridge()
        env = CelesteRoomEnv(bridge)
        env.reset()
        with self.assertRaises(ValueError):
            env.step(np.full(len(ACTION_INPUTS), 2))
        self.assertEqual(bridge.sent, [])
        env.step(noop())

    def test_reset_rejects_a_non_canonical_start(self):
        moved = copy.deepcopy(START["state"])
        moved["Player"]["Position"]["X"] = 40
        env = CelesteRoomEnv(ReplayBridge(start_state=moved))
        with self.assertRaisesRegex(BridgeFault, "position"):
            env.reset()
        with self.assertRaisesRegex(RuntimeError, "call reset"):
            env.step(noop())

    def test_disabled_inputs_and_applied_action(self):
        everything = np.ones(len(ACTION_INPUTS), dtype=np.int8)
        bridge = ReplayBridge()
        env = CelesteRoomEnv(bridge)  # default: pause, quick restart and journal disabled
        env.reset()
        obs, _, _, _, info = env.step(everything)
        self.assertEqual(bridge.sent[-1], ("LRUDJKXCZVGHO", "LRUD", "LRUD"))
        for name in ("S", "Q", "N"):
            self.assertEqual(info["applied_action"][INDEX[name]], 0)
        np.testing.assert_array_equal(obs["actions"][0], info["applied_action"])
        self.assertEqual(info["disabled_inputs"], ("S", "Q", "N"))

        bridge = ReplayBridge()
        env = CelesteRoomEnv(bridge, disabled_inputs=())
        env.reset()
        env.step(everything)
        self.assertEqual(bridge.sent[-1], ("LRUDJKXCZVGHSQNO", "LRUD", "LRUD"))

    def test_info_carries_versions_and_reward_stays_out_of_the_observation(self):
        env = CelesteRoomEnv(ReplayBridge("room1_exit_dash_route"))
        obs, info = env.reset()
        self.assertEqual((info["obs_version"], info["act_version"], info["reward_version"]), ("obs-v1", "act-v1", "rew-v1"))
        self.assertEqual(set(obs), {"player", "actions", "history_valid", "grid", "context"})
        _, reward, _, _, info = env.step(noop())
        self.assertEqual(reward, sum(info["reward_components"].values()))

    def test_close_closes_the_bridge(self):
        bridge = ReplayBridge()
        env = CelesteRoomEnv(bridge)
        env.close()
        self.assertTrue(bridge.closed)


if __name__ == "__main__":
    unittest.main()
