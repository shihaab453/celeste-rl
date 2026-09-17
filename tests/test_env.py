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
    FAILURES,
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
from celeste_rl.potential import RoomPotential
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


V2 = RewardConfig(version="rew-v2")


def episode_total(ending: str, elapsed: int, config: RewardConfig = V2) -> float:
    """What a whole episode of `elapsed` frames pays, ignoring shaping (which telescopes to a start constant)."""
    ending_step = reward_components(ending, config, elapsed)
    return sum(ending_step.values()) + (elapsed - 1) * config.time_per_frame


class RewardV2Tests(unittest.TestCase):
    """rew-v2: the unspent-deadline charge (Codex K1). Shaping is exercised by ShapingTests."""

    def test_every_failure_costs_exactly_the_same_whenever_it_happens(self):
        for elapsed in (1, 2, 60, 900, DEADLINE_FRAMES):
            for ending in FAILURES:
                with self.subTest(ending=ending, elapsed=elapsed):
                    self.assertAlmostEqual(episode_total(ending, elapsed), -1.03, places=10)

    def test_failing_earlier_is_never_better_than_failing_later(self):
        for ending in FAILURES:
            totals = [episode_total(ending, elapsed) for elapsed in range(1, DEADLINE_FRAMES + 1, 37)]
            for earlier, later in zip(totals, totals[1:]):
                self.assertLessEqual(earlier, later + 1e-12, f"{ending}: failing earlier pays more")

    def test_success_always_beats_every_failure(self):
        worst_success = 1 + DEADLINE_FRAMES * V2.time_per_frame
        self.assertGreater(worst_success, episode_total(DEATH, 1))
        for elapsed in (1, 900, DEADLINE_FRAMES):
            self.assertAlmostEqual(episode_total(SUCCESS, elapsed), 1 + elapsed * V2.time_per_frame, places=10)

    def test_the_charge_is_its_own_component_and_only_on_failures(self):
        self.assertEqual(reward_components(None, V2, 10)["unspent_deadline"], 0.0)
        self.assertEqual(reward_components(SUCCESS, V2, 10)["unspent_deadline"], 0.0)
        self.assertAlmostEqual(reward_components(DEATH, V2, 10)["unspent_deadline"], -(DEADLINE_FRAMES - 10) / 60_000)
        # A failure past the deadline cannot happen, but it must never pay the agent for the overrun.
        self.assertEqual(reward_components(DEATH, V2, DEADLINE_FRAMES + 50)["unspent_deadline"], 0.0)

    def test_rew_v1_is_unchanged_and_refuses_shaping(self):
        config = RewardConfig()
        self.assertEqual(config.version, "rew-v1")
        self.assertEqual(set(reward_components(DEATH, config, 10)), {"completion", "failure", "time", "shaping"})
        self.assertAlmostEqual(episode_total(DEATH, 10, config), -1 - 10 / 60_000, places=10)
        with self.assertRaises(ValueError):
            reward_components(None, config, 10, 0.5)

    def test_unknown_version_is_refused(self):
        with self.assertRaises(ValueError):
            RewardConfig(version="rew-v3")


class WalkingBridge(ReplayBridge):
    """A bridge whose player walks through given positions, so the shaping term is not constant.

    The last position is the last frame of the episode; `ending_events` are raised there.
    """

    def __init__(self, positions, ending_events):
        super().__init__()
        self.positions, self.ending_events = positions, ending_events

    def step(self, buttons="", dash_only="", move_only=""):
        observation = super().step(buttons, dash_only, move_only)
        x, y = self.positions[self.index - 1]
        observation.state["Player"]["Position"] = {"X": x, "Y": y}
        last = self.index == len(self.positions)
        return Observation(observation.episode_id, observation.step_id, observation.tas_frame, observation.room,
                           None if last and self.ending_events else observation.state, None, observation.extras,
                           self.ending_events if last else [])


class ShapingTests(unittest.TestCase):
    """Potential-based shaping: the sum over an episode is exactly -scale * potential(start), whatever happens
    in between, so shaping cannot change which ending the agent prefers."""

    WALK = [(27, 144), (35, 144), (43, 140), (51, 136), (60, 152), (60, 164)]
    DEATH_EVENTS = [{"type": "death", "room": "1"}]

    def run_walk(self, events):
        env = CelesteRoomEnv(WalkingBridge(self.WALK, events), reward_config=V2)
        _, info = env.reset()
        start_potential = info["potential"]
        shaping, total, steps = 0.0, 0.0, 0
        for _ in range(len(self.WALK)):
            _, reward, terminated, _, info = env.step(noop())
            shaping += info["reward_components"]["shaping"]
            total += reward
            steps += 1
            if terminated:
                break
        return start_potential, shaping, total, steps, info

    def test_shaping_telescopes_to_the_start_potential(self):
        """Over a finished episode: the terminal potential is 0, so the sum is exactly -scale * potential(start)."""
        start, shaping, _, _, info = self.run_walk(self.DEATH_EVENTS)
        self.assertEqual(info["ending"], DEATH)
        self.assertGreater(start, 0.0)
        self.assertAlmostEqual(shaping, -V2.shaping_scale * start, places=9)

    def test_an_unfinished_episode_telescopes_to_where_the_player_got_to(self):
        """The same sum with no ending: scale * (potential(now) - potential(start)), and nothing else."""
        start, shaping, _, _, info = self.run_walk([])
        self.assertIsNone(info["ending"])
        self.assertAlmostEqual(shaping, V2.shaping_scale * (info["potential"] - start), places=9)

    def test_the_shaped_return_is_the_unshaped_one_plus_that_constant(self):
        start, shaping, total, steps, info = self.run_walk(self.DEATH_EVENTS)
        self.assertEqual(info["ending"], DEATH)
        self.assertAlmostEqual(total, episode_total(DEATH, steps) + shaping, places=9)
        self.assertAlmostEqual(total, -1.03 - V2.shaping_scale * start, places=9)

    def test_moving_towards_the_exit_pays_and_moving_back_charges(self):
        env = CelesteRoomEnv(WalkingBridge([(84, 128), (19, 144)], []), reward_config=V2)
        env.reset()
        _, _, _, _, forward = env.step(noop())
        _, _, _, _, back = env.step(noop())
        self.assertGreater(forward["reward_components"]["shaping"], 0.0)
        self.assertAlmostEqual(forward["reward_components"]["shaping"] + back["reward_components"]["shaping"], 0.0,
                               places=9)

    def test_rew_v1_never_shapes_and_builds_no_potential(self):
        env = CelesteRoomEnv(WalkingBridge(self.WALK, []))
        _, info = env.reset()
        self.assertEqual(info["potential"], 0.0)
        for _ in range(3):
            _, _, _, _, info = env.step(noop())
            self.assertEqual(info["reward_components"]["shaping"], 0.0)
        self.assertIsNone(env._potential)

    def test_the_environment_reports_its_reward_version(self):
        env = CelesteRoomEnv(ReplayBridge(), reward_config=V2)
        _, info = env.reset()
        self.assertEqual(info["reward_version"], "rew-v2")
        _, _, _, _, info = env.step(noop())
        self.assertIn("unspent_deadline", info["reward_components"])

    def test_the_potential_is_reused_while_the_room_is_unchanged(self):
        env = CelesteRoomEnv(ReplayBridge(), reward_config=V2)
        env.reset()
        built = env._potential
        self.assertIsInstance(built, RoomPotential)
        env.reset()
        self.assertIs(env._potential, built)



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

    def test_unreachable_game_during_reset_is_a_fault(self):
        import socket
        import tempfile

        from celeste_rl.bridge import CelesteBridge, DebugRcClient
        from celeste_rl.lockstep import LockstepBridge

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        with tempfile.TemporaryDirectory() as directory:
            http = CelesteBridge(Path(directory) / "episode.tas", client=DebugRcClient(port, timeout=1.0))
            env = CelesteRoomEnv(LockstepBridge(http, port=port, timeout=1.0))
            with self.assertRaisesRegex(BridgeFault, "reset failed"):
                env.reset()
            env.close()

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
