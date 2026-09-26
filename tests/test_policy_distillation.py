"""Offline checks of recording a policy's own play and distilling it into a fresh network (no game)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from celeste_rl.endings import RoomTask  # noqa: E402
from celeste_rl.env import CelesteRoomEnv  # noqa: E402
from celeste_rl.schema import ACTION_INPUTS  # noqa: E402
from celeste_rl.starts import Start  # noqa: E402
from celeste_rl.training.policy import CelestePolicy, policy_kwargs  # noqa: E402
from distill_policy import agreement, fit_student, play_problems  # noqa: E402
from record_policy_play import record_episodes  # noqa: E402
from tests.test_cloning import _spaces_env, demonstrations  # noqa: E402
from tests.test_env import Room3Bridge  # noqa: E402


class FixedPolicy:
    """Stands in for a loaded policy: always presses the same inputs."""

    def __init__(self, pressed: tuple[str, ...]):
        self.action = np.array([name in pressed for name in ACTION_INPUTS], dtype=np.int8)

    def predict(self, obs, deterministic=False):
        return self.action.copy(), None


class RecordTests(unittest.TestCase):
    def test_every_frame_is_recorded_with_its_episode_and_applied_action(self):
        env = CelesteRoomEnv(Room3Bridge(), task=RoomTask("2", "3"), task_start=Start(("1,U", "1,U"), (261, 1), "2", 0))
        data, summaries = record_episodes(FixedPolicy(("R",)), env, episodes=3)
        self.assertEqual([s["episode"] for s in summaries], [0, 1, 2])
        self.assertEqual(len(data), sum(s["frames"] for s in summaries))
        self.assertEqual(sorted(set(data.trajectory.tolist())), [0, 1, 2])
        self.assertTrue(all(s["ending"] is not None for s in summaries))
        self.assertEqual(data.actions.shape[1], len(ACTION_INPUTS))


class DistillTests(unittest.TestCase):
    def test_a_fresh_student_moves_toward_the_donor(self):
        donor = PPO(CelestePolicy, _spaces_env(), policy_kwargs=policy_kwargs(), device="cpu", seed=5).policy
        data = demonstrations(4, 30)
        _, record = fit_student(donor, data, holdout=0.25, seed=0, epochs=30, batch_size=32, learning_rate=1e-3)
        before, after = record["before"]["fitted"], record["after"]["fitted"]
        self.assertLess(after["mean_abs_probability_difference"], before["mean_abs_probability_difference"])
        self.assertEqual(record["after"]["held_out"]["trajectories"], 1)

    def test_a_recording_must_be_this_donors_clean_attributable_play(self):
        identity = {"name": "chapter-1-room-1"}
        good = {"checkpoint_sha256": "d" * 64, "dataset_sha256": "a" * 64, "task": identity,
                "uncommitted_changes": False, "attributable": True, "runtime_problems": []}
        self.assertEqual(play_problems(good, "a" * 64, "d" * 64, identity), [])
        for change in ({"checkpoint_sha256": "e" * 64}, {"dataset_sha256": "b" * 64}, {"uncommitted_changes": True},
                       {"task": {"name": "chapter-1-room-2"}}, {"attributable": False},
                       {"runtime_problems": ["python.numpy differs"]}):
            with self.subTest(change=change):
                self.assertTrue(play_problems({**good, **change}, "a" * 64, "d" * 64, identity))

    def test_a_policy_agrees_with_itself(self):
        donor = PPO(CelestePolicy, _spaces_env(), policy_kwargs=policy_kwargs(), device="cpu", seed=5).policy
        result = agreement(donor, donor, demonstrations(2, 10))
        self.assertEqual((result["frame_agreement"], result["mean_abs_probability_difference"]), (1.0, 0.0))


if __name__ == "__main__":
    unittest.main()
