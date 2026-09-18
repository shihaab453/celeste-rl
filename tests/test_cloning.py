"""Behavioural cloning: the split, the accuracy measurement, and that fitting actually fits (no game).

Run from the repo root:
    .venv-rl/Scripts/python.exe -W error -m unittest
"""
from __future__ import annotations

import unittest

import numpy as np
import torch as th
from stable_baselines3 import PPO

from celeste_rl.cloning import OBS_KEYS, Demonstrations, accuracy, clone, split_by_trajectory
from celeste_rl.observation import observation_space
from celeste_rl.schema import ACTION_INPUTS, MENU_INPUTS
from celeste_rl.training.policy import CelestePolicy, policy_kwargs


def demonstrations(count: int, frames: int, seed: int = 0) -> Demonstrations:
    space, rng = observation_space(), np.random.default_rng(seed)
    total = count * frames
    obs = {key: np.stack([space[key].sample() for _ in range(total)]) for key in OBS_KEYS}
    actions = np.zeros((total, len(ACTION_INPUTS)), dtype=np.int8)
    # Two inputs per frame, as real play holds, and never a disabled one.
    enabled = [i for i, name in enumerate(ACTION_INPUTS) if name not in MENU_INPUTS]
    for row in range(total):
        actions[row, rng.choice(enabled, size=2, replace=False)] = 1
    trajectory = np.repeat(np.arange(count), frames)
    return Demonstrations(obs, actions, trajectory)


class SplitTests(unittest.TestCase):
    def test_whole_trajectories_are_held_out_never_frames(self):
        """A frame-wise split would leak: consecutive frames of one route are nearly identical."""
        data = demonstrations(10, 20)
        train, holdout = split_by_trajectory(data, holdout=0.3, seed=0)
        self.assertEqual(set(train.trajectories) & set(holdout.trajectories), set())
        self.assertEqual(len(set(train.trajectories) | set(holdout.trajectories)), 10)
        self.assertEqual(len(train) + len(holdout), len(data))
        self.assertEqual(len(holdout.trajectories), 3)

    def test_one_demonstration_is_always_held_out(self):
        train, holdout = split_by_trajectory(demonstrations(3, 10), holdout=0.01, seed=0)
        self.assertEqual(len(holdout.trajectories), 1)
        self.assertEqual(len(train.trajectories), 2)

    def test_a_single_demonstration_cannot_be_split(self):
        with self.assertRaises(ValueError):
            split_by_trajectory(demonstrations(1, 10), holdout=0.5, seed=0)

    def test_the_split_is_reproducible(self):
        data = demonstrations(8, 10)
        first = split_by_trajectory(data, 0.25, seed=3)[1].trajectories
        self.assertEqual(list(first), list(split_by_trajectory(data, 0.25, seed=3)[1].trajectories))


class AccuracyTests(unittest.TestCase):
    def test_the_always_zero_baseline_is_reported(self):
        """Inputs are sparse, so a per-input number means nothing without it."""
        data = demonstrations(4, 25)
        model = PPO(CelestePolicy, _spaces_env(), policy_kwargs=policy_kwargs(), device="cpu", seed=0)
        measured = accuracy(model.policy, data)
        self.assertAlmostEqual(measured["always_zero_input_accuracy"], 1 - 2 / 21, places=2)
        self.assertEqual(measured["always_zero_frame_accuracy"], 0.0)
        self.assertEqual((measured["frames"], measured["trajectories"]), (100, 4))
        self.assertAlmostEqual(measured["inputs_per_frame"], 2.0, places=2)


class CloneTests(unittest.TestCase):
    def test_fitting_improves_accuracy_on_the_frames_it_fits(self):
        data = demonstrations(3, 40)
        model = PPO(CelestePolicy, _spaces_env(), policy_kwargs=policy_kwargs(), device="cpu", seed=0)
        before = accuracy(model.policy, data)["input_accuracy"]
        result = clone(model.policy, data, None, epochs=30, batch_size=32, learning_rate=1e-3, seed=0)
        after = result["history"][-1]["train"]["input_accuracy"]
        self.assertGreater(after, before)
        self.assertGreater(after, before + 0.05, "30 epochs on 120 frames should move the fit")

    def test_the_holdout_is_measured_separately(self):
        train, holdout = split_by_trajectory(demonstrations(4, 30), holdout=0.25, seed=0)
        model = PPO(CelestePolicy, _spaces_env(), policy_kwargs=policy_kwargs(), device="cpu", seed=0)
        result = clone(model.policy, train, holdout, epochs=10, batch_size=32, learning_rate=1e-3, seed=0)
        last = result["history"][-1]
        self.assertIn("holdout", last)
        self.assertEqual(last["train"]["trajectories"], 3)
        self.assertEqual(last["holdout"]["trajectories"], 1)


def _spaces_env():
    import gymnasium as gym

    class SpacesOnly(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            self.observation_space = observation_space()
            self.action_space = gym.spaces.MultiBinary(len(ACTION_INPUTS))

        def reset(self, *, seed=None, options=None):
            raise RuntimeError("never stepped")

        def step(self, action):
            raise RuntimeError("never stepped")

    return SpacesOnly()


if __name__ == "__main__":
    unittest.main()
