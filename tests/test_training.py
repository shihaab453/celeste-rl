"""Learner smoke test: SB3 PPO with the obs-v1 feature extractor, and rollout discarding on bridge faults.

Run from the repo root:
    .venv-rl/Scripts/python.exe -W error -m unittest

No game: the environment runs on a fake bridge replaying recorded events. Every bridge fault starts a new
"epoch", written into the replayed player speed, so a rollout buffer that reaches an update can be checked to
contain only data collected after the latest fault.
"""
from __future__ import annotations

import copy
import unittest

import numpy as np
import torch as th
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from celeste_rl.bridge import BridgeError
from celeste_rl.env import BridgeFault, CelesteRoomEnv
from celeste_rl.schema import ACTION_INPUTS, HISTORY, PLAYER_FEATURE_COUNT, PLAYER_FEATURE_NAMES
from celeste_rl.training.policy import CelesteFeatures, policy_kwargs
from celeste_rl.training.supervisor import SupervisedPPO, TrainingAborted
from tests.test_env import ReplayBridge

SPEED_X = PLAYER_FEATURE_NAMES.index("speed_x")


class EpochBridge(ReplayBridge):
    """Replays the death route (an episode ends every 74 steps) and faults on chosen step or reset numbers."""

    def __init__(self, fault_steps=(), fault_resets=(), fault_every_step=False):
        super().__init__("room1_spike_death_route")
        self.fault_steps, self.fault_resets, self.fault_every_step = set(fault_steps), set(fault_resets), fault_every_step
        self.epoch = 0
        self.total_steps = self.total_resets = 0

    def _mark(self, observation):
        if observation.state is not None:
            observation.state["Player"]["Speed"]["X"] = self.epoch * 10.0
        return observation

    def reset(self):
        self.total_resets += 1
        if self.total_resets in self.fault_resets:
            self.epoch += 1
            raise BridgeError("Lockstep session ended: simulated reset timeout")
        return self._mark(super().reset())

    def step(self, buttons="", dash_only="", move_only=""):
        self.total_steps += 1
        if self.fault_every_step or self.total_steps in self.fault_steps:
            self.epoch += 1
            raise BridgeError("Lockstep session ended: simulated step timeout")
        return self._mark(super().step(buttons, dash_only, move_only))


class RecordingPPO(SupervisedPPO):
    """Checks every rollout that reaches an update."""

    def __init__(self, *args, bridge: EpochBridge, **kwargs):
        super().__init__(*args, **kwargs)
        self.bridge = bridge
        self.updates: list[set[float]] = []
        self.weights_after_update = copy.deepcopy(self.policy.state_dict())

    def train(self):
        speeds = self.rollout_buffer.observations["player"][:, :, 0, SPEED_X]
        self.updates.append(set(np.round(speeds.ravel() * 400).tolist()))
        # Nothing may have changed the weights since the previous update, including discarded rollouts.
        assert_same_weights(self.weights_after_update, self.policy.state_dict())
        super().train()
        self.weights_after_update = copy.deepcopy(self.policy.state_dict())


def assert_same_weights(expected: dict, actual: dict) -> None:
    for key, value in expected.items():
        if not th.equal(value, actual[key]):
            raise AssertionError(f"weights changed outside an update: {key}")


def make(bridge: EpochBridge, **kwargs) -> RecordingPPO:
    return RecordingPPO("MultiInputPolicy", CelesteRoomEnv(bridge), bridge=bridge, policy_kwargs=policy_kwargs(),
                        n_steps=32, batch_size=32, n_epochs=1, gamma=1.0, device="cpu", seed=0, **kwargs)


class PolicyTests(unittest.TestCase):
    def test_extractor_shapes_and_prediction(self):
        model = make(EpochBridge())
        extractor = model.policy.features_extractor
        self.assertIsInstance(extractor, CelesteFeatures)
        self.assertEqual(extractor.features_dim, 128 + HISTORY * PLAYER_FEATURE_COUNT + HISTORY * len(ACTION_INPUTS) + HISTORY + 2)
        self.assertFalse(model.policy.normalize_images)
        obs = model.env.reset()
        action, _ = model.predict(obs)
        self.assertEqual(action.shape, (1, len(ACTION_INPUTS)))
        self.assertTrue(np.isin(action, (0, 1)).all())
        # The grid goes through the CNN, not a 11,264-input flatten.
        with th.no_grad():
            tensors = model.policy.obs_to_tensor(obs)[0]
            self.assertEqual(extractor(tensors).shape, (1, extractor.features_dim))

    def test_rollout_buffer_keeps_the_grid_compact(self):
        model = make(EpochBridge())
        self.assertEqual(model.rollout_buffer.observations["grid"].dtype, np.uint8)


class SupervisorTests(unittest.TestCase):
    def test_learning_without_faults(self):
        bridge = EpochBridge()
        model = make(bridge)
        model.learn(96)
        self.assertEqual(len(model.updates), 3)
        self.assertEqual(model.num_timesteps, 96)
        self.assertEqual(model.fault_stats["accepted_transitions"], 96)
        self.assertEqual(model.fault_stats["discarded_rollouts"], 0)
        self.assertGreater(len(model.ep_info_buffer), 0, "episodes ended inside rollouts and were recorded")

    def test_faults_discard_whole_rollouts_and_updates_never_see_them(self):
        # Step 40 is 8 steps into the second rollout. Reset 3 is the autoreset when the episode started by the
        # recovery reset ends (74 steps later, at step 114), inside a later rollout.
        bridge = EpochBridge(fault_steps={40}, fault_resets={3})
        hook_calls = []
        model = make(bridge, on_fault=hook_calls.append)
        model.learn(128)

        self.assertEqual(len(model.updates), 4)
        for update in model.updates:
            self.assertEqual(len(update), 1, f"an update mixed data from before and after a fault: {update}")
        self.assertEqual(model.updates[-1], {float(bridge.epoch * 10)})
        self.assertEqual(model.num_timesteps, 128, "the budget counts accepted transitions only")
        self.assertEqual(model.fault_stats["accepted_transitions"], 128)
        self.assertEqual(model.fault_stats["discarded_rollouts"], 2)
        self.assertEqual(model.fault_stats["discarded_transitions"], sum(f["discarded_transitions"] for f in model.fault_stats["faults"]))
        self.assertEqual(model.fault_stats["discarded_transitions"], 7 + model.fault_stats["faults"][1]["discarded_transitions"])
        self.assertEqual(len(hook_calls), 2)
        self.assertTrue(all(isinstance(fault, BridgeFault) for fault in hook_calls))

    def test_episode_finished_inside_a_discarded_rollout_is_not_logged(self):
        # The first episode dies at step 74, inside the third rollout (steps 65-96), which faults at step 80. The
        # recovered episode never ends within the budget, so no episode statistics may remain.
        model = make(EpochBridge(fault_steps={80}))
        model.learn(96)
        self.assertEqual(model.fault_stats["discarded_rollouts"], 1)
        self.assertEqual(len(model.ep_info_buffer), 0)

    def test_discarded_rollouts_do_not_change_weights_or_optimizer(self):
        # Faults in the second rollout, after one real update. At each fault the weights and optimizer state must
        # still be exactly those left by the last update (RecordingPPO.train also checks this before every update).
        bridge = EpochBridge(fault_steps={40, 50})
        checks = []

        def on_fault(_fault):
            assert_same_weights(model.weights_after_update, model.policy.state_dict())
            checks.append(copy.deepcopy(model.policy.optimizer.state_dict()["state"]))

        model = make(bridge, on_fault=on_fault)
        optimizer_states = []
        original_train = model.train

        def train():
            original_train()
            optimizer_states.append(copy.deepcopy(model.policy.optimizer.state_dict()["state"]))

        model.train = train  # type: ignore[method-assign]
        model.learn(96)
        self.assertEqual(len(checks), 2)
        for state in checks:
            self.assertEqual(state.keys(), optimizer_states[0].keys())
            for key in state:
                for name, tensor in state[key].items():
                    self.assertTrue(th.equal(tensor, optimizer_states[0][key][name]) if th.is_tensor(tensor) else tensor == optimizer_states[0][key][name])

    def test_consecutive_faults_abort_without_updating(self):
        model = make(EpochBridge(fault_every_step=True))
        with self.assertRaisesRegex(TrainingAborted, "3 consecutive"):
            model.learn(96)
        self.assertEqual(model.updates, [])
        self.assertEqual(model.num_timesteps, 0)
        self.assertEqual(model.fault_stats["discarded_rollouts"], 3)

    def test_faulting_recovery_resets_count_toward_the_limit(self):
        # Initial reset is 1; step 3 faults; recovery resets 2 and 3 fault: three consecutive faults.
        model = make(EpochBridge(fault_steps={3}, fault_resets={2, 3}))
        with self.assertRaises(TrainingAborted):
            model.learn(32)
        stages = [f["stage"] for f in model.fault_stats["faults"]]
        self.assertEqual(stages, ["rollout", "recovery reset", "recovery reset"])
        self.assertEqual(model.updates, [])

    def test_only_dummy_vec_env_is_supported(self):
        env = VecMonitor(DummyVecEnv([lambda: CelesteRoomEnv(EpochBridge())]))
        with self.assertRaisesRegex(TypeError, "DummyVecEnv"):
            SupervisedPPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs(), n_steps=32, batch_size=32, device="cpu")


if __name__ == "__main__":
    unittest.main()
