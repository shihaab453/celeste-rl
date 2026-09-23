"""Learner smoke test: SB3 PPO with the obs-v1 feature extractor, and rollout discarding on bridge faults.

Run from the repo root:
    .venv-rl/Scripts/python.exe -W error -m unittest

No game: the environment runs on a fake bridge replaying recorded events. Every bridge fault starts a new
"epoch", written into the replayed player speed, so a rollout buffer that reaches an update can be checked to
contain only data collected after the latest fault.
"""
from __future__ import annotations

import copy
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from celeste_rl.bridge import BridgeError, BridgeTransportError
from celeste_rl.env import BridgeFault, CelesteRoomEnv
from celeste_rl.schema import ACTION_INPUTS, HISTORY, PLAYER_FEATURE_COUNT, PLAYER_FEATURE_NAMES
from celeste_rl.training.policy import (
    CelesteFeatures,
    CelestePolicy,
    bias_for_probability,
    policy_kwargs,
)
from celeste_rl.training.supervisor import SupervisedPPO, TrainingAborted
from tests.test_env import ReplayBridge

SPEED_X = PLAYER_FEATURE_NAMES.index("speed_x")


class EpochBridge(ReplayBridge):
    """Replays the death route (an episode ends every 74 steps) and faults on chosen step or reset numbers."""

    def __init__(self, fault_steps=(), fault_resets=(), fault_every_step=False, refused_resets=()):
        super().__init__("room1_spike_death_route")
        self.fault_steps, self.fault_resets, self.fault_every_step = set(fault_steps), set(fault_resets), fault_every_step
        self.refused_resets = set(refused_resets)
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
        if self.total_resets in self.refused_resets:
            # As when the game has crashed or is still starting: the HTTP request is refused.
            self.epoch += 1
            raise BridgeTransportError("GET /tas/info failed: ConnectionRefusedError: simulated")
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
    return RecordingPPO(CelestePolicy, CelesteRoomEnv(bridge), bridge=bridge, policy_kwargs=policy_kwargs(),
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

    def test_abort_saves_the_last_update_and_ends_training_callbacks(self):
        ended = []

        class EndRecorder(BaseCallback):
            def _on_step(self):
                return True

            def _on_training_end(self):
                ended.append(True)

        # Step 40 and every step after it fault: one update completes, then three consecutive faults abort.
        bridge = EpochBridge(fault_steps=set(range(40, 400)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aborted.zip"
            model = make(bridge, abort_checkpoint_path=path)
            with self.assertRaisesRegex(TrainingAborted, "model saved to"):
                model.learn(96, callback=EndRecorder())
            self.assertEqual(len(model.updates), 1)
            self.assertEqual(ended, [True])
            saved = PPO.load(path, device="cpu")
            assert_same_weights(model.weights_after_update, saved.policy.state_dict())

    def test_refused_recovery_reset_is_retried(self):
        # Step 10 faults; the first recovery reset is refused as if the game were restarting; the next succeeds.
        model = make(EpochBridge(fault_steps={10}, refused_resets={2}))
        model.learn(64)
        self.assertEqual([f["stage"] for f in model.fault_stats["faults"]], ["rollout", "recovery reset"])
        self.assertEqual(model.num_timesteps, 64)
        self.assertEqual(len(model.updates), 2)

    def test_faulting_recovery_resets_count_toward_the_limit(self):
        # Initial reset is 1; step 3 faults; recovery resets 2 and 3 fault: three consecutive faults.
        model = make(EpochBridge(fault_steps={3}, fault_resets={2, 3}))
        with self.assertRaises(TrainingAborted):
            model.learn(32)
        stages = [f["stage"] for f in model.fault_stats["faults"]]
        self.assertEqual(stages, ["rollout", "recovery reset", "recovery reset"])
        self.assertEqual(model.updates, [])

    def test_a_failed_relaunch_counts_toward_the_limit_and_aborts(self):
        """Review J7: a relaunch that raises must end in the recorded abort, not escape recovery."""
        calls = []

        def relaunch_fails(fault):
            calls.append(fault)
            raise RuntimeError("Celeste exited during startup with code 1")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aborted.zip"
            model = make(EpochBridge(fault_steps={40}), on_fault=relaunch_fails, abort_checkpoint_path=path)
            with self.assertRaisesRegex(TrainingAborted, "3 consecutive faults.*exited during startup"):
                model.learn(96)
            self.assertTrue(path.exists())
        self.assertEqual([f["stage"] for f in model.fault_stats["faults"]],
                         ["rollout", "recovery hook", "recovery hook"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(model.updates), 1)

    def test_a_relaunch_that_fails_once_is_retried(self):
        calls = []

        def fails_once(fault):
            calls.append(fault)
            if len(calls) == 1:
                raise RuntimeError("DebugRC did not answer on port 32279 within 120 s")

        model = make(EpochBridge(fault_steps={40}), on_fault=fails_once)
        model.learn(96)
        self.assertEqual([f["stage"] for f in model.fault_stats["faults"]], ["rollout", "recovery hook"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(model.num_timesteps, 96)

    def test_only_dummy_vec_env_is_supported(self):
        env = VecMonitor(DummyVecEnv([lambda: CelesteRoomEnv(EpochBridge())]))
        with self.assertRaisesRegex(TypeError, "DummyVecEnv"):
            SupervisedPPO(CelestePolicy, env, policy_kwargs=policy_kwargs(), n_steps=32, batch_size=32, device="cpu")


if __name__ == "__main__":
    unittest.main()


class ActionBiasTests(unittest.TestCase):
    """How many inputs a fresh policy holds at once.

    act-v1 is 24 independent on/off inputs, so a zero bias means every one starts at probability 0.5 and the
    policy samples about twelve buttons held at once, every frame. A recorded clear of room 1 holds 2.31 per
    frame and never more than three.
    """

    @staticmethod
    def expected_inputs_per_frame(action_bias: float) -> float:
        model = SupervisedPPO(CelestePolicy, CelesteRoomEnv(EpochBridge()), policy_kwargs=policy_kwargs(action_bias),
                              n_steps=32, batch_size=32, device="cpu", seed=0)
        with th.no_grad():
            return float(th.sigmoid(model.policy.action_net.bias).sum())

    def test_the_default_still_samples_twelve_inputs_at_once(self):
        """Unchanged from every run before the sparse initialisation, so those stay reproducible."""
        self.assertAlmostEqual(self.expected_inputs_per_frame(0.0), 12.0, delta=0.5)

    def test_a_negative_bias_gives_the_sparsity_real_play_has(self):
        self.assertAlmostEqual(self.expected_inputs_per_frame(bias_for_probability(0.10)), 2.4, delta=0.3)

    def test_the_bias_maps_to_the_probability_it_claims(self):
        for probability in (0.05, 0.10, 0.25, 0.5):
            with self.subTest(probability=probability):
                bias = bias_for_probability(probability)
                self.assertAlmostEqual(1 / (1 + math.exp(-bias)), probability, places=9)
        self.assertAlmostEqual(bias_for_probability(0.10), -2.197, places=3)
        for bad in (0.0, 1.0, -0.5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                bias_for_probability(bad)

    def test_the_bias_survives_a_save_and_load(self):
        """A resumed run must keep its learned biases, not have the initial one reapplied."""
        model = SupervisedPPO(CelestePolicy, CelesteRoomEnv(EpochBridge()),
                              policy_kwargs=policy_kwargs(bias_for_probability(0.10)), n_steps=32, batch_size=32,
                              device="cpu", seed=0)
        with th.no_grad():
            model.policy.action_net.bias.fill_(0.75)   # stand in for what training would have learned
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.zip"
            model.save(path)
            loaded = SupervisedPPO.load(path, device="cpu")
        self.assertTrue(th.allclose(loaded.policy.action_net.bias, th.full((len(ACTION_INPUTS),), 0.75)))
