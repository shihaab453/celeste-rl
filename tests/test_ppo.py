"""Fast unit tests for PPO components: GAE, clipping, probability ratio, entropy sign, and save/load.

Run from the repo root:
    .venv-rl/Scripts/python.exe -W error -m unittest
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from celeste_rl.ppo import (
    ActorCritic,
    compute_gae,
    compute_policy_loss,
)


class GAEComputationTests(unittest.TestCase):
    """GAE calculation against hand-computed values for termination and truncation."""

    def test_hand_computed_termination_and_truncation(self):
        # 3 time steps, 2 parallel environments.
        # gamma = 0.9, gae_lambda = 0.8.
        # Env 0 terminates at t=1 (task failed/ended: next value is 0.0).
        # Env 1 truncates at t=1 (time limit cut off: bootstraps from final observation value 4.0).
        # Both environments continue into a fresh episode at t=2.
        gamma = 0.9
        gae_lambda = 0.8

        rewards = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=torch.float32)
        values = torch.tensor([[2.0, 2.0], [1.5, 1.5], [2.5, 2.5]], dtype=torch.float32)
        terminations = torch.tensor([[False, False], [True, False], [False, False]], dtype=torch.bool)
        truncations = torch.tensor([[False, False], [False, True], [False, False]], dtype=torch.bool)
        next_value = torch.tensor([1.0, 1.0], dtype=torch.float32)
        final_values = torch.tensor([[0.0, 0.0], [0.0, 4.0], [0.0, 0.0]], dtype=torch.float32)

        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            terminations=terminations,
            truncations=truncations,
            next_value=next_value,
            final_values=final_values,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        # Hand calculations:
        # At t=2 (both envs ongoing):
        #   delta_2 = 3.0 + 0.9 * 1.0 - 2.5 = 1.4
        #   A_2 = 1.4
        #   R_2 = 1.4 + 2.5 = 3.9
        #
        # Env 0 at t=1 (terminated):
        #   next_val = 0.0
        #   delta_1 = 2.0 + 0.9 * 0.0 - 1.5 = 0.5
        #   A_1 = 0.5 (done=1 blocks propagation from t=2)
        #   R_1 = 0.5 + 1.5 = 2.0
        #
        # Env 0 at t=0 (ongoing):
        #   delta_0 = 1.0 + 0.9 * 1.5 - 2.0 = 0.35
        #   A_0 = 0.35 + 0.9 * 0.8 * A_1 = 0.35 + 0.72 * 0.5 = 0.71
        #   R_0 = 0.71 + 2.0 = 2.71
        #
        # Env 1 at t=1 (truncated, final_value=4.0):
        #   next_val = 4.0
        #   delta_1 = 2.0 + 0.9 * 4.0 - 1.5 = 4.1
        #   A_1 = 4.1 (done=1 blocks propagation from t=2)
        #   R_1 = 4.1 + 1.5 = 5.6
        #
        # Env 1 at t=0 (ongoing):
        #   delta_0 = 1.0 + 0.9 * 1.5 - 2.0 = 0.35
        #   A_0 = 0.35 + 0.9 * 0.8 * A_1 = 0.35 + 0.72 * 4.1 = 0.35 + 2.952 = 3.302
        #   R_0 = 3.302 + 2.0 = 5.302

        expected_adv_env0 = torch.tensor([0.71, 0.5, 1.4], dtype=torch.float32)
        expected_adv_env1 = torch.tensor([3.302, 4.1, 1.4], dtype=torch.float32)
        expected_ret_env0 = torch.tensor([2.71, 2.0, 3.9], dtype=torch.float32)
        expected_ret_env1 = torch.tensor([5.302, 5.6, 3.9], dtype=torch.float32)

        self.assertTrue(torch.allclose(advantages[:, 0], expected_adv_env0, atol=1e-5))
        self.assertTrue(torch.allclose(advantages[:, 1], expected_adv_env1, atol=1e-5))
        self.assertTrue(torch.allclose(returns[:, 0], expected_ret_env0, atol=1e-5))
        self.assertTrue(torch.allclose(returns[:, 1], expected_ret_env1, atol=1e-5))

    def test_termination_wins_over_truncation(self):
        # One step, one environment, both flags set (e.g. failing on exactly the time-limit step).
        # The task ended, so the final observation value must be ignored:
        #   delta = 2.0 + 0.9 * 0.0 - 1.5 = 0.5
        advantages, returns = compute_gae(
            rewards=torch.tensor([[2.0]]),
            values=torch.tensor([[1.5]]),
            terminations=torch.tensor([[True]]),
            truncations=torch.tensor([[True]]),
            next_value=torch.tensor([7.0]),
            final_values=torch.tensor([[4.0]]),
            gamma=0.9,
            gae_lambda=0.8,
        )
        self.assertTrue(torch.allclose(advantages, torch.tensor([[0.5]]), atol=1e-6))
        self.assertTrue(torch.allclose(returns, torch.tensor([[2.0]]), atol=1e-6))


class PolicyLossClippingTests(unittest.TestCase):
    """Verify clipped surrogate policy loss for positive and negative advantages."""

    def test_positive_advantage_clipping(self):
        adv = torch.tensor([2.0])
        old_lp = torch.tensor([0.0])  # log(1.0) = 0.0

        # Ratio = 1.5 (> 1 + eps = 1.2): clipped to 1.2 -> objective is 1.2 * 2.0 = 2.4 -> loss is -2.4.
        new_lp_high = torch.log(torch.tensor([1.5]))
        loss_high, ratio_high = compute_policy_loss(new_lp_high, old_lp, adv, clip_coef=0.2)
        self.assertTrue(torch.isclose(loss_high, torch.tensor(-2.4), atol=1e-5))
        self.assertTrue(torch.isclose(ratio_high, torch.tensor([1.5]), atol=1e-5))

        # Ratio = 1.1 (< 1.2): unclipped -> objective is 1.1 * 2.0 = 2.2 -> loss is -2.2.
        new_lp_mid = torch.log(torch.tensor([1.1]))
        loss_mid, _ = compute_policy_loss(new_lp_mid, old_lp, adv, clip_coef=0.2)
        self.assertTrue(torch.isclose(loss_mid, torch.tensor(-2.2), atol=1e-5))

        # Ratio = 0.5 (< 0.8): unclipped! min(0.5 * 2.0, 0.8 * 2.0) = min(1.0, 1.6) = 1.0 -> loss is -1.0.
        new_lp_low = torch.log(torch.tensor([0.5]))
        loss_low, _ = compute_policy_loss(new_lp_low, old_lp, adv, clip_coef=0.2)
        self.assertTrue(torch.isclose(loss_low, torch.tensor(-1.0), atol=1e-5))

    def test_negative_advantage_clipping(self):
        adv = torch.tensor([-2.0])
        old_lp = torch.tensor([0.0])

        # Ratio = 1.5 (> 1.2): unclipped! min(1.5 * -2.0, 1.2 * -2.0) = min(-3.0, -2.4) = -3.0 -> loss is 3.0.
        new_lp_high = torch.log(torch.tensor([1.5]))
        loss_high, _ = compute_policy_loss(new_lp_high, old_lp, adv, clip_coef=0.2)
        self.assertTrue(torch.isclose(loss_high, torch.tensor(3.0), atol=1e-5))

        # Ratio = 0.5 (< 0.8): clipped to 0.8! min(0.5 * -2.0, 0.8 * -2.0) = min(-1.0, -1.6) = -1.6 -> loss is 1.6.
        new_lp_low = torch.log(torch.tensor([0.5]))
        loss_low, _ = compute_policy_loss(new_lp_low, old_lp, adv, clip_coef=0.2)
        self.assertTrue(torch.isclose(loss_low, torch.tensor(1.6), atol=1e-5))


class ProbabilityRatioInitialTests(unittest.TestCase):
    """At the start of an update before any optimizer step, the probability ratio is exactly 1."""

    def test_probability_ratio_is_one_at_start(self):
        torch.manual_seed(42)
        model = ActorCritic(obs_dim=4, num_actions=2, hidden_dim=32)

        sample_obs = torch.randn(16, 4)
        with torch.no_grad():
            actions, old_logprobs, _, _ = model.get_action_and_value(sample_obs)

        # Before any parameter update, query the policy with the same observations and actions.
        _, current_logprobs, _, _ = model.get_action_and_value(sample_obs, actions)

        dummy_advantages = torch.ones(16)
        _, ratio = compute_policy_loss(current_logprobs, old_logprobs, dummy_advantages)

        expected_ratio = torch.ones(16, dtype=torch.float32)
        self.assertTrue(torch.allclose(ratio, expected_ratio, atol=1e-6))


class EntropyBonusSignTests(unittest.TestCase):
    """The entropy bonus must have the sign that increases distribution entropy when minimized."""

    def test_entropy_gradient_moves_toward_higher_entropy(self):
        # Start with uneven, low-entropy logits: [5.0, -5.0]
        logits = nn.Parameter(torch.tensor([5.0, -5.0]))
        optimizer = torch.optim.SGD([logits], lr=0.5)

        initial_dist = Categorical(logits=logits)
        initial_entropy = initial_dist.entropy().item()

        # In minimization loss: loss = - ent_coef * entropy.
        # Minimizing this loss must increase entropy.
        ent_coef = 1.0
        dist = Categorical(logits=logits)
        loss = -ent_coef * dist.entropy()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        updated_dist = Categorical(logits=logits)
        updated_entropy = updated_dist.entropy().item()

        self.assertGreater(updated_entropy, initial_entropy)


class SaveLoadRoundTripTests(unittest.TestCase):
    """Save and load round-trip gives identical action probabilities and values."""

    def test_save_load_preserves_weights_and_outputs(self):
        torch.manual_seed(123)
        model = ActorCritic(obs_dim=4, num_actions=2, hidden_dim=64)
        model.eval()

        sample_obs = torch.randn(10, 4)
        with torch.no_grad():
            orig_logits = model.actor(sample_obs)
            orig_probs = orig_logits.softmax(dim=-1)
            orig_values = model.get_value(sample_obs)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "test_model.pt"
            model.save(model_path)

            loaded_model = ActorCritic.load(model_path)
            loaded_model.eval()

            with torch.no_grad():
                loaded_logits = loaded_model.actor(sample_obs)
                loaded_probs = loaded_logits.softmax(dim=-1)
                loaded_values = loaded_model.get_value(sample_obs)

            self.assertEqual(loaded_model.obs_dim, model.obs_dim)
            self.assertEqual(loaded_model.num_actions, model.num_actions)
            self.assertEqual(loaded_model.hidden_dim, model.hidden_dim)
            self.assertTrue(torch.allclose(orig_probs, loaded_probs, atol=1e-7))
            self.assertTrue(torch.allclose(orig_values, loaded_values, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
