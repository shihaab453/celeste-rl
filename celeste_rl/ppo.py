"""Proximal Policy Optimization (PPO) for discrete-action Gymnasium environments.

Designed for readability and learning, following the standard clipped surrogate objective
with Generalized Advantage Estimation (GAE).
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium.vector import AutoresetMode
from torch.distributions.categorical import Categorical


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Initialize linear layer weights orthogonally with constant bias."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """Actor-Critic network with separate MLP trunks for policy and value function."""

    def __init__(self, obs_dim: int, num_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim

        # Policy network (Actor): maps observation to unnormalized action scores (logits).
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, num_actions), std=0.01),
        )

        # Value network (Critic): maps observation to estimated state value V(s).
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute estimated state value V(s).

        Args:
            obs: observation tensor, shape (batch_size, obs_dim)

        Returns:
            value: state value estimate, shape (batch_size,)
        """
        # Shape: (batch_size, 1) -> (batch_size,)
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or evaluate an action and return action, log probability, entropy, and value.

        Args:
            obs: observation tensor, shape (batch_size, obs_dim)
            action: optional action tensor to evaluate, shape (batch_size,)

        Returns:
            action: sampled or provided action, shape (batch_size,)
            log_prob: log probability of action under current policy, shape (batch_size,)
            entropy: entropy of categorical distribution, shape (batch_size,)
            value: estimated state value, shape (batch_size,)
        """
        logits = self.actor(obs)  # shape: (batch_size, num_actions)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()  # shape: (batch_size,)
        log_prob = dist.log_prob(action)  # shape: (batch_size,)
        entropy = dist.entropy()  # shape: (batch_size,)
        value = self.get_value(obs)  # shape: (batch_size,)
        return action, log_prob, entropy, value

    def get_greedy_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Select the highest probability (greedy) action for evaluation.

        Args:
            obs: observation tensor, shape (batch_size, obs_dim) or (obs_dim,)

        Returns:
            action: greedy action tensor, shape (batch_size,) or scalar
        """
        logits = self.actor(obs)
        return logits.argmax(dim=-1)

    def save(self, path: Path | str) -> None:
        """Save model architecture config and state dictionary."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "config": {
                "obs_dim": self.obs_dim,
                "num_actions": self.num_actions,
                "hidden_dim": self.hidden_dim,
            },
            "state_dict": self.state_dict(),
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: Path | str, device: str = "cpu") -> ActorCritic:
        """Load model from saved checkpoint file."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminations: torch.Tensor,
    truncations: torch.Tensor,
    next_value: torch.Tensor,
    final_values: torch.Tensor | None = None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation (GAE) and target returns.

    Properly handles episode boundaries:
    - Termination (true environment failure or goal reached): no future bootstrapping;
      value of terminal next state is 0.0.
    - Truncation (time limit cut the episode): the task did not end naturally, so we
      bootstrap from the value of the final observation before the cutoff.
    - Done (either): advantage does not propagate across the episode boundary into
      preceding steps from a different episode.

    Args:
        rewards: shape (num_steps, num_envs)
        values: shape (num_steps, num_envs)
        terminations: shape (num_steps, num_envs), bool
        truncations: shape (num_steps, num_envs), bool
        next_value: value of state following step T-1, shape (num_envs,)
        final_values: optional values of truncated final observations, shape (num_steps, num_envs)
        gamma: discount factor
        gae_lambda: GAE decay parameter

    Returns:
        advantages: shape (num_steps, num_envs)
        returns: shape (num_steps, num_envs)
    """
    num_steps, num_envs = rewards.shape
    advantages = torch.zeros_like(rewards)  # shape: (num_steps, num_envs)
    last_gae = torch.zeros(num_envs, dtype=torch.float32, device=rewards.device)

    for t in reversed(range(num_steps)):
        step_next_val = next_value if t == num_steps - 1 else values[t + 1]

        # Determine effective value of next state:
        # If truncated: bootstrap from final observation value.
        # If terminated: 0.0 (no future rewards). Termination wins when both flags are set,
        # because the task really ended on that step.
        # If ongoing: step_next_val.
        if final_values is not None:
            effective_next_val = torch.where(
                truncations[t] & ~terminations[t],
                final_values[t],
                step_next_val * (1.0 - terminations[t].float()),
            )
        else:
            effective_next_val = step_next_val * (1.0 - terminations[t].float())

        # TD error: delta = r + gamma * V(s') - V(s)
        delta = rewards[t] + gamma * effective_next_val - values[t]

        # When episode ended (termination or truncation), advantage from future steps must not propagate.
        dones = (terminations[t] | truncations[t]).float()
        last_gae = delta + gamma * gae_lambda * (1.0 - dones) * last_gae
        advantages[t] = last_gae

    returns = advantages + values  # shape: (num_steps, num_envs)
    return advantages, returns


def compute_policy_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_coef: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the clipped surrogate policy loss and probability ratio.

    Args:
        logprobs: log probabilities under current policy, shape (batch_size,)
        old_logprobs: log probabilities recorded during rollout, shape (batch_size,)
        advantages: normalized advantage values, shape (batch_size,)
        clip_coef: epsilon clipping parameter (default: 0.2)

    Returns:
        policy_loss: scalar loss to minimize
        ratio: probability ratio pi_theta(a|s) / pi_old(a|s), shape (batch_size,)
    """
    ratio = torch.exp(logprobs - old_logprobs)  # shape: (batch_size,)
    surr1 = ratio * advantages  # unclipped objective
    surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantages  # clipped objective
    # Negative because PyTorch optimizers minimize loss, but we want to maximize objective.
    policy_loss = -torch.min(surr1, surr2).mean()
    return policy_loss, ratio


@dataclass
class PPOConfig:
    """Hyperparameters for training PPO on discrete Gymnasium environments."""

    total_timesteps: int = 100_000
    learning_rate: float = 1e-3
    num_envs: int = 4
    num_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 42
    hidden_dim: int = 64

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches


def set_seed(seed: int) -> None:
    """Set global seeds across random, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def evaluate(
    model: ActorCritic,
    env_id: str,
    n_eval_episodes: int = 20,
    seed: int = 100,
) -> tuple[float, float, list[float]]:
    """Evaluate a trained model using greedy actions on fresh episodes.

    Args:
        model: trained ActorCritic policy
        env_id: Gymnasium environment ID (e.g. 'CartPole-v1')
        n_eval_episodes: number of episodes to evaluate
        seed: base seed for evaluation environments

    Returns:
        mean_return: average undiscounted return across episodes
        std_return: standard deviation of returns
        all_returns: list of individual episode returns
    """
    eval_env = gym.make(env_id)
    returns: list[float] = []

    for ep in range(n_eval_episodes):
        obs, _ = eval_env.reset(seed=seed + ep)
        ep_return = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                action = model.get_greedy_action(obs_tensor).item()
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            ep_return += float(reward)

        returns.append(ep_return)

    eval_env.close()
    return float(np.mean(returns)), float(np.std(returns)), returns


def train_ppo(
    env_id: str,
    config: PPOConfig,
    device: str = "cpu",
) -> tuple[ActorCritic, dict[str, Any]]:
    """Train an ActorCritic policy on a Gymnasium environment using PPO.

    Args:
        env_id: Gymnasium environment ID (e.g. 'CartPole-v1')
        config: training configuration and hyperparameters
        device: torch device ('cpu')

    Returns:
        model: trained ActorCritic network
        metrics: dictionary of training history and learning curve
    """
    set_seed(config.seed)

    # Vector environment setup:
    # Gymnasium 1.0+ SyncVectorEnv defaults to AutoresetMode.NEXT_STEP in gym.make_vec.
    # In NEXT_STEP, an extra step is inserted after episode end where the action is ignored
    # and reward is 0.0, causing transitions that span across episode boundaries.
    # Setting AutoresetMode.SAME_STEP ensures that resets happen immediately inside the
    # terminal step: the finished episode's final observation is stored in infos["final_obs"],
    # and next_obs contains the clean starting state for the next episode.
    envs = gym.make_vec(
        env_id,
        num_envs=config.num_envs,
        vectorization_mode="sync",
        vector_kwargs={"autoreset_mode": AutoresetMode.SAME_STEP},
    )

    obs_dim = int(np.prod(envs.single_observation_space.shape))
    num_actions = int(envs.single_action_space.n)

    model = ActorCritic(obs_dim=obs_dim, num_actions=num_actions, hidden_dim=config.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)

    num_updates = config.total_timesteps // config.batch_size
    obs, _ = envs.reset(seed=config.seed)

    # Tracking metrics
    learning_curve: list[dict[str, float]] = []
    episode_returns: list[float] = []
    current_episode_return = np.zeros(config.num_envs, dtype=np.float32)

    start_time = time.perf_counter()

    for update in range(1, num_updates + 1):
        # Rollout storage buffers
        obs_buf = torch.zeros((config.num_steps, config.num_envs, obs_dim), dtype=torch.float32, device=device)
        actions_buf = torch.zeros((config.num_steps, config.num_envs), dtype=torch.long, device=device)
        logprobs_buf = torch.zeros((config.num_steps, config.num_envs), dtype=torch.float32, device=device)
        rewards_buf = torch.zeros((config.num_steps, config.num_envs), dtype=torch.float32, device=device)
        terms_buf = torch.zeros((config.num_steps, config.num_envs), dtype=torch.bool, device=device)
        truncs_buf = torch.zeros((config.num_steps, config.num_envs), dtype=torch.bool, device=device)
        values_buf = torch.zeros((config.num_steps, config.num_envs), dtype=torch.float32, device=device)
        final_values = torch.zeros((config.num_steps, config.num_envs), dtype=torch.float32, device=device)

        for step in range(config.num_steps):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            obs_buf[step] = obs_tensor

            with torch.no_grad():
                action, logprob, _, value = model.get_action_and_value(obs_tensor)
            actions_buf[step] = action
            logprobs_buf[step] = logprob
            values_buf[step] = value

            # Step the vectorized environment
            next_obs, rewards, terminations, truncations, infos = envs.step(action.cpu().numpy())
            rewards_buf[step] = torch.as_tensor(rewards, dtype=torch.float32, device=device)
            terms_buf[step] = torch.as_tensor(terminations, dtype=torch.bool, device=device)
            truncs_buf[step] = torch.as_tensor(truncations, dtype=torch.bool, device=device)

            current_episode_return += rewards
            dones = np.logical_or(terminations, truncations)

            for env_idx in range(config.num_envs):
                if dones[env_idx]:
                    episode_returns.append(float(current_episode_return[env_idx]))
                    current_episode_return[env_idx] = 0.0

                    # For truncated episodes in SAME_STEP mode, the true final observation
                    # is available in infos["final_obs"][env_idx]. A terminated step never
                    # bootstraps, even if it also hit the time limit.
                    if truncations[env_idx] and not terminations[env_idx] and "final_obs" in infos:
                        fo = torch.as_tensor(infos["final_obs"][env_idx], dtype=torch.float32, device=device)
                        with torch.no_grad():
                            final_values[step, env_idx] = model.get_value(fo)

            obs = next_obs

        # Bootstrap value for ongoing episodes at the end of the rollout buffer
        with torch.no_grad():
            next_value = model.get_value(torch.as_tensor(obs, dtype=torch.float32, device=device))

        # Compute GAE advantages and target returns
        advantages, returns = compute_gae(
            rewards=rewards_buf,
            values=values_buf,
            terminations=terms_buf,
            truncations=truncs_buf,
            next_value=next_value,
            final_values=final_values,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )

        # Flatten rollout tensors from (num_steps, num_envs, ...) to (batch_size, ...)
        b_obs = obs_buf.reshape(-1, obs_dim)  # shape: (batch_size, obs_dim)
        b_actions = actions_buf.reshape(-1)  # shape: (batch_size,)
        b_logprobs = logprobs_buf.reshape(-1)  # shape: (batch_size,)
        b_advantages = advantages.reshape(-1)  # shape: (batch_size,)
        b_returns = returns.reshape(-1)  # shape: (batch_size,)

        # Normalize advantages across the entire rollout batch
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # Minibatch optimization over several epochs
        b_inds = np.arange(config.batch_size)
        clipfracs: list[float] = []

        for _ in range(config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, config.batch_size, config.minibatch_size):
                end = start + config.minibatch_size
                mb_inds = b_inds[start:end]

                _, new_logprob, entropy, new_value = model.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )

                # Clipped surrogate policy loss
                pg_loss, ratio = compute_policy_loss(
                    logprobs=new_logprob,
                    old_logprobs=b_logprobs[mb_inds],
                    advantages=b_advantages[mb_inds],
                    clip_coef=config.clip_coef,
                )

                # Value loss: mean squared error against GAE returns
                v_loss = 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()

                # Entropy bonus: encourage exploration by subtracting entropy from minimization objective
                entropy_loss = entropy.mean()

                # Combined loss
                loss = pg_loss + config.vf_coef * v_loss - config.ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    clipfrac = ((ratio - 1.0).abs() > config.clip_coef).float().mean().item()
                    clipfracs.append(clipfrac)

        # Record learning curve point periodically
        steps_so_far = update * config.batch_size
        recent_mean_return = float(np.mean(episode_returns[-20:])) if episode_returns else 0.0
        learning_curve.append({
            "step": steps_so_far,
            "mean_return": recent_mean_return,
            "pg_loss": float(pg_loss.item()),
            "v_loss": float(v_loss.item()),
            "entropy": float(entropy_loss.item()),
            "clipfrac": float(np.mean(clipfracs)),
        })

    envs.close()
    elapsed_time = time.perf_counter() - start_time

    metrics = {
        "wall_time": elapsed_time,
        "total_timesteps": num_updates * config.batch_size,
        "learning_curve": learning_curve,
        "final_train_return": float(np.mean(episode_returns[-20:])) if episode_returns else 0.0,
    }
    return model, metrics
