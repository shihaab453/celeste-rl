"""PPO that discards a whole rollout when the game bridge faults (Phase 2 spec, section 9.2).

Stable-Baselines3 has no recovery: an exception from env.step ends training, and catching it inside the
environment would force a fabricated transition. SupervisedPPO catches BridgeFault around rollout collection:

  1. No sample is stored for the failing step.
  2. The whole current rollout is discarded; no update ever sees it. Earlier completed updates are kept.
  3. The optional on_fault hook runs (for example to relaunch the game), then every environment slot is reset,
     and the rollout buffer, last observations and episode-start flags are replaced with fresh ones.
  4. Weights and optimizer state are untouched, because SB3 only updates them after a complete rollout. The
     step counter and episode statistics are restored to their values before the discarded rollout, so the
     training budget counts accepted transitions only.
  5. Discarded and accepted transitions and every fault are recorded in fault_stats. A reset that faults during
     recovery is retried. After max_consecutive_discards faults with no completed rollout in between (rollout
     or recovery reset faults), the model is saved to abort_checkpoint_path (if given), the callback's
     on_training_end runs, and TrainingAborted is raised. No further update happens.

Supported: a DummyVecEnv (one process), no VecNormalize (its running statistics are not rolled back). The first
reset inside learn() happens before collection and is not supervised; a fault there propagates to the caller.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
from pathlib import Path

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from celeste_rl.env import BridgeFault


class TrainingAborted(RuntimeError):
    """Too many consecutive rollouts were discarded because of bridge faults."""


class SupervisedPPO(PPO):
    def __init__(self, *args, max_consecutive_discards: int = 3,
                 on_fault: Callable[[BridgeFault], None] | None = None,
                 abort_checkpoint_path: str | Path | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        # A model loaded without an environment (for example to evaluate a checkpoint) has nothing to supervise.
        if self.env is not None and not isinstance(self.env, DummyVecEnv):
            raise TypeError(f"SupervisedPPO supports DummyVecEnv only, got {type(self.env).__name__}")
        self.max_consecutive_discards = max_consecutive_discards
        self.on_fault = on_fault
        self.abort_checkpoint_path = abort_checkpoint_path
        self.fault_stats = {"accepted_transitions": 0, "discarded_transitions": 0, "discarded_rollouts": 0,
                            "faults": []}

    # SB3's own training statistics, captured after each update. Without these, "is there a gradient at all"
    # cannot be answered from a run's records, which cost this project six diagnostics to notice.
    TRAIN_STATS = ("explained_variance", "clip_fraction", "approx_kl", "entropy_loss", "policy_gradient_loss",
                   "value_loss", "advantage_std", "advantage_abs_mean")

    def train(self) -> None:
        # Read before the update, because SB3 normalises advantages to unit variance inside each minibatch.
        # That normalisation is why a policy with almost no signal still takes ordinary-sized steps and every
        # other statistic looks healthy: the raw scale is the only number that distinguishes signal from
        # rescaled noise, and nothing else records it.
        advantages = self.rollout_buffer.advantages.flatten()
        super().train()
        recorded = self.logger.name_to_value
        self.last_train_stats = {name: recorded.get(f"train/{name}") for name in self.TRAIN_STATS}
        self.last_train_stats["advantage_std"] = float(np.std(advantages))
        self.last_train_stats["advantage_abs_mean"] = float(np.mean(np.abs(advantages)))

    def _excluded_save_params(self) -> list[str]:
        # The fault hook usually holds a game process handle; it belongs to the running session, not the checkpoint.
        return [*super()._excluded_save_params(), "on_fault"]

    def collect_rollouts(self, env: VecEnv, callback, rollout_buffer, n_rollout_steps: int) -> bool:
        consecutive = 0
        while True:
            timesteps_before = self.num_timesteps
            episode_infos_before = deque(self.ep_info_buffer, maxlen=self.ep_info_buffer.maxlen)
            episode_successes_before = deque(self.ep_success_buffer, maxlen=self.ep_success_buffer.maxlen)
            try:
                completed = super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
            except BridgeFault as fault:
                discarded = self.num_timesteps - timesteps_before
                self.fault_stats["discarded_transitions"] += discarded
                self.fault_stats["discarded_rollouts"] += 1
                self._record(fault, discarded, timesteps_before)
                self.num_timesteps = timesteps_before
                self.ep_info_buffer, self.ep_success_buffer = episode_infos_before, episode_successes_before
                rollout_buffer.reset()
                consecutive = self._recover(env, callback, fault, consecutive + 1)
                continue
            self.fault_stats["accepted_transitions"] += self.num_timesteps - timesteps_before
            return completed

    def _record(self, fault: BridgeFault, discarded: int, timesteps: int, stage: str = "rollout") -> None:
        self.fault_stats["faults"].append({"stage": stage, "error": str(fault), "discarded_transitions": discarded,
                                           "timesteps": timesteps})

    def _recover(self, env: VecEnv, callback, fault: BridgeFault, consecutive: int) -> int:
        """Reset every slot, retrying a faulting reset, until it succeeds or the consecutive limit is reached.
        Returns the consecutive failure count so far."""
        while True:
            if consecutive >= self.max_consecutive_discards:
                self._last_obs = None
                saved = ""
                if self.abort_checkpoint_path is not None:
                    # Weights and optimizer state are those of the last completed update.
                    self.save(self.abort_checkpoint_path)
                    saved = f"; model saved to {self.abort_checkpoint_path}"
                callback.on_training_end()
                raise TrainingAborted(
                    f"{consecutive} consecutive bridge faults without a completed rollout{saved}; last: {fault}") from fault
            if self.on_fault is not None:
                self.on_fault(fault)
            try:
                self._last_obs = env.reset()
            except BridgeFault as reset_fault:
                self._record(reset_fault, 0, self.num_timesteps, stage="recovery reset")
                fault, consecutive = reset_fault, consecutive + 1
                continue
            self._last_episode_starts = np.ones((env.num_envs,), dtype=bool)
            return consecutive
