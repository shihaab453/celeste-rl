"""A Phase 3 training run: SupervisedPPO on CelesteRoomEnv with durable records (roadmap Phase 3, spec section 9).

A run directory holds everything needed to audit or resume the run:

  manifest.json      config, git state, runtime manifest, schema versions; status and fault statistics, rewritten
                     at every rollout boundary and at the end
  progress.csv       one row per accepted rollout: accepted steps, episodes and endings, success rate, returns,
                     reward components, environment steps per second
  episodes.jsonl     one line per finished episode from an accepted rollout: length, ending, return, components
  evaluations.jsonl  one line per evaluation: stochastic success rate and clear times, deterministic episode
  checkpoints/       latest.zip, previous.zip, best.zip, step_<accepted>.zip, aborted.zip; each written to a
                     temporary file and renamed, so a crash never leaves a half-written checkpoint

Everything counts accepted transitions only (SupervisedPPO rolls the step counter back on a discarded rollout):
episodes that finish inside a discarded rollout are never written, and checkpoints and evaluations are scheduled on
the accepted step count after the update, not on SB3's callback call count.

Evaluation runs between rollouts on the training environment. It abandons the training episode in progress, which
fabricates nothing: PPO has already bootstrapped that episode at the rollout boundary, and the episode is not
recorded. Its stochastic episodes all start from the canonical start, so they measure reliability of the policy's
sampling at one start, not generalisation over starts (held-out entry states are a later step).
"""
from __future__ import annotations

import csv
import json
import os
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from celeste_rl.endings import SUCCESS
from celeste_rl.env import CelesteRoomEnv
from celeste_rl.schema import MENU_INPUTS
from celeste_rl.training.policy import policy_kwargs
from celeste_rl.training.supervisor import SupervisedPPO, TrainingAborted


@dataclass
class TrainConfig:
    """Starting values for the unshaped room 1 baseline, not tuned results."""

    seed: int = 0
    total_timesteps: int = 2_000_000  # accepted transitions (roadmap campaign budget per seed)
    n_steps: int = 2048
    batch_size: int = 512
    n_epochs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 1.0  # decision D2
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    device: str = "cpu"
    disabled_inputs: tuple[str, ...] = MENU_INPUTS  # decision D1
    # rew-v1 is the unshaped baseline; rew-v2 adds the unspent-deadline charge and progress shaping (Codex K1).
    reward_version: str = "rew-v1"
    shaping_scale: float = 0.2  # rew-v2 only
    max_consecutive_discards: int = 3
    checkpoint_every: int = 50_000  # accepted steps
    eval_every: int = 250_000  # accepted steps; 0 disables
    eval_episodes: int = 20


def _atomic_save(model: SupervisedPPO, path: Path) -> None:
    temporary = path.with_name(path.stem + ".tmp.zip")
    model.save(temporary)
    os.replace(temporary, path)


def _atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def run_episode(model: SupervisedPPO, env: CelesteRoomEnv, deterministic: bool) -> dict:
    obs, _ = env.reset()
    total, components, length = 0.0, Counter(), 0
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        length += 1
        total += reward
        components.update(info["reward_components"])
        if terminated or truncated:
            return {"ending": info["ending"], "length": length, "return": total, "components": dict(components)}


@dataclass
class _RolloutStats:
    started: float = field(default_factory=time.perf_counter)
    episodes: list = field(default_factory=list)


class RunRecorder(BaseCallback):
    """Writes episodes, progress, checkpoints and evaluations, counting accepted rollouts only.

    SB3 calls on_rollout_start at every collection attempt and on_rollout_end only when a rollout completes, so a
    rollout discarded by SupervisedPPO never reaches on_rollout_end: its buffered episodes are dropped at the next
    on_rollout_start, and episodes in progress at the fault are abandoned (the environment was reset).
    """

    def __init__(self, run_dir: Path, config: TrainConfig, env: CelesteRoomEnv, write_manifest: Callable[[str], None],
                 health: Callable[[], dict] | None = None):
        super().__init__()
        self.run_dir, self.config, self.env, self.write_manifest = run_dir, config, env, write_manifest
        self.health = health
        self.checkpoints = run_dir / "checkpoints"
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self._in_rollout = False
        self._rollout = _RolloutStats()
        self._running: dict[int, dict] = {}
        self.next_checkpoint = config.checkpoint_every
        self.next_eval = config.eval_every if config.eval_every > 0 else None
        self.best: tuple | None = None
        self.accepted_episodes = 0

    # Scheduling state that must survive a resume.
    def state(self) -> dict:
        return {"next_checkpoint": self.next_checkpoint, "next_eval": self.next_eval, "best": self.best,
                "accepted_episodes": self.accepted_episodes}

    def load_state(self, state: dict) -> None:
        self.next_checkpoint, self.next_eval = state["next_checkpoint"], state["next_eval"]
        self.best = tuple(state["best"]) if state["best"] is not None else None
        self.accepted_episodes = state["accepted_episodes"]

    def _on_rollout_start(self) -> None:
        if self._in_rollout:
            # The previous attempt faulted: its episodes are discarded and the ones in progress were abandoned.
            self._running.clear()
        self._in_rollout = True
        steps = self.model.num_timesteps  # accepted and already used by an update
        if steps >= self.next_checkpoint:
            self._checkpoint(steps)
            while self.next_checkpoint <= steps:
                self.next_checkpoint += self.config.checkpoint_every
        if self.next_eval is not None and steps >= self.next_eval:
            self._evaluate(steps)  # a fault here is a rollout fault; the evaluation reruns on the retry
            while self.next_eval <= steps:
                self.next_eval += self.config.eval_every
        self._rollout = _RolloutStats()  # started after any checkpoint or evaluation, so steps per second is collection only

    def _on_step(self) -> bool:
        infos, dones = self.locals["infos"], self.locals["dones"]
        for index, (info, done) in enumerate(zip(infos, dones)):
            episode = self._running.setdefault(index, {"length": 0, "return": 0.0, "components": Counter()})
            episode["length"] += 1
            episode["return"] += float(self.locals["rewards"][index])
            episode["components"].update(info["reward_components"])
            if done:
                self._rollout.episodes.append({"ending": info["ending"], "length": episode["length"],
                                               "return": episode["return"], "components": dict(episode["components"])})
                del self._running[index]
        return True

    def _on_rollout_end(self) -> None:
        self._in_rollout = False
        seconds = time.perf_counter() - self._rollout.started
        episodes = self._rollout.episodes
        with (self.run_dir / "episodes.jsonl").open("a", encoding="utf-8") as handle:
            for episode in episodes:
                self.accepted_episodes += 1
                handle.write(json.dumps({"index": self.accepted_episodes, "accepted_steps": self.model.num_timesteps,
                                         **episode}) + "\n")
        endings = Counter(e["ending"] for e in episodes)
        successes = [e for e in episodes if e["ending"] == SUCCESS]
        row = {
            "accepted_steps": self.model.num_timesteps,
            "episodes": len(episodes),
            "success_rate": len(successes) / len(episodes) if episodes else "",
            "mean_return": float(np.mean([e["return"] for e in episodes])) if episodes else "",
            "mean_success_length": float(np.mean([e["length"] for e in successes])) if successes else "",
            **{f"ending_{name}": endings.get(name, 0) for name in ("success", "death", "restart", "left_level", "wrong_room", "timeout")},
            # From the reward version, so a version that adds a component records it instead of dropping it.
            **{f"component_{name}": sum(e["components"].get(name, 0.0) for e in episodes)
               for name in self.env.reward_config.components},
            "env_steps_per_second": self.config.n_steps / seconds if seconds > 0 else "",
            "discarded_rollouts": self.model.fault_stats["discarded_rollouts"],
            # Process health for long runs (for example the game's memory); the keys must not change during a run.
            **(self.health() if self.health is not None else {}),
        }
        path = self.run_dir / "progress.csv"
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if new:
                writer.writeheader()
            writer.writerow(row)
        self.write_manifest("running")

    def _checkpoint(self, steps: int) -> None:
        latest, previous = self.checkpoints / "latest.zip", self.checkpoints / "previous.zip"
        if latest.exists():
            os.replace(latest, previous)
        _atomic_save(self.model, latest)
        _atomic_save(self.model, self.checkpoints / f"step_{steps:09d}.zip")
        self.write_manifest("running")

    def _evaluate(self, steps: int) -> None:
        stochastic = [run_episode(self.model, self.env, deterministic=False) for _ in range(self.config.eval_episodes)]
        deterministic = run_episode(self.model, self.env, deterministic=True)
        successes = [e["length"] for e in stochastic if e["ending"] == SUCCESS]
        record = {
            "accepted_steps": steps,
            "stochastic_episodes": len(stochastic),
            "stochastic_success_rate": len(successes) / len(stochastic) if stochastic else 0.0,
            "stochastic_endings": dict(Counter(e["ending"] for e in stochastic)),
            "stochastic_success_lengths": successes,
            "deterministic": deterministic,
        }
        with (self.run_dir / "evaluations.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        # Best: highest success rate, then shortest mean successful clear.
        score = (record["stochastic_success_rate"], -float(np.mean(successes)) if successes else -1e9)
        if self.best is None or score > tuple(self.best):
            self.best = score
            _atomic_save(self.model, self.checkpoints / "best.zip")
        # Evaluation used the training environment: start a fresh training episode.
        self.model._last_obs = self.model.env.reset()
        self.model._last_episode_starts = np.ones((self.model.env.num_envs,), dtype=bool)
        self._running.clear()


def train(config: TrainConfig, run_dir: Path, env: CelesteRoomEnv, provenance: dict,
          on_fault: Callable | None = None, resume: bool = False, health: Callable[[], dict] | None = None) -> SupervisedPPO:
    """Train, or resume from run_dir/checkpoints/latest.zip. Raises TrainingAborted after repeated bridge faults,
    with the model saved to checkpoints/aborted.zip."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    checkpoints = run_dir / "checkpoints"
    started = time.strftime("%Y-%m-%dT%H:%M:%S")

    if resume:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["config"] != json.loads(json.dumps(asdict(config))):
            raise ValueError("The resume config differs from the run's config")
        model = SupervisedPPO.load(checkpoints / "latest.zip", env=env, device=config.device,
                                   max_consecutive_discards=config.max_consecutive_discards)
        model.fault_stats = previous["fault_stats"]
        # The first resumed transition must start from a fresh reset, not an observation from when the checkpoint was
        # saved. SB3's load already clears it (force_reset); kept explicit so a library change cannot undo it.
        model._last_obs = None
        history = previous.get("sessions", [])
    else:
        if manifest_path.exists():
            raise FileExistsError(f"{run_dir} already holds a run; resume it or choose another directory")
        model = SupervisedPPO(
            "MultiInputPolicy", env, policy_kwargs=policy_kwargs(), n_steps=config.n_steps, batch_size=config.batch_size,
            n_epochs=config.n_epochs, learning_rate=config.learning_rate, gamma=config.gamma, gae_lambda=config.gae_lambda,
            clip_range=config.clip_range, ent_coef=config.ent_coef, vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm, seed=config.seed, device=config.device,
            max_consecutive_discards=config.max_consecutive_discards)
        history = []
        previous = None
    model.on_fault = on_fault
    model.abort_checkpoint_path = checkpoints / "aborted.zip"

    recorder: RunRecorder | None = None

    def write_manifest(status: str) -> None:
        _atomic_json(manifest_path, {
            "status": status,
            "config": asdict(config),
            "accepted_steps": model.num_timesteps,
            "fault_stats": model.fault_stats,
            "recorder": recorder.state() if recorder is not None else None,
            "sessions": history + [{"started": started, "provenance": provenance, "resumed_at": resume_steps}],
        })

    resume_steps = model.num_timesteps
    recorder = RunRecorder(run_dir, config, env, write_manifest, health)
    recorder.init_callback(model)  # also needed when a resumed run has nothing left to learn
    if previous is not None and previous.get("recorder"):
        recorder.load_state(previous["recorder"])
    write_manifest("running")
    try:
        # SB3 adds the steps already taken to total_timesteps when not resetting the counter, so pass the remainder.
        remaining = config.total_timesteps - model.num_timesteps if resume else config.total_timesteps
        if remaining > 0:
            model.learn(remaining, callback=recorder, reset_num_timesteps=not resume)
    except TrainingAborted:
        write_manifest("aborted")
        raise
    except BaseException:
        write_manifest("crashed")
        raise
    recorder._checkpoint(model.num_timesteps)
    write_manifest("finished")
    return model
