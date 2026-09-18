"""Behavioural cloning for Phase 3B: fit the training policy to demonstrated play.

Behavioural cloning is supervised learning, not reinforcement learning. It never uses a reward: it is shown
states paired with the action a demonstration took there and learns to reproduce them. That makes it immune to
everything that stopped Phase 3 (the flat advantages, the near-uniform sampling), and vulnerable to something
else, **distribution shift**: a cloned policy only ever saw states on the demonstrated path, so its first small
mistake puts it somewhere it has no idea what to do, and errors compound from there. That is why the honest
measurement is not accuracy on the demonstrated frames but whether the policy clears the room when it plays.

Two measurements, and they answer different questions:

- **Held-out trajectory accuracy.** Whole demonstrations are kept out of training, never individual frames.
  Splitting by frame would leak: consecutive frames of one route are nearly identical, so a frame-wise split
  measures memorisation and reads far too high. `scripts/route_fit_check.py` deliberately fits and measures on
  the same frames, because it was asking about capacity; this is asking about generalisation, so it must not.
- **Playing the game.** Done by the caller under the Phase 3 evaluation protocol, because only that answers the
  question the project is actually asking.

The policy built here is exactly the one training uses (`CelestePolicy`, `policy_kwargs()`), so its weights can
be handed to PPO for fine-tuning afterwards.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch as th
from torch.nn import functional as F

from celeste_rl.schema import ACTION_INPUTS, MENU_INPUTS

OBS_KEYS = ("player", "actions", "history_valid", "grid", "context")


@dataclass
class Demonstrations:
    """Observation and action pairs, grouped by the trajectory they came from.

    `trajectory` gives each frame's demonstration index, which is what makes a leak-free split possible.
    """

    obs: dict[str, np.ndarray]
    actions: np.ndarray
    trajectory: np.ndarray
    provenance: list[dict] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.actions)

    @property
    def trajectories(self) -> np.ndarray:
        return np.unique(self.trajectory)

    def subset(self, wanted) -> "Demonstrations":
        mask = np.isin(self.trajectory, list(wanted))
        return Demonstrations({key: value[mask] for key, value in self.obs.items()}, self.actions[mask],
                              self.trajectory[mask], self.provenance)

    def inputs_per_frame(self) -> float:
        return float(self.actions.sum(axis=1).mean())


def split_by_trajectory(data: Demonstrations, holdout: float, seed: int) -> tuple[Demonstrations, Demonstrations]:
    """Keep whole demonstrations back, never individual frames: see the module docstring."""
    trajectories = data.trajectories
    if len(trajectories) < 2:
        raise ValueError(f"need at least 2 demonstrations to hold one out, got {len(trajectories)}")
    order = np.random.default_rng(seed).permutation(trajectories)
    kept = max(1, round(len(trajectories) * holdout))
    return data.subset(order[kept:]), data.subset(order[:kept])


def accuracy(policy, data: Demonstrations, batch_size: int = 512) -> dict:
    """How often the policy's most likely action matches the demonstration, on these frames."""
    enabled = th.as_tensor([name not in MENU_INPUTS for name in ACTION_INPUTS])
    targets = th.as_tensor(data.actions).float()
    correct = []
    policy.set_training_mode(False)
    with th.no_grad():
        for start in range(0, len(data), batch_size):
            stop = start + batch_size
            batch = {key: th.as_tensor(value[start:stop]) for key, value in data.obs.items()}
            predicted = (policy.get_distribution(batch).distribution.logits > 0).float()
            correct.append(predicted == targets[start:stop])
    correct = th.cat(correct)
    zeros = (targets[:, enabled] == 0).float()
    return {
        "frames": len(data),
        "trajectories": int(len(data.trajectories)),
        "input_accuracy": round(correct[:, enabled].float().mean().item(), 4),
        "frame_accuracy": round(correct[:, enabled].all(dim=1).float().mean().item(), 4),
        "always_zero_input_accuracy": round(zeros.mean().item(), 4),
        "always_zero_frame_accuracy": round(zeros.all(dim=1).float().mean().item(), 4),
        "inputs_per_frame": round(data.inputs_per_frame(), 2),
    }


def clone(policy, train: Demonstrations, holdout: Demonstrations | None, epochs: int, batch_size: int,
          learning_rate: float, seed: int, report_every: int = 10) -> dict:
    """Fit `policy` to the demonstrated actions with binary cross entropy. Returns the fitting history."""
    th.manual_seed(seed)
    optimizer = th.optim.Adam(policy.parameters(), lr=learning_rate)
    observations = {key: th.as_tensor(value) for key, value in train.obs.items()}
    targets = th.as_tensor(train.actions).float()
    generator = th.Generator().manual_seed(seed)
    history = []
    for epoch in range(epochs):
        policy.set_training_mode(True)
        order = th.randperm(len(train), generator=generator)
        losses = []
        for start in range(0, len(train), batch_size):
            index = order[start:start + batch_size]
            batch = {key: value[index] for key, value in observations.items()}
            loss = F.binary_cross_entropy_with_logits(
                policy.get_distribution(batch).distribution.logits, targets[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if epoch % report_every == 0 or epoch == epochs - 1:
            entry = {"epoch": epoch, "loss": round(float(np.mean(losses)), 5),
                     "train": accuracy(policy, train)}
            if holdout is not None:
                entry["holdout"] = accuracy(policy, holdout)
            history.append(entry)
    return {"epochs": epochs, "history": history}
