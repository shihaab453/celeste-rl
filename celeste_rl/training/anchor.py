"""A Room 1 anchor inside PPO: keep the policy near a frozen donor's own action probabilities on recorded frames.

Why: in the mixed self-distillation pilot, cloning kept a donor's Room 1 skill (median retained share 0.78), and
500,000 PPO steps on Room 2 then wore it away (median one-unit drop 0.87), because nothing in PPO refers to Room 1.
The anchor adds one term to every PPO minibatch: on a batch of the donor's own recorded Room 1 frames, the binary
cross entropy between the policy's logits and the donor's probabilities, over the enabled inputs only, times a fixed
lambda. It is the soft-target loss the coverage-ceiling copies and the SD-k clones were fitted with.

Design (reviewed by the orchestrator, 2026-09-27, with its required changes):
- **Frames and targets.** The donor's recording, verified with `celeste_rl.policy_play.verified_play` against the
  donor's sha256; only the fitted episodes of the seed-k split (the held-back ones are the early warning). Targets
  are the frozen donor's probabilities, computed once before training; no donor network is kept.
- **(a) Disabled inputs are masked.** The donor gives the disabled menu inputs 0.15 to 0.20 while `distilled()`
  sets them to 0, so an unmasked anchor would pull even at the donor itself. Masked, the anchor's excess (BCE minus
  the targets' own entropy, the mean per-input KL from donor to policy) is 0 at the donor, and so is its gradient.
- **How it joins SB3 without copying its code.** SB3 2.9.0 `PPO.train` runs, per minibatch: the PPO forward pass
  and loss, the target_kl break, `optimizer.zero_grad()`, `loss.backward()`, `clip_grad_norm_`, `optimizer.step()`.
  The anchor's backward runs right after `zero_grad`, so SB3's own backward adds the PPO gradient, the clip acts on
  the sum and the step follows: exactly the gradient of PPO loss + lambda * anchor loss. The order is pinned by the
  sha256 of `PPO.train`'s source; any other SB3 refuses to train with an anchor.
- **(b)** The hooks exist only inside `train()` and are removed in a `finally`.
- **(c)** Each anchor batch is drawn from (seed, SB3's update counter, minibatch index), so a run resumed from
  `latest.zip` draws the same batches as an uninterrupted one.
- **(d)** The anchor frames are never saved in a checkpoint; anchored checkpoints load with plain
  `SupervisedPPO.load`. Nothing the evaluators import imports this module.
- **(e)** On the first minibatch of each update: the anchor-only gradient norm (as applied, lambda included), the
  PPO-only gradient norm (the clipped sum minus the anchor part, measured before the clip) and the combined norm
  before clipping. With the mean anchor excess and the policy's agreement with the donor on the held-back episodes,
  one row per update goes to `anchor.csv` in the run directory.
"""
from __future__ import annotations

import csv
import hashlib
import inspect
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3 import PPO
from torch.nn import functional as F

from celeste_rl.cloning import Demonstrations, distilled, split_by_trajectory
from celeste_rl.policy_play import verified_play
from celeste_rl.schema import ACTION_INPUTS, MENU_INPUTS
from celeste_rl.training.supervisor import SupervisedPPO

# sha256 of the source of stable_baselines3 2.9.0 PPO.train, whose call order the hook relies on.
PPO_TRAIN_SHA256 = "7c8d09940123242f0ac401c5ec2cc858b8bf3339f571c222382f048ffd8306ae"
DISABLED = [ACTION_INPUTS.index(name) for name in MENU_INPUTS]
ENABLED = [i for i in range(len(ACTION_INPUTS)) if i not in DISABLED]
ANCHOR_CSV_FIELDS = ("update", "minibatches", "anchor_excess_mean", "anchor_grad_norm", "ppo_grad_norm",
                     "combined_grad_norm_before_clip", "heldback_input_agreement", "heldback_frame_agreement")


def ppo_train_source_sha256() -> str:
    return hashlib.sha256(inspect.getsource(PPO.train).encode()).hexdigest()


@dataclass
class Anchor:
    """The frozen donor's targets on its fitted recorded frames, the held-back frames, and the anchor settings."""

    fitted: Demonstrations
    held: Demonstrations
    coef: float
    seed: int
    record: dict = field(default_factory=dict)
    run_dir: Path | None = None

    def __len__(self) -> int:
        return len(self.fitted)

    def obs(self, data: Demonstrations, index) -> dict[str, th.Tensor]:
        return {key: th.as_tensor(value[index]) for key, value in data.obs.items()}


def build_anchor(play: Path, donor: Path, donor_sha256: str, identity: dict, holdout: float, split_seed: int,
                 coef: float, seed: int) -> Anchor:
    """Verify the recording against the donor, split it by episode as the SD-k clones were, and take the donor's
    probabilities on both parts as targets (disabled inputs 0; they are masked out of the loss anyway)."""
    if hashlib.sha256(Path(donor).read_bytes()).hexdigest() != donor_sha256:
        raise ValueError(f"{donor} does not match the pinned sha256")
    data, states = verified_play(Path(play), donor_sha256, identity)
    fitted, held = split_by_trajectory(data, holdout, split_seed)
    donor_policy = SupervisedPPO.load(donor, device="cpu").policy
    record = {"play": states, "donor": Path(donor).as_posix(), "donor_sha256": donor_sha256, "holdout": holdout,
              "split_seed": split_seed, "coef": coef, "seed": seed, "fitted_frames": len(fitted),
              "held_frames": len(held), "fitted_episodes": int(len(fitted.trajectories)),
              "held_episodes": int(len(held.trajectories)), "ppo_train_sha256": PPO_TRAIN_SHA256}
    return Anchor(distilled(donor_policy, fitted, DISABLED), distilled(donor_policy, held, DISABLED), coef, seed,
                  record)


def anchor_loss(policy, obs: dict[str, th.Tensor], targets: np.ndarray) -> tuple[th.Tensor, float]:
    """(BCE over the enabled inputs, the mean excess over the targets' own entropy). Both are 0-minimal at targets."""
    logits = policy.get_distribution(obs).distribution.logits[:, ENABLED]
    target = th.as_tensor(targets[:, ENABLED], dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    with th.no_grad():
        clipped = target.clamp(1e-7, 1 - 1e-7)
        entropy = -(target * clipped.log() + (1 - target) * (1 - clipped).log()).mean()
    return bce, float(bce.detach() - entropy)


def batch_indices(size: int, batch: int, seed: int, update: int, minibatch: int) -> np.ndarray:
    """A resume-stable anchor batch: a function of (seed, update counter, minibatch index) only."""
    return np.random.default_rng([seed, update, minibatch]).integers(0, size, min(batch, size))


def heldback_agreement(policy, anchor: Anchor) -> tuple[float, float]:
    """How often the policy's most likely buttons equal the donor's, on the held-back episodes (enabled inputs)."""
    policy.set_training_mode(False)
    with th.no_grad():
        probs = policy.get_distribution(anchor.obs(anchor.held, slice(None))).distribution.probs.numpy()[:, ENABLED]
    same = (probs >= 0.5) == (anchor.held.targets[:, ENABLED] >= 0.5)
    return round(float(same.mean()), 4), round(float(same.all(axis=1).mean()), 4)


def _norm(grads: list[th.Tensor | None]) -> float:
    present = [g for g in grads if g is not None]
    return float(th.linalg.vector_norm(th.stack([th.linalg.vector_norm(g) for g in present]))) if present else 0.0


class AnchoredPPO(SupervisedPPO):
    """SupervisedPPO whose every minibatch also fits the anchor. `pending` is attached by the first train() call,
    which is how scripts/train_anchored.py hands the anchor to models that run.train builds or loads."""

    pending: Anchor | None = None

    def __init__(self, *args, anchor: Anchor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.anchor = anchor

    def _excluded_save_params(self) -> list[str]:
        return [*super()._excluded_save_params(), "anchor"]

    def train(self) -> None:
        if getattr(self, "anchor", None) is None:
            self.anchor = AnchoredPPO.pending
        anchor = self.anchor
        if anchor is None:
            raise RuntimeError("AnchoredPPO has no anchor; use SupervisedPPO for unanchored training")
        if ppo_train_source_sha256() != PPO_TRAIN_SHA256:
            raise RuntimeError("stable_baselines3 PPO.train differs from the pinned source; the anchor hook depends "
                               "on its call order, so anchored training refuses")
        update = int(self._n_updates)
        state, restore = self._install_hooks(anchor, update)
        try:
            super().train()
        finally:
            restore()
        agree_inputs, agree_frames = heldback_agreement(self.policy, anchor)
        self.last_anchor_row = {"update": update, "minibatches": state["minibatch"],
                                "anchor_excess_mean": round(float(np.mean(state["excess"])), 6)
                                if state["excess"] else "",
                                "anchor_grad_norm": state["anchor_norm"], "ppo_grad_norm": state["ppo_norm"],
                                "combined_grad_norm_before_clip": state["combined_norm"],
                                "heldback_input_agreement": agree_inputs, "heldback_frame_agreement": agree_frames}
        if anchor.run_dir is not None:
            path = Path(anchor.run_dir) / "anchor.csv"
            new = not path.exists()
            with path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(ANCHOR_CSV_FIELDS))
                if new:
                    writer.writeheader()
                writer.writerow(self.last_anchor_row)

    def _install_hooks(self, anchor: Anchor, update: int):
        """Hook the optimizer's zero_grad and torch's clip_grad_norm_ for one update. Returns (state, restore)."""
        params = list(self.policy.parameters())
        optimizer = self.policy.optimizer
        original_zero_grad = optimizer.zero_grad
        original_clip = th.nn.utils.clip_grad_norm_
        state = {"minibatch": 0, "excess": [], "anchor_grads": None, "anchor_norm": None, "ppo_norm": None,
                 "combined_norm": None}

        def zero_grad(*args, **kwargs):
            original_zero_grad(*args, **kwargs)
            index = batch_indices(len(anchor), self.batch_size, anchor.seed, update, state["minibatch"])
            bce, excess = anchor_loss(self.policy, anchor.obs(anchor.fitted, index), anchor.fitted.targets[index])
            (anchor.coef * bce).backward()
            if state["minibatch"] == 0:
                state["anchor_grads"] = [None if p.grad is None else p.grad.detach().clone() for p in params]
                state["anchor_norm"] = _norm(state["anchor_grads"])
            state["excess"].append(excess)
            state["minibatch"] += 1

        def clip_grad_norm_(parameters, max_norm, *args, **kwargs):
            parameters = list(parameters)
            first = state["minibatch"] == 1 and state["combined_norm"] is None
            if first:
                ppo = [None if p.grad is None else (p.grad if a is None else p.grad - a)
                       for p, a in zip(parameters, state["anchor_grads"])]
                state["ppo_norm"] = _norm(ppo)
            total = original_clip(parameters, max_norm, *args, **kwargs)
            if first:
                state["combined_norm"] = float(total)
            return total

        def restore() -> None:
            optimizer.__dict__.pop("zero_grad", None)  # back to the optimizer class's own method
            th.nn.utils.clip_grad_norm_ = original_clip

        optimizer.zero_grad = zero_grad
        th.nn.utils.clip_grad_norm_ = clip_grad_norm_
        return state, restore
