"""Offline checks of the Room 1 anchor inside PPO (celeste_rl/training/anchor.py): no game, a replayed bridge."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch as th
from stable_baselines3 import PPO
from torch.nn import functional as F

from celeste_rl.env import CelesteRoomEnv
from celeste_rl.tasks import resolve_task_definition, task_identity
from celeste_rl.training import anchor as anchor_module
from celeste_rl.training.anchor import (
    DISABLED,
    ENABLED,
    Anchor,
    AnchoredPPO,
    anchor_loss,
    batch_indices,
    build_anchor,
)
from celeste_rl.cloning import distilled, split_by_trajectory
from celeste_rl.training.policy import CelestePolicy, policy_kwargs
from celeste_rl.training.supervisor import SupervisedPPO
from tests.test_cloning import _spaces_env, demonstrations
from tests.test_training import EpochBridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

SETTINGS = dict(policy_kwargs=policy_kwargs(), n_steps=32, batch_size=16, n_epochs=2, gamma=1.0, device="cpu", seed=0)


def donor_policy(seed: int = 5, action_bias: float = 0.0):
    return PPO(CelestePolicy, _spaces_env(), policy_kwargs=policy_kwargs(action_bias), device="cpu", seed=seed).policy


def toy_anchor(coef: float, donor=None, seed: int = 3) -> Anchor:
    donor = donor if donor is not None else donor_policy()
    fitted, held = split_by_trajectory(demonstrations(4, 20), 0.25, 0)
    return Anchor(distilled(donor, fitted, DISABLED), distilled(donor, held, DISABLED), coef, seed)


def weights(model) -> dict:
    return {k: v.clone() for k, v in model.policy.state_dict().items()}


class LossTests(unittest.TestCase):
    def test_masked_anchor_is_zero_at_the_donor_and_unmasked_is_not(self):
        donor = donor_policy()
        anchor = toy_anchor(1.0, donor)
        obs, targets = anchor.obs(anchor.fitted, slice(None)), anchor.fitted.targets
        bce, excess = anchor_loss(donor, obs, targets)
        donor.zero_grad()
        bce.backward()
        norm = th.linalg.vector_norm(th.stack([p.grad.norm() for p in donor.parameters() if p.grad is not None]))
        self.assertLess(abs(excess), 1e-5)
        self.assertLess(float(norm), 1e-4)
        # Without the mask, the disabled inputs' targets (0) disagree with the donor's own probabilities.
        logits = donor.get_distribution(obs).distribution.logits
        unmasked = F.binary_cross_entropy_with_logits(logits, th.as_tensor(targets))
        donor.zero_grad()
        unmasked.backward()
        unmasked_norm = th.linalg.vector_norm(th.stack([p.grad.norm() for p in donor.parameters() if p.grad is not None]))
        self.assertGreater(float(unmasked_norm), 100 * float(norm))

    def test_entropy_bits_counts_fair_coin_tosses(self):
        from celeste_rl.training.anchor import entropy_bits
        self.assertAlmostEqual(entropy_bits(np.full((5, 3), 0.5)), 3.0)  # three fair coins per frame
        self.assertAlmostEqual(entropy_bits(np.zeros((5, 3))), 0.0, places=6)
        self.assertAlmostEqual(entropy_bits(np.ones((5, 3))), 0.0, places=6)

    def test_only_enabled_inputs_are_scored(self):
        self.assertEqual(sorted(ENABLED + DISABLED), list(range(len(ENABLED) + len(DISABLED))))
        self.assertFalse(set(ENABLED) & set(DISABLED))

    def test_batches_depend_only_on_seed_update_and_minibatch(self):
        first = batch_indices(100, 16, 3, 7, 2)
        self.assertTrue(np.array_equal(first, batch_indices(100, 16, 3, 7, 2)))
        self.assertFalse(np.array_equal(first, batch_indices(100, 16, 3, 8, 2)))
        self.assertFalse(np.array_equal(first, batch_indices(100, 16, 3, 7, 3)))


class HookTests(unittest.TestCase):
    def test_one_hooked_step_equals_a_joint_loss_step(self):
        """Drive the real hooks through SB3's pinned order: zero_grad, backward, clip, step."""
        anchor = toy_anchor(2.0)
        model = AnchoredPPO(CelestePolicy, CelesteRoomEnv(EpochBridge()), anchor=anchor, **SETTINGS)
        reference = copy.deepcopy(model.policy)
        obs = anchor.obs(anchor.held, slice(None))

        def ppo_like_loss(policy):  # any differentiable stand-in for the PPO loss
            return policy.get_distribution(obs).distribution.logits.pow(2).mean() + policy.predict_values(obs).mean()

        state, restore = model._install_hooks(anchor, update=3)
        try:
            loss = ppo_like_loss(model.policy)
            model.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            model.policy.optimizer.step()
        finally:
            restore()

        index = batch_indices(len(anchor), SETTINGS["batch_size"], anchor.seed, 3, 0)
        bce, _ = anchor_loss(reference, anchor.obs(anchor.fitted, index), anchor.fitted.targets[index])
        total = ppo_like_loss(reference) + anchor.coef * bce
        reference.optimizer.zero_grad()
        total.backward()
        combined = float(th.nn.utils.clip_grad_norm_(reference.parameters(), 0.5))
        reference.optimizer.step()
        for (name, a), b in zip(model.policy.state_dict().items(), reference.state_dict().values()):
            self.assertTrue(th.allclose(a, b, atol=1e-6), name)
        self.assertAlmostEqual(state["combined_norm"], combined, places=4)
        self.assertEqual(state["minibatch"], 1)
        self.assertIsNotNone(state["ppo_norm"])
        self.assertNotIn("zero_grad", model.policy.optimizer.__dict__)

    def test_lambda_zero_trains_exactly_like_unanchored_ppo(self):
        anchor = toy_anchor(0.0)  # built first: making its donor re-seeds the global generators
        # Each model is built (which seeds the global generators) and trained before the next, so both learn from
        # the same random stream.
        plain = SupervisedPPO(CelestePolicy, CelesteRoomEnv(EpochBridge()), **SETTINGS)
        plain.learn(96)
        anchored = AnchoredPPO(CelestePolicy, CelesteRoomEnv(EpochBridge()), anchor=anchor, **SETTINGS)
        anchored.learn(96)
        for (name, a), b in zip(weights(plain).items(), weights(anchored).values()):
            self.assertTrue(th.equal(a, b), name)

    def test_the_anchor_pulls_the_policy_toward_the_donor(self):
        # A donor that prefers about 10% per input, unlike a fresh network's 50%, so there is something to pull toward.
        anchor = toy_anchor(10.0, donor_policy(action_bias=-2.2))
        model = AnchoredPPO(CelestePolicy, CelesteRoomEnv(EpochBridge()), anchor=anchor, **SETTINGS)
        everything = anchor.obs(anchor.fitted, slice(None))
        _, before = anchor_loss(model.policy, everything, anchor.fitted.targets)
        model.learn(160)
        _, after = anchor_loss(model.policy, everything, anchor.fitted.targets)
        self.assertLess(after, before)
        self.assertGreater(model.last_anchor_row["anchor_grad_norm"], 0)

    def test_a_different_ppo_train_refuses_and_leaves_no_hooks(self):
        model = AnchoredPPO(CelestePolicy, CelesteRoomEnv(EpochBridge()), anchor=toy_anchor(1.0), **SETTINGS)
        original_clip = th.nn.utils.clip_grad_norm_
        with mock.patch.object(anchor_module, "PPO_TRAIN_SHA256", "0" * 64), \
                self.assertRaisesRegex(RuntimeError, "differs from the pinned source"):
            model.learn(32)
        self.assertIs(th.nn.utils.clip_grad_norm_, original_clip)
        self.assertNotIn("zero_grad", model.policy.optimizer.__dict__)

    def test_hooks_are_removed_when_the_update_raises(self):
        model = AnchoredPPO(CelestePolicy, CelesteRoomEnv(EpochBridge()), anchor=toy_anchor(1.0), **SETTINGS)
        original_clip = th.nn.utils.clip_grad_norm_
        with mock.patch.object(SupervisedPPO, "train", side_effect=RuntimeError("boom")), \
                self.assertRaisesRegex(RuntimeError, "boom"):
            model.learn(32)
        self.assertIs(th.nn.utils.clip_grad_norm_, original_clip)
        self.assertNotIn("zero_grad", model.policy.optimizer.__dict__)


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder)

    def test_checkpoints_hold_no_anchor_and_load_as_plain_policies(self):
        anchor = toy_anchor(1.0)
        anchor.run_dir = self.folder
        model = AnchoredPPO(CelestePolicy, CelesteRoomEnv(EpochBridge()), anchor=anchor, **SETTINGS)
        model.learn(64)
        model.save(self.folder / "anchored.zip")
        plain = SupervisedPPO.load(self.folder / "anchored.zip", device="cpu")
        self.assertFalse(hasattr(plain, "anchor"))
        for (name, a), b in zip(weights(model).items(), weights(plain).values()):
            self.assertTrue(th.equal(a, b), name)
        self.assertIsNone(AnchoredPPO.load(self.folder / "anchored.zip", device="cpu").anchor)
        rows = (self.folder / "anchor.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 1 + 2)  # header and one row per update (64 steps / 32 per rollout)
        self.assertIn("heldback_frame_agreement", rows[0])
        self.assertTrue(rows[0].rstrip().endswith("heldback_entropy_bits"))
        self.assertGreater(float(rows[1].split(",")[-1]), 0)  # a fresh policy is random: positive bits

    def test_build_anchor_accepts_only_this_donors_recording(self):
        from record_policy_play import save
        donor = SupervisedPPO(CelestePolicy, _spaces_env(), **{**SETTINGS, "seed": 5})
        donor_path = self.folder / "donor.zip"
        donor.save(donor_path)
        donor_sha = hashlib.sha256(donor_path.read_bytes()).hexdigest()
        play = self.folder / "play"
        play.mkdir()
        save(play / "dataset.npz", demonstrations(8, 10))
        identity = task_identity(resolve_task_definition(None))
        record = {"checkpoint_sha256": donor_sha, "task": identity, "uncommitted_changes": False, "attributable": True,
                  "runtime_problems": [], "commit": "c" * 40, "seed": 1, "episodes": 8,
                  "starts": "canonical only; no held-out state is read",
                  "dataset_sha256": hashlib.sha256((play / "dataset.npz").read_bytes()).hexdigest()}
        (play / "play.json").write_text(json.dumps(record), encoding="utf-8")
        built = build_anchor(play, donor_path, donor_sha, identity, 0.25, 0, 1.0, 0)
        self.assertEqual((built.record["fitted_episodes"], built.record["held_episodes"]), (6, 2))
        self.assertEqual(len(built) + len(built.held), 80)
        with self.assertRaises(ValueError):  # the pinned sha256 is not this file's
            build_anchor(play, donor_path, "e" * 64, identity, 0.25, 0, 1.0, 0)
        (play / "play.json").write_text(json.dumps({**record, "checkpoint_sha256": "e" * 64}), encoding="utf-8")
        with self.assertRaises(ValueError):  # the recording names another policy
            build_anchor(play, donor_path, donor_sha, identity, 0.25, 0, 1.0, 0)


class ScriptTests(unittest.TestCase):
    def test_anchor_arguments_are_consumed_and_the_rest_passed_on(self):
        from train_anchored import parse_anchor_args
        args, rest = parse_anchor_args(["--anchor-play", "p", "--anchor-donor", "d.zip", "--anchor-donor-sha256", "x",
                                        "--anchor-coef", "1.0", "--anchor-split-seed", "2", "--anchor-seed", "2",
                                        "--task-definition", "config/room2.json", "--seed", "40"])
        self.assertEqual((args.anchor_coef, args.anchor_split_seed), (1.0, 2))
        self.assertEqual(rest, ["--task-definition", "config/room2.json", "--seed", "40"])


if __name__ == "__main__":
    unittest.main()
