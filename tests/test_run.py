"""Training run records, checkpoints, evaluation, resume and abort, on the fake bridge (no game).

Run from the repo root:
    .venv-rl/Scripts/python.exe -W error -m unittest
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch as th

import celeste_rl.training.run as run_module
from celeste_rl.env import CelesteRoomEnv
from celeste_rl.reward import RewardConfig
from celeste_rl.starts import Start, StartArchive
from celeste_rl.training.run import (
    PROGRESS_FIELDS,
    TrainConfig,
    build_environment,
    new_progress,
    roll_back_to_checkpoint,
    train,
    update_progress,
)
from celeste_rl.training.policy import CelestePolicy, policy_kwargs
from celeste_rl.training.supervisor import SupervisedPPO, TrainingAborted
from tests.test_training import EpochBridge

PROVENANCE = {"commit": "test", "uncommitted_changes": False, "attributable": False}


def config(**overrides) -> TrainConfig:
    values = dict(total_timesteps=160, n_steps=32, batch_size=32, n_epochs=1, checkpoint_every=64, eval_every=96,
                  eval_episodes=2)
    values.update(overrides)
    return TrainConfig(**values)


def read(run_dir: Path):
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    progress = list(csv.DictReader((run_dir / "progress.csv").open(encoding="utf-8")))
    episodes = [json.loads(line) for line in (run_dir / "episodes.jsonl").open(encoding="utf-8")] \
        if (run_dir / "episodes.jsonl").exists() else []
    evaluations = [json.loads(line) for line in (run_dir / "evaluations.jsonl").open(encoding="utf-8")] \
        if (run_dir / "evaluations.jsonl").exists() else []
    return manifest, progress, episodes, evaluations


class HardKill(BaseException):
    """The process is killed: nothing after this moment reaches the disk. SupervisedPPO only catches bridge faults,
    so this passes straight through training, as a kill would."""


class KillableBridge(EpochBridge):
    """Killed `after` steps once `armed()` first holds."""

    def __init__(self, armed, after=5, **kwargs):
        super().__init__(**kwargs)
        self.armed, self.remaining = armed, None
        self.after = after

    def step(self, *args, **kwargs):
        if self.remaining is None and self.armed():
            self.remaining = self.after
        if self.remaining is not None:
            if self.remaining == 0:
                raise HardKill()
            self.remaining -= 1
        return super().step(*args, **kwargs)


def train_until_killed(run_config: TrainConfig, run_dir: Path, env: CelesteRoomEnv) -> None:
    """Train until a HardKill. A killed process never writes the "crashed" manifest, so that write is dropped."""
    real = run_module._atomic_json

    def atomic_json(path, data):
        if data.get("status") != "crashed":
            real(path, data)

    with mock.patch.object(run_module, "_atomic_json", atomic_json):
        with unittest.TestCase().assertRaises(HardKill):
            train(run_config, run_dir, env, PROVENANCE)


class ProgressRecordTests(unittest.TestCase):
    """update_progress folds one step's info into an episode's record."""

    @staticmethod
    def info(x=None, y=None, potential=None):
        player = None if x is None else {"x": x, "y": y, "speed_x": 0, "speed_y": 0, "dashes": 1}
        return {"player": player, "potential": potential}

    def test_keeps_the_furthest_right_the_highest_and_the_last(self):
        progress = new_progress()
        for x, y, potential in ((19, 144, 0.2), (80, 120, 0.3), (60, 130, 0.2)):
            update_progress(progress, self.info(x, y, potential))
        self.assertEqual(progress["max_x"], 80)
        self.assertEqual(progress["min_y"], 120)          # y grows downwards, so this is the highest point
        self.assertEqual((progress["end_x"], progress["end_y"]), (60, 130))
        self.assertEqual(progress["max_potential"], 0.3)

    def test_a_step_without_a_player_keeps_the_last_known_position(self):
        progress = update_progress(new_progress(), self.info(80, 120, 0.3))
        update_progress(progress, self.info())             # the death step: no player, no potential
        self.assertEqual((progress["end_x"], progress["end_y"], progress["max_x"]), (80, 120, 80))
        self.assertEqual(progress["max_potential"], 0.3)

    def test_fields_never_reported_stay_none(self):
        self.assertEqual(update_progress(new_progress(), self.info()), dict.fromkeys(PROGRESS_FIELDS))


class BuildEnvironmentTests(unittest.TestCase):
    """How a run's environment is assembled from its config."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.run_dir = Path(directory.name)

    def test_varied_starts_off_means_no_archive_and_no_sampler(self):
        env = build_environment(EpochBridge(), config(), self.run_dir)
        self.assertIsNone(env.archive)
        self.assertIsNone(env.start_sampler)

    def test_a_new_empty_archive_still_gets_a_sampler(self):
        """The bug this test exists for: StartArchive defines __len__, so an empty one is falsy, and
        `archive.sample if archive else None` handed the environment no sampler at all. Every episode then
        started canonically while the archive filled up unused."""
        env = build_environment(EpochBridge(), config(varied_starts=True), self.run_dir)
        self.assertIsNotNone(env.archive)
        self.assertEqual(len(env.archive), 0)
        self.assertIsNotNone(env.start_sampler, "an empty archive must still be sampled from")

    def test_the_settings_reach_the_archive(self):
        env = build_environment(EpochBridge(), config(varied_starts=True, canonical_fraction=0.5,
                                                      max_start_frames=120), self.run_dir)
        self.assertEqual((env.archive.canonical_fraction, env.archive.max_frames), (0.5, 120))

    def test_an_existing_archive_is_loaded_rather_than_replaced(self):
        archive = StartArchive(canonical_fraction=0.25, max_frames=600, seed=0)
        archive.offer(Start(("1,R",), (27, 144), "1", 1))
        archive.save(self.run_dir / "archive.json")
        env = build_environment(EpochBridge(), config(varied_starts=True), self.run_dir)
        self.assertEqual(len(env.archive), 1)

    def test_a_later_room_definition_reaches_the_environment(self):
        env = build_environment(EpochBridge(), config(task_definition="config/room2.json"), self.run_dir)

        self.assertEqual((env.task.start_room, env.task.target_room), ("2", "3"))
        self.assertEqual((env.task_start.room, env.task_start.position, env.task_start.frames),
                         ("2", (261, 1), 286))

    def test_later_room_archive_limit_excludes_the_task_setup_frames(self):
        env = build_environment(EpochBridge(), config(task_definition="config/room2.json", varied_starts=True,
                                                      max_start_frames=600), self.run_dir)

        self.assertEqual(env.archive.max_frames, 886)


class RunTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.run_dir = Path(directory.name) / "run"

    def test_rew_v2_records_the_unspent_deadline_charge(self):
        """The progress columns follow the reward version, so the new charge is recorded and not dropped."""
        env = CelesteRoomEnv(EpochBridge(), reward_config=RewardConfig(version="rew-v2"))
        train(config(reward_version="rew-v2", eval_every=0), self.run_dir, env, PROVENANCE)
        _, progress, episodes, _ = read(self.run_dir)

        self.assertIn("component_unspent_deadline", progress[0])
        self.assertLess(sum(float(row["component_unspent_deadline"]) for row in progress), 0.0)
        self.assertTrue(episodes, "the fake bridge dies inside the first rollout")
        for episode in episodes:
            # Every death costs -1.03 before shaping, whenever it happened (Codex K1).
            unshaped = sum(value for name, value in episode["components"].items() if name != "shaping")
            self.assertAlmostEqual(unshaped, -1.03, places=9)

    def test_every_evaluation_names_the_checkpoint_it_measured(self):
        """Codex J4: an evaluation is only evidence if you can say which weights produced it."""
        train(config(), self.run_dir, CelesteRoomEnv(EpochBridge()), PROVENANCE)
        _, _, _, evaluations = read(self.run_dir)

        for record in evaluations:
            path = self.run_dir / "checkpoints" / record["checkpoint"]
            self.assertTrue(path.exists(), record["checkpoint"])
            self.assertEqual(record["checkpoint_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(record["checkpoint"], f"step_{record['accepted_steps']:09d}.zip")

    def test_the_final_policy_is_evaluated_once_and_not_twice(self):
        """The last scheduled evaluation runs before the final updates, so the policy a run ends with was never
        measured. Evaluating it again when a run ends exactly on an evaluation boundary would double-count."""
        train(config(eval_every=80), self.run_dir, CelesteRoomEnv(EpochBridge()), PROVENANCE)
        _, _, _, evaluations = read(self.run_dir)
        steps = [e["accepted_steps"] for e in evaluations]
        self.assertEqual(steps[-1], 160)
        self.assertEqual(len(steps), len(set(steps)), f"an evaluation was repeated: {steps}")

    def test_a_configuration_that_cannot_run_is_refused(self):
        """Codex J10: a zero checkpoint interval never advances the scheduler and hangs the run."""
        for field in ("checkpoint_every", "eval_episodes", "total_timesteps", "n_steps"):
            with self.subTest(field=field), self.assertRaises(ValueError) as caught:
                config(**{field: 0})
            self.assertIn(field, str(caught.exception))
        with self.assertRaises(ValueError):
            config(eval_every=-1)
        self.assertEqual(TrainConfig().eval_episodes, 50)  # Codex K7

    def test_varied_starts_are_recorded_and_the_archive_is_saved(self):
        """starts-v1: the archive is the run's other learned artefact, so it is saved with the checkpoints."""
        archive = StartArchive(canonical_fraction=0.0, seed=0)
        archive.offer(Start(("1,R", "1,R"), (19, 144), "1", 1))
        env = CelesteRoomEnv(EpochBridge(), archive=archive, start_sampler=archive.sample)
        train(config(eval_every=0), self.run_dir, env, PROVENANCE)
        _, progress, episodes, _ = read(self.run_dir)

        self.assertTrue((self.run_dir / "archive.json").exists())
        reloaded = StartArchive.load(self.run_dir / "archive.json")
        self.assertEqual(reloaded.starts, archive.starts)
        for column in ("episodes_from_archive", "archive_cells", "archive_furthest_x"):
            self.assertIn(column, progress[0])
        self.assertTrue(all(e["start"] == "archive" for e in episodes), "every episode should use the archive here")
        self.assertTrue(all(e["start_frames"] == 2 for e in episodes))

    def test_a_run_without_varied_starts_writes_no_archive(self):
        train(config(eval_every=0), self.run_dir, CelesteRoomEnv(EpochBridge()), PROVENANCE)
        _, progress, episodes, _ = read(self.run_dir)
        self.assertFalse((self.run_dir / "archive.json").exists())
        self.assertEqual(progress[0]["archive_cells"], "")
        self.assertTrue(all(e["start"] == "canonical" and e["start_frames"] == 0 for e in episodes))

    def test_evaluation_always_uses_the_canonical_start(self):
        """A run that trains on varied starts is still measured on the task it claims to solve."""
        archive = StartArchive(canonical_fraction=0.0, seed=0)
        archive.offer(Start(("1,R", "1,R"), (19, 144), "1", 1))
        env = CelesteRoomEnv(EpochBridge(), archive=archive, start_sampler=archive.sample)
        train(config(), self.run_dir, env, PROVENANCE)
        _, _, _, evaluations = read(self.run_dir)
        for record in evaluations:
            self.assertEqual(record["deterministic"]["length"], 74,
                             "an evaluation episode should start canonically, so it dies where it always did")

    def test_varied_start_settings_are_validated(self):
        for field, value in (("canonical_fraction", 1.5), ("canonical_fraction", -0.1), ("max_start_frames", 0)):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError) as caught:
                config(**{field: value})
            self.assertIn(field, str(caught.exception))

    def test_a_run_can_start_from_another_policy_s_weights(self):
        """Phase 3B fine-tuning: the policy is inherited, everything else starts fresh."""
        donor = SupervisedPPO(CelestePolicy, CelesteRoomEnv(EpochBridge()), policy_kwargs=policy_kwargs(),
                              n_steps=32, batch_size=32, device="cpu", seed=0)
        with th.no_grad():
            donor.policy.action_net.bias.fill_(1.25)
        donor.save(self.run_dir.parent / "donor.zip")

        model = train(config(init_from=str(self.run_dir.parent / "donor.zip"), eval_every=0),
                      self.run_dir, CelesteRoomEnv(EpochBridge()), PROVENANCE)
        manifest, _, _, _ = read(self.run_dir)
        self.assertEqual(manifest["accepted_steps"], 160, "the step counter must not be inherited")
        self.assertEqual(manifest["config"]["init_from"], str(self.run_dir.parent / "donor.zip"))
        # The weights moved during training, but not back to a fresh policy's zero bias.
        self.assertGreater(float(model.policy.action_net.bias.detach().mean()), 0.5)

    def test_init_from_does_not_steal_the_run_s_seed(self):
        """SB3's load() re-seeds torch, numpy and python globally from the donor's saved seed, so without a
        re-seed two fine-tuning runs with different --seed values produce byte-identical episodes."""
        donor = SupervisedPPO(CelestePolicy, CelesteRoomEnv(EpochBridge()), policy_kwargs=policy_kwargs(),
                              n_steps=32, batch_size=32, device="cpu", seed=0)
        donor.save(self.run_dir.parent / "donor.zip")

        draws = []
        for seed in (1, 2):
            directory = self.run_dir.parent / f"run{seed}"
            train(config(seed=seed, init_from=str(self.run_dir.parent / "donor.zip"), eval_every=0,
                         total_timesteps=32), directory, CelesteRoomEnv(EpochBridge()), PROVENANCE)
            draws.append(th.randint(0, 10 ** 6, (3,)).tolist())
        self.assertNotEqual(draws[0], draws[1], "two seeds must not share the donor's random stream")

    def test_progress_records_say_how_far_each_episode_got(self):
        """Codex J9 and K8: the unshaped campaign could not say where episodes ended, only that they ended."""
        train(config(), self.run_dir, CelesteRoomEnv(EpochBridge()), PROVENANCE)
        _, progress, episodes, evaluations = read(self.run_dir)

        for episode in episodes:
            for field in PROGRESS_FIELDS:
                self.assertIsNotNone(episode[field], f"{field} missing from an episode record")
            self.assertEqual((episode["end_x"], episode["end_y"]), (19, 144))  # the fake bridge never moves
        for column in ("median_max_potential", "best_max_potential", "median_max_x", "best_max_x", "best_min_y"):
            self.assertIn(column, progress[0])
        self.assertEqual(float(progress[2]["median_max_x"]), 19.0)
        self.assertIn("median_max_potential", evaluations[0])
        self.assertEqual(len(evaluations[0]["stochastic_progress"]), 2)

    def test_a_rollout_with_no_finished_episode_records_blanks_not_zeros(self):
        train(config(total_timesteps=32, eval_every=0), self.run_dir, CelesteRoomEnv(EpochBridge()), PROVENANCE)
        _, progress, _, _ = read(self.run_dir)
        self.assertEqual(progress[0]["median_max_x"], "")
        self.assertEqual(progress[0]["best_max_potential"], "")

    def test_records_checkpoints_and_evaluation(self):
        model = train(config(), self.run_dir, CelesteRoomEnv(EpochBridge()), PROVENANCE)
        manifest, progress, episodes, evaluations = read(self.run_dir)

        self.assertEqual(manifest["status"], "finished")
        self.assertEqual(manifest["accepted_steps"], 160)
        self.assertEqual(manifest["sessions"][0]["provenance"], PROVENANCE)
        self.assertEqual([int(row["accepted_steps"]) for row in progress], [32, 64, 96, 128, 160])
        # Training episodes in the records are exactly SB3's (evaluation episodes run outside the Monitor wrapper).
        self.assertEqual(len(episodes), len(model.ep_info_buffer))
        self.assertTrue(all(e["ending"] == "death" and e["length"] == 74 for e in episodes))
        self.assertEqual(sum(int(row["episodes"]) for row in progress), len(episodes))

        checkpoints = self.run_dir / "checkpoints"
        # 96 is there because an evaluation writes the weights it measured before measuring them (Codex J4).
        self.assertEqual(sorted(p.name for p in checkpoints.glob("step_*.zip")),
                         ["step_000000064.zip", "step_000000096.zip", "step_000000128.zip", "step_000000160.zip"])
        for name in ("latest.zip", "previous.zip", "best.zip"):
            self.assertTrue((checkpoints / name).exists(), name)
        self.assertEqual(list(checkpoints.glob("*.tmp*")), [])

        # 96 was scheduled; 160 is the final policy, which no scheduled evaluation ever reached (Codex J4).
        self.assertEqual([e["accepted_steps"] for e in evaluations], [96, 160])
        self.assertEqual(evaluations[0]["stochastic_episodes"], 2)
        self.assertEqual(evaluations[0]["deterministic"]["ending"], "death")
        loaded = SupervisedPPO.load(checkpoints / "latest.zip", device="cpu")
        self.assertEqual(loaded.num_timesteps, 160)

    def test_health_columns_are_recorded_per_rollout(self):
        calls = []
        train(config(total_timesteps=64, eval_every=0), self.run_dir, CelesteRoomEnv(EpochBridge()), PROVENANCE,
              health=lambda: calls.append(1) or {"game_private_mb": 100.0 + len(calls)})
        _, progress, _, _ = read(self.run_dir)
        self.assertEqual([row["game_private_mb"] for row in progress], ["101.0", "102.0"])

    def test_episodes_in_a_discarded_rollout_are_not_recorded(self):
        # The first episode dies at step 74 inside the third rollout (steps 65-96), which faults at step 80.
        model = train(config(eval_every=0, total_timesteps=128), self.run_dir, CelesteRoomEnv(EpochBridge(fault_steps={80})),
                      PROVENANCE)
        manifest, progress, episodes, _ = read(self.run_dir)
        self.assertEqual(manifest["fault_stats"]["discarded_rollouts"], 1)
        self.assertEqual([int(row["accepted_steps"]) for row in progress], [32, 64, 96, 128])
        self.assertEqual(episodes, [])
        self.assertEqual(len(model.ep_info_buffer), 0)

    def test_checkpoints_follow_accepted_steps_not_callback_calls(self):
        # Step 60 faults 27 steps into the second rollout. SB3's callback call count then passes 48 while only 32
        # steps are accepted; the first checkpoint must still wait for 48 accepted steps (taken at 64).
        train(config(total_timesteps=96, checkpoint_every=48, eval_every=0), self.run_dir,
              CelesteRoomEnv(EpochBridge(fault_steps={60})), PROVENANCE)
        names = sorted(p.name for p in (self.run_dir / "checkpoints").glob("step_*.zip"))
        self.assertEqual(names, ["step_000000064.zip", "step_000000096.zip"])

    def test_resume_continues_the_budget_from_a_fresh_reset(self):
        train(config(total_timesteps=96, eval_every=0), self.run_dir, CelesteRoomEnv(EpochBridge()), PROVENANCE)
        bridge = EpochBridge()
        model = train(config(total_timesteps=96, eval_every=0), self.run_dir, CelesteRoomEnv(bridge), PROVENANCE, resume=True)
        self.assertEqual(model.num_timesteps, 96, "a finished run resumed with the same budget does nothing more")

        with self.assertRaisesRegex(ValueError, "config differs"):
            train(config(total_timesteps=160, eval_every=0), self.run_dir, CelesteRoomEnv(EpochBridge()), PROVENANCE, resume=True)

    def test_resume_after_an_interrupted_run(self):
        # Simulate an interruption: a run that stopped after 64 accepted steps with its config set to 160.
        interrupted = config(total_timesteps=160, eval_every=0, checkpoint_every=32)
        train(TrainConfig(**{**interrupted.__dict__, "total_timesteps": 64}), self.run_dir, CelesteRoomEnv(EpochBridge()),
              PROVENANCE)
        manifest_path = self.run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["config"]["total_timesteps"] = 160
        manifest["status"] = "crashed"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        bridge = EpochBridge()
        model = train(interrupted, self.run_dir, CelesteRoomEnv(bridge), PROVENANCE, resume=True)
        manifest, progress, _, _ = read(self.run_dir)
        self.assertEqual(model.num_timesteps, 160)
        self.assertEqual(manifest["status"], "finished")
        self.assertEqual(len(manifest["sessions"]), 2)
        self.assertEqual(manifest["sessions"][1]["resumed_at"], 64)
        self.assertEqual([int(row["accepted_steps"]) for row in progress], [32, 64, 96, 128, 160])
        self.assertGreaterEqual(bridge.total_resets, 1, "the resumed run starts from a fresh reset, not the saved observation")

    def test_resume_between_checkpoints_rolls_the_records_back_to_the_checkpoint(self):
        """Review J1: latest.zip is from step 64, but the records reached 96 before the run aborted. The resume
        must replay 65 to 96 once, not record it twice, and keep what it rolls back."""
        run_config = config(eval_every=0, checkpoint_every=64)
        with self.assertRaises(TrainingAborted):
            train(run_config, self.run_dir, CelesteRoomEnv(EpochBridge(fault_steps=set(range(100, 10000)))),
                  PROVENANCE)
        manifest, progress, episodes, _ = read(self.run_dir)
        self.assertEqual((manifest["accepted_steps"], manifest["checkpoint_state"]["accepted_steps"]), (96, 64))
        self.assertEqual(len(episodes), 1, "the first death, at step 74, is after the checkpoint")

        model = train(run_config, self.run_dir, CelesteRoomEnv(EpochBridge()), PROVENANCE, resume=True)

        manifest, progress, episodes, _ = read(self.run_dir)
        self.assertEqual(model.num_timesteps, 160)
        self.assertEqual([int(row["accepted_steps"]) for row in progress], [32, 64, 96, 128, 160])
        self.assertEqual([episode["index"] for episode in episodes], list(range(1, len(episodes) + 1)))
        self.assertEqual(manifest["fault_stats"]["accepted_transitions"], 160)
        rolled_back = manifest["sessions"][1]["rolled_back"]
        self.assertEqual((rolled_back["from_steps"], rolled_back["to_steps"]), (96, 64))
        self.assertEqual(rolled_back["lines_moved"], {"episodes.jsonl": 1, "progress.csv": 1})
        self.assertEqual(rolled_back["fault_stats_before"]["discarded_rollouts"], 3)
        moved = self.run_dir / rolled_back["moved_to"]
        self.assertEqual(moved, self.run_dir / "rolled_back" / "session-2")
        self.assertIn("96,", (moved / "progress.csv").read_text(encoding="utf-8"))
        self.assertTrue((moved / "aborted.zip").exists())
        self.assertFalse((self.run_dir / "checkpoints" / "aborted.zip").exists())

    def test_resume_rolls_back_an_evaluation_and_best_checkpoint_made_after_latest(self):
        # Checkpoint at 64, evaluation and best.zip at 96 (its episodes use bridge steps 97 to 318), then every
        # step from 330 faults and the run aborts at 96.
        run_config = config(checkpoint_every=64, eval_every=96)
        with self.assertRaises(TrainingAborted):
            train(run_config, self.run_dir, CelesteRoomEnv(EpochBridge(fault_steps=set(range(330, 10000)))),
                  PROVENANCE)
        manifest, _, _, evaluations = read(self.run_dir)
        self.assertEqual([record["accepted_steps"] for record in evaluations], [96])
        self.assertEqual(manifest["checkpoint_state"]["accepted_steps"], 64)

        train(run_config, self.run_dir, CelesteRoomEnv(EpochBridge()), PROVENANCE, resume=True)

        manifest, _, _, evaluations = read(self.run_dir)
        self.assertEqual([record["accepted_steps"] for record in evaluations], [96, 160])
        for record in evaluations:
            path = self.run_dir / "checkpoints" / record["checkpoint"]
            self.assertEqual(record["checkpoint_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        # best.zip holds the same weights as the step checkpoint it names (the zip bytes differ by timestamps).
        best = SupervisedPPO.load(self.run_dir / "checkpoints" / "best.zip", device="cpu").policy.state_dict()
        named = SupervisedPPO.load(self.run_dir / "checkpoints" / manifest["recorder"]["best_checkpoint"],
                                   device="cpu").policy.state_dict()
        self.assertTrue(all(th.equal(best[key], named[key]) for key in best))
        moved = self.run_dir / manifest["sessions"][1]["rolled_back"]["moved_to"]
        self.assertEqual(manifest["sessions"][1]["rolled_back"]["checkpoints_moved"], ["step_000000096.zip"])
        self.assertTrue((moved / "step_000000096.zip").exists())
        self.assertTrue((moved / "best.zip").exists())
        self.assertEqual(len((moved / "evaluations.jsonl").read_text(encoding="utf-8").splitlines()), 1)

    def test_resume_refuses_records_ahead_of_latest_without_saved_state(self):
        """A run from before the saved checkpoint state cannot be rolled back safely, so it is not resumed."""
        run_config = config(eval_every=0, checkpoint_every=64)
        with self.assertRaises(TrainingAborted):
            train(run_config, self.run_dir, CelesteRoomEnv(EpochBridge(fault_steps=set(range(100, 10000)))),
                  PROVENANCE)
        manifest_path = self.run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        del manifest["checkpoint_state"]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "cannot be rolled back"):
            train(run_config, self.run_dir, CelesteRoomEnv(EpochBridge()), PROVENANCE, resume=True)
        _, progress, _, _ = read(self.run_dir)
        self.assertEqual([int(row["accepted_steps"]) for row in progress], [32, 64, 96])
        self.assertFalse((self.run_dir / "rolled_back").exists())

    def test_a_fault_on_the_reset_after_an_evaluation_does_not_repeat_it(self):
        """Review J2: reset 4 is the one after the evaluation at step 32 (1 initial, 2 and 3 for its episodes).
        Its record is already written, so the retried rollout must not evaluate again."""
        train(config(total_timesteps=96, eval_every=32, eval_episodes=1), self.run_dir,
              CelesteRoomEnv(EpochBridge(fault_resets={4})), PROVENANCE)
        manifest, _, _, evaluations = read(self.run_dir)
        self.assertEqual(manifest["fault_stats"]["discarded_rollouts"], 1)
        self.assertEqual([record["accepted_steps"] for record in evaluations], [32, 64, 96])

    def test_a_fault_during_an_evaluation_reruns_it_once(self):
        # Step 40 falls inside the first evaluation episode at step 32: nothing is recorded, and the retry
        # evaluates the same weights once.
        train(config(total_timesteps=96, eval_every=32, eval_episodes=1), self.run_dir,
              CelesteRoomEnv(EpochBridge(fault_steps={40})), PROVENANCE)
        manifest, _, _, evaluations = read(self.run_dir)
        self.assertEqual(manifest["fault_stats"]["discarded_rollouts"], 1)
        self.assertEqual([record["accepted_steps"] for record in evaluations], [32, 64, 96])

    def test_a_failed_relaunch_is_recorded_as_an_abort(self):
        """Review J7 at run level: the run ends aborted with its checkpoint, not crashed without one."""
        def relaunch_fails(fault):
            raise RuntimeError("Celeste exited during startup with code 1")

        with self.assertRaises(TrainingAborted):
            train(config(eval_every=0), self.run_dir, CelesteRoomEnv(EpochBridge(fault_steps={40})), PROVENANCE,
                  on_fault=relaunch_fails)
        manifest, _, _, _ = read(self.run_dir)
        self.assertEqual(manifest["status"], "aborted")
        self.assertTrue((self.run_dir / "checkpoints" / "aborted.zip").exists())

    def test_a_hard_kill_after_an_evaluation_at_a_checkpoint_does_not_repeat_it(self):
        """The checkpoint and the evaluation share step 64, and the run is killed a few steps into the next
        rollout. The resume starts from latest.zip at 64 and must not evaluate those weights again."""
        run_config = config(checkpoint_every=64, eval_every=64, eval_episodes=1)
        evaluations = self.run_dir / "evaluations.jsonl"
        train_until_killed(run_config, self.run_dir, CelesteRoomEnv(KillableBridge(armed=evaluations.exists)))
        manifest, _, _, recorded = read(self.run_dir)
        self.assertEqual([record["accepted_steps"] for record in recorded], [64])
        self.assertEqual(manifest["status"], "running", "a killed run writes nothing more")

        train(run_config, self.run_dir, CelesteRoomEnv(EpochBridge()), PROVENANCE, resume=True)

        manifest, progress, episodes, recorded = read(self.run_dir)
        self.assertEqual([record["accepted_steps"] for record in recorded], [64, 128, 160])
        self.assertEqual([int(row["accepted_steps"]) for row in progress], [32, 64, 96, 128, 160])
        self.assertEqual([episode["index"] for episode in episodes], list(range(1, len(episodes) + 1)))
        self.assertEqual(manifest["status"], "finished")

    def test_a_hard_kill_while_saving_a_checkpoint_leaves_latest_zip_to_resume_from(self):
        """Killed while the weights at 128 are being written: the checkpoint at 64 must still be there."""
        run_config = config(total_timesteps=192, checkpoint_every=64, eval_every=0)
        real_save = SupervisedPPO.save

        def save(model, path, *args, **kwargs):
            if Path(path).name == "latest.tmp.zip" and model.num_timesteps == 128:
                raise HardKill()
            return real_save(model, path, *args, **kwargs)

        with mock.patch.object(SupervisedPPO, "save", save):
            train_until_killed(run_config, self.run_dir, CelesteRoomEnv(EpochBridge()))
        self.assertTrue((self.run_dir / "checkpoints" / "latest.zip").exists())

        train(run_config, self.run_dir, CelesteRoomEnv(EpochBridge()), PROVENANCE, resume=True)

        manifest, progress, episodes, _ = read(self.run_dir)
        self.assertEqual(manifest["status"], "finished")
        self.assertEqual(manifest["sessions"][1]["rolled_back"]["to_steps"], 64)
        self.assertEqual([int(row["accepted_steps"]) for row in progress], [32, 64, 96, 128, 160, 192])
        self.assertEqual([episode["index"] for episode in episodes], list(range(1, len(episodes) + 1)))

    def test_refuses_to_overwrite_a_run(self):
        train(config(total_timesteps=32, eval_every=0), self.run_dir, CelesteRoomEnv(EpochBridge()), PROVENANCE)
        with self.assertRaises(FileExistsError):
            train(config(total_timesteps=32, eval_every=0), self.run_dir, CelesteRoomEnv(EpochBridge()), PROVENANCE)

    def test_abort_is_recorded_with_a_checkpoint(self):
        with self.assertRaises(TrainingAborted):
            train(config(eval_every=0), self.run_dir, CelesteRoomEnv(EpochBridge(fault_steps=set(range(40, 1000)))),
                  PROVENANCE)
        manifest, progress, _, _ = read(self.run_dir)
        self.assertEqual(manifest["status"], "aborted")
        self.assertEqual([int(row["accepted_steps"]) for row in progress], [32])
        self.assertTrue((self.run_dir / "checkpoints" / "aborted.zip").exists())


class RollBackTests(unittest.TestCase):
    """roll_back_to_checkpoint on synthetic files: the parts a fake-bridge run cannot steer."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run_dir = Path(self.temp.name)
        checkpoints = self.run_dir / "checkpoints"
        checkpoints.mkdir()
        for name in ("step_000000064.zip", "step_000000096.zip", "step_000000096.tmp.zip", "latest.zip"):
            (checkpoints / name).write_bytes(name.encode())
        (checkpoints / "best.zip").write_bytes(b"best from step 96")
        (self.run_dir / "episodes.jsonl").write_bytes(b'{"index": 1}\n{"index": 2}\n')
        (self.run_dir / "evaluations.jsonl").write_bytes(b'{"accepted_steps": 64}\n{"accepted_steps": 96}\n')
        (self.run_dir / "progress.csv").write_bytes(b"accepted_steps\n32\n64\n96\n")
        snapshot_recorder = {"next_checkpoint": 128, "next_eval": 96, "best": [0.0, 10.0, -1e9],
                             "best_checkpoint": "step_000000064.zip", "accepted_episodes": 1, "last_eval_steps": 64}
        self.previous = {
            "accepted_steps": 96,
            "fault_stats": {"accepted_transitions": 96, "discarded_transitions": 5, "discarded_rollouts": 3,
                            "faults": ["a", "b", "c"]},
            "recorder": {**snapshot_recorder, "best": [0.0, 20.0, -1e9], "best_checkpoint": "step_000000096.zip",
                         "accepted_episodes": 2, "next_eval": 192},
            "checkpoint_state": {
                "accepted_steps": 64,
                "fault_stats": {"accepted_transitions": 64, "discarded_transitions": 0, "discarded_rollouts": 0,
                                "faults": []},
                "recorder": snapshot_recorder,
                "records": {"episodes.jsonl": len(b'{"index": 1}\n'),
                            "progress.csv": len(b"accepted_steps\n32\n64\n"),
                            "evaluations.jsonl": len(b'{"accepted_steps": 64}\n')},
            },
        }

    def test_a_newer_best_is_replaced_by_the_one_at_the_checkpoint_and_a_repeat_is_harmless(self):
        for attempt in range(2):
            fault_stats, recorder, rolled_back = roll_back_to_checkpoint(self.run_dir, self.previous, 64, "session-2")
            checkpoints = self.run_dir / "checkpoints"
            moved = self.run_dir / "rolled_back" / "session-2"
            self.assertEqual((checkpoints / "best.zip").read_bytes(), b"step_000000064.zip")
            self.assertEqual((moved / "best.zip").read_bytes(), b"best from step 96")
            self.assertEqual((moved / "step_000000096.zip").read_bytes(), b"step_000000096.zip")
            self.assertTrue((checkpoints / "step_000000096.tmp.zip").exists(), "only step checkpoints move")
            self.assertEqual((self.run_dir / "progress.csv").read_bytes(), b"accepted_steps\n32\n64\n")
            self.assertEqual((moved / "progress.csv").read_bytes(), b"96\n")
            self.assertEqual((moved / "evaluations.jsonl").read_bytes(), b'{"accepted_steps": 96}\n')
            self.assertEqual(fault_stats["accepted_transitions"], 64)
            self.assertEqual((recorder["accepted_episodes"], recorder["next_eval"]), (1, 96))
            self.assertEqual(rolled_back["lines_moved"], {} if attempt else
                             {"episodes.jsonl": 1, "progress.csv": 1, "evaluations.jsonl": 1})

    def test_an_evaluation_after_the_checkpoint_rolls_best_back_even_if_the_manifest_missed_it(self):
        # A hard kill after an evaluation and before the next manifest write: the manifest still shows the
        # checkpoint's best, but the evaluation line after latest.zip shows best.zip may have changed.
        previous = {**self.previous, "recorder": self.previous["checkpoint_state"]["recorder"]}

        roll_back_to_checkpoint(self.run_dir, previous, 64, "session-2")

        moved = self.run_dir / "rolled_back" / "session-2"
        self.assertEqual((self.run_dir / "checkpoints" / "best.zip").read_bytes(), b"step_000000064.zip")
        self.assertEqual((moved / "best.zip").read_bytes(), b"best from step 96")

    def test_records_shorter_than_at_the_checkpoint_are_refused_before_anything_moves(self):
        (self.run_dir / "episodes.jsonl").write_bytes(b"")

        with self.assertRaisesRegex(ValueError, "episodes.jsonl is shorter"):
            roll_back_to_checkpoint(self.run_dir, self.previous, 64, "session-2")
        self.assertFalse((self.run_dir / "rolled_back").exists())
        self.assertEqual((self.run_dir / "checkpoints" / "best.zip").read_bytes(), b"best from step 96")

    def _cut_records_to_the_checkpoint(self):
        for name, size in self.previous["checkpoint_state"]["records"].items():
            with (self.run_dir / name).open("r+b") as handle:
                handle.truncate(size)

    def test_a_run_that_stopped_at_its_checkpoint_needs_no_rollback(self):
        self._cut_records_to_the_checkpoint()
        previous = {**self.previous, "accepted_steps": 64}

        fault_stats, recorder, rolled_back = roll_back_to_checkpoint(self.run_dir, previous, 64, "session-2")

        self.assertIsNone(rolled_back)
        self.assertEqual(fault_stats, previous["fault_stats"])
        self.assertEqual(recorder, previous["recorder"])
        self.assertFalse((self.run_dir / "rolled_back").exists())

    def test_records_ahead_of_a_manifest_that_agrees_with_latest_are_rolled_back(self):
        # A hard kill between a progress row and the manifest write after it, in the first rollout after the
        # checkpoint: the manifest still says 64, like latest.zip, but the records reached 96.
        checkpoints = self.run_dir / "checkpoints"
        (checkpoints / "step_000000096.zip").unlink()
        (checkpoints / "best.zip").write_bytes(b"step_000000064.zip")
        (self.run_dir / "evaluations.jsonl").write_bytes(b'{"accepted_steps": 64}\n')
        snapshot = self.previous["checkpoint_state"]
        previous = {**self.previous, "accepted_steps": 64, "fault_stats": snapshot["fault_stats"],
                    "recorder": snapshot["recorder"]}

        fault_stats, recorder, rolled_back = roll_back_to_checkpoint(self.run_dir, previous, 64, "session-2")

        self.assertEqual((rolled_back["from_steps"], rolled_back["to_steps"]), (64, 64))
        self.assertEqual(rolled_back["lines_moved"], {"episodes.jsonl": 1, "progress.csv": 1})
        self.assertEqual((self.run_dir / "progress.csv").read_bytes(), b"accepted_steps\n32\n64\n")
        self.assertEqual((self.run_dir / "episodes.jsonl").read_bytes(), b'{"index": 1}\n')
        self.assertEqual((checkpoints / "best.zip").read_bytes(), b"step_000000064.zip")
        self.assertEqual(recorder["accepted_episodes"], 1)

    def test_a_rollback_interrupted_by_a_locked_file_is_finished_by_a_retry(self):
        # best.zip is held open (an antivirus scan, a reader) the first time, as on Windows, and the manifest
        # missed the evaluation that changed it: only the evaluation records show that best.zip must go back.
        previous = {**self.previous, "recorder": self.previous["checkpoint_state"]["recorder"]}
        real_replace = os.replace

        def best_zip_locked(source, target):
            if Path(source).name == "best.zip" and Path(source).parent.name == "checkpoints":
                raise PermissionError(32, "The process cannot access the file because it is being used by "
                                          "another process")
            real_replace(source, target)

        with mock.patch.object(os, "replace", best_zip_locked):
            with self.assertRaises(PermissionError):
                roll_back_to_checkpoint(self.run_dir, previous, 64, "session-2")
        roll_back_to_checkpoint(self.run_dir, previous, 64, "session-2")

        moved = self.run_dir / "rolled_back" / "session-2"
        self.assertEqual((self.run_dir / "checkpoints" / "best.zip").read_bytes(), b"step_000000064.zip")
        self.assertEqual((moved / "best.zip").read_bytes(), b"best from step 96")
        self.assertEqual((moved / "step_000000096.zip").read_bytes(), b"step_000000096.zip")
        self.assertEqual((self.run_dir / "evaluations.jsonl").read_bytes(), b'{"accepted_steps": 64}\n')
        self.assertEqual((moved / "evaluations.jsonl").read_bytes(), b'{"accepted_steps": 96}\n')

    def test_a_checkpoint_without_a_best_leaves_no_best_zip(self):
        snapshot = self.previous["checkpoint_state"]
        previous = {**self.previous, "checkpoint_state": {
            **snapshot, "recorder": {**snapshot["recorder"], "best": None, "best_checkpoint": None}}}

        roll_back_to_checkpoint(self.run_dir, previous, 64, "session-2")

        self.assertFalse((self.run_dir / "checkpoints" / "best.zip").exists())
        self.assertEqual((self.run_dir / "rolled_back" / "session-2" / "best.zip").read_bytes(),
                         b"best from step 96")


if __name__ == "__main__":
    unittest.main()
