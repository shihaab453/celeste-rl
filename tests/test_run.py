"""Training run records, checkpoints, evaluation, resume and abort, on the fake bridge (no game).

Run from the repo root:
    .venv-rl/Scripts/python.exe -W error -m unittest
"""
from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from celeste_rl.env import CelesteRoomEnv
from celeste_rl.reward import RewardConfig
from celeste_rl.starts import Start, StartArchive
from celeste_rl.training.run import PROGRESS_FIELDS, TrainConfig, new_progress, train, update_progress
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


if __name__ == "__main__":
    unittest.main()
