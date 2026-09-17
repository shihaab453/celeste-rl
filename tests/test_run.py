"""Training run records, checkpoints, evaluation, resume and abort, on the fake bridge (no game).

Run from the repo root:
    .venv-rl/Scripts/python.exe -W error -m unittest
"""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from celeste_rl.env import CelesteRoomEnv
from celeste_rl.training.run import TrainConfig, train
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


class RunTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.run_dir = Path(directory.name) / "run"

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
        self.assertEqual(sorted(p.name for p in checkpoints.glob("step_*.zip")),
                         ["step_000000064.zip", "step_000000128.zip", "step_000000160.zip"])
        for name in ("latest.zip", "previous.zip", "best.zip"):
            self.assertTrue((checkpoints / name).exists(), name)
        self.assertEqual(list(checkpoints.glob("*.tmp*")), [])

        self.assertEqual([e["accepted_steps"] for e in evaluations], [96])
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
