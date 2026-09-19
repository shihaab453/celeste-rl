from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import run_overnight


class BuildCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.entry = {
            "id": "seed-1",
            "run_dir": "runs/seed-1",
            "command": [
                "scripts/train_room1.py",
                "--config", "config/train.json",
                "--seed", "1",
                "--run-dir", "runs/seed-1",
            ],
        }

    def test_new_run_keeps_declared_arguments_and_forwards_game_dir(self) -> None:
        game_dir = Path("C:/games/celeste")

        command = run_overnight.build_command(self.entry, game_dir)

        self.assertEqual(command[1:-2], self.entry["command"])
        self.assertEqual(command[-2], "--game-dir")
        self.assertEqual(command[-1], str(game_dir.resolve()))

    def test_resume_uses_only_resume_contract_and_game_dir(self) -> None:
        command = run_overnight.build_command(self.entry, Path("C:/games/celeste"), resume=True)

        self.assertEqual(command[1], "scripts/train_room1.py")
        self.assertEqual(command[2:4], ["--resume", "runs/seed-1"])
        self.assertNotIn("--config", command)
        self.assertNotIn("--seed", command)
        self.assertNotIn("--run-dir", command)
        self.assertEqual(command[-2], "--game-dir")

    def test_runner_game_dir_overrides_plan_value(self) -> None:
        self.entry["command"].extend(["--game-dir", "C:/wrong-game"])

        command = run_overnight.build_command(self.entry, Path("C:/right-game"))

        self.assertNotIn("C:/wrong-game", command)
        self.assertEqual(command[-1], str(Path("C:/right-game").resolve()))
        self.assertEqual(command.count("--game-dir"), 1)

    def test_equals_form_game_dir_is_removed(self) -> None:
        self.entry["command"].append("--game-dir=C:/wrong-game")

        command = run_overnight.build_command(self.entry, Path("C:/right-game"))

        self.assertFalse(any(argument.startswith("--game-dir=") for argument in command))
        self.assertEqual(command[-1], str(Path("C:/right-game").resolve()))

    def test_resume_rejects_non_training_command(self) -> None:
        self.entry["command"][0] = "scripts/evaluate_heldout.py"

        with self.assertRaisesRegex(ValueError, "does not support --resume"):
            run_overnight.build_command(self.entry, Path("C:/games/celeste"), resume=True)


class ResultArtifactTests(unittest.TestCase):
    def test_captures_result_episode_hashes_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            result_dir = repo / "runs" / "heldout" / "eval-1"
            result_dir.mkdir(parents=True)
            episodes = result_dir / "episodes.jsonl"
            episodes.write_text('{"success": true}\n', encoding="utf-8")
            result = {
                "checkpoint": "runs/model/checkpoint.pt",
                "checkpoint_sha256": "checkpoint-hash",
                "heldout_set": "config/heldout_starts.json",
                "heldout_sha256": "heldout-hash",
                "heldout_format_version": 2,
                "heldout_states": 10,
                "repeats": 3,
                "episodes_file": "episodes.jsonl",
                "successes": 21,
                "episodes": 30,
                "success_rate": 0.7,
                "uncertainty": {"method": "none"},
                "route_macro_success_rate": 0.65,
            }
            result_file = result_dir / "results.json"
            result_file.write_text(json.dumps(result), encoding="utf-8")

            artifact = run_overnight.result_artifact(
                f"evaluation complete\nResults: {result_dir}\n", repo=repo)

            self.assertIsNotNone(artifact)
            self.assertEqual(artifact["result_file"], "runs/heldout/eval-1/results.json")
            self.assertEqual(artifact["episodes_file"], "runs/heldout/eval-1/episodes.jsonl")
            self.assertEqual(artifact["result_sha256"], hashlib.sha256(result_file.read_bytes()).hexdigest())
            self.assertEqual(artifact["episodes_sha256"], hashlib.sha256(episodes.read_bytes()).hexdigest())
            self.assertEqual(artifact["summary"]["checkpoint_sha256"], "checkpoint-hash")
            self.assertEqual(artifact["summary"]["success_rate"], 0.7)
            self.assertEqual(artifact["summary"]["route_macro_success_rate"], 0.65)

    def test_no_result_announcement_returns_none(self) -> None:
        self.assertIsNone(run_overnight.result_artifact("ordinary output\n"))

    def test_result_outside_repository_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as repository, tempfile.TemporaryDirectory() as outside:
            artifact = run_overnight.result_artifact(
                f"Results: {Path(outside) / 'results.json'}\n", repo=Path(repository))

            self.assertEqual(artifact["problem"], "announced result is outside the repository")


class ExecuteTests(unittest.TestCase):
    @mock.patch.object(run_overnight.subprocess, "run")
    def test_failed_child_then_successful_resume_uses_clean_subprocess_command(self, run: mock.Mock) -> None:
        run.side_effect = [
            run_overnight.subprocess.CompletedProcess([], 1, stdout="first attempt failed\n", stderr=""),
            run_overnight.subprocess.CompletedProcess([], 0, stdout="resume completed\n", stderr=""),
        ]
        entry = {
            "id": "seed-3",
            "run_dir": "runs/train/seed-3",
            "limit_minutes": 1,
            "command": [
                "scripts/train_room1.py",
                "--seed", "3",
                "--total-timesteps", "500000",
                "--run-dir", "runs/train/seed-3",
            ],
        }
        game_dir = Path("C:/games/celeste")
        with tempfile.TemporaryDirectory() as temporary:
            logfile = Path(temporary) / "campaign.log"

            first = run_overnight.execute(entry, logfile, game_dir)
            retry = run_overnight.execute(entry, logfile, game_dir, resume=True)
            result = run_overnight.merge_attempts(first, retry)

        self.assertEqual(result["status"], "ok")
        self.assertEqual([attempt["status"] for attempt in result["attempts"]], ["exit_1", "ok"])
        self.assertEqual(run.call_count, 2)
        first_command = run.call_args_list[0].args[0]
        resume_command = run.call_args_list[1].args[0]
        self.assertIn("--total-timesteps", first_command)
        self.assertEqual(resume_command[1:], [
            "scripts/train_room1.py",
            "--resume", "runs/train/seed-3",
            "--game-dir", str(game_dir.resolve()),
        ])


class CampaignOutcomeTests(unittest.TestCase):
    def test_successful_retry_is_final_success_and_retains_attempts(self) -> None:
        first = {"status": "exit_1", "seconds": 5, "command": ["first"]}
        retry = {"status": "ok", "seconds": 7, "command": ["retry"], "artifact": {"result_file": "result"}}

        merged = run_overnight.merge_attempts(first, retry)

        self.assertEqual(merged["status"], "ok")
        self.assertEqual(merged["seconds"], 12)
        self.assertEqual(merged["attempts"], [first, retry])
        self.assertEqual(merged["artifact"], retry["artifact"])

    def test_campaign_exit_code_requires_every_entry_to_succeed(self) -> None:
        self.assertEqual(run_overnight.campaign_exit_code([{"status": "ok"}, {"status": "ok"}]), 0)
        self.assertEqual(run_overnight.campaign_exit_code([{"status": "ok"}, {"status": "exit_1"}]), 1)
        self.assertEqual(run_overnight.campaign_exit_code([{"status": "skipped_out_of_time"}]), 1)

    def test_reported_success_rate_supports_heldout_artifact(self) -> None:
        record = {"artifact": {"summary": {"success_rate": 0.75}}}

        self.assertEqual(run_overnight.reported_success_rate(record), 0.75)


if __name__ == "__main__":
    unittest.main()
