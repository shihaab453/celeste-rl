"""Offline checks of running campaign jobs side by side on separate game copies (no game)."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from scripts import run_overnight
from scripts.run_overnight import GameCopy, copy_problems, parse_copy, side_by_side

REPO = Path(__file__).resolve().parents[1]
PORTS = ("from celeste_rl import bridge, game_process, lockstep; import inspect; "
         "print(bridge.DEFAULT_PORT, lockstep.DEFAULT_LOCKSTEP_PORT, "
         "inspect.signature(game_process.launch).parameters['port'].default, "
         "list(inspect.signature(game_process.stop).parameters['ports'].default))")


def ports_seen(extra_env: dict) -> str:
    env = {k: v for k, v in os.environ.items() if k not in ("CELESTE_RL_DEBUGRC_PORT", "CELESTE_RL_LOCKSTEP_PORT")}
    return subprocess.run([sys.executable, "-c", PORTS], cwd=REPO, env={**env, **extra_env}, capture_output=True,
                          text=True, check=True).stdout.strip()


class PortDefaultTests(unittest.TestCase):
    def test_unset_variables_keep_the_ports_every_run_used(self):
        self.assertEqual(ports_seen({}), "32279 32280 32279 [32279, 32280]")

    def test_the_variables_move_every_default_together(self):
        self.assertEqual(ports_seen({"CELESTE_RL_DEBUGRC_PORT": "32289", "CELESTE_RL_LOCKSTEP_PORT": "32290"}),
                         "32289 32290 32289 [32289, 32290]")


class CopyTests(unittest.TestCase):
    def game_copy(self, root: Path, name: str, port: int) -> Path:
        saves = root / name / "probe-profile" / "Saves"
        saves.mkdir(parents=True)
        (saves / "modsettings-Everest.celeste").write_text(f"DebugModeInEverest: true\nDebugRCPort: {port}\n",
                                                          encoding="utf-8")
        return root / name

    def test_parse_copy_allows_a_drive_letter(self):
        self.assertEqual(parse_copy("C:/games/copy-2:32289:32290"), GameCopy(Path("C:/games/copy-2"), 32289, 32290))
        with self.assertRaises(ValueError):
            parse_copy("C:/games/copy-2:32289")

    def test_the_child_environment_always_sets_threads_and_only_its_own_ports(self):
        default = GameCopy(Path("x")).env(10)
        self.assertEqual((default["OMP_NUM_THREADS"], default["MKL_NUM_THREADS"]), ("10", "10"))
        with mock.patch.dict(os.environ, {"CELESTE_RL_DEBUGRC_PORT": "1", "CELESTE_RL_LOCKSTEP_PORT": "2"}):
            self.assertNotIn("CELESTE_RL_DEBUGRC_PORT", GameCopy(Path("x")).env(10))  # never an inherited port
            env = GameCopy(Path("x"), 32289, 32290).env(1)
        self.assertEqual((env["CELESTE_RL_DEBUGRC_PORT"], env["CELESTE_RL_LOCKSTEP_PORT"], env["OMP_NUM_THREADS"]),
                         ("32289", "32290", "1"))

    def test_copies_must_have_their_own_ports_directories_and_matching_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a, b = self.game_copy(root, "a", 32279), self.game_copy(root, "b", 32289)
            closed = lambda port: False  # noqa: E731
            good = [GameCopy(a, 32279, 32280), GameCopy(b, 32289, 32290)]
            self.assertEqual(copy_problems(good, port_open=closed), [])
            self.assertTrue(copy_problems([GameCopy(a, 32279, 32280), GameCopy(b, 32289, 32280)], port_open=closed))
            self.assertTrue(copy_problems([GameCopy(a, 32279, 32280), GameCopy(a, 32289, 32290)], port_open=closed))
            wrong = copy_problems([GameCopy(a, 32279, 32280), GameCopy(b, 32299, 32300)], port_open=closed)
            self.assertTrue(any("settings name DebugRC port 32289" in problem for problem in wrong))
            self.assertTrue(copy_problems(good, port_open=lambda port: port == 32290))


class SideBySideTests(unittest.TestCase):
    def test_each_copy_runs_one_job_at_a_time_and_records_keep_plan_order(self):
        copies = [GameCopy(Path(f"copy-{i}"), 32279 + 10 * i, 32280 + 10 * i) for i in range(3)]
        entries = [{"id": f"run-{i}", "seconds": 0.05 * (i % 3 + 1)} for i in range(9)]
        busy, overlaps, peak, lock = set(), [], [0], threading.Lock()
        published = []

        def run_one(entry, copy):
            with lock:
                if copy in busy:
                    overlaps.append(copy)
                busy.add(copy)
                peak[0] = max(peak[0], len(busy))
            time.sleep(entry["seconds"])
            with lock:
                busy.discard(copy)
            return {"id": entry["id"], "copy": copy.game_dir.name}

        results = side_by_side(entries, copies, run_one, published.append)
        self.assertEqual([r["id"] for r in results], [e["id"] for e in entries])
        self.assertEqual(overlaps, [])
        self.assertEqual(peak[0], 3)  # the three copies really ran at the same time
        self.assertEqual(len(published), 9)
        self.assertEqual(published[-1], results)

    def test_a_failing_job_still_returns_its_copy(self):
        copies = [GameCopy(Path("only"), 32279, 32280)]

        def run_one(entry, copy):
            if entry["id"] == "bad":
                raise RuntimeError("boom")
            return {"id": entry["id"]}

        with self.assertRaises(RuntimeError):
            side_by_side([{"id": "bad"}, {"id": "good"}], copies, run_one, lambda done: None)



class StartSpacingTests(unittest.TestCase):
    def test_concurrent_starts_are_at_least_the_spacing_apart(self):
        starts, lock = [], threading.Lock()

        def start():
            run_overnight.wait_for_start_slot()
            with lock:
                starts.append(time.monotonic())

        threads = [threading.Thread(target=start) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        starts.sort()
        gaps = [later - earlier for earlier, later in zip(starts, starts[1:])]
        self.assertTrue(all(gap >= run_overnight.START_SPACING_SECONDS - 0.01 for gap in gaps), gaps)


class RunEntryTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.folder, ignore_errors=True))
        self.logfile = self.folder / "campaign.log"
        self.entry = {"id": "eval-1", "run_dir": "runs/evaluation/_planned-never-exists", "limit_minutes": 1}
        self.copy = GameCopy(Path("copy-2"), 32289, 32290)

    def fake(self, outcomes):
        calls = []

        def run(entry, logfile, game_dir, resume=False, env=None):
            calls.append({"resume": resume, "threads": env["OMP_NUM_THREADS"], "port": env["CELESTE_RL_DEBUGRC_PORT"]})
            return dict(outcomes[len(calls) - 1])
        return run, calls

    def test_a_folder_collision_is_retried_once_on_the_same_copy_and_both_attempts_kept(self):
        run, calls = self.fake([{"status": "exit_1", "seconds": 1, "folder_collision": True},
                                {"status": "ok", "seconds": 60}])
        with mock.patch.object(run_overnight, "clear_game"):
            record = run_overnight.run_entry(self.entry, self.copy, self.logfile, 10, side=True, run=run,
                                             free_memory=lambda: 8.0)
        self.assertEqual(record["status"], "ok")
        self.assertEqual(len(record["attempts"]), 2)
        self.assertEqual(calls, [{"resume": False, "threads": "10", "port": "32289"}] * 2)
        self.assertEqual((record["threads_per_job"], record["available_memory_gb_at_start"]), (10, 8.0))
        self.assertEqual(record["game_copy"], "copy-2")

    def test_an_ordinary_failure_is_not_retried(self):
        run, calls = self.fake([{"status": "exit_2", "seconds": 1}])
        with mock.patch.object(run_overnight, "clear_game"):
            record = run_overnight.run_entry(self.entry, self.copy, self.logfile, 10, side=True, run=run,
                                             free_memory=lambda: 8.0)
        self.assertEqual((record["status"], len(calls)), ("exit_2", 1))

    def test_side_by_side_jobs_wait_for_memory_and_sequential_ones_do_not(self):
        readings, sleeps = iter([1.0, 2.0, 3.0]), []
        free = run_overnight.wait_for_memory(2.5, self.logfile, "eval-1", read=lambda: next(readings),
                                             sleep=sleeps.append)
        self.assertEqual((free, len(sleeps)), (3.0, 2))
        self.assertIn("waits: 1.0 GB available", self.logfile.read_text(encoding="utf-8"))
        run, _ = self.fake([{"status": "ok", "seconds": 1}])
        with mock.patch.object(run_overnight, "clear_game"), \
                mock.patch.object(run_overnight, "wait_for_memory") as waited:
            record = run_overnight.run_entry(self.entry, self.copy, self.logfile, 10, side=False, run=run,
                                             free_memory=lambda: 0.5)
        waited.assert_not_called()
        self.assertEqual(record["available_memory_gb_at_start"], 0.5)

    def test_unreadable_memory_never_blocks(self):
        self.assertIsNone(run_overnight.wait_for_memory(2.5, self.logfile, "x", read=lambda: None,
                                                        sleep=self.fail))

    def test_the_summary_records_the_thread_count(self):
        path = self.folder / "summary.json"
        run_overnight.write_summary(path, {"name": "p"}, "h", "c", [], None, 10)
        self.assertEqual(__import__("json").loads(path.read_text(encoding="utf-8"))["threads_per_job"], 10)

if __name__ == "__main__":
    unittest.main()
