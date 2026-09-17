"""Tests for process discovery, launch guards, and shutdown in game_process (no game needed).

Run from the repo root:
    .venv-rl/Scripts/python.exe -m unittest
"""
from __future__ import annotations

import subprocess
import time
import unittest
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

from celeste_rl.game_process import _is_game_exe, launch, running_game_pids, stop


class RunningGamePidsMatchingTests(unittest.TestCase):
    """Path comparison for Celeste.exe processes inside vs outside a target game directory."""

    def setUp(self):
        self.game_dir = Path("C:/Projects/celeste-research-scratch/game-probe")

    def test_matches_exe_directly_inside_game_dir(self):
        candidates = [(101, self.game_dir / "Celeste.exe")]
        self.assertEqual(running_game_pids(self.game_dir, _processes=candidates), [101])

    def test_matches_exe_in_subdirectory(self):
        candidates = [(102, self.game_dir / "sub" / "Celeste.exe")]
        self.assertEqual(running_game_pids(self.game_dir, _processes=candidates), [102])

    def test_excludes_real_steam_install(self):
        steam_exe = Path(r"C:\Program Files (x86)\Steam\steamapps\common\Celeste\Celeste.exe")
        candidates = [(201, steam_exe)]
        self.assertEqual(running_game_pids(self.game_dir, _processes=candidates), [])

    def test_excludes_other_game_directories(self):
        other_dir_exe = Path("C:/Projects/celeste-research-scratch/other-probe/Celeste.exe")
        candidates = [(202, other_dir_exe)]
        self.assertEqual(running_game_pids(self.game_dir, _processes=candidates), [])

    def test_excludes_parent_directory(self):
        parent_exe = self.game_dir.parent / "Celeste.exe"
        candidates = [(203, parent_exe)]
        self.assertEqual(running_game_pids(self.game_dir, _processes=candidates), [])

    def test_excludes_different_executable_names(self):
        candidates = [
            (301, self.game_dir / "Everest.exe"),
            (302, self.game_dir / "Celeste.dll"),
            (303, self.game_dir / "python.exe"),
        ]
        self.assertEqual(running_game_pids(self.game_dir, _processes=candidates), [])

    def test_case_insensitivity_and_path_normalization(self):
        variations = [
            Path("c:/projects/celeste-research-scratch/game-probe/celeste.exe"),
            Path("C:/PROJECTS/CELESTE-RESEARCH-SCRATCH/GAME-PROBE/CELESTE.EXE"),
            Path("c:/projects/./celeste-research-scratch/../celeste-research-scratch/game-probe/Celeste.exe"),
            Path(r"c:\projects\celeste-research-scratch\game-probe\Celeste.exe"),
        ]
        for idx, path in enumerate(variations):
            with self.subTest(path=str(path)):
                candidates = [(400 + idx, path)]
                self.assertEqual(running_game_pids(self.game_dir, _processes=candidates), [400 + idx])

    def test_multiple_processes_returned_sorted(self):
        candidates = [
            (503, self.game_dir / "Celeste.exe"),
            (101, self.game_dir / "sub" / "Celeste.exe"),
            (999, Path(r"C:\Program Files (x86)\Steam\steamapps\common\Celeste\Celeste.exe")),
            (205, self.game_dir / "Celeste.exe"),
        ]
        self.assertEqual(running_game_pids(self.game_dir, _processes=candidates), [101, 205, 503])

    def test_helper_is_game_exe_direct(self):
        self.assertTrue(_is_game_exe(self.game_dir / "Celeste.exe", self.game_dir))
        self.assertFalse(_is_game_exe(Path(r"C:\Steam\Celeste\Celeste.exe"), self.game_dir))


class LaunchRefusalTests(unittest.TestCase):
    """launch() refusal when another game-copy process is running or port is answering."""

    def setUp(self):
        self.game_dir = Path("C:/Projects/celeste-research-scratch/game-probe")

    @patch("celeste_rl.game_process.running_game_pids")
    def test_refuses_when_game_copy_process_is_running(self, mock_running_pids):
        mock_running_pids.return_value = [1234, 5678]
        with self.assertRaises(RuntimeError) as ctx:
            launch(self.game_dir)

        message = str(ctx.exception)
        self.assertIn("1234", message)
        self.assertIn("5678", message)
        self.assertIn(str(self.game_dir), message)

    @patch("celeste_rl.game_process.DebugRcClient")
    @patch("celeste_rl.game_process.running_game_pids")
    def test_refuses_when_port_is_already_answering(self, mock_running_pids, mock_client_cls):
        mock_running_pids.return_value = []
        mock_client = MagicMock()
        mock_client.is_available.return_value = True
        mock_client_cls.return_value = mock_client

        with self.assertRaises(RuntimeError) as ctx:
            launch(self.game_dir, port=32279)

        self.assertIn("32279", message := str(ctx.exception))
        self.assertIn("already answering", message)


class StopShutdownTests(unittest.TestCase):
    """stop() process termination, single-target restriction, and port cleanup wait."""

    def test_stop_waits_for_ports_to_close(self):
        mock_process = MagicMock(spec=subprocess.Popen)
        mock_process.poll.return_value = 0

        # Simulate port answering for 2 checks, then closing.
        calls = 0

        def fake_checker(port: int) -> bool:
            nonlocal calls
            calls += 1
            return calls < 3

        self.assertTrue(stop(mock_process, timeout=1.0, _port_checker=fake_checker))
        self.assertGreaterEqual(calls, 3)

    def test_stop_times_out_cleanly_when_port_stays_open(self):
        mock_process = MagicMock(spec=subprocess.Popen)
        mock_process.poll.return_value = 0

        # Port stays open indefinitely; stop() must report it instead of hanging or claiming success.
        self.assertFalse(stop(mock_process, timeout=0.05, _port_checker=lambda port: True))

    @patch("subprocess.run")
    def test_stop_shares_one_timeout_budget(self, _):
        mock_process = MagicMock(spec=subprocess.Popen)
        mock_process.pid = 4321
        mock_process.poll.return_value = None
        # The graceful close succeeds after using most of the budget; the port wait must not get a fresh one.
        mock_process.wait.side_effect = lambda timeout: time.sleep(0.15)

        began = time.perf_counter()
        self.assertFalse(stop(mock_process, timeout=0.2, _port_checker=lambda port: True))
        self.assertLess(time.perf_counter() - began, 0.4)

    @patch("subprocess.run")
    def test_stop_only_targets_given_process(self, mock_subproc_run):
        mock_process = MagicMock(spec=subprocess.Popen)
        mock_process.pid = 9876
        # Simulate running process that fails graceful wait and requires kill.
        mock_process.poll.return_value = None
        mock_process.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="taskkill", timeout=0.01),
            0,
        ]

        stop(mock_process, timeout=0.01, _port_checker=lambda port: False)

        # Ensure taskkill targeted only PID 9876.
        mock_subproc_run.assert_called_once_with(["taskkill", "/PID", "9876"], capture_output=True, timeout=ANY)
        # Ensure only the given process instance was killed.
        mock_process.kill.assert_called_once()

    @patch("subprocess.run")
    def test_stop_bounds_a_stalled_taskkill(self, mock_subproc_run):
        mock_process = MagicMock(spec=subprocess.Popen)
        mock_process.pid = 5555
        mock_process.poll.return_value = None
        mock_process.wait.side_effect = [subprocess.TimeoutExpired(cmd="Celeste.exe", timeout=0), 0]

        # taskkill uses its whole timeout and then times out; stop() must still kill and stay in budget.
        def stalled_taskkill(args, capture_output, timeout):
            time.sleep(timeout)
            raise subprocess.TimeoutExpired(cmd=args, timeout=timeout)

        mock_subproc_run.side_effect = stalled_taskkill

        began = time.perf_counter()
        self.assertFalse(stop(mock_process, timeout=0.2, _port_checker=lambda port: True))
        self.assertLess(time.perf_counter() - began, 0.4)
        self.assertLessEqual(mock_subproc_run.call_args.kwargs["timeout"], 0.2)
        mock_process.kill.assert_called_once()

    def test_stop_returns_immediately_when_ports_already_closed(self):
        mock_process = MagicMock(spec=subprocess.Popen)
        mock_process.poll.return_value = 0

        checked_ports = []

        def fake_checker(port: int) -> bool:
            checked_ports.append(port)
            return False

        self.assertTrue(stop(mock_process, timeout=5.0, _port_checker=fake_checker))
        self.assertEqual(checked_ports, [32279, 32280])


if __name__ == "__main__":
    unittest.main()
