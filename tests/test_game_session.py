"""GameSession's fault hook relaunches only a game whose process has exited (no game needed).

Run from the repo root:
    .venv-rl/Scripts/python.exe -W error -m unittest
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from celeste_rl.training.game import GameSession


class GameSessionTests(unittest.TestCase):
    @patch("celeste_rl.training.game.game_process")
    def test_relaunches_only_after_the_process_exited(self, game_process):
        first, second = MagicMock(pid=1), MagicMock(pid=2)
        first.poll.return_value = None
        game_process.launch.side_effect = [first, second]
        game_process.stop.return_value = True

        session = GameSession("C:/game")
        session.on_fault(RuntimeError("step failed while the game still runs"))
        self.assertIs(session.process, first)
        self.assertEqual(session.relaunches, [])
        game_process.stop.assert_not_called()

        first.poll.return_value = 1
        first.returncode = 1
        session.on_fault(RuntimeError("step failed: connection reset"))
        self.assertIs(session.process, second)
        game_process.stop.assert_called_once_with(first)
        self.assertEqual(session.relaunches, [{"exit_code": 1, "ports_closed": True, "fault": "step failed: connection reset"}])
        self.assertEqual(game_process.set_window_mode.call_count, 2)

        game_process.process_memory.return_value = {"private_mb": 1300.0, "working_set_mb": 600.0}
        self.assertEqual(session.health(), {"game_private_mb": 1300.0, "game_working_set_mb": 600.0, "game_relaunches": 1})
        game_process.process_memory.side_effect = OSError("process gone")
        self.assertEqual(session.health(), {"game_private_mb": None, "game_working_set_mb": None, "game_relaunches": 1})

        session.close()
        game_process.stop.assert_called_with(second)


if __name__ == "__main__":
    unittest.main()
