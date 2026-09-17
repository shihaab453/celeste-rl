"""The game process behind a training run, relaunched when a bridge fault finds it gone.

Used as SupervisedPPO's on_fault hook. The hook only relaunches: the supervisor then resets every environment slot,
and the lockstep bridge recovers its session through the HTTP reset (which plays the episode file from the level
load, recreating the savestate in the new process) and a new socket connection. A fault with the game still running
needs no relaunch; the recovery reset alone handles it.
"""
from __future__ import annotations

from pathlib import Path

from celeste_rl import game_process


class GameSession:
    def __init__(self, game_dir: Path, window: str = "minimized"):
        self.game_dir = Path(game_dir)
        self.window = window
        self.relaunches: list[dict] = []
        self.process = self._launch()

    def _launch(self):
        process = game_process.launch(self.game_dir, focus=False)
        game_process.set_window_mode(process.pid, self.window)
        return process

    def on_fault(self, fault) -> None:
        if self.process.poll() is None:
            return
        exit_code = self.process.returncode
        # Wait for the old ports to close before launching into them.
        ports_closed = game_process.stop(self.process)
        self.process = self._launch()
        self.relaunches.append({"exit_code": exit_code, "ports_closed": ports_closed, "fault": str(fault)[:300]})

    def health(self) -> dict:
        """The game's memory and relaunch count, for progress records. Keys never change; values are None if the
        process cannot be read (for example between a crash and its relaunch)."""
        try:
            memory = game_process.process_memory(self.process.pid)
        except OSError:
            memory = {"private_mb": None, "working_set_mb": None}
        return {"game_private_mb": memory["private_mb"], "game_working_set_mb": memory["working_set_mb"],
                "game_relaunches": len(self.relaunches)}

    def close(self) -> None:
        game_process.stop(self.process)
