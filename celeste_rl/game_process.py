"""Launch and stop the instrumented Celeste copy (Everest + CelesteTAS + Speedrun Tool).

Requirements learned while getting the first benchmark running:

- EVEREST_SAVEPATH must point at the isolated profile, so the game never touches personal saves.
- `--debug` only takes effect if the profile's modsettings-Everest.celeste has
  `DebugModeInEverest: true`; otherwise Everest switches debug mode off when settings load and
  the DebugRC HTTP server never starts.
- The window should be focused while the game starts, or it can stall on the loading screen.
- The copy needs its own steam_appid.txt, or Steam redirects the launch to the real install.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import time
from pathlib import Path

from celeste_rl.bridge import DebugRcClient

_user32 = ctypes.windll.user32 if os.name == "nt" else None


def check_game_dir(game_dir: Path) -> Path:
    """Fail early with a clear message if the game copy is not set up the way the bridge needs."""
    game_dir = Path(game_dir).resolve()
    profile = game_dir / "probe-profile"
    everest_settings = profile / "Saves" / "modsettings-Everest.celeste"

    problems = []
    if not (game_dir / "Celeste.exe").exists():
        problems.append(f"no Celeste.exe in {game_dir}")
    if not (game_dir / "steam_appid.txt").exists():
        problems.append("no steam_appid.txt, so Steam would redirect to the real install")
    if not everest_settings.exists():
        problems.append(f"no Everest settings at {everest_settings}")
    elif "DebugModeInEverest: true" not in everest_settings.read_text(encoding="utf-8-sig"):
        problems.append("DebugModeInEverest is not true, so DebugRC would never start")
    if problems:
        raise RuntimeError("Game copy is not ready: " + "; ".join(problems))
    return profile


def _focus_window(pid: int) -> bool:
    """Bring the game's window to the front. Returns False if it has no window yet."""
    if _user32 is None:
        return True
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def visit(hwnd, _):
        window_pid = ctypes.c_ulong()
        _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(window_pid))
        if window_pid.value == pid and _user32.IsWindowVisible(hwnd):
            found.append(hwnd)
        return True

    _user32.EnumWindows(visit, 0)
    if not found:
        return False
    # Windows only lets a process take focus right after a key event; a lone Alt tap satisfies that.
    alt = 0x12
    _user32.keybd_event(alt, 0, 0, 0)
    _user32.keybd_event(alt, 0, 2, 0)
    return bool(_user32.SetForegroundWindow(found[0]))


def launch(game_dir: Path, port: int = 32279, timeout: float = 120.0) -> subprocess.Popen:
    """Start the game and wait until DebugRC answers."""
    client = DebugRcClient(port)
    if client.is_available():
        raise RuntimeError(f"Something is already answering on port {port}; close that game first")

    game_dir = Path(game_dir).resolve()
    profile = check_game_dir(game_dir)
    env = dict(os.environ, EVEREST_SAVEPATH=str(profile))
    process = subprocess.Popen(
        [str(game_dir / "Celeste.exe"), "--debug", "--disable-splash"],
        cwd=game_dir,
        env=env,
    )

    deadline = time.perf_counter() + timeout
    focused = False
    while time.perf_counter() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Celeste exited during startup with code {process.returncode}")
        if not focused:
            focused = _focus_window(process.pid)
        if client.is_available():
            client.close()
            return process
        time.sleep(0.5)

    process.terminate()
    raise RuntimeError(f"DebugRC did not answer on port {port} within {timeout:.0f} s")


def stop(process: subprocess.Popen, timeout: float = 10.0) -> None:
    """Ask the game to close, then force it if it does not."""
    if process.poll() is not None:
        return
    subprocess.run(["taskkill", "/PID", str(process.pid)], capture_output=True)
    try:
        process.wait(timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout)
