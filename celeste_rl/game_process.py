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
if _user32 is not None:
    # Window handles are pointer-sized; ctypes would otherwise truncate them to 32-bit ints.
    _user32.GetForegroundWindow.restype = ctypes.c_void_p
    _user32.GetShellWindow.restype = ctypes.c_void_p
    _user32.SetForegroundWindow.argtypes = [ctypes.c_void_p]
    _user32.ShowWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _user32.IsWindowVisible.argtypes = [ctypes.c_void_p]
    _user32.IsIconic.argtypes = [ctypes.c_void_p]
    _user32.GetWindowThreadProcessId.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]


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


def _game_window(pid: int) -> int | None:
    if _user32 is None:
        return None
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def visit(hwnd, _):
        window_pid = ctypes.c_ulong()
        _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(window_pid))
        if window_pid.value == pid and _user32.IsWindowVisible(hwnd):
            found.append(hwnd)
        return True

    _user32.EnumWindows(visit, 0)
    return found[0] if found else None


def _take_focus(hwnd: int) -> bool:
    # Windows only lets a process move focus right after a key event; a lone Alt tap satisfies that.
    alt = 0x12
    _user32.keybd_event(alt, 0, 0, 0)
    _user32.keybd_event(alt, 0, 2, 0)
    return bool(_user32.SetForegroundWindow(hwnd))


def _focus_window(pid: int) -> bool:
    """Bring the game's window to the front. Returns False if it has no window yet."""
    if _user32 is None:
        return True
    hwnd = _game_window(pid)
    return hwnd is not None and _take_focus(hwnd)


def set_window_mode(pid: int, mode: str) -> None:
    """Put the game window in the front ("focused"), behind the desktop ("background") or minimized."""
    if _user32 is None:
        return
    hwnd = _game_window(pid)
    if hwnd is None:
        raise RuntimeError("Game window not found")
    if mode == "focused":
        _user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        _take_focus(hwnd)
    elif mode == "background":
        _user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        _take_focus(_user32.GetShellWindow())  # hand focus to the desktop
    elif mode == "minimized":
        _user32.ShowWindow(hwnd, 6)  # SW_MINIMIZE
    else:
        raise ValueError(f"Unknown window mode {mode!r}")


def window_is_focused(pid: int) -> bool:
    if _user32 is None:
        return True
    return _user32.GetForegroundWindow() == _game_window(pid)


def window_is_minimized(pid: int) -> bool:
    if _user32 is None:
        return False
    hwnd = _game_window(pid)
    return hwnd is not None and bool(_user32.IsIconic(hwnd))


class _ProcessMemoryCountersEx(ctypes.Structure):
    """PROCESS_MEMORY_COUNTERS_EX. ctypes inserts the same alignment padding as the C compiler."""

    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def process_memory(pid: int) -> dict[str, float]:
    """Working set and private bytes of a process in MB, read with GetProcessMemoryInfo."""
    if os.name != "nt":
        return {"working_set_mb": float("nan"), "private_mb": float("nan")}
    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        raise OSError(f"OpenProcess failed for pid {pid}")
    try:
        counters = _ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        if not kernel32.K32GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            raise OSError(f"GetProcessMemoryInfo failed for pid {pid}")
        return {"working_set_mb": counters.WorkingSetSize / 2**20, "private_mb": counters.PrivateUsage / 2**20}
    finally:
        kernel32.CloseHandle(handle)


def memory_mb(pid: int) -> float:
    """Working set of the game process in MB, read from tasklist."""
    output = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True, text=True
    ).stdout
    # e.g. "Celeste.exe","1234","Console","1","542,236 K"
    fields = [field.strip('"') for field in output.strip().split('","')]
    return int(fields[-1].rstrip(' K"').replace(",", "")) / 1024 if len(fields) >= 5 else float("nan")


def launch(game_dir: Path, port: int = 32279, timeout: float = 120.0, focus: bool = True,
           extra_args: list[str] | None = None) -> subprocess.Popen:
    """Start the game and wait until DebugRC answers.

    `focus=False` leaves the window unfocused during startup, to check whether startup needs focus.
    """
    client = DebugRcClient(port)
    if client.is_available():
        raise RuntimeError(f"Something is already answering on port {port}; close that game first")

    game_dir = Path(game_dir).resolve()
    profile = check_game_dir(game_dir)
    env = dict(os.environ, EVEREST_SAVEPATH=str(profile))
    process = subprocess.Popen(
        [str(game_dir / "Celeste.exe"), "--debug", "--disable-splash", *(extra_args or [])],
        cwd=game_dir,
        env=env,
    )

    deadline = time.perf_counter() + timeout
    focused = not focus
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
