"""Record and check the exact runtime a live run used: game build, mod loader, mods, profile settings, TAS prefix,
Python packages and schema.

Live results are only comparable when they come from the same runtime. `collect` builds a manifest that live
scripts save in results.json; `check` compares it with the pinned runtime in config/pinned_runtime.json and with
the profile settings every run requires, and returns the problems. Scripts refuse to run on a problem unless told
otherwise, so a changed game, mod build or setting is noticed instead of silently mixing results.

The pins are written deliberately with scripts/runtime_pins.py after a runtime change has been verified (for
example, after rebuilding the mod and re-running the live checks).
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PINS_PATH = REPO / "config" / "pinned_runtime.json"

# Files that define the game and the instrumentation, relative to the game copy.
HASHED_FILES = (
    "Celeste.exe",
    "Celeste.dll",  # patched by the Everest installer, so it also identifies the Everest build
    "Celeste.Mod.mm.dll",
    "FNA.dll",
    "MMHOOK_Celeste.dll",
    "Mods/CelesteTAS.zip",
    "Mods/SpeedrunTool.zip",
    "Mods/CelesteRLLockstep/CelesteRLLockstep.dll",
)
MOD_MANIFESTS = {
    "CelesteTAS": ("Mods/CelesteTAS.zip", "everest.yaml"),
    "SpeedrunTool": ("Mods/SpeedrunTool.zip", "everest.yaml"),
    "CelesteRLLockstep": ("Mods/CelesteRLLockstep/everest.yaml", None),
}
# Profile settings every live run needs (see the README): without them DebugRC does not start, the intro plays,
# an unfocused fullscreen window stalls, or CelesteTAS behaves differently from the verified configuration.
REQUIRED_SETTINGS = {
    "modsettings-Everest.celeste": {"DebugModeInEverest": "true", "LaunchWithoutIntro": "true"},
    "settings.celeste": {"Fullscreen": "false"},
    "modsettings-CelesteTAS.celeste": {"AutoPauseDraft": "true", "HideFreezeFrames": "false"},
    "modsettings-SpeedrunTool.celeste": {"GcAfterLoadState": "false"},
}
PACKAGES = ("torch", "gymnasium", "stable-baselines3", "numpy")


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mod_version(game_dir: Path, source: str, member: str | None) -> str | None:
    path = game_dir / source
    try:
        if member is None:
            text = path.read_text(encoding="utf-8-sig")
        else:
            with zipfile.ZipFile(path) as archive:
                text = archive.read(member).decode("utf-8-sig")
    except (OSError, KeyError, zipfile.BadZipFile):
        return None
    match = re.search(r"^\s*-?\s*Version:\s*(\S+)", text, re.MULTILINE)
    return match[1] if match else None


def _setting(text: str, key: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(key)}:\s*(\S+)", text, re.MULTILINE) or re.search(rf"<{key}>([^<]*)</{key}>", text)
    return match[1].strip().lower() if match else None


def _profile_settings(game_dir: Path) -> dict[str, dict[str, str | None]]:
    saves = game_dir / "probe-profile" / "Saves"
    settings = {}
    for file_name, keys in REQUIRED_SETTINGS.items():
        try:
            text = (saves / file_name).read_text(encoding="utf-8-sig")
        except OSError:
            text = ""
        settings[file_name] = {key: _setting(text, key) for key in keys}
    return settings


def _running_versions(game_dir: Path) -> dict[str, str | None]:
    """Versions the running game reported in its log (read after launch)."""
    try:
        text = (game_dir / "log.txt").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"celeste": None, "everest": None, "dotnet_runtime": None}
    celeste = re.search(r"VersionCelesteString: (\S+) \[Everest: ([^\]]+)\]", text)
    runtime = re.search(r"RuntimeVersion: (\S+)", text)
    return {"celeste": celeste[1] if celeste else None, "everest": celeste[2] if celeste else None,
            "dotnet_runtime": runtime[1] if runtime else None}


def collect(game_dir: Path, tas_prefix_lines: list[str] | None = None) -> dict:
    """The runtime manifest. Call after the game has launched, so the log shows the running versions."""
    from celeste_rl.reward import REWARD_VERSIONS
    from celeste_rl.schema import ACT_VERSION, FINGERPRINT, OBS_VERSION

    game_dir = Path(game_dir)
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    prefix = "\n".join(tas_prefix_lines) if tas_prefix_lines is not None else None
    return {
        "files_sha256": {name: _sha256(game_dir / name) for name in HASHED_FILES},
        "mod_versions": {name: _mod_version(game_dir, *where) for name, where in MOD_MANIFESTS.items()},
        "running": _running_versions(game_dir),
        "settings": _profile_settings(game_dir),
        "tas_prefix_sha256": hashlib.sha256(prefix.encode()).hexdigest() if prefix is not None else None,
        "python": {"version": platform.python_version(), **packages},
        # Every reward version the code implements, not the one a run chose: that belongs to the run's own
        # manifest. Adding or removing a version is code drift and should trip the pins.
        "schema": {"obs": OBS_VERSION, "act": ACT_VERSION, "reward": list(REWARD_VERSIONS),
                   "fingerprint": FINGERPRINT},
    }


def pinned_view(manifest: dict) -> dict:
    """The parts of a manifest that must match the pins (everything except settings, which have fixed values)."""
    return {key: manifest[key] for key in ("files_sha256", "mod_versions", "running", "tas_prefix_sha256", "python", "schema")}


def check(manifest: dict, pins: dict | None) -> list[str]:
    """Problems with a manifest: required settings that differ, and any difference from the pins."""
    problems = []
    for file_name, keys in REQUIRED_SETTINGS.items():
        for key, expected in keys.items():
            actual = manifest["settings"].get(file_name, {}).get(key)
            if actual != expected:
                problems.append(f"setting {file_name} {key} is {actual!r}, required {expected!r}")
    if pins is None:
        problems.append(f"no pinned runtime at {PINS_PATH.relative_to(REPO)}; write it with scripts/runtime_pins.py")
        return problems

    def compare(prefix: str, actual, expected):
        if isinstance(expected, dict) and isinstance(actual, dict):
            for key in sorted(set(expected) | set(actual)):
                compare(f"{prefix}.{key}" if prefix else key, actual.get(key), expected.get(key))
        elif actual != expected:
            problems.append(f"{prefix} is {actual!r}, pinned {expected!r}")

    compare("", pinned_view(manifest), pins)
    return problems


def git_state() -> dict:
    """The commit a run used and whether the working tree differed from it."""
    git = lambda *a: subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()
    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"), "uncommitted_changes": bool(status),
            "changed_paths": [line[3:] for line in status.splitlines()][:50]}


DIRTY_MESSAGE = ("The working tree has uncommitted changes, so these results could not be attributed to a commit. "
                 "Commit first, or pass --allow-dirty for an exploratory run.")


def load_pins(path: Path = PINS_PATH) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def write_pins(manifest: dict, path: Path = PINS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pinned_view(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
