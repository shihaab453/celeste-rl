"""Runtime manifest and pin checks, on a fake game copy (no game).

Run from the repo root:
    .venv-rl/Scripts/python.exe -W error -m unittest
"""
from __future__ import annotations

import copy
import tempfile
import unittest
import zipfile
from pathlib import Path

from celeste_rl import runtime


def fake_game(directory: Path) -> Path:
    game = directory / "game"
    (game / "Mods" / "CelesteRLLockstep").mkdir(parents=True)
    saves = game / "probe-profile" / "Saves"
    saves.mkdir(parents=True)
    for name in runtime.HASHED_FILES:
        path = game / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not name.endswith(".zip"):
            path.write_bytes(name.encode())
    for mod, version in (("CelesteTAS", "3.47.1"), ("SpeedrunTool", "3.27.21")):
        with zipfile.ZipFile(game / "Mods" / f"{mod}.zip", "w") as archive:
            # Speedrun Tool's manifest starts with a byte order mark.
            archive.writestr("everest.yaml", f"﻿- Name: {mod}\n  Version: {version}\n  DLL: x.dll\n".encode())
    (game / "Mods" / "CelesteRLLockstep" / "everest.yaml").write_text(
        "- Name: CelesteRLLockstep\n  Version: 0.1.0\n  Dependencies:\n    - Name: CelesteTAS\n      Version: 3.47.1\n")
    (game / "log.txt").write_text("[core] VersionCelesteString: 1.4.0.0-fna [Everest: 6531-azure-d72e9-stable]\n"
                                  "[core] RuntimeVersion: 8.0.14\n")
    (saves / "modsettings-Everest.celeste").write_text("DebugModeInEverest: true\nLaunchWithoutIntro: true\n")
    (saves / "settings.celeste").write_text("<Settings>\n  <Fullscreen>false</Fullscreen>\n</Settings>\n")
    (saves / "modsettings-CelesteTAS.celeste").write_text("AutoPauseDraft: true\nHideFreezeFrames: false\n")
    (saves / "modsettings-SpeedrunTool.celeste").write_text("GcAfterLoadState: false\n")
    return game


class RuntimeManifestTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.game = fake_game(self.directory)
        self.prefix = ["console load 1", "299", "***S", "1"]

    def test_collects_versions_settings_and_hashes(self):
        manifest = runtime.collect(self.game, self.prefix)
        self.assertEqual(manifest["mod_versions"], {"CelesteTAS": "3.47.1", "SpeedrunTool": "3.27.21", "CelesteRLLockstep": "0.1.0"})
        self.assertEqual(manifest["running"], {"celeste": "1.4.0.0-fna", "everest": "6531-azure-d72e9-stable", "dotnet_runtime": "8.0.14"})
        self.assertEqual(manifest["settings"]["settings.celeste"], {"Fullscreen": "false"})
        self.assertTrue(all(manifest["files_sha256"].values()))
        self.assertEqual(len(manifest["tas_prefix_sha256"]), 64)
        self.assertEqual(manifest["schema"]["obs"], "obs-v1")

    def test_matching_pins_have_no_problems_and_every_difference_is_reported(self):
        pins_path = self.directory / "pins.json"
        manifest = runtime.collect(self.game, self.prefix)
        runtime.write_pins(manifest, pins_path)
        pins = runtime.load_pins(pins_path)
        self.assertEqual(runtime.check(manifest, pins), [])

        rebuilt_mod = copy.deepcopy(manifest)
        rebuilt_mod["files_sha256"]["Mods/CelesteRLLockstep/CelesteRLLockstep.dll"] = "0" * 64
        self.assertEqual(len(runtime.check(rebuilt_mod, pins)), 1)
        self.assertIn("CelesteRLLockstep.dll", runtime.check(rebuilt_mod, pins)[0])

        other_prefix = runtime.collect(self.game, ["console load 1", "300", "***S", "1"])
        self.assertIn("tas_prefix_sha256", " ".join(runtime.check(other_prefix, pins)))

        (self.game / "Mods" / "CelesteRLLockstep" / "everest.yaml").write_text("- Name: CelesteRLLockstep\n  Version: 0.2.0\n")
        self.assertIn("mod_versions.CelesteRLLockstep", " ".join(runtime.check(runtime.collect(self.game, self.prefix), pins)))

    def test_required_settings_are_checked_without_pins(self):
        saves = self.game / "probe-profile" / "Saves"
        (saves / "settings.celeste").write_text("<Settings><Fullscreen>true</Fullscreen></Settings>")
        (saves / "modsettings-CelesteTAS.celeste").write_text("HideFreezeFrames: false\n")  # AutoPauseDraft missing
        problems = runtime.check(runtime.collect(self.game, self.prefix), None)
        self.assertIn("setting settings.celeste Fullscreen is 'true', required 'false'", problems)
        self.assertIn("setting modsettings-CelesteTAS.celeste AutoPauseDraft is None, required 'true'", problems)
        self.assertTrue(any("no pinned runtime" in p for p in problems))

    def test_missing_files_are_recorded_as_none(self):
        (self.game / "Mods" / "SpeedrunTool.zip").unlink()
        manifest = runtime.collect(self.game, self.prefix)
        self.assertIsNone(manifest["files_sha256"]["Mods/SpeedrunTool.zip"])
        self.assertIsNone(manifest["mod_versions"]["SpeedrunTool"])

    def test_git_state_names_the_commit(self):
        state = runtime.git_state()
        self.assertEqual(len(state["commit"]), 40)
        self.assertIsInstance(state["uncommitted_changes"], bool)


if __name__ == "__main__":
    unittest.main()
