"""Show, check or write the pinned runtime (config/pinned_runtime.json).

Run from the repo root with the RL interpreter (Steam running):
    .venv-rl/Scripts/python.exe scripts/runtime_pins.py            # launch the game, print the manifest and problems
    .venv-rl/Scripts/python.exe scripts/runtime_pins.py --write    # the same, then write the pins

The game is launched so its log reports the running Celeste and Everest versions, then closed. Write the pins
only after the runtime has been verified with the live checks; every live script compares against them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from celeste_rl import game_process, runtime  # noqa: E402
from celeste_rl.bridge import CelesteBridge  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game-dir", type=Path, default=Path("C:/Projects/celeste-research-scratch/game-probe"))
    parser.add_argument("--write", action="store_true", help="write config/pinned_runtime.json from this runtime")
    args = parser.parse_args()

    process = game_process.launch(args.game_dir, focus=False)
    try:
        manifest = runtime.collect(args.game_dir, CelesteBridge(REPO / "runs" / "scratch" / "episode.tas")._prefix_lines())
    finally:
        game_process.stop(process)
    print(json.dumps(manifest, indent=2))
    problems = runtime.check(manifest, runtime.load_pins())
    if args.write:
        settings_problems = [p for p in problems if p.startswith("setting ")]
        if settings_problems:
            print("Not writing pins; required settings differ:\n  " + "\n  ".join(settings_problems))
            return 1
        runtime.write_pins(manifest)
        print(f"Wrote {runtime.PINS_PATH}")
        return 0
    print("Runtime matches the pins." if not problems else "Problems:\n  " + "\n  ".join(problems))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
