"""PPO training with a Room 1 anchor (celeste_rl/training/anchor.py): train_room1.py unchanged, plus the anchor.

Run from the repo root with the RL interpreter (Steam running, mod installed):
    .venv-rl/Scripts/python.exe scripts/train_anchored.py \
        --anchor-play runs/policy-play/chapter-1-room-1/<timestamp> \
        --anchor-donor runs/train/<donor run>/checkpoints/latest.zip --anchor-donor-sha256 <sha256> \
        --anchor-coef 1.0 --anchor-split-seed <k> --anchor-seed <k> \
        <every train_room1.py argument, for example --task-definition config/room2.json --init-from ... --run-dir ...>

Every train_room1.py argument keeps its meaning and its checks. The anchor arguments are consumed here: the recording
is verified against the donor's sha256 (same donor, unchanged, Room 1, clean, attributable, canonical starts only),
split by episode with --anchor-split-seed exactly as the SD-k clones were, and the frozen donor's probabilities
become the targets. The run is trained by AnchoredPPO instead of SupervisedPPO; its checkpoints load with plain
SupervisedPPO.load. The anchor's settings are recorded in the run manifest's session provenance, and a resumed run
refuses unless its anchor settings are identical. Per-update anchor records go to <run dir>/anchor.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import train_room1  # noqa: E402
from celeste_rl.tasks import resolve_task_definition, task_identity  # noqa: E402
from celeste_rl.training import run as run_module  # noqa: E402
from celeste_rl.training.anchor import AnchoredPPO, build_anchor  # noqa: E402


def parse_anchor_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--anchor-play", type=Path, required=True)
    parser.add_argument("--anchor-donor", type=Path, required=True)
    parser.add_argument("--anchor-donor-sha256", required=True)
    parser.add_argument("--anchor-coef", type=float, required=True)
    parser.add_argument("--anchor-holdout", type=float, default=0.25)
    parser.add_argument("--anchor-split-seed", type=int, required=True)
    parser.add_argument("--anchor-seed", type=int, required=True)
    args, rest = parser.parse_known_args(argv)
    if args.anchor_coef < 0:
        parser.error("--anchor-coef must be 0 or greater")
    return args, rest


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if "-h" in argv or "--help" in argv:
        print(__doc__)
        return 0
    args, rest = parse_anchor_args(argv)
    try:
        anchor = build_anchor(args.anchor_play, args.anchor_donor, args.anchor_donor_sha256,
                              task_identity(resolve_task_definition(None)), args.anchor_holdout,
                              args.anchor_split_seed, args.anchor_coef, args.anchor_seed)
    except (OSError, KeyError, json.JSONDecodeError, ValueError) as error:
        print(f"Cannot build the anchor: {error}")
        return 2
    print(f"Anchor: {anchor.record['fitted_frames']} fitted frames from {anchor.record['fitted_episodes']} episodes "
          f"({anchor.record['held_frames']} held back), lambda {anchor.coef}")

    original_train = train_room1.train

    def anchored_train(config, run_dir, env, provenance, **kwargs):
        run_dir = Path(run_dir)
        if kwargs.get("resume"):
            sessions = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["sessions"]
            if sessions[0]["provenance"].get("anchor") != json.loads(json.dumps(anchor.record)):
                raise ValueError("the resumed run was trained with different anchor settings")
        anchor.run_dir = run_dir
        AnchoredPPO.pending = anchor
        return original_train(config, run_dir, env, {**provenance, "anchor": anchor.record}, **kwargs)

    # run.train builds and loads its models through the name SupervisedPPO in its own module; point that name at
    # AnchoredPPO for this process only. Nothing the evaluators import is changed.
    run_module.SupervisedPPO = AnchoredPPO
    train_room1.train = anchored_train
    sys.argv = [sys.argv[0], *rest]
    return train_room1.main()


if __name__ == "__main__":
    sys.exit(main())
