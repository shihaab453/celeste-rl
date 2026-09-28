"""Checks on a policy's recorded play (scripts/record_policy_play.py) before its states are used for distillation.

A recording is a folder with dataset.npz (observations, applied actions, each frame's episode as `trajectory`) and
play.json (who played, from where, and under what runtime). Its states may stand in for a donor's own behaviour only
if the recording names that donor by sha256, is unchanged since it was written, is for the right task, came from a
clean tree and a pinned runtime, and started every episode at the canonical start (so no held-out state was read).
Shared by scripts/distill_policy.py and scripts/clone_room1.py --mix-play.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from celeste_rl.cloning import OBS_KEYS, Demonstrations

CANONICAL_ONLY = "canonical only"


def play_problems(record: dict, dataset_sha256: str, donor_sha256: str, identity: dict) -> list[str]:
    """Reasons a record_policy_play.py recording cannot be used as this donor's states (empty when it can)."""
    problems = []
    if record.get("checkpoint_sha256") != donor_sha256:
        problems.append("the play was recorded from a different policy")
    if record.get("dataset_sha256") != dataset_sha256:
        problems.append("dataset.npz changed since it was recorded")
    if record.get("task") != identity or record.get("uncommitted_changes") is not False:
        problems.append("the play is for another task or was recorded from a dirty tree")
    if record.get("attributable") is not True or record.get("runtime_problems"):
        problems.append("the play is not attributable (recorded with runtime problems or an override)")
    return problems


def verified_play(folder: Path, donor_sha256: str, identity: dict) -> tuple[Demonstrations, dict]:
    """Load a recording's pairs only after play_problems finds nothing and every episode began at the canonical start.

    Returns the pairs and a short record of where they came from. Raises ValueError naming every problem.
    """
    record = json.loads((folder / "play.json").read_text(encoding="utf-8"))
    dataset = folder / "dataset.npz"
    problems = play_problems(record, hashlib.sha256(dataset.read_bytes()).hexdigest(), donor_sha256, identity)
    if not str(record.get("starts", "")).startswith(CANONICAL_ONLY):
        problems.append("the play did not start only from the canonical start")
    if problems:
        raise ValueError(f"{folder}: " + "; ".join(problems))
    stored = np.load(dataset)
    data = Demonstrations({key: stored[f"obs_{key}"] for key in OBS_KEYS}, stored["actions"], stored["trajectory"])
    return data, {"kind": "donor play", "play": folder.as_posix(), "dataset_sha256": record["dataset_sha256"],
                  "episodes": record["episodes"], "seed": record["seed"], "recorded_at": record["commit"]}
