"""Two descriptive numbers for the coverage-ceiling write-up that the analyzer does not compute (read-only, no game).

Run from the repo root with the RL interpreter:
    .venv-rl/Scripts/python.exe scripts/describe_coverage_ceiling.py

Inputs come from the committed records: config/campaign-coverage-ceiling-eval.json (the eight students and the four
recordings), config/retention-pilot.json (the donors) and each student's results.json (its dataset, seed and fitted
frame count). Every dataset is checked against the sha256 its record names, and every fitted split is rebuilt with
the same split_by_trajectory call and checked against the student's recorded fitted frame count.

1. **Donor choices against the demonstrated ones.** On all Room 1 demonstration frames, how often each donor's most
   likely buttons (probability at least 0.5) equal the demonstrated buttons: per button and for whole frames (all
   buttons at once), enabled inputs only.
2. **How close the held-out starts are to each copy's fitted frames.** For each of the 200 Room 1 held-out starts,
   the distance in pixels to the nearest frame the copy was fitted on with the same dash count. Position only: speed
   and timers are not compared (the held-out manifest does not store them). Room 1's bounds are 320 by 180 pixels
   from the origin, so a normalised position times 320 or 180 is the pixel position; every recorded episode's first
   frame must convert to the canonical start (19, 144) with 1 dash, which checks that conversion.

Writes docs/results/coverage-ceiling-descriptive.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch as th

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from celeste_rl.cloning import OBS_KEYS, Demonstrations, split_by_trajectory  # noqa: E402
from celeste_rl.schema import ACTION_INPUTS, MENU_INPUTS, PLAYER_FEATURE_NAMES  # noqa: E402
from celeste_rl.training.supervisor import SupervisedPPO  # noqa: E402

CAMPAIGN = REPO / "config" / "campaign-coverage-ceiling-eval.json"
DONORS = REPO / "config" / "retention-pilot.json"
HELDOUT = REPO / "config" / "heldout_starts.json"
OUTPUT = REPO / "docs" / "results" / "coverage-ceiling-descriptive.json"
ROOM_W, ROOM_H, CANONICAL = 320, 180, (19, 144, 1)
ENABLED = [i for i, name in enumerate(ACTION_INPUTS) if name not in MENU_INPUTS]
FEATURE = {name: i for i, name in enumerate(PLAYER_FEATURE_NAMES)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path, expected_sha256: str) -> Demonstrations:
    if sha256(path) != expected_sha256:
        raise SystemExit(f"{path} does not match its recorded sha256")
    stored = np.load(path)
    return Demonstrations({key: stored[f"obs_{key}"] for key in OBS_KEYS}, stored["actions"], stored["trajectory"])


def pixels(data: Demonstrations) -> np.ndarray:
    """(x, y, dashes) per frame, from the current frame of the player features."""
    player = data.obs["player"][:, 0, :].astype(np.float64)
    return np.stack([np.round(player[:, FEATURE["position_x"]] * ROOM_W), np.round(player[:, FEATURE["position_y"]] * ROOM_H),
                     np.round(player[:, FEATURE["dashes"]] * 2)], axis=1)


def likely_buttons(policy, data: Demonstrations) -> np.ndarray:
    policy.set_training_mode(False)
    chunks = []
    with th.no_grad():
        for start in range(0, len(data), 512):
            batch = {key: th.as_tensor(value[start:start + 512]) for key, value in data.obs.items()}
            chunks.append(policy.get_distribution(batch).distribution.probs.numpy())
    return np.concatenate(chunks)[:, ENABLED] >= 0.5


def distances(fitted: np.ndarray, starts: np.ndarray) -> tuple[np.ndarray, int]:
    """Distance from each start to the nearest fitted frame with its dash count (any frame if none has it)."""
    result, fallbacks = [], 0
    for x, y, dashes in starts:
        pool = fitted[fitted[:, 2] == dashes]
        if not len(pool):
            pool, fallbacks = fitted, fallbacks + 1
        result.append(np.sqrt(((pool[:, :2] - (x, y)) ** 2).sum(axis=1)).min())
    return np.array(result), fallbacks


def main() -> int:
    campaign = json.loads(CAMPAIGN.read_text(encoding="utf-8"))
    donors = json.loads(DONORS.read_text(encoding="utf-8"))["donors"]["runs"]
    heldout = json.loads(HELDOUT.read_text(encoding="utf-8"))["entries"]
    starts = np.array([[e["position"][0], e["position"][1], e["dashes"]] for e in heldout], dtype=np.float64)
    out = {"label": "descriptive; companion to docs/results/coverage-ceiling-pilot.json; aggregates only",
           "analysis_code_sha256": sha256(Path(__file__)), "inputs": {}, "donor_vs_demonstrations": {}, "coverage": {}}
    for path in (CAMPAIGN, DONORS, HELDOUT):
        out["inputs"][path.relative_to(REPO).as_posix()] = sha256(path)

    demo_record = None
    for name, student in campaign["students"].items():
        record = json.loads((REPO / student["dir"] / "results.json").read_text(encoding="utf-8"))
        states = record["states"]
        if states["kind"] == "demonstration routes":
            data = load(REPO / states["dataset"], states["dataset_sha256"])
            demo_record = demo_record or (states["dataset"], states["dataset_sha256"])
        else:
            play = REPO / states["play"]
            data = load(play / "dataset.npz", states["dataset_sha256"])
            firsts = {tuple(row) for row in pixels(data)[np.r_[0, np.flatnonzero(np.diff(data.trajectory)) + 1]]}
            if firsts != {CANONICAL}:
                raise SystemExit(f"{play}: episode first frames {sorted(firsts)} are not the canonical start {CANONICAL}")
        fitted, _ = split_by_trajectory(data, float(record["args"]["holdout"]), int(record["args"]["seed"]))
        if len(fitted) != record["after"]["fitted"]["frames"]:
            raise SystemExit(f"{name}: rebuilt {len(fitted)} fitted frames, the record says {record['after']['fitted']['frames']}")
        d, fallbacks = distances(pixels(fitted), starts)
        out["coverage"][name] = {"fitted_frames": len(fitted), "median_px": round(float(np.median(d)), 2),
                                 "within_4px": round(float(np.mean(d <= 4)), 3), "within_8px": round(float(np.mean(d <= 8)), 3),
                                 "starts_without_same_dash_frame": fallbacks}

    demo = load(REPO / demo_record[0], demo_record[1])
    demonstrated = demo.actions[:, ENABLED].astype(bool)
    out["inputs"][demo_record[0]] = demo_record[1]
    for seed, info in sorted(donors.items()):
        if sha256(REPO / info["checkpoint"]) != info["sha256"]:
            raise SystemExit(f"donor {seed} does not match its pinned sha256")
        same = likely_buttons(SupervisedPPO.load(REPO / info["checkpoint"], device="cpu").policy, demo) == demonstrated
        out["donor_vs_demonstrations"][seed] = {"frames": len(demo), "button_match": round(float(same.mean()), 3),
                                                "whole_frame_match": round(float(same.all(axis=1).mean()), 3)}

    OUTPUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    for seed, row in out["donor_vs_demonstrations"].items():
        print(f"donor {seed}: buttons match {row['button_match']:.3f}, whole frames {row['whole_frame_match']:.3f}")
    for name, row in out["coverage"].items():
        print(f"{name}: {row['fitted_frames']} fitted frames, median {row['median_px']} px, "
              f"within 4 px {row['within_4px']:.2f}, within 8 px {row['within_8px']:.2f}")
    print(f"Results: {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
