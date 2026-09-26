"""Descriptive analysis of the coverage-ceiling pilot, as config/coverage-ceiling-pilot.json declares (read-only).

Run from the repo root with the RL interpreter, after the evaluation campaign:
    .venv-rl/Scripts/python.exe scripts/analyze_coverage_ceiling.py runs/campaign/<ts>-coverage-ceiling-eval-v1/summary.json \
        --output <new path>

For the 8 fresh students (S-demo-k on the demonstration states, S-play-k on donor 3 + k's own play): Room 1 held-out
route-macro and state-weighted success (aggregates only), the retained share against the clone floor
((student - floor) / (donor before - floor)), the donor's before value and the imitation-only reference beside it, and
each student's open-loop agreement with its donor. Then the declared reading, applied mechanically: P when the median
S-play retained share is at least 0.5; otherwise coverage first (S-play-wide) before Q or R, with the provisional
Q or R shown. Also, descriptively, whether S-play beats S-demo by more than 10 points in at least 3 of 4 donors, and
how many held-out start states coincide with a recorded state (same task frame, position and dashes).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from celeste_rl.schema import PLAYER_FEATURE_NAMES  # noqa: E402

FEATURE = {name: index for index, name in enumerate(PLAYER_FEATURE_NAMES)}
ROOM1_BOUNDS = (0.0, 0.0, 320.0, 180.0)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def retained_share(after: float, before: float, floor: float) -> float:
    return round((after - floor) / (before - floor), 4)


def decide(play_shares: list[float]) -> dict:
    """The declared reading of the four S-play retained shares."""
    median = statistics.median(play_shares)
    if median >= 0.5:
        return {"branch": "P", "median_s_play_retained_share": round(median, 4),
                "next": "a mixed self-distillation pilot on donor-play states, cloning stage first"}
    provisional = "Q" if median >= 0.2 else "R"
    return {"branch": "coverage first", "provisional_after_s_play_wide": provisional,
            "median_s_play_retained_share": round(median, 4),
            "next": "S-play-wide (donor play also from training-side starts, never held-out states) before choosing Q or R"}


def exact_integers(normalised: np.ndarray, origin: float, size: float) -> np.ndarray:
    values = np.round(origin + normalised.astype(np.float64) * size).astype(np.int64)
    if not np.array_equal(((values - origin) / size).astype(np.float32), normalised.astype(np.float32)):
        raise ValueError("a recorded feature does not decode to an exact integer")
    return values


def recorded_states(dataset: Path) -> set[tuple[int, int, int, int]]:
    """(task frame, x, y, dashes) of every recorded frame; task frame = index within its episode."""
    stored = np.load(dataset)
    player = stored["obs_player"][:, 0, :]
    bx, by, bw, bh = ROOM1_BOUNDS
    xs = exact_integers(player[:, FEATURE["position_x"]], bx, bw)
    ys = exact_integers(player[:, FEATURE["position_y"]], by, bh)
    dashes = exact_integers(player[:, FEATURE["dashes"]], 0, 2)
    trajectory, states, frame, previous = stored["trajectory"], set(), 0, None
    for i, episode in enumerate(trajectory):
        frame = 0 if episode != previous else frame + 1
        previous = episode
        states.add((frame, int(xs[i]), int(ys[i]), int(dashes[i])))
    return states


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} already exists")
    plan_path = REPO / "config/campaign-coverage-ceiling-eval.json"
    plan, summary = load(plan_path), load(REPO / args.summary)
    if summary["plan_sha256"] != sha(plan_path):
        raise SystemExit("the evaluation summary does not match the committed evaluation plan")
    planned = {run["id"]: run for run in plan["runs"]}
    results = {}
    for entry in summary["results"]:
        if entry["status"] != "ok":
            raise SystemExit(f"{entry['id']} ended {entry['status']}")
        result_path = REPO / entry["artifact"]["result_file"]
        if sha(result_path) != entry["artifact"]["result_sha256"]:
            raise SystemExit(f"{entry['id']}: result file changed since the run")
        result = load(result_path)
        if result["checkpoint_sha256"] != planned[entry["id"]]["checkpoint_sha256"] or not result["attributable"]:
            raise SystemExit(f"{entry['id']}: wrong checkpoint or not attributable")
        results[entry["id"]] = result
    if set(results) != set(planned):
        raise SystemExit(f"missing evaluations: {sorted(set(planned) - set(results))}")

    before = load(REPO / "config/retention-pilot.json")["donors"]["before"]
    floor = load(REPO / "docs/results/retention-pilot.json")["floor"]["clone"]["route_macro"]
    reference = load(REPO / "docs/results/mixed-imitation-pilot.json")["imitation_only_reference"]["route_macro"]
    heldout = load(REPO / "config/heldout_starts.json")["entries"]
    students, play_shares, demo_shares = {}, [], []
    for k in range(4):
        donor = str(3 + k)
        donor_before = before["room1_heldout_route_macro"][donor]
        for kind in ("demo", "play"):
            entry_id = f"room1-S-{kind}-k{k}"
            r = results[entry_id]
            record = load(REPO / plan["students"][entry_id]["dir"] / "results.json")
            value = round(r["route_macro_success_rate"], 4)
            share = retained_share(value, donor_before, floor)
            (play_shares if kind == "play" else demo_shares).append(share)
            students[entry_id] = {"donor_seed": 3 + k, "route_macro": value, "state_weighted": round(r["success_rate"], 4),
                                  "retained_share": share, "donor_before": donor_before,
                                  "open_loop_agreement": record["after"]}
    coincide = {}
    for seed, folder in plan["recordings"].items():
        states = recorded_states(REPO / folder / "dataset.npz")
        coincide[seed] = sum((e["frames"], e["position"][0], e["position"][1], e["dashes"]) in states for e in heldout)
    play_beats_demo = sum(students[f"room1-S-play-k{k}"]["route_macro"] - students[f"room1-S-demo-k{k}"]["route_macro"]
                          > 0.10 for k in range(4))
    report = {"label": load(REPO / "config/coverage-ceiling-pilot.json")["label"],
              "analysis_code_sha256": sha(Path(__file__)), "evaluation_plan_sha256": summary["plan_sha256"],
              "evaluation_commit": summary["commit"], "clone_floor": floor, "imitation_only_reference": reference,
              "students": students, "decision": decide(play_shares),
              "descriptive": {"s_play_beats_s_demo_by_more_than_10_points": f"{play_beats_demo} of 4",
                              "median_s_demo_retained_share": round(statistics.median(demo_shares), 4),
                              "heldout_states_coinciding_with_a_recorded_state": coincide,
                              "heldout_states": len(heldout)}}
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"clone floor {floor:.3f}, imitation-only reference {reference:.3f}")
    for entry_id, s in students.items():
        print(f"{entry_id}: route-macro {s['route_macro']:.3f} (donor before {s['donor_before']:.3f}), "
              f"retained share {s['retained_share']:.3f}")
    print("decision:", json.dumps(report["decision"]))
    print("descriptive:", json.dumps(report["descriptive"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
