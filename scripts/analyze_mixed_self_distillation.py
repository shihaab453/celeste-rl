"""Descriptive analysis of the mixed self-distillation pilot, clone point, as config/mixed-self-distillation-pilot.json
declares (read-only).

Run from the repo root with the RL interpreter, after the clone evaluation campaign:
    .venv-rl/Scripts/python.exe scripts/analyze_mixed_self_distillation.py \
        runs/campaign/<ts>-mixed-self-distillation-clone-eval-v1/summary.json --output <new path>

For the 8 clones (SD-k: Room 2 demonstrations plus donor 3 + k's own recorded Room 1 play, both rooms weighted
equally, Room 1 fitted to the donor's own probabilities; D-k: the same with the Room 1 demonstration states instead):
Room 1 held-out route-macro and state-weighted success (aggregates only) and the retained share against the clone floor
((clone - floor) / (donor before - floor)); Room 2 canonical clears of 50; held-back imitation accuracy per room; the
training frames and per-frame weights. Beside them: M-k and R-k at the clone point, S-play-k and S-demo-k, and the
imitation-only reference. Then the declared branch on the median SD-k share: K (at least 0.5), Partial (0.2 to below
0.5), Lost (below 0.2). D-k is descriptive, never a branch. `ppo_drop` is the plan's one-unit drop for the PPO stage,
used only if K is reached.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from celeste_rl.texthash import matches_text_hash  # noqa: E402

ROOM1, ROOM2 = "chapter-1-room-1", "chapter-1-room-2"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def retained_share(after: float, before: float, floor: float) -> float:
    return round((after - floor) / (before - floor), 4)


def ppo_drop(clone_value: float, value_500k: float, before: float, clone_floor: float) -> float:
    """The PPO-stage drop in one unit: (clone - 500k) / (donor before - clone floor)."""
    return round((clone_value - value_500k) / (before - clone_floor), 4)


def decide_clone(sd_shares: list[float]) -> dict:
    """The declared reading of the four SD-k clone retained shares."""
    median = round(statistics.median(sd_shares), 4)
    if median >= 0.5:
        return {"branch": "K", "median_sd_retained_share": median,
                "next": "the PPO stage (declared fine-tuning), then K-hold, K-decay or K-cost on the one-unit drop"}
    if median >= 0.2:
        return {"branch": "Partial", "median_sd_retained_share": median,
                "next": "no PPO stage; more Room 1 weight or fewer epochs, chosen by the owner and the orchestrator"}
    return {"branch": "Lost", "median_sd_retained_share": median,
            "next": "no PPO stage; a weight anchor or frozen layers, chosen by the owner and the orchestrator"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} already exists")
    plan_path = REPO / "config/campaign-mixed-self-distillation-clone-eval.json"
    plan, summary = load(plan_path), load(REPO / args.summary)
    if not matches_text_hash(plan_path, summary["plan_sha256"]):
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
    retention = load(REPO / "docs/results/retention-pilot.json")
    mixed = load(REPO / "docs/results/mixed-imitation-pilot.json")
    coverage = load(REPO / "docs/results/coverage-ceiling-pilot.json")
    floor = retention["floor"]["clone"]["route_macro"]
    reference = mixed["imitation_only_reference"]["route_macro"]
    clones, shares = {}, {"SD": [], "D": []}
    for arm in ("SD", "D"):
        for k in range(4):
            donor = str(3 + k)
            donor_before = before["room1_heldout_route_macro"][donor]
            r1, r2 = results[f"room1-{arm}-k{k}-clone"], results[f"room2-{arm}-k{k}-clone"]
            record = load(REPO / plan["clones"][f"{arm}-k{k}"]["dir"] / "results.json")
            value = round(r1["route_macro_success_rate"], 4)
            share = retained_share(value, donor_before, floor)
            shares[arm].append(share)
            clones[f"{arm}-k{k}"] = {
                "donor_seed": 3 + k, "donor_before": donor_before,
                "room1": {"route_macro": value, "state_weighted": round(r1["success_rate"], 4),
                          "successes": r1["successes"], "retained_share": share},
                "room2_clone_clears_of_50": r2["successes"],
                "heldback_imitation": {room: v["holdout"] for room, v in record["after_per_room"].items()},
                "train_frames": record["mix"]["train_frames"], "per_frame_weight": record["mix"]["per_frame_weight"],
                "beside": {"M-k_clone_route_macro": mixed["runs"][f"M-k{k}"]["room1_curve"][0]["route_macro"],
                           "R-k_clone_route_macro": retention["runs"][f"R-k{k}"]["room1_curve"][0]["route_macro"],
                           "S-play-k_route_macro": coverage["students"][f"room1-S-play-k{k}"]["route_macro"],
                           "S-demo-k_route_macro": coverage["students"][f"room1-S-demo-k{k}"]["route_macro"],
                           "M-k_room2_clone_clears_of_50": mixed["runs"][f"M-k{k}"]["room2"]["clone_clears_of_50"],
                           "R-k_room2_clone_clears_of_50": retention["runs"][f"R-k{k}"]["room2"]["clone_clears_of_50"],
                           "M-k_room2_heldback": mixed["runs"][f"M-k{k}"]["per_room_imitation_holdout"]
                           ["mixed_clone_after"][ROOM2],
                           "R-k_room2_heldback": mixed["runs"][f"M-k{k}"]["per_room_imitation_holdout"]
                           ["retention_clone_room2_after"]}}
    report = {"label": load(REPO / "config/mixed-self-distillation-pilot.json")["label"],
              "analysis_code_sha256": sha(Path(__file__)), "evaluation_plan_sha256": summary["plan_sha256"],
              "evaluation_commit": summary["commit"], "clone_floor": floor, "imitation_only_reference": reference,
              "clones": clones, "decision": decide_clone(shares["SD"]),
              "descriptive": {"median_d_retained_share": round(statistics.median(shares["D"]), 4)}}
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"clone floor {floor:.3f}, imitation-only reference {reference:.3f}")
    for name, c in clones.items():
        print(f"{name}: Room 1 route-macro {c['room1']['route_macro']:.3f} (donor before {c['donor_before']:.3f}), "
              f"retained share {c['room1']['retained_share']:.3f}; Room 2 clears {c['room2_clone_clears_of_50']}/50")
    print("decision:", json.dumps(report["decision"]))
    print("descriptive:", json.dumps(report["descriptive"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
