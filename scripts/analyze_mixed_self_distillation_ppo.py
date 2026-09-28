"""Descriptive analysis of the mixed self-distillation pilot's PPO stage (branch K only), as
config/mixed-self-distillation-pilot.json declares (read-only).

Run from the repo root with the RL interpreter, after the PPO-stage evaluation campaign:
    .venv-rl/Scripts/python.exe scripts/analyze_mixed_self_distillation_ppo.py \
        runs/campaign/<ts>-mixed-self-distillation-ppo-eval-v1/summary.json \
        --clone-analysis docs/results/mixed-self-distillation-clone.json --output <new path>

For SD-k0 to SD-k3 (fine-tuned on Room 2 from the SD-k clones): the Room 1 held-out curve (the clone point from the
committed clone analysis, then every 100k checkpoint), aggregates only; each run's drop in one unit,
(clone - 500k) / (donor before - clone floor); Room 2 canonical final clears and training clear rates; the critic's
explained variance over the first 10 updates; and, descriptively, Room 2 v1 held-out success (a development set,
aggregates only) for the SD-k, M-k and R-k finals. Then the declared reading: K-hold (median drop at most 0.25 and at
least 3 of 4 Room 2 finals clear 45 of 50), K-decay (median drop above 0.25), otherwise K-cost. Step shares against the
final floor are reported descriptively only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from analyze_mixed_self_distillation import ppo_drop, retained_share  # noqa: E402
from analyze_retention_pilot import explained_variance, room2_training  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def decide_ppo(drops: list[float], room2_final_clears: list[int]) -> dict:
    """The declared reading after PPO: K-hold, K-decay or K-cost."""
    median = round(statistics.median(drops), 4)
    room2_ok = sum(clears >= 45 for clears in room2_final_clears)
    if median > 0.25:
        branch, next_step = "K-decay", "an anchor inside PPO (design reviewed; code and its own review first)"
    elif room2_ok >= 3:
        branch, next_step = "K-hold", "the fresh Room 1 v2 set before any claim"
    else:
        branch, next_step = "K-cost", "a Room 2 cost; the owner and the orchestrator choose"
    return {"branch": branch, "median_drop": median, "room2_finals_clearing_45_of_50": f"{room2_ok} of 4",
            "next": next_step}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--clone-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} already exists")
    plan_path = REPO / "config/campaign-mixed-self-distillation-ppo-eval.json"
    plan, summary = load(plan_path), load(REPO / args.summary)
    clone_analysis = load(REPO / args.clone_analysis)
    if summary["plan_sha256"] != sha(plan_path):
        raise SystemExit("the evaluation summary does not match the committed evaluation plan")
    if clone_analysis["decision"]["branch"] != "K":
        raise SystemExit("the clone-point analysis is not branch K, so there is no PPO stage to analyse")
    planned = {run["id"]: run for run in plan["runs"]}
    results = {}
    for item in summary["results"]:
        if item["status"] != "ok":
            raise SystemExit(f"{item['id']} ended {item['status']}")
        result_path = REPO / item["artifact"]["result_file"]
        if sha(result_path) != item["artifact"]["result_sha256"]:
            raise SystemExit(f"{item['id']}: result file changed since the run")
        result = load(result_path)
        if result["checkpoint_sha256"] != planned[item["id"]]["checkpoint_sha256"] or not result["attributable"]:
            raise SystemExit(f"{item['id']}: wrong checkpoint or not attributable")
        results[item["id"]] = result
    if set(results) != set(planned):
        raise SystemExit(f"missing evaluations: {sorted(set(planned) - set(results))}")

    retention = load(REPO / "docs/results/retention-pilot.json")
    clone_floor = retention["floor"]["clone"]["route_macro"]
    final_floor = retention["floor"]["final"]["route_macro"]
    runs, drops, room2_finals = {}, [], []
    for k in range(4):
        clone = clone_analysis["clones"][f"SD-k{k}"]
        before = clone["donor_before"]
        curve = [{"checkpoint": "clone", "route_macro": clone["room1"]["route_macro"],
                  "retained_share_clone_floor": clone["room1"]["retained_share"]}]
        for entry_id in sorted(i for i in planned if i.startswith(f"room1-SD-k{k}-step_")):
            r = results[entry_id]
            value = round(r["route_macro_success_rate"], 4)
            curve.append({"checkpoint": entry_id.split(f"SD-k{k}-")[1], "route_macro": value,
                          "state_weighted": round(r["success_rate"], 4), "successes": r["successes"],
                          "retained_share_final_floor_descriptive": retained_share(value, before, final_floor)})
        drop = ppo_drop(curve[0]["route_macro"], curve[-1]["route_macro"], before, clone_floor)
        drops.append(drop)
        clears = results[f"room2-SD-k{k}-final"]["successes"]
        room2_finals.append(clears)
        run_dir = REPO / f"runs/train/mixed-self-distillation-pilot-SD-k{k}"
        runs[f"SD-k{k}"] = {"donor_seed": 3 + k, "donor_before": before, "room1_curve": curve, "drop_one_unit": drop,
                            "room2": {"final_clears_of_50": clears, "training": room2_training(run_dir)},
                            "explained_variance_first_10_updates": explained_variance(run_dir)}
    headroom = {}
    for arm in ("SD", "M", "R"):
        for k in range(4):
            r = results[f"room2v1-{arm}-k{k}-final"]
            headroom[f"{arm}-k{k}"] = {"route_macro": round(r["route_macro_success_rate"], 4),
                                       "state_weighted": round(r["success_rate"], 4), "successes": r["successes"]}
    report = {"label": load(REPO / "config/mixed-self-distillation-pilot.json")["label"],
              "analysis_code_sha256": sha(Path(__file__)), "evaluation_plan_sha256": summary["plan_sha256"],
              "evaluation_commit": summary["commit"], "clone_analysis_sha256": sha(REPO / args.clone_analysis),
              "clone_floor": clone_floor, "final_floor": final_floor, "runs": runs,
              "room2_v1_heldout_finals_descriptive": headroom, "decision": decide_ppo(drops, room2_finals)}
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    for name, run in runs.items():
        points = ", ".join(f"{p['checkpoint']} {p['route_macro']:.3f}" for p in run["room1_curve"])
        print(f"{name}: Room 1 {points}; drop {run['drop_one_unit']:.3f}; Room 2 final {run['room2']['final_clears_of_50']}/50")
    print("Room 2 v1 held-out finals:", json.dumps({n: h["route_macro"] for n, h in headroom.items()}))
    print("decision:", json.dumps(report["decision"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
