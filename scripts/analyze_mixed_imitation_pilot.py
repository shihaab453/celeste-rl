"""Descriptive analysis of the mixed-imitation pilot, exactly as config/mixed-imitation-pilot.json declares.

Run from the repo root with the RL interpreter, after the evaluation campaign:
    .venv-rl/Scripts/python.exe scripts/analyze_mixed_imitation_pilot.py runs/campaign/<ts>-mixed-imitation-pilot-eval-v1/summary.json \
        --output <new path>

Room 1 (aggregates only): route-macro and state-weighted held-out success for every M-k clone and step checkpoint,
the retained share against the retention pilot's floor (reused, as declared), the imitation-only reference (4
Room 1-only clones) and the retention pilot's R-k curve beside it; then the pre-stated decision branch, applied
mechanically. Per-room held-back imitation accuracy for each mixed clone against the R-k clone (Room 2) and the
donor before cloning (Room 1). Room 2 (secondary): clone and final canonical clears, training clear rate per 100k
and the first 50k window at 50%, beside R-k. Critic record: explained variance over the first 10 updates.
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
from analyze_retention_pilot import explained_variance, retained_share, room2_training  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def decide(post_clone: list[float], post_clone_share: list[float], final_share: list[float],
           reference_mean: float) -> dict:
    """The declaration's branches A to D, in order, from Room 1 route-macro values of the four runs."""
    above_reference = sum(value - reference_mean > 0.10 for value in post_clone)
    near_reference = sum(abs(value - reference_mean) <= 0.10 for value in post_clone)
    median_clone, median_final = statistics.median(post_clone_share), statistics.median(final_share)
    facts = {"median_retained_share_after_cloning": round(median_clone, 4),
             "median_retained_share_at_500k": round(median_final, 4),
             "runs_more_than_10_points_above_reference": above_reference,
             "runs_within_10_points_of_reference": near_reference}
    if median_clone >= 0.5 and above_reference >= 3:
        erodes = median_clone - median_final >= 0.25
        return {"branch": "A", "next": "mixed PPO" if erodes else "mixed imitation suffices for now",
                "ppo_erodes": erodes, **facts}
    if near_reference >= 3:
        return {"branch": "B", "next": "a recipe that targets the donor's own play (for example self-distillation)",
                **facts}
    if median_clone < 0.2:
        return {"branch": "C", "next": "owner and orchestrator choose: self-distillation or a different weight",
                **facts}
    return {"branch": "D", "next": "report descriptively; orchestrator and owner decide", **facts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} already exists")
    pilot_path = REPO / "config/mixed-imitation-pilot.json"
    plan_path = REPO / "config/campaign-mixed-imitation-pilot-eval.json"
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

    def room1(entry_id: str) -> dict:
        r = results[entry_id]
        return {"route_macro": round(r["route_macro_success_rate"], 4), "state_weighted": round(r["success_rate"], 4),
                "successes": r["successes"]}

    retention = load(REPO / "docs/results/retention-pilot.json")
    floor = {stage: {m: retention["floor"][stage][m] for m in ("route_macro", "state_weighted")}
             for stage in ("clone", "final")}
    references = [room1(f"reference-room1-only-seed{k}") for k in range(4)]
    reference = {"per_seed": references,
                 **{m: round(sum(r[m] for r in references) / 4, 4) for m in ("route_macro", "state_weighted")}}
    before = load(REPO / "config/retention-pilot.json")["donors"]["before"]
    runs, post_clone, post_share, final_share = {}, [], [], []
    for k in range(4):
        donor = str(3 + k)
        base = {"route_macro": before["room1_heldout_route_macro"][donor],
                "state_weighted": before["room1_heldout_successes_of_200"][donor] / 200}
        ids = [f"room1-M-k{k}-clone"] + sorted(i for i in planned if i.startswith(f"room1-M-k{k}-step_"))
        curve = []
        for entry_id in ids:
            point = room1(entry_id)
            reference_floor = floor["clone" if entry_id.endswith("clone") else "final"]
            for m in ("route_macro", "state_weighted"):
                point[f"{m}_retained_share"] = retained_share(point[m], base[m], reference_floor[m])
            curve.append({"checkpoint": entry_id.split(f"M-k{k}-")[1], **point})
        post_clone.append(curve[0]["route_macro"])
        post_share.append(curve[0]["route_macro_retained_share"])
        final_share.append(curve[-1]["route_macro_retained_share"])
        clone = plan["clones"][str(k)]
        clone_record = load(REPO / clone["dir"] / "results.json")
        retention_clone = load(REPO / load(REPO / "config/campaign-retention-pilot-train.json")["clones"][str(k)]["dir"]
                               / "results.json")
        imitation = {"mixed_clone_after": {room: v["holdout"] for room, v in clone_record["after_per_room"].items()},
                     "donor_before": {room: v["holdout"] for room, v in clone_record["before_per_room"].items()},
                     "retention_clone_room2_after": retention_clone["cloning"]["history"][-1]["holdout"],
                     "train_frames": clone_record["mix"]["train_frames"]}
        run_dir = REPO / f"runs/train/mixed-imitation-pilot-M-k{k}"
        retention_dir = REPO / f"runs/train/retention-pilot-R-k{k}"
        runs[f"M-k{k}"] = {
            "donor_seed": 3 + k, "room1_before": base, "room1_curve": curve,
            "retention_R_k_curve": retention["runs"][f"R-k{k}"]["room1_curve"],
            "per_room_imitation_holdout": imitation,
            "room2": {"clone_clears_of_50": results[f"room2-M-k{k}-clone"]["successes"],
                      "final_clears_of_50": results[f"room2-M-k{k}-final"]["successes"],
                      "training": room2_training(run_dir), "retention_R_k_training": room2_training(retention_dir)},
            "explained_variance_first_10_updates": {"M-k": explained_variance(run_dir),
                                                    "R-k": explained_variance(retention_dir)}}
    report = {"label": load(pilot_path)["label"], "analysis_code_sha256": sha(Path(__file__)),
              "pilot_plan_sha256": sha(pilot_path), "evaluation_plan_sha256": summary["plan_sha256"],
              "evaluation_commit": summary["commit"], "floor_reused_from": "docs/results/retention-pilot.json",
              "floor": floor, "imitation_only_reference": reference,
              "decision": decide(post_clone, post_share, final_share, reference["route_macro"]), "runs": runs}
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"floor route-macro: clones {floor['clone']['route_macro']:.3f}, finals {floor['final']['route_macro']:.3f}; "
          f"imitation-only reference {reference['route_macro']:.3f} {[r['route_macro'] for r in references]}")
    for name, run in runs.items():
        cells = " ".join(f"{p['checkpoint'].replace('step_000', '')[:4]}:{p['route_macro']:.2f}" for p in run["room1_curve"])
        print(f"{name} before {run['room1_before']['route_macro']:.2f} | {cells} | Room 2 clone "
              f"{run['room2']['clone_clears_of_50']}/50 final {run['room2']['final_clears_of_50']}/50")
    print("decision:", json.dumps(report["decision"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
