"""Descriptive analysis of the retention pilot, exactly as config/retention-pilot.json declares (read-only).

Run from the repo root with the RL interpreter, after the evaluation campaign:
    .venv-rl/Scripts/python.exe scripts/analyze_retention_pilot.py runs/campaign/<ts>-retention-pilot-eval-v1/summary.json \
        --output <new path>

Room 1 (aggregates only): route-macro and state-weighted held-out success for every R-k clone and step checkpoint,
the drop from the donor's before value, the floor (mean over H2 B-k0 to B-k3 finals for fine-tuned points, over
their clones for the post-clone point) and the retained share (after - floor) / (before - floor). Room 2
(secondary): canonical clears of 50 for each R-k clone and final, training clear rate per 100k window and the first
50k window at 50%, beside H2 B-k. Critic record: explained variance over the first 10 PPO updates for R-k and H2 B-k.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from summarize_room2_stability import first_window_reaching, training_windows  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def retained_share(after: float, before: float, floor: float) -> float:
    """Share of the skill above the floor that remains: 1 kept everything, 0 is back at the floor."""
    return round((after - floor) / (before - floor), 4)


def explained_variance(run_dir: Path, updates: int = 10) -> list[float]:
    rows = list(csv.DictReader((run_dir / "progress.csv").read_text(encoding="utf-8").splitlines()))
    return [round(float(row["explained_variance"]), 3) for row in rows if row["explained_variance"]][:updates]


def room2_training(run_dir: Path) -> dict:
    episodes = [json.loads(line) for line in (run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    windows = training_windows(episodes)
    return {"clear_per_100k": [round(w["success"] / w["episodes"], 3) for w in windows],
            "first_50k_window_at_half": first_window_reaching(episodes)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} already exists")
    pilot = load(REPO / "config/retention-pilot.json")
    plan_path = REPO / "config/campaign-retention-pilot-eval.json"
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
                "successes": r["successes"], "episodes": r["episodes"]}

    floor = {"clone": {}, "final": {}}
    for stage in ("clone", "final"):
        points = [room1(f"floor-room1-H2-B-k{k}-{stage}") for k in range(4)]
        floor[stage] = {"per_run": points, **{m: round(sum(p[m] for p in points) / 4, 4)
                                              for m in ("route_macro", "state_weighted")}}
    before = pilot["donors"]["before"]
    runs = {}
    for k in range(4):
        donor = str(3 + k)
        base = {"route_macro": before["room1_heldout_route_macro"][donor],
                "state_weighted": before["room1_heldout_successes_of_200"][donor] / 200}
        curve = []
        ids = [f"room1-R-k{k}-clone"] + sorted(i for i in planned if i.startswith(f"room1-R-k{k}-step_"))
        for entry_id in ids:
            point = room1(entry_id)
            reference = floor["clone" if entry_id.endswith("clone") else "final"]
            for m in ("route_macro", "state_weighted"):
                point[f"{m}_drop"] = round(base[m] - point[m], 4)
                point[f"{m}_retained_share"] = retained_share(point[m], base[m], reference[m])
            curve.append({"checkpoint": entry_id.split(f"R-k{k}-")[1], **point})
        run_dir = REPO / f"runs/train/retention-pilot-R-k{k}"
        h2_dir = REPO / f"runs/train/room2-demos-stall-B-k{k}"
        runs[f"R-k{k}"] = {
            "donor_seed": 3 + k, "room1_before": base, "room1_curve": curve,
            "room2": {"clone_clears_of_50": results[f"room2-R-k{k}-clone"]["successes"],
                      "final_clears_of_50": results[f"room2-R-k{k}-final"]["successes"],
                      "training": room2_training(run_dir), "h2_B_k_training": room2_training(h2_dir)},
            "explained_variance_first_10_updates": {"R-k": explained_variance(run_dir),
                                                    "H2_B-k": explained_variance(h2_dir)}}
    report = {"label": pilot["label"], "analysis_code_sha256": sha(Path(__file__)),
              "pilot_plan_sha256": sha(REPO / "config/retention-pilot.json"),
              "evaluation_plan_sha256": summary["plan_sha256"], "evaluation_commit": summary["commit"],
              "aggregates_only": pilot["aggregates_only"], "floor": floor, "runs": runs}
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"floor route-macro: clones {floor['clone']['route_macro']:.3f}, finals {floor['final']['route_macro']:.3f}")
    for name, run in runs.items():
        cells = " ".join(f"{p['checkpoint'].replace('step_000', '')[:4]}:{p['route_macro']:.2f}" for p in run["room1_curve"])
        print(f"{name} before {run['room1_before']['route_macro']:.2f} | {cells} | Room 2 clone "
              f"{run['room2']['clone_clears_of_50']}/50 final {run['room2']['final_clears_of_50']}/50")
    return 0


if __name__ == "__main__":
    sys.exit(main())
