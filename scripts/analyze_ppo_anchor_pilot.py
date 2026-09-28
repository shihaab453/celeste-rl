"""Descriptive analysis of the PPO anchor pilot, as config/ppo-anchor-pilot.json declares (read-only).

Run from the repo root with the RL interpreter, after the evaluation campaign:
    .venv-rl/Scripts/python.exe scripts/analyze_ppo_anchor_pilot.py \
        runs/campaign/<ts>-ppo-anchor-pilot-eval-v1/summary.json --output <new path>

For every arm (A1, A10, E0) and run k: the Room 1 held-out curve (the SD-k clone point from the committed clone
analysis, then every 100k), aggregates only; the one-unit drop (clone - 500k) / (donor before - clone floor); Room 2
canonical final clears; Room 2 v1 held-out finals. The control (the SD-k PPO runs) is read from the committed
docs/results/mixed-self-distillation-ppo.json. Then the declared branches per arm, applied mechanically:
- Holds: median drop at most 0.25, the Room 2 v1 median at least the M-k median minus 0.05 (the M-k median computed
  from the committed file), and at least 3 of 4 canonical finals at 45 or more;
- Holds with a Room 2 cost: the drop is at most 0.25 but a Room 2 condition fails;
- Partial: median drop above 0.25 and at most 0.6;
- Does not hold: median drop above 0.6.
and the declared reading of the arms together (amended before any data): rank by branch (a clean Holds beats Holds with
a Room 2 cost); within a branch prefer E0 unless its Room 2 v1 median is more than 0.05 below the best anchor arm's;
among anchor arms rank by branch, then median drop; and "tie" when the top two are within 0.15 on median drop and 0.05
on the Room 2 v1 median.

Record checks, refusing on failure: the evaluation summary matches the committed plan (line-ending safe); every
result is attributable and pins the planned checkpoint; the training summary matches the committed training plan with
every run ok at 10 threads; E0's manifest records ent_coef 0 and no anchor; every session of an anchored run carries
the identical anchor record with the planned lambda (check (b)); every update in anchor.csv has 16 minibatches
(check (c)); nothing names the fresh v2 held-out sets. Resumed runs are reported by name (sessions and runner
attempts), not refused: a resumed run is valid but not bit-identical to an uninterrupted one.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from celeste_rl.texthash import matches_text_hash, text_sha256  # noqa: E402

ARMS = ("A1", "A10", "E0")
LAMBDA = {"A1": 1.0, "A10": 10.0}
MINIBATCHES_PER_UPDATE = 16  # n_steps 2048 / batch 512 * n_epochs 4
FINAL = "step_000501760"
UNREAD_SETS = ("heldout_starts-room1-v2.json", "heldout_starts-room2-v2.json")  # kept unread by construction
RANK = {"Holds": 0, "Holds with a Room 2 cost": 1}
TIE_DROP, TIE_ROOM2 = 0.15, 0.05


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def one_unit_drop(clone: float, final: float, before: float, clone_floor: float) -> float:
    return round((clone - final) / (before - clone_floor), 4)


def decide_arm(drops: list[float], room2v1: list[float], finals: list[int], room2_threshold: float) -> dict:
    """The declared per-arm branch."""
    median_drop = round(statistics.median(drops), 4)
    median_v1 = round(statistics.median(room2v1), 4)
    finals_ok = sum(clears >= 45 for clears in finals)
    room2_ok = median_v1 >= room2_threshold and finals_ok >= 3
    if median_drop <= 0.25:
        branch = "Holds" if room2_ok else "Holds with a Room 2 cost"
    elif median_drop <= 0.6:
        branch = "Partial"
    else:
        branch = "Does not hold"
    return {"branch": branch, "median_drop": median_drop, "room2_v1_median": median_v1,
            "room2_v1_threshold": round(room2_threshold, 4), "room2_finals_at_45_or_more": f"{finals_ok} of 4"}


def read_together(decisions: dict[str, dict]) -> dict:
    """The declared reading (amended before any data): (a) rank by branch, (b) within a branch prefer E0 unless its
    Room 2 v1 median is more than 0.05 below the best anchor arm's, (c) anchor arms by branch then median drop,
    (d) "tie" when the top two are within 0.15 on median drop and 0.05 on the Room 2 v1 median."""
    candidates = [arm for arm in ARMS if decisions[arm]["branch"] in RANK]
    if not candidates:
        return {"lever": None, "preferred_arm": None, "candidates": [], "tie": False,
                "why": "nothing holds: next, anchor the value head too, or a weight anchor (each its own reviewed change)"}
    best_rank = min(RANK[decisions[arm]["branch"]] for arm in candidates)
    top = [arm for arm in candidates if RANK[decisions[arm]["branch"]] == best_rank]
    anchors = sorted((arm for arm in top if arm != "E0"), key=lambda arm: decisions[arm]["median_drop"])
    if "E0" in top and anchors:
        best_v1 = max(decisions[arm]["room2_v1_median"] for arm in anchors)
        order = (["E0", *anchors] if decisions["E0"]["room2_v1_median"] >= best_v1 - TIE_ROOM2
                 else [*anchors, "E0"])
    else:
        order = anchors or ["E0"]
    ranked = order + sorted((arm for arm in candidates if arm not in order),
                            key=lambda arm: (RANK[decisions[arm]["branch"]], decisions[arm]["median_drop"]))
    first = ranked[0]
    lever = "entropy control" if first == "E0" else "the anchor"
    if len(ranked) > 1:
        second = ranked[1]
        close = (abs(decisions[first]["median_drop"] - decisions[second]["median_drop"]) <= TIE_DROP
                 and abs(decisions[first]["room2_v1_median"] - decisions[second]["room2_v1_median"]) <= TIE_ROOM2)
        if close:
            return {"lever": "tie", "preferred_arm": None, "tied": [first, second], "candidates": ranked, "tie": True,
                    "why": (f"{first} and {second} are within {TIE_DROP} on median drop and {TIE_ROOM2} on the Room 2 v1 "
                            "median: noise with 4 runs per arm; the owner and the orchestrator choose, recorded as a "
                            "choice")}
    return {"lever": lever, "preferred_arm": first, "candidates": ranked, "tie": False,
            "why": f"{first} ranks first on branch, then the declared E0 preference or median drop"}


def anchored_run_problems(run_dir: Path, coef: float) -> list[str]:
    """Checks (b) and (c) for one anchored run."""
    problems = []
    sessions = load(run_dir / "manifest.json")["sessions"]
    records = [session["provenance"].get("anchor") for session in sessions]
    if any(record is None for record in records) or any(record != records[0] for record in records):
        problems.append(f"{run_dir}: not every session carries the identical anchor record")
    elif records[0].get("coef") != coef:
        problems.append(f"{run_dir}: anchor lambda {records[0].get('coef')}, planned {coef}")
    rows = list(csv.DictReader((run_dir / "anchor.csv").read_text(encoding="utf-8").splitlines()))
    wrong = [row["update"] for row in rows if int(row["minibatches"]) != MINIBATCHES_PER_UPDATE]
    if not rows or wrong:
        problems.append(f"{run_dir}: anchor.csv has {len(rows)} rows; updates without {MINIBATCHES_PER_UPDATE} "
                        f"minibatches: {wrong[:5]}")
    return problems


def e0_problems(run_dir: Path) -> list[str]:
    manifest = load(run_dir / "manifest.json")
    problems = []
    if manifest["config"].get("ent_coef") != 0:
        problems.append(f"{run_dir}: ent_coef {manifest['config'].get('ent_coef')}, planned 0")
    if any(session["provenance"].get("anchor") for session in manifest["sessions"]):
        problems.append(f"{run_dir}: E0 carries an anchor record")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} already exists")
    plan_path = REPO / "config/campaign-ppo-anchor-eval.json"
    plan, summary = load(plan_path), load(REPO / args.summary)
    if not matches_text_hash(plan_path, summary["plan_sha256"]):
        raise SystemExit("the evaluation summary does not match the committed evaluation plan")
    train_path = REPO / "config/campaign-ppo-anchor-train.json"
    train_summary = load(REPO / plan["training_campaign"])
    if (not matches_text_hash(train_path, train_summary["plan_sha256"])
            or any(r["status"] != "ok" for r in train_summary["results"])
            or train_summary.get("threads_per_job") != 10):
        raise SystemExit("the training summary does not match the committed training plan, has a run not ok, or did "
                         "not record 10 threads")
    planned = {run["id"]: run for run in plan["runs"]}
    named = [run["id"] for run in plan["runs"] if any(name in " ".join(run["command"]) for name in UNREAD_SETS)]
    if named:
        raise SystemExit(f"the evaluation plan names a fresh held-out set that must stay unread: {named}")
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
        if any(name in json.dumps(result) for name in UNREAD_SETS):
            raise SystemExit(f"{item['id']}: the result names a fresh held-out set that must stay unread")
        results[item["id"]] = result
    if set(results) != set(planned):
        raise SystemExit(f"missing evaluations: {sorted(set(planned) - set(results))}")

    problems = []
    for k in range(4):
        for arm in ARMS:
            folder = REPO / f"runs/train/ppo-anchor-pilot-{arm}-k{k}"
            problems += e0_problems(folder) if arm == "E0" else anchored_run_problems(folder, LAMBDA[arm])
    if problems:
        raise SystemExit("record checks failed:\n  " + "\n  ".join(problems))

    resumed = {}
    for record in train_summary["results"]:
        sessions = len(load(REPO / record["run_dir"] / "manifest.json")["sessions"])
        attempts = len(record.get("attempts", [])) or 1
        if sessions > 1 or attempts > 1:
            resumed[record["id"]] = {"sessions": sessions, "runner_attempts": attempts,
                                     "note": "valid, but not bit-identical to an uninterrupted run"}

    retention = load(REPO / "docs/results/retention-pilot.json")
    clone_floor, final_floor = retention["floor"]["clone"]["route_macro"], retention["floor"]["final"]["route_macro"]
    clones = load(REPO / "docs/results/mixed-self-distillation-clone.json")["clones"]
    control = load(REPO / "docs/results/mixed-self-distillation-ppo.json")
    m_k = [control["room2_v1_heldout_finals_descriptive"][f"M-k{k}"]["route_macro"] for k in range(4)]
    threshold = statistics.median(m_k) - 0.05
    runs, decisions = {}, {}
    for arm in ARMS:
        drops, room2v1, finals = [], [], []
        for k in range(4):
            clone = clones[f"SD-k{k}"]
            curve = [{"checkpoint": "clone", "route_macro": clone["room1"]["route_macro"]}]
            for entry_id in sorted(i for i in planned if i.startswith(f"room1-{arm}-k{k}-step_")):
                r = results[entry_id]
                curve.append({"checkpoint": entry_id.split(f"{arm}-k{k}-")[1], "route_macro": round(r["route_macro_success_rate"], 4),
                              "successes": r["successes"]})
            if curve[-1]["checkpoint"] != FINAL:
                raise SystemExit(f"{arm}-k{k}: no {FINAL} evaluation")
            drop = one_unit_drop(curve[0]["route_macro"], curve[-1]["route_macro"], clone["donor_before"], clone_floor)
            v1 = round(results[f"room2v1-{arm}-k{k}-final"]["route_macro_success_rate"], 4)
            clears = results[f"room2-{arm}-k{k}-final"]["successes"]
            drops.append(drop), room2v1.append(v1), finals.append(clears)
            runs[f"{arm}-k{k}"] = {"donor_seed": 3 + k, "donor_before": clone["donor_before"], "room1_curve": curve,
                                   "drop_one_unit": drop, "room2_final_clears_of_50": clears, "room2_v1_final": v1}
        decisions[arm] = decide_arm(drops, room2v1, finals, threshold)
    control_row = {"median_drop": control["decision"]["median_drop"],
                   "room2_v1_median": round(statistics.median(
                       control["room2_v1_heldout_finals_descriptive"][f"SD-k{k}"]["route_macro"] for k in range(4)), 4),
                   "room2_finals": [control["runs"][f"SD-k{k}"]["room2"]["final_clears_of_50"] for k in range(4)]}
    report = {"label": load(REPO / "config/ppo-anchor-pilot.json")["label"],
              "analysis_code_sha256": text_sha256(Path(__file__)), "evaluation_plan_sha256": summary["plan_sha256"],
              "evaluation_commit": summary["commit"], "training_commit": train_summary["commit"],
              "clone_floor": clone_floor, "final_floor": final_floor,
              "room2_v1_reference": {"m_k_median": round(statistics.median(m_k), 4), "threshold": round(threshold, 4)},
              "runs": runs, "decisions": decisions, "control": control_row, "reading": read_together(decisions),
              "resumed_runs": resumed or "none",
              "record_checks": "passed: plan hashes, attributable results, 10 threads, E0 ent_coef 0, (b), (c)"}
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    for arm, decision in decisions.items():
        print(f"{arm}: {json.dumps(decision)}")
    print("control:", json.dumps(control_row))
    print("reading:", json.dumps(report["reading"]))
    print("resumed runs:", json.dumps(report["resumed_runs"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
