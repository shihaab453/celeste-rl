"""Verify and analyse the predeclared six-pair held-out A/B campaign.

This script intentionally implements only the analysis declared in the campaign plan. It verifies every
artifact and row before calculating results, then writes one machine-readable analysis.json beside the campaign
summary. It does not fall back to a different checkpoint, held-out set, missing-row rule, or statistical test.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from celeste_rl.heldout import validate_manifest  # noqa: E402


class AnalysisError(ValueError):
    """The campaign does not satisfy its predeclared analysis contract."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_sign_flip_pvalue(pair_differences: list[float]) -> float:
    """Exact two-sided paired randomisation test over all 2^n sign assignments."""
    if not pair_differences:
        raise AnalysisError("the paired test needs at least one seed pair")
    observed = abs(sum(pair_differences) / len(pair_differences))
    assignments = list(itertools.product((-1, 1), repeat=len(pair_differences)))
    at_least_as_extreme = sum(
        abs(sum(sign * value for sign, value in zip(signs, pair_differences)) / len(pair_differences))
        >= observed - 1e-15
        for signs in assignments
    )
    return at_least_as_extreme / len(assignments)


def percentile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def crossed_cluster_interval(matrix: list[list[float]], samples: int, seed: int) -> tuple[float, float]:
    """Percentile interval after independently resampling paired seeds and complete route clusters."""
    if samples <= 0 or not matrix or not matrix[0]:
        raise AnalysisError("crossed cluster bootstrap needs samples, seeds, and routes")
    route_count = len(matrix[0])
    if any(len(row) != route_count for row in matrix):
        raise AnalysisError("crossed cluster matrix is ragged")
    rng = random.Random(seed)
    seed_count = len(matrix)
    draws = []
    denominator = seed_count * route_count
    for _ in range(samples):
        selected_seeds = [rng.randrange(seed_count) for _ in range(seed_count)]
        selected_routes = [rng.randrange(route_count) for _ in range(route_count)]
        draws.append(sum(matrix[s][r] for s in selected_seeds for r in selected_routes) / denominator)
    draws.sort()
    return percentile(draws, 0.025), percentile(draws, 0.975)


def _repo_path(value: str) -> Path:
    repository = REPO.resolve()
    path = Path(value)
    if not path.is_absolute():
        path = repository / path
    path = path.resolve()
    try:
        path.relative_to(repository)
    except ValueError as error:
        raise AnalysisError(f"artifact is outside the repository: {value}") from error
    return path


def same_repo_path(left: str, right: str) -> bool:
    """Compare repository paths by identity, not by the slash style emitted on this platform."""
    return _repo_path(left) == _repo_path(right)


def _jsonl(path: Path) -> list[dict]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisError(f"cannot read episode rows {path}: {error}") from error


def validate_run(plan_entry: dict, campaign_record: dict, campaign_commit: str, protocol: dict, heldout: dict,
                 heldout_by_id: dict[str, dict]) -> tuple[dict, list[dict]]:
    run_id = plan_entry["id"]
    if campaign_record.get("status") != "ok":
        raise AnalysisError(f"{run_id} did not complete successfully: {campaign_record.get('status')}")
    artifact = campaign_record.get("artifact")
    if not isinstance(artifact, dict) or artifact.get("problem"):
        raise AnalysisError(f"{run_id} has no valid captured result artifact")
    result_path = _repo_path(artifact["result_file"])
    episodes_path = _repo_path(artifact["episodes_file"])
    if file_sha256(result_path) != artifact.get("result_sha256"):
        raise AnalysisError(f"{run_id} result hash does not match the campaign summary")
    if file_sha256(episodes_path) != artifact.get("episodes_sha256"):
        raise AnalysisError(f"{run_id} episode hash does not match the campaign summary")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (result.get("commit") != campaign_commit or result.get("uncommitted_changes") is not False
            or result.get("runtime_problems") != [] or result.get("attributable") is not True):
        raise AnalysisError(f"{run_id} is not attributable to the clean campaign commit and pinned runtime")
    expected = {
        "checkpoint": plan_entry["checkpoint"],
        "checkpoint_sha256": plan_entry["checkpoint_sha256"],
        "heldout_set": protocol["heldout_set"],
        "heldout_sha256": heldout["sha256"],
        "heldout_format_version": heldout["format_version"],
        "heldout_states": heldout["states"],
        "repeats": protocol["repeats"],
        "deterministic": protocol["deterministic"],
        "evaluation_seed": protocol["evaluation_seed"],
    }
    for field, value in expected.items():
        matches = (same_repo_path(result.get(field), value) if field in {"checkpoint", "heldout_set"}
                   and isinstance(result.get(field), str) else result.get(field) == value)
        if not matches:
            raise AnalysisError(f"{run_id} result {field} is {result.get(field)!r}; expected {value!r}")
    rows = _jsonl(episodes_path)
    expected_ids = set(heldout_by_id)
    actual_ids = [row.get("state_id") for row in rows]
    if len(rows) != len(expected_ids) or set(actual_ids) != expected_ids or len(actual_ids) != len(set(actual_ids)):
        raise AnalysisError(f"{run_id} does not contain exactly one row for every held-out state")
    for row in rows:
        state = heldout_by_id[row["state_id"]]
        if row.get("problem") is not None:
            raise AnalysisError(f"{run_id} contains a stale or failed start {row['state_id']}: {row['problem']}")
        if row.get("repeat") != 0:
            raise AnalysisError(f"{run_id} contains an undeclared repeat index")
        if row.get("checkpoint_sha256") != plan_entry["checkpoint_sha256"]:
            raise AnalysisError(f"{run_id} row checkpoint hash mismatch")
        if row.get("route") != state["route"] or row.get("search_seed") != state["search_seed"]:
            raise AnalysisError(f"{run_id} row provenance mismatch for {row['state_id']}")
    return result, rows


def analyse(run_rows: dict[tuple[int, str], list[dict]], seeds: list[int], routes: list[str],
            bootstrap_samples: int, bootstrap_seed: int) -> dict:
    route_rates: dict[tuple[int, str], dict[str, float]] = {}
    per_seed = {}
    for seed in seeds:
        per_seed[str(seed)] = {}
        for arm in ("A", "B"):
            rows = run_rows[(seed, arm)]
            grouped = {route: [row for row in rows if row["route"] == route] for route in routes}
            if any(not values for values in grouped.values()):
                raise AnalysisError(f"seed {seed} arm {arm} is missing a route cluster")
            rates = {route: sum(row["ending"] == "success" for row in values) / len(values)
                     for route, values in grouped.items()}
            route_rates[(seed, arm)] = rates
            successes = sum(row["ending"] == "success" for row in rows)
            per_seed[str(seed)][arm] = {
                "successes": successes,
                "episodes": len(rows),
                "state_weighted_success_rate": successes / len(rows),
                "route_macro_success_rate": sum(rates.values()) / len(routes),
                "endings": dict(sorted(Counter(row["ending"] for row in rows).items())),
            }
        per_seed[str(seed)]["B_minus_A_route_macro"] = (
            per_seed[str(seed)]["B"]["route_macro_success_rate"]
            - per_seed[str(seed)]["A"]["route_macro_success_rate"])
        per_seed[str(seed)]["B_minus_A_state_weighted"] = (
            per_seed[str(seed)]["B"]["state_weighted_success_rate"]
            - per_seed[str(seed)]["A"]["state_weighted_success_rate"])

    difference_matrix = [
        [route_rates[(seed, "B")][route] - route_rates[(seed, "A")][route] for route in routes]
        for seed in seeds
    ]
    pair_differences = [per_seed[str(seed)]["B_minus_A_route_macro"] for seed in seeds]
    lower, upper = crossed_cluster_interval(difference_matrix, bootstrap_samples, bootstrap_seed)

    per_route = {}
    for route in routes:
        arm_values = {}
        for arm in ("A", "B"):
            rows = [row for seed in seeds for row in run_rows[(seed, arm)] if row["route"] == route]
            successes = sum(row["ending"] == "success" for row in rows)
            arm_values[arm] = {"successes": successes, "episodes": len(rows),
                               "success_rate": successes / len(rows)}
        per_route[route] = {**arm_values,
                            "B_minus_A": arm_values["B"]["success_rate"] - arm_values["A"]["success_rate"]}

    arms = {}
    for arm in ("A", "B"):
        rows = [row for seed in seeds for row in run_rows[(seed, arm)]]
        successes = sum(row["ending"] == "success" for row in rows)
        arms[arm] = {
            "successes": successes,
            "episodes": len(rows),
            "state_weighted_success_rate": successes / len(rows),
            "route_macro_success_rate": sum(
                route_rates[(seed, arm)][route] for seed in seeds for route in routes
            ) / (len(seeds) * len(routes)),
        }

    primary_difference = sum(pair_differences) / len(pair_differences)
    return {
        "primary": {
            "estimand": "mean paired B-minus-A route-macro success-rate difference",
            "A_route_macro_success_rate": arms["A"]["route_macro_success_rate"],
            "B_route_macro_success_rate": arms["B"]["route_macro_success_rate"],
            "B_minus_A": primary_difference,
            "pair_differences": {str(seed): difference for seed, difference in zip(seeds, pair_differences)},
            "exact_two_sided_paired_sign_flip_p": exact_sign_flip_pvalue(pair_differences),
            "crossed_seed_route_bootstrap_95_interval": [lower, upper],
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
        },
        "secondary": {
            "arms": arms,
            "B_minus_A_state_weighted": (
                arms["B"]["state_weighted_success_rate"] - arms["A"]["state_weighted_success_rate"]),
            "per_seed": per_seed,
            "per_route": per_route,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        campaign = json.loads(args.campaign.read_text(encoding="utf-8"))
        plan_hash = file_sha256(args.plan)
        if campaign.get("plan") != plan.get("name") or campaign.get("plan_sha256") != plan_hash:
            raise AnalysisError("campaign summary does not identify the exact predeclared plan")
        protocol = plan["evaluation_protocol"]
        heldout_path = _repo_path(protocol["heldout_set"])
        heldout = json.loads(heldout_path.read_text(encoding="utf-8"))
        entries = validate_manifest(heldout)
        heldout_metadata = {
            "sha256": protocol["heldout_sha256"],
            "format_version": protocol["heldout_format_version"],
            "states": protocol["heldout_states"],
        }
        for field, expected in heldout_metadata.items():
            if heldout.get(field) != expected:
                raise AnalysisError(f"held-out manifest {field} differs from the predeclared plan")
        route_hashes = {entry["route_sha256"] for entry in entries}
        if (len(heldout["routes"]) != protocol["independent_route_hashes"]
                or len(route_hashes) != protocol["independent_route_hashes"]):
            raise AnalysisError("held-out route count differs from the predeclared plan")
        for route in heldout["routes"]:
            hashes = {entry["route_sha256"] for entry in entries if entry["route"] == route}
            if len(hashes) != 1:
                raise AnalysisError(f"held-out route {route} does not identify exactly one route hash")
        heldout_by_id = {entry["state_id"]: entry for entry in entries}
        planned_runs = {entry["id"]: entry for entry in plan["runs"]}
        if len(planned_runs) != len(plan["runs"]):
            raise AnalysisError("predeclared run IDs are not unique")
        campaign_runs = {entry["id"]: entry for entry in campaign["results"]}
        if len(campaign_runs) != len(campaign["results"]) or set(planned_runs) != set(campaign_runs):
            raise AnalysisError("campaign entries differ from the predeclared run list")
        run_rows = {}
        run_artifacts = {}
        for run_id, plan_entry in planned_runs.items():
            checkpoint_path = _repo_path(plan_entry["checkpoint"])
            if file_sha256(checkpoint_path) != plan_entry["checkpoint_sha256"]:
                raise AnalysisError(f"{run_id} checkpoint differs from its predeclared hash")
            result, rows = validate_run(plan_entry, campaign_runs[run_id], campaign["commit"], protocol,
                                        heldout, heldout_by_id)
            key = (plan_entry["train_seed"], plan_entry["arm"])
            if key in run_rows:
                raise AnalysisError(f"duplicate paired cell {key}")
            run_rows[key] = rows
            run_artifacts[run_id] = {
                "checkpoint": result["checkpoint"],
                "checkpoint_sha256": result["checkpoint_sha256"],
                "results": campaign_runs[run_id]["artifact"]["result_file"],
                "results_sha256": campaign_runs[run_id]["artifact"]["result_sha256"],
                "episodes": campaign_runs[run_id]["artifact"]["episodes_file"],
                "episodes_sha256": campaign_runs[run_id]["artifact"]["episodes_sha256"],
            }
        declaration = plan["analysis"]
        seeds = declaration["matched_training_seeds"]
        routes = heldout["routes"]
        expected_cells = {(seed, arm) for seed in seeds for arm in ("A", "B")}
        if set(run_rows) != expected_cells:
            raise AnalysisError("campaign does not contain all predeclared matched A/B cells")
        statistics = analyse(run_rows, seeds, routes, declaration["bootstrap_samples"],
                             declaration["bootstrap_seed"])
        report = {
            "plan": str(args.plan),
            "plan_sha256": plan_hash,
            "campaign": str(args.campaign),
            "campaign_commit": campaign["commit"],
            "heldout_set": protocol["heldout_set"],
            "heldout_sha256": heldout["sha256"],
            "validated_runs": len(run_rows),
            "validated_episodes": sum(len(rows) for rows in run_rows.values()),
            "heldout_states_per_run": len(entries),
            "independent_routes": len(routes),
            "run_artifacts": run_artifacts,
            **statistics,
        }
        output = args.output or args.campaign.with_name("analysis.json")
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except (OSError, KeyError, json.JSONDecodeError, AnalysisError, ValueError) as error:
        print(f"Analysis refused: {error}")
        return 2

    primary = report["primary"]
    arms = report["secondary"]["arms"]
    print(f"Validated {report['validated_runs']} runs and {report['validated_episodes']} episode rows")
    print(f"Route-macro A {primary['A_route_macro_success_rate']:.1%}, "
          f"B {primary['B_route_macro_success_rate']:.1%}, difference {primary['B_minus_A']:+.1%}")
    print(f"Crossed seed-route bootstrap 95% interval "
          f"[{primary['crossed_seed_route_bootstrap_95_interval'][0]:+.1%}, "
          f"{primary['crossed_seed_route_bootstrap_95_interval'][1]:+.1%}]")
    print(f"Exact paired sign-flip p = {primary['exact_two_sided_paired_sign_flip_p']:.6f}")
    print(f"State-weighted A {arms['A']['successes']}/{arms['A']['episodes']} = "
          f"{arms['A']['state_weighted_success_rate']:.1%}, "
          f"B {arms['B']['successes']}/{arms['B']['episodes']} = "
          f"{arms['B']['state_weighted_success_rate']:.1%}")
    print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
