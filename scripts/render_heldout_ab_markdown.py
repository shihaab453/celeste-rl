"""Render the numeric tables in the Phase 3B report from a verified analysis.json.

The report keeps interpretation in prose, but its tables must not be transcribed by hand. Run:

    .venv-rl/Scripts/python.exe scripts/render_heldout_ab_markdown.py docs/results/phase3b-heldout-ab-200.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def percent(value: float) -> str:
    return f"{value:.1%}"


def points(value: float) -> str:
    return f"{value * 100:+.1f} points"


def primary_table(analysis: dict) -> str:
    primary = analysis["primary"]
    lower, upper = primary["crossed_seed_route_bootstrap_95_interval"]
    return "\n".join([
        "| primary result | A, canonical starts | B, varied starts | B minus A |",
        "|---|---:|---:|---:|",
        f"| Route-macro success | {percent(primary['A_route_macro_success_rate'])} | "
        f"**{percent(primary['B_route_macro_success_rate'])}** | **{points(primary['B_minus_A'])}** |",
        f"| Crossed seed-route 95% interval | | | **{points(lower)} to {points(upper)}** |",
        f"| Exact paired sign-flip test | | | "
        f"**p = {primary['exact_two_sided_paired_sign_flip_p']:.5f}** |",
    ])


def seed_table(analysis: dict) -> str:
    lines = [
        "| seed | A route-macro | B route-macro | paired difference | A successes | B successes |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for seed, values in analysis["secondary"]["per_seed"].items():
        lines.append(
            f"| {seed} | {percent(values['A']['route_macro_success_rate'])} | "
            f"{percent(values['B']['route_macro_success_rate'])} | "
            f"{points(values['B_minus_A_route_macro'])} | "
            f"{values['A']['successes']} | {values['B']['successes']} |"
        )
    return "\n".join(lines)


def route_table(analysis: dict) -> str:
    lines = [
        "| route | states per checkpoint | A | B | B minus A |",
        "|---|---:|---:|---:|---:|",
    ]
    for route, values in analysis["secondary"]["per_route"].items():
        states = values["A"]["episodes"] // (analysis["validated_runs"] // 2)
        lines.append(
            f"| `{route}` | {states} | {percent(values['A']['success_rate'])} | "
            f"{percent(values['B']['success_rate'])} | {points(values['B_minus_A'])} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    print(primary_table(analysis))
    print()
    print(seed_table(analysis))
    print()
    print(route_table(analysis))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
