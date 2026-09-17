"""What the policy actually became, checkpoint by checkpoint: entropy, per-input marginals and value (Codex K4).

Run from the repo root with the RL interpreter (offline, no game needed):
    .venv-rl/Scripts/python.exe scripts/checkpoint_policy.py --run-dir runs/train/<run>

The unshaped campaign's records said only that no episode cleared the room. They could not say whether the
policy had changed at all, so "it learned to die sooner" stayed a guess. This reads a run's checkpoints and
measures, at one fixed set of observations:

  entropy         the action distribution's entropy in nats, and its share of the 24-Bernoulli maximum
                  (24 * ln 2 = 16.64). Near the maximum means the policy is still near coin flips on every
                  input, whatever its returns did.
  marginals       P(input on) per input, averaged over the probe observations. This is what "the policy
                  concentrated" would actually look like: some inputs moving away from 0.5.
  value           the value head's mean prediction, which under rew-v1 sat at -1.00 (it had learned only that
                  every episode ends in a failure).

The probe observations are a fixed set replayed through the environment earlier and saved by
`scripts/route_fit_check.py` as `dataset.npz` (arrays `obs_<key>`). They are a measuring stick, never training
data: nothing here writes a checkpoint or touches a run. Using the recorded clear's frames is fine for that
and is not a demonstration method, but the same file is also the reason these numbers describe the policy on
sensible room 1 states rather than on states it visits itself.

Results go to runs/checkpoint-policy/<timestamp>/results.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch as th

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from celeste_rl import runtime  # noqa: E402
from celeste_rl.schema import ACTION_INPUTS, MENU_INPUTS  # noqa: E402
from celeste_rl.training.supervisor import SupervisedPPO  # noqa: E402

# The observation keys route_fit_check.py stores, prefixed with obs_ in the npz.
OBS_KEYS = ("player", "actions", "history_valid", "grid", "context")
MAX_ENTROPY = len(ACTION_INPUTS) * math.log(2)


def newest_dataset() -> Path | None:
    candidates = sorted((REPO / "runs" / "route-fit").glob("*/dataset.npz"))
    return candidates[-1] if candidates else None


def load_observations(path: Path) -> dict[str, th.Tensor]:
    stored = np.load(path)
    return {key: th.as_tensor(stored[f"obs_{key}"]) for key in OBS_KEYS}


def checkpoints_of(run_dir: Path) -> list[tuple[str, Path]]:
    """Every step checkpoint in order, then the named ones that exist."""
    steps = sorted((run_dir / "checkpoints").glob("step_*.zip"))
    named = [(name, run_dir / "checkpoints" / f"{name}.zip") for name in ("latest", "best", "aborted")]
    return [(p.stem, p) for p in steps] + [(name, p) for name, p in named if p.exists()]


def measure(path: Path, observations: dict[str, th.Tensor]) -> dict:
    model = SupervisedPPO.load(path, device="cpu")
    policy = model.policy
    policy.set_training_mode(False)
    with th.no_grad():
        distribution = policy.get_distribution(observations)
        entropy = distribution.entropy()
        marginals = th.sigmoid(distribution.distribution.logits).mean(dim=0)
        values = policy.predict_values(observations)
    enabled = [name not in MENU_INPUTS for name in ACTION_INPUTS]
    return {
        "checkpoint": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "accepted_steps": int(model.num_timesteps),
        "mean_entropy": round(float(entropy.mean()), 4),
        "entropy_share_of_maximum": round(float(entropy.mean()) / MAX_ENTROPY, 4),
        "mean_value": round(float(values.mean()), 4),
        # How far the policy has moved off coin flips: 0 is uniform, 0.5 is deterministic.
        "mean_marginal_distance_from_half": round(float((marginals[enabled] - 0.5).abs().mean()), 4),
        "marginals": {name: round(float(marginals[i]), 4) for i, name in enumerate(ACTION_INPUTS)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="a training run directory")
    parser.add_argument("--observations", type=Path, default=None,
                        help="probe observations (default: the newest runs/route-fit/*/dataset.npz)")
    args = parser.parse_args()

    dataset = args.observations or newest_dataset()
    if dataset is None or not dataset.exists():
        print("No probe observations. Record some with scripts/route_fit_check.py, or pass --observations.")
        return 2
    observations = load_observations(dataset)
    found = checkpoints_of(args.run_dir)
    if not found:
        print(f"No checkpoints in {args.run_dir / 'checkpoints'}")
        return 2

    count = len(next(iter(observations.values())))
    print(f"{args.run_dir}: {len(found)} checkpoints, {count} probe observations from {dataset}")
    print(f"{'checkpoint':>16s} {'steps':>10s} {'entropy':>8s} {'of max':>7s} {'value':>8s} {'off half':>9s}")
    measurements = []
    for _, path in found:
        record = measure(path, observations)
        measurements.append(record)
        print(f"{record['checkpoint']:>16s} {record['accepted_steps']:>10,d} {record['mean_entropy']:>8.3f} "
              f"{record['entropy_share_of_maximum']:>7.3f} {record['mean_value']:>8.3f} "
              f"{record['mean_marginal_distance_from_half']:>9.4f}")

    last = measurements[-1]
    extremes = sorted(((name, value) for name, value in last["marginals"].items() if name not in MENU_INPUTS),
                      key=lambda item: item[1])
    print(f"\n{last['checkpoint']}: least pressed " +
          ", ".join(f"{n} {v:.3f}" for n, v in extremes[:4]) + " | most pressed " +
          ", ".join(f"{n} {v:.3f}" for n, v in reversed(extremes[-4:])))

    output_dir = REPO / "runs" / "checkpoint-policy" / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True)
    (output_dir / "results.json").write_text(json.dumps({
        **runtime.git_state(), "run_dir": str(args.run_dir), "observations": str(dataset),
        "probe_observations": count, "max_entropy": round(MAX_ENTROPY, 4), "checkpoints": measurements,
    }, indent=2, default=str), encoding="utf-8")
    print(f"Results: {output_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
