"""The room task's reward. Two versions, selected by `RewardConfig.version`.

**rew-v1** (Phase 2 spec, section 8), the unshaped baseline, kept selectable so that campaign stays reproducible:

  +1 on success (once), -1 on any failure ending, and a time cost on every step including the last
  (-0.001 per second of normal play, -1/60,000 per frame). No survival bonus, no shaping.

  Known trade-off (decision D5): a policy certain to fail scores about -1.00002 by failing at once and -1.03 by
  timing out, so it slightly prefers failing early.

**rew-v2** (Codex finding K1), the shaped version. Two changes that only work together, so they ship together:

  1. *The unspent-deadline charge.* A failure at frame t is also charged for the deadline it did not use,
     `-(1800 - t)/60,000`. With the per-step time cost that makes every failure total exactly -1.03 whenever it
     happens, so the early-death gradient of rew-v1 is gone. This keeps D5 (time still costs from the first
     frame) and adds no survival bonus: surviving longer is worth nothing by itself, it only stops costing
     extra to die sooner. Reported as its own component.
  2. *Progress shaping.* A potential-based term over `celeste_rl.potential`, scale 0.2, computed by the
     environment and passed in here. Potential-based shaping leaves the optimal policy unchanged, and with a
     terminal potential of 0 an episode's shaping sums to exactly `-scale * potential(start)`, a constant per
     start, so it cannot change which ending is preferred.

  Shipping the potential alone would leave the die-early gradient intact and waste a run; shipping the charge
  alone gives the agent no signal about where to go.

The reward is computed from the ending cause, the elapsed frames and the potential only, never from anything
the observation encoder produces, and is returned outside the observation.
"""
from __future__ import annotations

from dataclasses import dataclass

from celeste_rl.endings import FAILURES, SUCCESS
from celeste_rl.schema import DEADLINE_FRAMES

REWARD_VERSION = "rew-v1"  # the default; a run records its own choice in RewardConfig.version
REWARD_VERSIONS = ("rew-v1", "rew-v2")
# The components each version reports, in the order records should show them. A run's progress columns are
# built from this, so a version that adds a component cannot have it silently dropped from the records.
COMPONENTS = {
    "rew-v1": ("completion", "failure", "time", "shaping"),
    "rew-v2": ("completion", "failure", "unspent_deadline", "time", "shaping"),
}


@dataclass(frozen=True)
class RewardConfig:
    version: str = REWARD_VERSION
    success: float = 1.0
    failure: float = -1.0
    time_per_frame: float = -1.0 / 60_000
    # The learner's discount (decision D2). Any shaping term must use the same value.
    gamma: float = 1.0
    # rew-v2 only. The charge rate is the time cost, so an unused deadline costs exactly what using it would have.
    deadline_frames: int = DEADLINE_FRAMES
    shaping_scale: float = 0.2

    def __post_init__(self):
        if self.version not in REWARD_VERSIONS:
            raise ValueError(f"Unknown reward version {self.version!r}; expected one of {REWARD_VERSIONS}")

    @property
    def shaped(self) -> bool:
        return self.version == "rew-v2"

    @property
    def components(self) -> tuple[str, ...]:
        return COMPONENTS[self.version]


def reward_components(ending: str | None, config: RewardConfig, elapsed: int = 0,
                      shaping: float = 0.0) -> dict[str, float]:
    """The step's reward, split into the components a run records. `elapsed` is the frame this step ended on and
    `shaping` the environment's potential difference; both are ignored by rew-v1, which rejects a nonzero one."""
    if ending is not None and ending != SUCCESS and ending not in FAILURES:
        raise ValueError(f"Unknown ending {ending!r}")
    failed = ending in FAILURES
    if not config.shaped and shaping:
        raise ValueError(f"{config.version} has no shaping term, but shaping={shaping!r} was passed")
    # max(0, ...) so an episode that somehow ran past the deadline is never paid for the overrun.
    unspent = max(0, config.deadline_frames - elapsed) if failed else 0
    values = {
        "completion": config.success if ending == SUCCESS else 0.0,
        "failure": config.failure if failed else 0.0,
        "unspent_deadline": config.time_per_frame * unspent,
        "time": config.time_per_frame,
        "shaping": shaping,
    }
    return {name: values[name] for name in config.components}
