"""The policy network input for obs-v1 (Phase 2 spec, section 9.1).

Stable-Baselines3's default Dict handling would flatten the 11 x 32 x 32 grid into 11,264 inputs, and its
image network (NatureCNN) does not fit a 32 x 32 grid. CelesteFeatures instead passes the grid through a small
channels-first CNN and concatenates the result with the flattened player features, action history, history
mask and context. The policy and value heads each add two 128-unit layers on top.

The grid is 0 or 1 in the environment (uint8). SB3 only divides by 255 for 1, 3 or 4 channel image spaces with
normalize_images=True; normalize_images=False is set anyway so the grid can never be scaled.
"""
from __future__ import annotations

import math

import gymnasium as gym
import torch as th
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

VECTOR_KEYS = ("player", "actions", "history_valid", "context")


class CelesteFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, grid_features: int = 128):
        channels, height, width = observation_space["grid"].shape
        cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1), nn.ReLU(),           # 32 x 32
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),       # 16 x 16
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),       # 8 x 8
            nn.Flatten(),
        )
        with th.no_grad():
            cnn_out = cnn(th.zeros(1, channels, height, width)).shape[1]
        vector_size = sum(int(th.tensor(observation_space[key].shape).prod()) for key in VECTOR_KEYS)
        super().__init__(observation_space, features_dim=grid_features + vector_size)
        self.grid = nn.Sequential(cnn, nn.Linear(cnn_out, grid_features), nn.ReLU())

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        grid = self.grid(observations["grid"].float())
        vectors = [observations[key].float().flatten(start_dim=1) for key in VECTOR_KEYS]
        return th.cat([grid, *vectors], dim=1)


class CelestePolicy(MultiInputActorCriticPolicy):
    """The training policy, with one addition: the action head's bias can start negative.

    act-v1 is 24 independent on/off inputs. A zero bias means every input starts at probability 0.5, so a fresh
    policy samples about twelve buttons held at once, every frame. A recorded clear of room 1 holds 2.31 inputs
    per frame and never more than three, so the default initialisation explores a part of the action space that
    has almost no overlap with where solutions live, and maximum entropy over 21 Bernoullis is exactly that
    twelve-button behaviour, which the entropy bonus then pays to keep.

    `action_bias` is the initial bias on every action logit: -2.2 gives probability 0.10 per input, about 2.3
    held per frame. It only moves where sampling starts; the head is free to learn any bias from there. It is a
    declared run parameter, recorded in each run's manifest, and 0.0 reproduces the earlier runs exactly.
    """

    def __init__(self, *args, action_bias: float = 0.0, **kwargs):
        # Set before super().__init__, which calls _build.
        self.action_bias = action_bias
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        if self.action_bias:
            with th.no_grad():
                self.action_net.bias.fill_(self.action_bias)


def bias_for_probability(probability: float) -> float:
    """The action-head bias that makes each input start at this probability. 0.10 gives about -2.2."""
    if not 0.0 < probability < 1.0:
        raise ValueError(f"probability must be between 0 and 1, got {probability}")
    return math.log(probability / (1 - probability))


def policy_kwargs(action_bias: float = 0.0) -> dict:
    """Keyword arguments for PPO(CelestePolicy, ..., policy_kwargs=policy_kwargs())."""
    return {
        "features_extractor_class": CelesteFeatures,
        "net_arch": {"pi": [128, 128], "vf": [128, 128]},
        "normalize_images": False,
        "action_bias": action_bias,
    }
