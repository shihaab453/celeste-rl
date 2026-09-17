"""The policy network input for obs-v1 (Phase 2 spec, section 9.1).

Stable-Baselines3's default Dict handling would flatten the 11 x 32 x 32 grid into 11,264 inputs, and its
image network (NatureCNN) does not fit a 32 x 32 grid. CelesteFeatures instead passes the grid through a small
channels-first CNN and concatenates the result with the flattened player features, action history, history
mask and context. The policy and value heads each add two 128-unit layers on top.

The grid is 0 or 1 in the environment (uint8). SB3 only divides by 255 for 1, 3 or 4 channel image spaces with
normalize_images=True; normalize_images=False is set anyway so the grid can never be scaled.
"""
from __future__ import annotations

import gymnasium as gym
import torch as th
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


def policy_kwargs() -> dict:
    """Keyword arguments for PPO("MultiInputPolicy", ..., policy_kwargs=policy_kwargs())."""
    return {
        "features_extractor_class": CelesteFeatures,
        "net_arch": {"pi": [128, 128], "vf": [128, 128]},
        "normalize_images": False,
    }
