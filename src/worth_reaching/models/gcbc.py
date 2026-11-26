"""Continuous-action goal-conditioned behavioral cloning for public state inputs."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from worth_reaching.data.ogbench import StateArchive
from worth_reaching.models.codec import mlp
from worth_reaching.models.training import tensor


class GCBC(nn.Module):
    def __init__(
        self,
        width: int,
        observation_size: int,
        action_size: int,
        relative_coordinates: int = 2,
    ):
        super().__init__()
        self.relative_coordinates = relative_coordinates
        self.net = mlp(
            2 * observation_size + relative_coordinates, width, action_size, layers=3
        )

    def forward(self, observation, goal):
        leading = self.relative_coordinates
        features = torch.cat(
            (observation, goal, goal[..., :leading] - observation[..., :leading]),
            dim=-1,
        )
        return self.net(features)


@dataclass(frozen=True)
class BehaviorBatches:
    archive: StateArchive
    mean: np.ndarray
    scale: np.ndarray
    seed: tuple[int, ...]
    batch_size: int
    max_goal_steps: int | None
    alternate_horizons: bool = False

    def __call__(self, step: int):
        cap = None if self.alternate_horizons and step % 2 else self.max_goal_steps
        batch = self.archive.sample(
            np.random.default_rng([*self.seed, step]),
            self.batch_size,
            max_goal_steps=cap,
        )
        return normalized_batch(batch, self.mean, self.scale)


def normalized_batch(batch, mean, scale):
    return (
        tensor((batch.observations - mean) / scale),
        tensor((batch.goals - mean) / scale),
        tensor(batch.actions),
    )


def policy_from_model(model: GCBC, mean: np.ndarray, scale: np.ndarray):
    @torch.no_grad()
    def act(observation: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return (
            model(tensor((observation - mean) / scale), tensor((goal - mean) / scale))
            .clamp(-1, 1)
            .numpy()
        )

    return act
