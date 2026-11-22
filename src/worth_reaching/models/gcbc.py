"""Continuous-action goal-conditioned behavioral cloning for public state inputs."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from flax import serialization
from torch import nn

from worth_reaching.data.antmaze import ACTION_SIZE, OBSERVATION_SIZE
from worth_reaching.data.ogbench import StateArchive
from worth_reaching.models.codec import legacy_dense, mlp
from worth_reaching.models.training import restore_weights, tensor

CHECKPOINT_SCHEMA = "antmaze_gcbc_v2"
LEGACY_SCHEMA = "antmaze_gcbc_v1"


class GCBC(nn.Module):
    def __init__(
        self,
        width: int,
        observation_size: int = OBSERVATION_SIZE,
        action_size: int = ACTION_SIZE,
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


def validation_error(model, batch):
    observation, goal, actions = batch
    return (model(observation, goal).clamp(-1, 1) - actions).square().mean()


def load_policy(directory: Path):
    """Read frozen Lightning actors and previously published Flax actors."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") not in (CHECKPOINT_SCHEMA, LEGACY_SCHEMA):
        raise ValueError("Expected an AntMaze GCBC checkpoint")
    model = GCBC(manifest["training"]["width"])
    values = serialization.msgpack_restore((directory / "weights.msgpack").read_bytes())
    if manifest["schema"] == LEGACY_SCHEMA:
        values = legacy_dense(
            values["params"], "net", *(f"Dense_{index}" for index in range(4))
        )
    restore_weights(model, values)
    with np.load(directory / "normalization.npz", allow_pickle=False) as arrays:
        mean, scale = arrays["mean"], arrays["scale"]
    if (
        mean.shape != (OBSERVATION_SIZE,)
        or scale.shape != mean.shape
        or not np.isfinite(mean).all()
        or not (np.isfinite(scale) & (scale > 0)).all()
    ):
        raise ValueError("Invalid GCBC normalization")
    return policy_from_model(model, mean, scale)


def policy_from_model(model: GCBC, mean: np.ndarray, scale: np.ndarray):
    @torch.no_grad()
    def act(observation: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return (
            model(tensor((observation - mean) / scale), tensor((goal - mean) / scale))
            .clamp(-1, 1)
            .numpy()
        )

    return act
