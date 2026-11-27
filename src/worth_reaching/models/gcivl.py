"""Goal-conditioned implicit V-learning (GCIVL), following OGBench.

A goal-conditioned value V(s, g) is learned by expectile regression toward
r + γ V̄(s', g) with r = 0 at the goal and −1 elsewhere, using two value heads and a
Polyak-averaged target. The policy is extracted by advantage-weighted regression with
the advantage V(s', g) − V(s, g). With a constant standard deviation, the actor's
log-likelihood loss is a weighted squared error on the mean action.

The goal can be a state or, for a latent-conditioned executor, a feature row aligned
with the archive, such as the frozen encoding of every observation.
"""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from worth_reaching.data.ogbench import StateArchive
from worth_reaching.models.training import tensor, train_stage


@dataclass(frozen=True, slots=True)
class GCIVLConfig:
    updates: int = 1_000_000
    batch_size: int = 1024
    width: int = 512
    depth: int = 3
    learning_rate: float = 3e-4
    discount: float = 0.99
    expectile: float = 0.9
    alpha: float = 10.0
    tau: float = 0.005
    value_p_current: float = 0.2
    value_p_trajectory: float = 0.5
    value_p_random: float = 0.3

    def __post_init__(self) -> None:
        probabilities = (
            self.value_p_current,
            self.value_p_trajectory,
            self.value_p_random,
        )
        if (
            min(self.updates, self.batch_size, self.width, self.depth) < 1
            or not 0 < self.discount < 1
            or not 0 < self.expectile < 1
            or not 0 < self.tau <= 1
            or self.alpha <= 0
            or min(probabilities) < 0
            or not np.isclose(sum(probabilities), 1)
        ):
            raise ValueError(
                "Use positive sizes, rates in (0, 1) and goal probabilities summing to one"
            )


def layer_norm_mlp(
    input_size: int, width: int, output_size: int, depth: int
) -> nn.Sequential:
    layers = []
    for _ in range(depth):
        layers += [nn.Linear(input_size, width), nn.GELU(), nn.LayerNorm(width)]
        input_size = width
    layers.append(nn.Linear(input_size, output_size))
    return nn.Sequential(*layers)


class ValuePair(nn.Module):
    def __init__(self, observation_size: int, goal_size: int, config: GCIVLConfig):
        super().__init__()
        size = observation_size + goal_size
        self.heads = nn.ModuleList(
            layer_norm_mlp(size, config.width, 1, config.depth) for _ in range(2)
        )

    def forward(self, observation: Tensor, goal: Tensor) -> tuple[Tensor, Tensor]:
        features = torch.cat((observation, goal), dim=-1)
        first, second = (head(features)[..., 0] for head in self.heads)
        return first, second


class GCIVL(nn.Module):
    def __init__(
        self,
        observation_size: int,
        goal_size: int,
        action_size: int,
        config: GCIVLConfig,
    ):
        super().__init__()
        self.config = config
        self.value = ValuePair(observation_size, goal_size, config)
        self.target = deepcopy(self.value).requires_grad_(False)
        self.actor = layer_norm_mlp(
            observation_size + goal_size, config.width, action_size, config.depth
        )

    def forward(self, observation: Tensor, goal: Tensor) -> Tensor:
        return self.actor(torch.cat((observation, goal), dim=-1))

    @torch.no_grad()
    def act(self, observation: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return (
            self(tensor(observation)[None], tensor(goal)[None])[0].clamp(-1, 1).numpy()
        )

    @torch.no_grad()
    def update_target(self) -> None:
        for target, online in zip(
            self.target.parameters(), self.value.parameters(), strict=True
        ):
            target.lerp_(online, self.config.tau)


@dataclass(frozen=True)
class GCIVLBatches:
    """Sample OGBench-style value and actor goals from complete trajectories."""

    archive: StateArchive
    config: GCIVLConfig
    seed: tuple[int, ...]
    goal_features: np.ndarray | None = None
    """Goal inputs aligned with archive rows; the observations when omitted."""

    def __call__(self, step: int):
        archive, config = self.archive, self.config
        rng = np.random.default_rng([*self.seed, step])
        size = config.batch_size
        rows = rng.choice(archive.rows, size=size)
        final = archive.ends[np.searchsorted(archive.ends, rows)]
        offsets = rng.geometric(1 - config.discount, size=size)
        trajectory = np.minimum(rows + offsets, final)
        random = rng.integers(len(archive.observations), size=size)
        draw = rng.random(size)
        value_goals = np.where(
            draw < config.value_p_trajectory / (1 - config.value_p_current),
            trajectory,
            random,
        )
        value_goals = np.where(
            rng.random(size) < config.value_p_current, rows, value_goals
        )
        fraction = rng.random(size)
        actor_goals = np.round(
            np.minimum(rows + 1, final) * fraction + final * (1 - fraction)
        ).astype(np.int64)
        successes = (rows == value_goals).astype(np.float32)
        observations = archive.observations
        return (
            tensor(observations[rows]),
            tensor(observations[rows + 1]),
            tensor(archive.actions[rows]),
            tensor(self._goals[value_goals]),
            tensor(self._goals[actor_goals]),
            tensor(successes - 1),
            tensor(1 - successes),
        )

    @property
    def _goals(self) -> np.ndarray:
        if self.goal_features is None:
            return self.archive.observations
        return self.goal_features


def gcivl_loss(model: GCIVL, batch):
    """Return error terms whose squared means equal the OGBench value and actor losses."""
    observation, following, action, value_goal, actor_goal, reward, mask = batch
    config = model.config
    model.update_target()
    with torch.no_grad():
        next_first, next_second = model.target(following, value_goal)
        current_first, current_second = model.target(observation, value_goal)
        bootstrap = reward + config.discount * mask * torch.minimum(
            next_first, next_second
        )
        advantage = bootstrap - (current_first + current_second) / 2
        weight = torch.where(advantage >= 0, config.expectile, 1 - config.expectile)
        first_target = reward + config.discount * mask * next_first
        second_target = reward + config.discount * mask * next_second
    first, second = model.value(observation, value_goal)
    with torch.no_grad():
        value = sum(model.value(observation, actor_goal)) / 2
        next_value = sum(model.value(following, actor_goal)) / 2
        scale = torch.clamp(torch.exp(config.alpha * (next_value - value)), max=100.0)
        # A unit-variance Gaussian log-likelihood sums half squared errors over actions.
        scale = (scale * action.shape[-1] / 2).sqrt()[:, None]
    return (
        (weight.sqrt() * (first_target - first), None),
        (weight.sqrt() * (second_target - second), None),
        (scale * (model(observation, actor_goal) - action), None),
    )


def fit_gcivl(
    archive: StateArchive,
    action_size: int,
    config: GCIVLConfig,
    *,
    seed: int,
    out: Path,
    goal_features: np.ndarray | None = None,
    resume: Path | None = None,
) -> GCIVL:
    if goal_features is not None and len(goal_features) != len(archive.observations):
        raise ValueError("Goal features must align with the archive rows")
    torch.manual_seed(seed)
    observation_size = archive.observations.shape[-1]
    goals = archive.observations if goal_features is None else goal_features
    model = GCIVL(observation_size, goals.shape[-1], action_size, config)
    return train_stage(
        model,
        GCIVLBatches(archive, config, (seed, 900), goal_features),
        gcivl_loss,
        steps=config.updates,
        learning_rate=config.learning_rate,
        out=out,
        checkpoint_every=10_000,
        resume=resume,
    )
