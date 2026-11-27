"""Goal-conditioned behavior cloning for discrete actions.

The same architecture serves two goal forms: an observation to reach, or a latent goal
z that the policy pursues without knowing what it denotes.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from worth_reaching.models.codec import mlp
from worth_reaching.models.training import RandomBatches, tensor, train_stage


@dataclass(frozen=True, slots=True)
class ExecutorConfig:
    updates: int = 2000
    width: int = 128
    batch_size: int = 128
    horizon: int = 8
    learning_rate: float = 0.001

    def __post_init__(self) -> None:
        if min(self.updates, self.width, self.batch_size, self.horizon) < 1:
            raise ValueError("Executor sizes and update counts must be positive")


class GoalPolicy(nn.Module):
    def __init__(self, observation_size: int, goal_size: int, actions: int, width: int):
        super().__init__()
        self.net = mlp(observation_size + goal_size, width, actions, layers=2)

    def forward(self, observation, goal):
        return self.net(torch.cat((observation, goal), dim=-1))

    @torch.no_grad()
    def act(self, observation: np.ndarray, goal: np.ndarray) -> int:
        logits = self(tensor(observation)[None], tensor(goal)[None])
        return int(logits.argmax(dim=-1))


def hindsight_pairs(
    trajectories: list[tuple[np.ndarray, np.ndarray]], horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pair each action with every observation reached within the next horizon steps."""
    observations, goals, actions = [], [], []
    for states, taken in trajectories:
        for step, action in enumerate(taken):
            for offset in range(1, horizon + 1):
                observations.append(states[step])
                goals.append(states[min(step + offset, len(taken))])
                actions.append(action)
    if not actions:
        raise ValueError("Executor training needs at least one action")
    return np.stack(observations), np.stack(goals), np.asarray(actions, np.int64)


def _cross_entropy(model, batch):
    observation, goal, action = batch
    loss = F.cross_entropy(model(observation, goal), action.long(), reduction="none")
    # The shared trainer squares each error term, so pass the square root.
    return ((loss.clamp_min(1e-12).sqrt(), None),)


def fit_policy(
    observations: np.ndarray,
    goals: np.ndarray,
    actions: np.ndarray,
    action_count: int,
    config: ExecutorConfig,
    *,
    seed: int,
    out: Path,
) -> GoalPolicy:
    torch.manual_seed(seed)
    model = GoalPolicy(
        observations.shape[-1], goals.shape[-1], action_count, config.width
    )
    return train_stage(
        model,
        RandomBatches(
            (tensor(observations), tensor(goals), tensor(actions)),
            config.batch_size,
            (seed, 5),
        ),
        _cross_entropy,
        steps=config.updates,
        learning_rate=config.learning_rate,
        out=out,
    )


def accuracy(model: GoalPolicy, observations, goals, actions) -> float:
    with torch.no_grad():
        predicted = model(tensor(observations), tensor(goals)).argmax(dim=-1)
    return float((predicted.numpy() == actions).mean())
