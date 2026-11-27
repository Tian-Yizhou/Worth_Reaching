"""Conditional flow matching for residuals around a latent forecast."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from worth_reaching.models.codec import legacy_dense, mlp
from worth_reaching.models.training import (
    RandomBatches,
    regression,
    restore_weights,
    tensor,
    train_stage,
)


class RouteFlow(nn.Module):
    def __init__(self, size: int, condition_size: int):
        super().__init__()
        self.size = size
        self.net = mlp(
            size + condition_size + 1, 128, size, layers=2, activation=nn.SiLU
        )

    def forward(self, noisy: Tensor, time: Tensor, condition: Tensor) -> Tensor:
        return self.net(torch.cat((noisy, time, condition), dim=-1))


def flow_path(target: Tensor, rng: np.random.Generator, mask=1):
    noise = tensor(rng.standard_normal(target.shape, dtype=np.float32))
    time = tensor(rng.random((len(target), 1), dtype=np.float32))
    velocity = target - noise
    return noise + time * velocity * mask, time, velocity


@dataclass(frozen=True)
class FlowBatches:
    batches: RandomBatches
    seed: int

    def __call__(self, step: int):
        target, context = self.batches(step)
        current, time, velocity = flow_path(
            target, np.random.default_rng([self.seed, 2, step])
        )
        return current, time, context, velocity


def fit(
    condition,
    residual,
    *,
    seed: int,
    updates: int,
    out: Path,
    learning_rate: float = 0.001,
    resume: Path | None = None,
):
    condition, residual = tensor(condition), tensor(residual)
    scale = residual.std(dim=(0, 1), correction=0).clamp_min(0.01)
    normalized = (residual / scale).flatten(1)
    torch.manual_seed(seed)
    model = RouteFlow(normalized.shape[-1], condition.shape[-1])
    batches = FlowBatches(RandomBatches((normalized, condition), 64, (seed, 1)), seed)
    return train_stage(
        model,
        batches,
        regression,
        steps=updates,
        learning_rate=learning_rate,
        out=out,
        resume=resume,
    ), scale


def route_conditions(inputs, forecast: Tensor, goals: int) -> Tensor:
    return torch.cat(
        (
            forecast.flatten(1),
            tensor(inputs.observations[:, 0]),
            F.one_hot(torch.as_tensor(inputs.goal).long(), goals),
        ),
        dim=-1,
    )


def load_parameters(size: int, values: dict) -> RouteFlow:
    if "params" in values:
        values = legacy_dense(values["params"], "net", "Dense_0", "Dense_1", "Dense_2")
    conditions = values["net.0.weight"].shape[1] - size - 1
    model = RouteFlow(size, conditions)
    restore_weights(model, values)
    return model


@torch.no_grad()
def sample(
    model, forecast, condition, scale, seed: int, *, candidates: int, steps: int
):
    repeated = condition.repeat_interleave(candidates, dim=0)
    noise = torch.randn(
        (len(repeated), model.size), generator=torch.Generator().manual_seed(seed)
    )
    residual = noise
    for index in range(steps):
        time = residual.new_full((len(residual), 1), index / steps)
        residual = residual + model(residual, time, repeated) / steps
    return (
        forecast[:, None]
        + residual.reshape(len(forecast), candidates, *forecast.shape[1:]) * scale
    )
