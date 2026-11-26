"""Generative goal regions: the states that realize a latent goal z.

The decoder models p(s | z) with conditional flow matching. Training pairs interpolate
the latent between the world model's prediction and the encoding of the state that
actually occurred, z = ẑ + α (enc(s) − ẑ) with α uniform in [0, 1]. At α = 0 the region
is the full predicted distribution of future states. As α approaches 1 it narrows to
the states whose encoding is z. One decoder therefore serves both mean predictions and
flow samples, and the width of a region follows from z itself.

The decoder also conditions on the public observation at the decision and generates the
change from it. Features that stay fixed during an episode, such as a layout, then
decode exactly, and the region describes only what the goal asks to change.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from worth_reaching.models import flow
from worth_reaching.models.training import tensor


@dataclass(frozen=True, slots=True)
class RegionConfig:
    updates: int = 3000
    interpolations: int = 16
    sampling_steps: int = 16
    learning_rate: float = 0.001
    width: int = 128
    batch_size: int = 64

    def __post_init__(self) -> None:
        if (
            min(
                self.updates,
                self.interpolations,
                self.sampling_steps,
                self.width,
                self.batch_size,
            )
            < 1
        ):
            raise ValueError("Region decoder counts must be positive")


def interpolated_pairs(
    predicted: np.ndarray,
    encoded: np.ndarray,
    states: np.ndarray,
    interpolations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return latent conditions between ẑ and enc(s) with their realized states."""
    if not predicted.shape == encoded.shape or len(predicted) != len(states):
        raise ValueError("Expected one prediction and encoding per realized state")
    alpha = np.random.default_rng(seed).random(
        (interpolations, len(states), 1), dtype=np.float32
    )
    latents = predicted[None] + alpha * (encoded - predicted)[None]
    return latents.reshape(-1, predicted.shape[-1]), np.tile(
        states, (interpolations, 1)
    )


@dataclass
class RegionDecoder:
    model: flow.RouteFlow
    scale: Tensor
    steps: int

    @classmethod
    def fit(
        cls,
        latents: np.ndarray,
        contexts: np.ndarray,
        states: np.ndarray,
        config: RegionConfig,
        *,
        seed: int,
        out: Path,
    ) -> "RegionDecoder":
        model, scale = flow.fit(
            np.concatenate((latents, contexts), axis=-1),
            (tensor(states) - tensor(contexts))[:, None],
            seed=seed,
            updates=config.updates,
            out=out,
            learning_rate=config.learning_rate,
            width=config.width,
            batch_size=config.batch_size,
        )
        return cls(model, scale, config.sampling_steps)

    @classmethod
    def from_weights(cls, values: dict, steps: int) -> "RegionDecoder":
        scale = tensor(values["scale"])
        return cls(flow.load_parameters(scale.numel(), values["flow"]), scale, steps)

    @property
    def weights(self) -> dict:
        return {
            "flow": {
                name: value.detach().cpu().numpy()
                for name, value in self.model.state_dict().items()
            },
            "scale": self.scale.numpy(),
        }

    def sample(self, latents, context, count: int, seed: int) -> np.ndarray:
        """Draw realizations of each latent goal from one decision-time observation.

        Returns an array shaped (goals, count, state).
        """
        latents = np.asarray(latents, np.float32)
        latents = latents.reshape(-1, latents.shape[-1])
        context = np.asarray(context, np.float32)
        conditions = np.concatenate(
            (latents, np.broadcast_to(context, (len(latents), context.size))), axis=-1
        )
        draws = flow.sample(
            self.model,
            torch.zeros(len(latents), 1, context.size),
            tensor(conditions),
            self.scale,
            seed,
            candidates=count,
            steps=self.steps,
        )
        return context + draws[:, :, 0].numpy()


def gaussian_support(state: np.ndarray, samples: np.ndarray, floor: float) -> float:
    """Log support under a diagonal Gaussian matched to the region samples, at most 0.

    Supports are compared with quantile-calibrated thresholds, so the log scale only
    avoids underflow in high-dimensional observations.
    """
    center = samples.mean(axis=0)
    scale = np.maximum(samples.std(axis=0), floor)
    return float(-0.5 * np.sum(np.square((state - center) / scale)))


def kernel_support(state: np.ndarray, samples: np.ndarray, bandwidth: float) -> float:
    """Log Gaussian-kernel density of state under region samples, at most 0."""
    exponents = -0.5 * np.sum(np.square(samples - state), axis=-1) / bandwidth**2
    peak = exponents.max()
    return float(peak + np.log(np.mean(np.exp(exponents - peak))))


@torch.no_grad()
def nearest_neighbor_distance(encoded: np.ndarray) -> float:
    """Median distance from each distinct encoded state to its nearest other state."""
    unique = np.unique(np.round(encoded, 6), axis=0)
    if len(unique) < 2:
        raise ValueError("Calibrating latent distances needs two distinct states")
    points = tensor(unique)
    distances = torch.cdist(points, points)
    distances.fill_diagonal_(float("inf"))
    return float(distances.min(dim=1).values.median())
