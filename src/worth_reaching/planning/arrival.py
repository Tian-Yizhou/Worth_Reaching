"""Arrival rules for latent goals, judged without naming what the goal denotes.

Each rule receives the current public observation and a LatentGoal. Encoder
consistency compares the encoded observation with z. Region support asks whether the
observation lies where the region decoder places realizations of z. The Gaussian rule is
the DisCo-style ablation: a diagonal Gaussian matched to the same decoder samples.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from worth_reaching.models.region import gaussian_support, kernel_support


@dataclass(frozen=True, slots=True)
class LatentGoal:
    latent: np.ndarray
    samples: np.ndarray | None = None
    """Realizations of z drawn from the region decoder before the attempt."""
    realization: np.ndarray | None = None
    """One sampled realization for an observation-conditioned executor."""


@dataclass(frozen=True, slots=True)
class EncoderConsistency:
    encode: Callable[[np.ndarray], np.ndarray]
    epsilon: float

    def __call__(self, observation: np.ndarray, goal: LatentGoal) -> bool:
        distance = np.linalg.norm(self.encode(observation) - goal.latent)
        return bool(distance <= self.epsilon)


@dataclass(frozen=True, slots=True)
class RegionSupport:
    bandwidth: float
    threshold: float

    def __call__(self, observation: np.ndarray, goal: LatentGoal) -> bool:
        if goal.samples is None:
            raise ValueError("Region support needs decoder samples for the goal")
        return (
            kernel_support(observation, goal.samples, self.bandwidth) >= self.threshold
        )


@dataclass(frozen=True, slots=True)
class GaussianSupport:
    floor: float
    threshold: float

    def __call__(self, observation: np.ndarray, goal: LatentGoal) -> bool:
        if goal.samples is None:
            raise ValueError("Gaussian support needs decoder samples for the goal")
        return gaussian_support(observation, goal.samples, self.floor) >= self.threshold
