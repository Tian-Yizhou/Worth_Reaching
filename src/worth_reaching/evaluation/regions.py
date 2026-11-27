"""Shared pieces of the region-goal comparison (E1) for continuous tasks.

The task runner supplies the world model's predictions on masked windows, the
environment and the observation-space executor. This module fits the region decoder,
calibrates every arrival rule to the same recall on held-out windows, defines the arms
and summarizes coherence.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from worth_reaching.models.region import (
    RegionConfig,
    RegionDecoder,
    gaussian_support,
    interpolated_pairs,
    kernel_support,
)
from worth_reaching.models.training import stream_seed
from worth_reaching.planning.arrival import (
    EncoderConsistency,
    GaussianSupport,
    RegionSupport,
)

LATENT_RULES = ("latent_point", "region_gaussian", "region_flow")


@dataclass(frozen=True, slots=True)
class RegionWindows:
    """Predictions and realized outcomes at the queried offsets of masked windows.

    States and contexts are in the space where regions and kernels are defined, for
    example observations normalized by the training scale.
    """

    predicted: np.ndarray
    encoded: np.ndarray
    contexts: np.ndarray
    states: np.ndarray

    def __post_init__(self) -> None:
        if not (
            self.predicted.shape == self.encoded.shape
            and self.contexts.shape == self.states.shape
            and len(self.predicted) == len(self.states)
        ):
            raise ValueError("Expected one prediction, encoding and context per state")


@dataclass(frozen=True, slots=True)
class Arm:
    executor: str
    """observation (the task's controller on a state target) or latent."""
    target: str
    """decoded (deterministic decoding of z), sample (one region realization) or
    latent (z itself)."""
    arrival: object = None
    """A latent arrival rule, or None for the task's native arrival predicate."""


def fit_decoder(
    windows: RegionWindows, config: RegionConfig, *, seed: int, out: Path
) -> RegionDecoder:
    latents, states = interpolated_pairs(
        windows.predicted,
        windows.encoded,
        windows.states,
        config.interpolations,
        stream_seed(seed, 6101),
    )
    return RegionDecoder.fit(
        latents,
        np.tile(windows.contexts, (config.interpolations, 1)),
        states,
        config,
        seed=stream_seed(seed, 6102),
        out=out,
    )


def calibrate(
    decoder: RegionDecoder,
    encode: Callable[[np.ndarray], np.ndarray],
    windows: RegionWindows,
    *,
    levels: tuple[float, ...],
    samples: int,
    bandwidth: float,
    limit: int,
    seed: int,
) -> tuple[dict, dict]:
    """Set each latent rule's threshold so it accepts a share of realized states.

    Goals interpolate between the forecast and the realized encoding, as in decoder
    training. A different window's state measures false arrival at each threshold.
    """
    rng = np.random.default_rng(stream_seed(seed, 6104))
    count = len(windows.states)
    chosen = rng.permutation(count)[:limit]
    alpha = rng.random((len(chosen), 1))
    goals = windows.predicted[chosen] + alpha * (
        windows.encoded[chosen] - windows.predicted[chosen]
    )
    others = (chosen + 1 + rng.integers(count - 1, size=len(chosen))) % count
    draws = np.stack(
        [
            decoder.sample(
                goal, windows.contexts[index], samples, stream_seed(seed, 6105, i)
            )[0]
            for i, (goal, index) in enumerate(zip(goals, chosen, strict=True))
        ]
    )
    scores, impostor = {}, {}
    for sink, rows in ((scores, chosen), (impostor, others)):
        states = windows.states[rows]
        encoded = np.stack([encode(state) for state in states])
        sink["latent_point"] = -np.linalg.norm(encoded - goals, axis=-1)
        sink["region_gaussian"] = np.array(
            [
                gaussian_support(s, d, bandwidth)
                for s, d in zip(states, draws, strict=True)
            ]
        )
        sink["region_flow"] = np.array(
            [
                kernel_support(s, d, bandwidth)
                for s, d in zip(states, draws, strict=True)
            ]
        )
    thresholds, report = {}, {}
    for name in LATENT_RULES:
        for level in levels:
            threshold = float(np.quantile(scores[name], 1 - level))
            thresholds[name, level] = threshold
            report[f"{name}@{level:g}"] = {
                "threshold": -threshold if name == "latent_point" else threshold,
                "false_arrival": float(np.mean(impostor[name] >= threshold)),
            }
    return thresholds, report


def arms(
    thresholds: dict,
    levels: tuple[float, ...],
    encode: Callable[[np.ndarray], np.ndarray],
    bandwidth: float,
) -> dict[str, Arm]:
    """Cross target and arrival for the task controller, plus the latent executor arms."""
    result = {
        "point_native": Arm("observation", "decoded"),
        "sample_native": Arm("observation", "sample"),
    }
    for level in levels:
        tag = f"@{level:g}"
        support = RegionSupport(bandwidth, thresholds["region_flow", level])
        result["decoded_support" + tag] = Arm("observation", "decoded", support)
        result["sample_support" + tag] = Arm("observation", "sample", support)
        result["latent_point" + tag] = Arm(
            "latent",
            "latent",
            EncoderConsistency(encode, -thresholds["latent_point", level]),
        )
        result["region_gaussian" + tag] = Arm(
            "latent",
            "latent",
            GaussianSupport(bandwidth, thresholds["region_gaussian", level]),
        )
        result["region_flow" + tag] = Arm("latent", "latent", support)
    return result


def describe(arm: Arm) -> dict:
    rule = arm.arrival
    if rule is None:
        arrival = {"rule": "native"}
    elif isinstance(rule, EncoderConsistency):
        arrival = {"rule": "encoder_consistency", "radius": rule.epsilon}
    else:
        arrival = {"rule": type(rule).__name__, "threshold": rule.threshold}
    return {"executor": arm.executor, "target": arm.target, "arrival": arrival}


def coherence(values: np.ndarray, repeats: int) -> dict:
    """Variance of value within regions, between regions, and binomial noise."""
    values = np.asarray(values, dtype=float)
    return {
        "within_region_variance": float(values.var(axis=1).mean()),
        "between_region_variance": float(values.mean(axis=1).var()),
        "binomial_noise_variance": float((values * (1 - values)).mean() / repeats),
    }
