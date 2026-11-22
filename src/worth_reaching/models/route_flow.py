"""Conditional flow matching over complete masked latent routes."""

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax import serialization
from flax.training.train_state import TrainState

from worth_reaching.models.completion import (
    LatentPredictor,
    ModelSpec,
    ObservationCodec,
    PackedInput,
    TrainingConfig,
    encode_observations,
)


@dataclass(frozen=True, slots=True)
class FrozenCompletion:
    spec: ModelSpec
    config: TrainingConfig
    weights: dict
    source: dict

    @classmethod
    def load(cls, directory: Path) -> "FrozenCompletion":
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema") != "latent_completion_v1":
            raise ValueError("Expected a latent completion checkpoint")
        spec = ModelSpec(
            **{**manifest["model"], "goals": tuple(manifest["model"]["goals"])}
        )
        weights = serialization.msgpack_restore(
            (directory / "weights.msgpack").read_bytes()
        )
        return cls(
            spec, TrainingConfig(**manifest["config"]), weights, manifest["source"]
        )

    def codec(self) -> ObservationCodec:
        return ObservationCodec(
            self.config.width, self.config.latent_size, self.spec.observation_size
        )

    def forecast(self, inputs: PackedInput) -> jax.Array:
        codec = self.codec()
        history = encode_observations(codec, self.weights["codec"], inputs.observations)
        predictor = LatentPredictor(
            self.spec, self.config.width, self.config.latent_size
        )
        return predictor.apply(self.weights["conditioned"], inputs, history)

    def encode(self, observations: np.ndarray) -> jax.Array:
        return encode_observations(self.codec(), self.weights["codec"], observations)

    def decode(self, latents: jax.Array) -> jax.Array:
        codec = self.codec()
        return codec.apply(self.weights["codec"], latents, method=codec.decode)


class RouteFlow(nn.Module):
    size: int

    @nn.compact
    def __call__(
        self, noisy: jax.Array, time: jax.Array, condition: jax.Array
    ) -> jax.Array:
        hidden = jnp.concatenate((noisy, time, condition), axis=-1)
        hidden = nn.silu(nn.Dense(128)(hidden))
        hidden = nn.silu(nn.Dense(128)(hidden))
        return nn.Dense(self.size)(hidden)


def route_conditions(inputs: PackedInput, forecast: jax.Array, goals: int) -> jax.Array:
    return jnp.concatenate(
        (
            forecast.reshape((len(forecast), -1)),
            inputs.observations[:, 0],
            jax.nn.one_hot(inputs.goal, goals),
        ),
        axis=-1,
    )


@jax.jit
def _update(
    state: TrainState, target: jax.Array, condition: jax.Array, key: jax.Array
) -> TrainState:
    noise_key, time_key = jax.random.split(key)
    noise = jax.random.normal(noise_key, target.shape)
    time = jax.random.uniform(time_key, (len(target), 1))
    current = noise + time * (target - noise)

    def loss(params):
        prediction = state.apply_fn(params, current, time, condition)
        return jnp.square(prediction - (target - noise)).mean()

    return state.apply_gradients(grads=jax.grad(loss)(state.params))


def fit(
    condition: jax.Array,
    residual: jax.Array,
    *,
    seed: int,
    updates: int,
) -> tuple[RouteFlow, dict, jax.Array]:
    scale = jnp.maximum(jnp.std(residual, axis=(0, 1)), 0.01)
    normalized = (residual / scale).reshape((len(residual), -1))
    model = RouteFlow(normalized.shape[-1])
    key = jax.random.key(seed)
    state = TrainState.create(
        apply_fn=model.apply,
        params=model.init(key, normalized[:1], jnp.zeros((1, 1)), condition[:1]),
        tx=optax.adam(0.001),
    )
    rng = np.random.default_rng([seed, 1])
    for index in range(updates):
        batch = rng.integers(len(residual), size=64)
        state = _update(
            state,
            normalized[batch],
            condition[batch],
            jax.random.fold_in(key, index + 1),
        )
    return model, state.params, scale


@partial(jax.jit, static_argnames=("model", "candidates", "steps"))
def sample(
    model: RouteFlow,
    params: dict,
    forecast: jax.Array,
    condition: jax.Array,
    scale: jax.Array,
    key: jax.Array,
    *,
    candidates: int,
    steps: int,
) -> jax.Array:
    repeated = jnp.repeat(condition, candidates, axis=0)
    noise = jax.random.normal(key, (len(repeated), model.size))

    def advance(index, latent):
        time = jnp.full((len(latent), 1), index / steps)
        return latent + model.apply(params, latent, time, repeated) / steps

    residual = jax.lax.fori_loop(0, steps, advance, noise)
    return (
        forecast[:, None]
        + residual.reshape((len(forecast), candidates, *forecast.shape[1:])) * scale
    )
