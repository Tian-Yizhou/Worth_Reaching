"""Continuous-action goal-conditioned behavioral cloning for public state inputs."""

import json
from collections.abc import Callable
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax import serialization
from flax.training.train_state import TrainState

from worth_reaching.data.antmaze import ACTION_SIZE, OBSERVATION_SIZE
from worth_reaching.models.parallel import mean_squared

CHECKPOINT_SCHEMA = "antmaze_gcbc_v1"


class GCBC(nn.Module):
    width: int

    @nn.compact
    def __call__(self, observation: jax.Array, goal: jax.Array) -> jax.Array:
        # Keep the relative displacement explicit for the state-based executor.
        hidden = jnp.concatenate(
            (observation, goal, goal[..., :2] - observation[..., :2]), axis=-1
        )
        for _ in range(3):
            hidden = nn.gelu(nn.Dense(self.width)(hidden))
        return nn.Dense(ACTION_SIZE)(hidden)


def initialize(width: int, seed: int, learning_rate: float) -> tuple[GCBC, TrainState]:
    model = GCBC(width)
    observation = jnp.zeros((1, OBSERVATION_SIZE))
    return model, TrainState.create(
        apply_fn=model.apply,
        params=model.init(jax.random.key(seed), observation, observation),
        tx=optax.adam(learning_rate),
    )


@jax.jit
def update(
    state: TrainState,
    observation: np.ndarray,
    goal: np.ndarray,
    actions: np.ndarray,
    valid: jax.Array | None = None,
) -> TrainState:
    def loss(params):
        predicted = state.apply_fn(params, observation, goal)
        return mean_squared(predicted - actions, valid)

    return state.apply_gradients(grads=jax.grad(loss)(state.params))


@partial(jax.jit, static_argnames=("model",))
def predict(
    model: GCBC,
    weights: dict,
    observation: np.ndarray,
    goal: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> jax.Array:
    predicted = model.apply(
        weights, (observation - mean) / scale, (goal - mean) / scale
    )
    return jnp.clip(predicted, -1, 1)


def load_policy(directory: Path) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Load one frozen actor with the same normalization as the training run."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("Expected an AntMaze GCBC checkpoint")
    model = GCBC(manifest["training"]["width"])
    weights = serialization.msgpack_restore(
        (directory / "weights.msgpack").read_bytes()
    )
    with np.load(directory / "normalization.npz", allow_pickle=False) as values:
        mean, scale = values["mean"], values["scale"]
    if (
        mean.shape != (OBSERVATION_SIZE,)
        or scale.shape != mean.shape
        or not np.isfinite(mean).all()
        or not (np.isfinite(scale) & (scale > 0)).all()
    ):
        raise ValueError("Invalid GCBC normalization")

    def act(observation: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return np.asarray(predict(model, weights, observation, goal, mean, scale))

    return act
