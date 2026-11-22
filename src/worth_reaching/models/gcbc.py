"""Continuous-action goal-conditioned behavioral cloning for public state inputs."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState

from worth_reaching.data.antmaze import ACTION_SIZE, OBSERVATION_SIZE


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
) -> TrainState:
    def loss(params):
        predicted = state.apply_fn(params, observation, goal)
        return jnp.square(predicted - actions).mean()

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
