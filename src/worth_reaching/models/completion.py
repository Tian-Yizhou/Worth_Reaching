"""Learn latent predictions from public histories and masked observations."""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState

from worth_reaching.data.completion import CompletionExample, CompletionInput
from worth_reaching.models.parallel import (
    mean_squared,
    replicate,
    shard_batch,
    single_device,
)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int = 7
    encoder_updates: int = 250
    updates: int = 600
    batch_size: int = 32
    history_length: int = 16
    queries_per_window: int = 16
    width: int = 64
    latent_size: int = 16
    learning_rate: float = 0.003

    def __post_init__(self) -> None:
        if not 0 <= self.seed < 2**32:
            raise ValueError("Use a training seed in the 32-bit unsigned range")
        if (
            min(
                self.encoder_updates,
                self.updates,
                self.batch_size,
                self.history_length,
                self.queries_per_window,
                self.width,
                self.latent_size,
            )
            < 1
            or not np.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError(
                "Training dimensions, update counts and rate must be positive"
            )


@dataclass(frozen=True, slots=True)
class ModelSpec:
    observation_size: int
    action_count: int
    goals: tuple[str, ...]
    budget: int


class PackedInput(NamedTuple):
    observations: jax.Array
    actions: jax.Array
    lengths: jax.Array
    goal: jax.Array
    remaining: jax.Array
    queries: jax.Array

    def without_goal(self) -> "PackedInput":
        return self._replace(goal=jnp.full_like(self.goal, -1))


def _query_indices(span: int, limit: int) -> np.ndarray:
    if limit == 1:
        return np.array([span // 2])
    return np.linspace(0, span - 1, min(limit, span), dtype=np.int32)


def pack_inputs(
    histories: Sequence[CompletionInput], spec: ModelSpec, config: TrainingConfig
) -> PackedInput:
    history = min(config.history_length, spec.budget)
    queries = min(config.queries_per_window, spec.budget - 1)
    count = len(histories)
    observations = np.zeros((count, history, spec.observation_size), dtype=np.float32)
    actions = np.full((count, history), -1, dtype=np.int32)
    lengths = np.empty(count, dtype=np.int32)
    goals = np.empty(count, dtype=np.int32)
    remaining = np.empty(count, dtype=np.float32)
    query_steps = np.zeros((count, queries), dtype=np.int32)
    for index, public in enumerate(histories):
        length = min(len(public.observations), history)
        observations[index, :length] = public.observations[-length:]
        incoming_actions = min(length, len(public.actions))
        if incoming_actions:
            actions[index, length - incoming_actions : length] = public.actions[
                -incoming_actions:
            ]
        lengths[index] = length
        goals[index] = spec.goals.index(public.goal)
        remaining[index] = public.remaining_actions
        selected = _query_indices(len(public.query_steps), queries)
        query_steps[index, : len(selected)] = public.query_steps[selected]
    return PackedInput(observations, actions, lengths, goals, remaining, query_steps)


def pack_windows(
    examples: Sequence[CompletionExample], spec: ModelSpec, config: TrainingConfig
) -> tuple[PackedInput, np.ndarray]:
    inputs = pack_inputs([example.inputs for example in examples], spec, config)
    targets = np.zeros((*inputs.queries.shape, spec.observation_size), dtype=np.float32)
    for index, example in enumerate(examples):
        selected = _query_indices(
            len(example.inputs.query_steps), inputs.queries.shape[1]
        )
        targets[index, : len(selected)] = example.target.observations[selected]
    return inputs, targets


class ObservationCodec(nn.Module):
    width: int
    latent_size: int
    observation_size: int

    def setup(self) -> None:
        self.encoder = nn.Sequential(
            (nn.Dense(self.width), nn.gelu, nn.Dense(self.latent_size))
        )
        self.decoder = nn.Sequential(
            (nn.Dense(self.width), nn.gelu, nn.Dense(self.observation_size))
        )

    def encode(self, observations: jax.Array) -> jax.Array:
        return self.encoder(observations)

    def decode(self, latents: jax.Array) -> jax.Array:
        return self.decoder(latents)

    def __call__(self, observations: jax.Array) -> jax.Array:
        return self.decode(self.encode(observations))


class LatentPredictor(nn.Module):
    spec: ModelSpec
    width: int
    latent_size: int

    @nn.compact
    def __call__(self, inputs: PackedInput, encoded_history: jax.Array) -> jax.Array:
        actions = jax.nn.one_hot(inputs.actions, self.spec.action_count)
        history = jnp.concatenate((encoded_history, actions), axis=-1)
        context, _ = nn.RNN(nn.GRUCell(self.width), return_carry=True)(
            history, seq_lengths=inputs.lengths
        )
        goal = jax.nn.one_hot(inputs.goal, len(self.spec.goals))
        context = jnp.concatenate((context, goal), axis=-1)
        context = jnp.broadcast_to(
            context[:, None, :], (*inputs.queries.shape, context.shape[-1])
        )
        remaining = jnp.broadcast_to(inputs.remaining[:, None], inputs.queries.shape)
        time = jnp.stack(
            (
                remaining / self.spec.budget,
                inputs.queries / self.spec.budget,
                inputs.queries / remaining,
            ),
            axis=-1,
        )
        hidden = nn.gelu(
            nn.Dense(self.width)(jnp.concatenate((context, time), axis=-1))
        )
        hidden = nn.gelu(nn.Dense(self.width)(hidden))
        return nn.Dense(self.latent_size)(hidden)


def _loss(predicted: jax.Array, target: jax.Array, valid: jax.Array) -> jax.Array:
    per_step = jnp.mean(jnp.square(predicted - target), axis=-1)
    return jnp.sum(jnp.where(valid, per_step, 0.0)) / jnp.sum(valid)


@jax.jit
def fit_codec_step(
    state: TrainState, observations: jax.Array, valid: jax.Array | None
) -> TrainState:
    def objective(params):
        return mean_squared(state.apply_fn(params, observations) - observations, valid)

    return state.apply_gradients(grads=jax.grad(objective)(state.params))


@jax.jit
def _fit_predictor(
    state: TrainState,
    inputs: PackedInput,
    history: jax.Array,
    targets: jax.Array,
    valid: jax.Array | None,
) -> TrainState:
    def objective(params):
        query_mask = inputs.queries > 0
        if valid is not None:
            query_mask = query_mask & valid[:, None]
        return _loss(state.apply_fn(params, inputs, history), targets, query_mask)

    return state.apply_gradients(grads=jax.grad(objective)(state.params))


@partial(jax.jit, static_argnames=("codec",))
def encode_observations(
    codec: ObservationCodec, params, observations: jax.Array
) -> jax.Array:
    return codec.apply(params, observations, method=codec.encode)


@partial(jax.jit, static_argnames=("codec", "predictor"))
def predict(
    codec: ObservationCodec,
    predictor: LatentPredictor,
    codec_params,
    predictor_params,
    inputs: PackedInput,
) -> tuple[jax.Array, jax.Array]:
    """Predict latent values and decode them for observation-space diagnostics."""
    history = encode_observations(codec, codec_params, inputs.observations)
    latent = predictor.apply(predictor_params, inputs, history)
    return latent, codec.apply(codec_params, latent, method=codec.decode)


def fit(
    training: Sequence[CompletionExample], spec: ModelSpec, config: TrainingConfig
) -> tuple[ObservationCodec, LatentPredictor, dict]:
    if not training:
        raise ValueError("Latent completion requires training windows")
    codec = ObservationCodec(config.width, config.latent_size, spec.observation_size)
    predictor = LatentPredictor(spec, config.width, config.latent_size)
    codec_key, predictor_key = jax.random.split(jax.random.key(config.seed))
    encoder_rng = np.random.default_rng([config.seed, 0])
    predictor_rng = np.random.default_rng([config.seed, 1])
    optimizer = optax.adam(config.learning_rate)
    codec_state = TrainState.create(
        apply_fn=codec.apply,
        params=codec.init(codec_key, training[0].target.observations[:1]),
        tx=optimizer,
    )
    codec_state = replicate(codec_state)
    for _ in range(config.encoder_updates):
        observations = np.stack(
            [
                training[index].target.observations[
                    encoder_rng.integers(len(training[index].target.observations))
                ]
                for index in encoder_rng.integers(len(training), size=config.batch_size)
            ]
        )
        observations, valid = shard_batch(observations)
        codec_state = fit_codec_step(codec_state, observations, valid)

    codec_params = codec_state.params
    example_input, _ = pack_windows(training[:1], spec, config)
    encoded = encode_observations(
        codec, single_device(codec_params), example_input.observations
    )
    initial = TrainState.create(
        apply_fn=predictor.apply,
        params=predictor.init(predictor_key, example_input, encoded),
        tx=optimizer,
    )
    conditioned = goal_blind = replicate(initial)
    for _ in range(config.updates):
        indices = predictor_rng.integers(len(training), size=config.batch_size)
        inputs, targets = pack_windows(
            [training[index] for index in indices], spec, config
        )
        (inputs, targets), valid = shard_batch((inputs, targets))
        history = encode_observations(codec, codec_params, inputs.observations)
        targets = encode_observations(codec, codec_params, targets)
        conditioned = _fit_predictor(conditioned, inputs, history, targets, valid)
        goal_blind = _fit_predictor(
            goal_blind, inputs.without_goal(), history, targets, valid
        )
    return (
        codec,
        predictor,
        single_device(
            {
                "codec": codec_params,
                "conditioned": conditioned.params,
                "goal_blind": goal_blind.params,
            }
        ),
    )
