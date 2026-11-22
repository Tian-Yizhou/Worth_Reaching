"""Predict and sample masked AntMaze trajectories in the learned observation space."""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState

from worth_reaching.data.antmaze import OBSERVATION_SIZE, CompletionContext, MaskedBatch
from worth_reaching.models import route_flow
from worth_reaching.models.completion import (
    ObservationCodec,
    encode_observations,
    fit_codec_step,
)
from worth_reaching.models.parallel import (
    mean_squared,
    replicate,
    shard_batch,
    single_device,
)


@dataclass(frozen=True, slots=True)
class CompletionConfig:
    seed: int = 7
    windows: int = 16000
    history: int = 4
    queries: int = 8
    max_horizon: int = 1000
    batch_size: int = 128
    width: int = 128
    latent_size: int = 16
    codec_updates: int = 500
    predictor_updates: int = 1200
    flow_updates: int = 700
    candidates: int = 4

    def __post_init__(self) -> None:
        if (
            not 0 <= self.seed < 2**32
            or min(
                self.windows,
                self.history,
                self.queries,
                self.batch_size,
                self.width,
                self.latent_size,
                self.codec_updates,
                self.predictor_updates,
                self.flow_updates,
                self.candidates,
            )
            < 1
            or self.max_horizon <= self.queries
        ):
            raise ValueError(
                "Use positive completion dimensions and a horizon longer than the query count"
            )


class LatentTrajectory(nn.Module):
    width: int
    queries: int
    latent_size: int

    @nn.compact
    def __call__(self, features: jax.Array) -> jax.Array:
        for _ in range(3):
            features = nn.gelu(nn.Dense(self.width)(features))
        latent = nn.Dense(self.queries * self.latent_size)(features)
        return latent.reshape(-1, self.queries, self.latent_size)


@partial(jax.jit, static_argnames=("codec", "max_horizon"))
def public_features(
    codec: ObservationCodec,
    weights: dict,
    context: CompletionContext,
    mean: np.ndarray,
    scale: np.ndarray,
    max_horizon: int,
) -> jax.Array:
    observations = (context.observations - mean) / scale
    goal = (context.goal - mean) / scale
    history = encode_observations(codec, weights, observations)
    history = jnp.where(context.present[..., None], history, 0)
    actions = jnp.where(context.present[..., None], context.incoming_actions, 0)
    goal_latent = encode_observations(codec, weights, goal)
    count = len(goal)
    return jnp.concatenate(
        (
            history.reshape(count, -1),
            actions.reshape(count, -1),
            context.present,
            goal_latent,
            observations[:, -1, :2],
            goal[:, :2],
            context.remaining[:, None] / max_horizon,
            context.offsets / max_horizon,
        ),
        axis=-1,
    )


@jax.jit
def _fit_predictor(state: TrainState, features, target, valid) -> TrainState:
    def objective(params):
        return mean_squared(state.apply_fn(params, features) - target, valid)

    return state.apply_gradients(grads=jax.grad(objective)(state.params))


@partial(jax.jit, static_argnames=("predictor",))
def _predict(
    predictor: LatentTrajectory, weights: dict, features: jax.Array
) -> jax.Array:
    return predictor.apply(weights, features)


@partial(jax.jit, static_argnames=("codec",))
def _decode(codec: ObservationCodec, weights: dict, latent: jax.Array) -> jax.Array:
    return codec.apply(weights, latent, method=codec.decode)


def _flow_condition(features: jax.Array, forecast: jax.Array) -> jax.Array:
    return jnp.concatenate((features, forecast.reshape(len(forecast), -1)), axis=-1)


@dataclass(frozen=True, slots=True)
class LearnedCompletion:
    config: CompletionConfig
    codec: ObservationCodec
    predictor: LatentTrajectory
    flow: route_flow.RouteFlow
    weights: dict
    mean: np.ndarray
    scale: np.ndarray
    flow_scale: jax.Array

    def forecast(self, context: CompletionContext) -> tuple[jax.Array, jax.Array]:
        features = public_features(
            self.codec,
            self.weights["codec"],
            context,
            self.mean,
            self.scale,
            self.config.max_horizon,
        )
        return features, _predict(self.predictor, self.weights["predictor"], features)

    def decode(self, latent: jax.Array) -> np.ndarray:
        return np.asarray(
            _decode(self.codec, self.weights["codec"], latent) * self.scale + self.mean
        )

    def propose(
        self, context: CompletionContext, key: jax.Array
    ) -> tuple[np.ndarray, np.ndarray]:
        features, forecast = self.forecast(context)
        routes = self._sample(features, forecast, key)
        return np.asarray(routes), self.decode(routes)

    def _sample(
        self, features: jax.Array, forecast: jax.Array, key: jax.Array
    ) -> jax.Array:
        return route_flow.sample(
            self.flow,
            self.weights["flow"],
            forecast,
            _flow_condition(features, forecast),
            self.flow_scale,
            key,
            candidates=self.config.candidates,
            steps=12,
        )

    def validate(self, batch: MaskedBatch) -> dict[str, float]:
        features, forecast = self.forecast(batch.context)
        target = encode_observations(
            self.codec, self.weights["codec"], (batch.targets - self.mean) / self.scale
        )
        reconstructed = self.decode(target)
        replay_xy = batch.targets[..., :2]
        samples = self._sample(
            features,
            forecast,
            jax.random.fold_in(jax.random.key(self.config.seed), 802),
        )
        sampled_errors = np.linalg.norm(
            self.decode(samples)[..., :2] - replay_xy[:, None], axis=-1
        ).mean(axis=-1)
        start = batch.context.observations[:, -1, None, :2]
        fraction = batch.context.offsets / batch.context.remaining[:, None]
        interpolated = start + fraction[..., None] * (
            batch.context.goal[:, None, :2] - start
        )
        return {
            "normalized_reconstruction_mse": float(
                np.mean(np.square((reconstructed - batch.targets) / self.scale))
            ),
            "reconstructed_xy_mean_error": float(
                np.linalg.norm(reconstructed[..., :2] - replay_xy, axis=-1).mean()
            ),
            "latent_prediction_mse": float(jnp.square(forecast - target).mean()),
            "decoded_xy_mean_error": float(
                np.linalg.norm(
                    self.decode(forecast)[..., :2] - replay_xy, axis=-1
                ).mean()
            ),
            "sampled_xy_mean_error": float(sampled_errors.mean()),
            "best_sampled_route_xy_mean_error": float(
                sampled_errors.min(axis=1).mean()
            ),
            "linear_xy_mean_error": float(
                np.linalg.norm(interpolated - replay_xy, axis=-1).mean()
            ),
        }


def fit(
    training: MaskedBatch,
    mean: np.ndarray,
    scale: np.ndarray,
    config: CompletionConfig,
) -> LearnedCompletion:
    codec = ObservationCodec(config.width, config.latent_size, OBSERVATION_SIZE)
    predictor = LatentTrajectory(config.width, config.queries, config.latent_size)
    codec_key, predictor_key = jax.random.split(jax.random.key(config.seed))
    targets = (training.targets - mean) / scale
    observations = targets.reshape(-1, OBSERVATION_SIZE)
    optimizer = optax.adam(0.001)
    codec_state = replicate(
        TrainState.create(
            apply_fn=codec.apply,
            params=codec.init(codec_key, observations[:1]),
            tx=optimizer,
        )
    )
    codec_rng = np.random.default_rng([config.seed, 601])
    for _ in range(config.codec_updates):
        rows = codec_rng.integers(len(observations), size=config.batch_size)
        batch, valid = shard_batch(observations[rows])
        codec_state = fit_codec_step(codec_state, batch, valid)
    codec_weights = single_device(codec_state.params)
    features = public_features(
        codec, codec_weights, training.context, mean, scale, config.max_horizon
    )
    encoded_targets = encode_observations(codec, codec_weights, targets)
    state = replicate(
        TrainState.create(
            apply_fn=predictor.apply,
            params=predictor.init(predictor_key, features[:1]),
            tx=optimizer,
        )
    )
    predictor_rng = np.random.default_rng([config.seed, 602])
    for _ in range(config.predictor_updates):
        rows = predictor_rng.integers(len(features), size=config.batch_size)
        batch, valid = shard_batch((features[rows], encoded_targets[rows]))
        state = _fit_predictor(state, *batch, valid)
    predictor_weights = single_device(state.params)
    forecast = _predict(predictor, predictor_weights, features)
    flow, flow_weights, flow_scale = route_flow.fit(
        _flow_condition(features, forecast),
        encoded_targets - forecast,
        seed=config.seed,
        updates=config.flow_updates,
    )
    weights = {
        "codec": codec_weights,
        "predictor": predictor_weights,
        "flow": flow_weights,
    }
    return LearnedCompletion(
        config, codec, predictor, flow, weights, mean, scale, flow_scale
    )
