"""Predict and sample masked continuous-control trajectories in latent space."""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
from torch import Tensor, nn

from worth_reaching.data.trajectory_windows import CompletionContext, MaskedBatch
from worth_reaching.models import flow
from worth_reaching.models.codec import ObservationCodec, mlp, reconstruction
from worth_reaching.models.training import (
    PredictorStopping,
    RandomBatches,
    regression,
    restore_weights,
    stage_resume,
    stream_seed,
    tensor,
    train_stage,
    weights,
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
    flow_width: int = 128
    flow_batch_size: int = 64
    candidates: int = 4
    predictor_stopping: PredictorStopping = PredictorStopping()
    learning_rate: float = 0.001
    max_action_budget: ClassVar[int | None] = None

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
                self.flow_width,
                self.flow_batch_size,
                self.candidates,
            )
            < 1
            or not np.isfinite(self.learning_rate)
            or self.learning_rate <= 0
            or self.max_horizon <= self.queries
            or (
                self.max_action_budget is not None
                and self.max_action_budget < self.max_horizon
            )
        ):
            raise ValueError(
                "Use positive completion dimensions and a horizon longer than the query count"
            )


@dataclass(frozen=True, slots=True)
class BudgetedCompletionConfig(CompletionConfig):
    max_action_budget: int | None = None


class LatentTrajectory(nn.Module):
    def __init__(self, input_size: int, config: CompletionConfig):
        super().__init__()
        self.queries = config.queries
        self.latent_size = config.latent_size
        self.net = mlp(
            input_size, config.width, self.queries * self.latent_size, layers=3
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features).reshape(-1, self.queries, self.latent_size)


@torch.no_grad()
def public_features(
    codec: ObservationCodec,
    context: CompletionContext,
    mean: np.ndarray,
    scale: np.ndarray,
    config: CompletionConfig,
) -> Tensor:
    observations = tensor((context.observations - mean) / scale)
    goal = tensor((context.goal - mean) / scale)
    present = torch.as_tensor(context.present)
    history = codec.encode(observations).masked_fill(~present[..., None], 0)
    actions = tensor(context.incoming_actions).masked_fill(~present[..., None], 0)
    return torch.cat(
        (
            history.flatten(1),
            actions.flatten(1),
            present,
            codec.encode(goal),
            observations[:, -1, :2],
            goal[:, :2],
            tensor(context.remaining[:, None])
            / (config.max_action_budget or config.max_horizon),
            tensor(context.offsets) / config.max_horizon,
        ),
        dim=-1,
    )


def _flow_condition(features: Tensor, forecast: Tensor) -> Tensor:
    return torch.cat((features, forecast.flatten(1)), dim=-1)


@dataclass(frozen=True, slots=True)
class LearnedCompletion:
    config: CompletionConfig
    codec: ObservationCodec
    predictor: LatentTrajectory
    flow: flow.RouteFlow
    mean: np.ndarray
    scale: np.ndarray
    flow_scale: Tensor

    @classmethod
    def from_weights(
        cls, config: CompletionConfig, values: dict
    ) -> "LearnedCompletion":
        mean, scale = values["mean"], values["scale"]
        flow_scale = tensor(values["flow_scale"])
        if (
            mean.ndim != 1
            or scale.shape != mean.shape
            or not np.isfinite(mean).all()
            or not (np.isfinite(scale) & (scale > 0)).all()
            or flow_scale.shape != (config.latent_size,)
            or not bool((torch.isfinite(flow_scale) & (flow_scale > 0)).all())
        ):
            raise ValueError("Invalid completion normalization or latent scale")
        codec = ObservationCodec(config.width, config.latent_size, mean.size)
        predictor = LatentTrajectory(
            values["predictor"]["net.0.weight"].shape[1], config
        )
        restore_weights(codec, values["codec"])
        restore_weights(predictor, values["predictor"])
        generator = flow.load_parameters(
            config.queries * config.latent_size, values["flow"]
        )
        return cls(config, codec, predictor, generator, mean, scale, flow_scale)

    @property
    def weights(self) -> dict:
        return {
            "codec": weights(self.codec),
            "predictor": weights(self.predictor),
            "flow": weights(self.flow),
        }

    @torch.no_grad()
    def forecast(self, context: CompletionContext) -> tuple[Tensor, Tensor]:
        features = public_features(
            self.codec, context, self.mean, self.scale, self.config
        )
        return features, self.predictor(features)

    @torch.no_grad()
    def encode_observations(
        self, observations: np.ndarray, chunk: int = 65536
    ) -> np.ndarray:
        """Frozen encodings of raw observations, row by row."""
        normalized = (observations - self.mean) / self.scale
        return np.concatenate(
            [
                self.codec.encode(tensor(normalized[begin : begin + chunk])).numpy()
                for begin in range(0, len(normalized), chunk)
            ]
        ).astype(np.float32)

    @torch.no_grad()
    def decode(self, latent: Tensor) -> np.ndarray:
        return self.codec.decode(latent).numpy() * self.scale + self.mean

    def propose(
        self, context: CompletionContext, seed: int, *, candidates: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        features, forecast = self.forecast(context)
        routes = self._sample(features, forecast, seed, candidates=candidates)
        return routes.numpy(), self.decode(routes)

    def _sample(
        self,
        features: Tensor,
        forecast: Tensor,
        seed: int,
        *,
        candidates: int | None = None,
    ) -> Tensor:
        return flow.sample(
            self.flow,
            forecast,
            _flow_condition(features, forecast),
            self.flow_scale,
            seed,
            candidates=self.config.candidates if candidates is None else candidates,
            steps=12,
        )

    @torch.no_grad()
    def validate(
        self, batch: MaskedBatch, *, compare_xy: bool = True
    ) -> dict[str, float]:
        features, forecast = self.forecast(batch.context)
        target = self.codec.encode(tensor((batch.targets - self.mean) / self.scale))
        reconstructed = self.decode(target)
        metrics = {
            "normalized_reconstruction_mse": float(
                np.mean(np.square((reconstructed - batch.targets) / self.scale))
            ),
            "latent_prediction_mse": float((forecast - target).square().mean()),
        }
        if not compare_xy:
            return {
                **metrics,
                "normalized_prediction_mse": float(
                    np.mean(
                        np.square((self.decode(forecast) - batch.targets) / self.scale)
                    )
                ),
            }
        replay_xy = batch.targets[..., :2]
        samples = self._sample(
            features,
            forecast,
            stream_seed(self.config.seed, 802),
        )
        sampled_errors = np.linalg.norm(
            self.decode(samples)[..., :2] - replay_xy[:, None], axis=-1
        ).mean(axis=-1)
        start = batch.context.observations[:, -1, None, :2]
        span = np.minimum(batch.context.remaining, self.config.max_horizon)
        fraction = batch.context.offsets / span[:, None]
        interpolated = start + fraction[..., None] * (
            batch.context.goal[:, None, :2] - start
        )
        return {
            **metrics,
            "reconstructed_xy_mean_error": float(
                np.linalg.norm(reconstructed[..., :2] - replay_xy, axis=-1).mean()
            ),
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
    validation: MaskedBatch,
    mean: np.ndarray,
    scale: np.ndarray,
    config: CompletionConfig,
    out: Path,
    resume: Path | None = None,
) -> LearnedCompletion:
    targets = tensor((training.targets - mean) / scale)
    observations = targets.flatten(0, 1)
    torch.manual_seed(stream_seed(config.seed, 601))
    codec = ObservationCodec(config.width, config.latent_size, observations.shape[-1])
    codec = train_stage(
        codec,
        RandomBatches((observations,), config.batch_size, (config.seed, 601)),
        reconstruction,
        steps=config.codec_updates,
        learning_rate=config.learning_rate,
        out=out / "codec",
        resume=stage_resume(resume, "codec"),
    )
    features = public_features(codec, training.context, mean, scale, config)
    validation_features = public_features(
        codec, validation.context, mean, scale, config
    )
    with torch.no_grad():
        encoded_targets = codec.encode(targets)
        validation_targets = codec.encode(tensor((validation.targets - mean) / scale))
    torch.manual_seed(stream_seed(config.seed, 602))
    predictor = train_stage(
        LatentTrajectory(features.shape[-1], config),
        RandomBatches(
            (features, encoded_targets), config.batch_size, (config.seed, 602)
        ),
        regression,
        steps=config.predictor_updates,
        learning_rate=config.learning_rate,
        out=out / "predictor",
        resume=stage_resume(resume, "predictor"),
        validation=(validation_features, validation_targets),
        stopping=config.predictor_stopping,
    )
    with torch.no_grad():
        forecast = predictor(features)
    generator, flow_scale = flow.fit(
        _flow_condition(features, forecast),
        encoded_targets - forecast,
        seed=config.seed,
        updates=config.flow_updates,
        learning_rate=config.learning_rate,
        width=config.flow_width,
        batch_size=config.flow_batch_size,
        out=out / "flow",
        resume=stage_resume(resume, "flow"),
    )
    return LearnedCompletion(
        config, codec, predictor, generator, mean, scale, flow_scale
    )
