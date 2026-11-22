"""Learn latent predictions from public histories and masked observations."""

import json
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from flax import serialization
from torch import nn
from torch.nn import functional as F

from worth_reaching.data.completion import CompletionExample, CompletionInput
from worth_reaching.models.codec import (
    ObservationCodec,
    legacy_dense,
    mlp,
    reconstruction,
)
from worth_reaching.models.training import (
    RandomBatches,
    restore_weights,
    stage_resume,
    stream_seed,
    tensor,
    train_stage,
)

CHECKPOINT_SCHEMA = "latent_completion_v2"
LEGACY_SCHEMA = "latent_completion_v1"


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
    observations: np.ndarray
    actions: np.ndarray
    lengths: np.ndarray
    goal: np.ndarray
    remaining: np.ndarray
    queries: np.ndarray

    def without_goal(self) -> "PackedInput":
        return self._replace(goal=np.full_like(self.goal, -1))


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


def one_hot(indices, count: int):
    return F.one_hot(indices.clamp_min(0).long(), count) * (indices[..., None] >= 0)


class LatentPredictor(nn.Module):
    def __init__(self, spec: ModelSpec, width: int, latent_size: int):
        super().__init__()
        self.spec = spec
        self.history = nn.GRU(latent_size + spec.action_count, width, batch_first=True)
        self.net = mlp(width + len(spec.goals) + 3, width, latent_size, layers=2)

    def forward(self, history, actions, lengths, goal, remaining, queries):
        features = torch.cat(
            (history, one_hot(actions, self.spec.action_count)), dim=-1
        )
        hidden, _ = self.history(features)
        context = hidden[
            torch.arange(len(hidden), device=hidden.device), lengths.long() - 1
        ]
        context = torch.cat((context, one_hot(goal, len(self.spec.goals))), dim=-1)
        context = context[:, None].expand(-1, queries.shape[1], -1)
        remaining = remaining[:, None].expand_as(queries)
        time = torch.stack(
            (
                remaining / self.spec.budget,
                queries / self.spec.budget,
                queries / remaining,
            ),
            dim=-1,
        )
        return self.net(torch.cat((context, time), dim=-1))


@torch.no_grad()
def predictor_inputs(codec: ObservationCodec, inputs: PackedInput):
    return (
        codec.encode(tensor(inputs.observations)),
        *(tensor(value) for value in inputs[1:]),
    )


@torch.no_grad()
def predict(codec: ObservationCodec, predictor: LatentPredictor, inputs: PackedInput):
    latent = predictor(*predictor_inputs(codec, inputs))
    return latent, codec.decode(latent)


@dataclass(frozen=True)
class EncoderBatches:
    observations: tuple[np.ndarray, ...]
    batch_size: int
    seed: int

    def __call__(self, step: int):
        rng = np.random.default_rng([self.seed, 0, step])
        return (
            tensor(
                np.stack(
                    [
                        self.observations[index][
                            rng.integers(len(self.observations[index]))
                        ]
                        for index in rng.integers(
                            len(self.observations), size=self.batch_size
                        )
                    ]
                )
            ),
        )


def prediction_loss(model, batch):
    *inputs, expected = batch
    valid = inputs[-1][..., None] > 0
    return ((model(*inputs) - expected, valid),)


def fit(
    training: Sequence[CompletionExample],
    spec: ModelSpec,
    config: TrainingConfig,
    out: Path,
    resume: Path | None = None,
) -> tuple[ObservationCodec, LatentPredictor, LatentPredictor]:
    if not training:
        raise ValueError("Latent completion requires training windows")
    torch.manual_seed(stream_seed(config.seed, 0))
    codec = train_stage(
        ObservationCodec(config.width, config.latent_size, spec.observation_size),
        EncoderBatches(
            tuple(example.target.observations for example in training),
            config.batch_size,
            config.seed,
        ),
        reconstruction,
        steps=config.encoder_updates,
        learning_rate=config.learning_rate,
        out=out / "codec",
        resume=stage_resume(resume, "codec"),
    )
    inputs, targets = pack_windows(training, spec, config)
    with torch.no_grad():
        encoded = codec.encode(tensor(targets))
    torch.manual_seed(stream_seed(config.seed, 1))
    initial = LatentPredictor(spec, config.width, config.latent_size)
    predictors = []
    for stage, public in (
        ("conditioned", inputs),
        ("goal_blind", inputs.without_goal()),
    ):
        arrays = (*predictor_inputs(codec, public), encoded)
        predictors.append(
            train_stage(
                deepcopy(initial),
                RandomBatches(arrays, config.batch_size, (config.seed, 1)),
                prediction_loss,
                steps=config.updates,
                learning_rate=config.learning_rate,
                out=out / stage,
                resume=stage_resume(resume, stage),
            )
        )
    return codec, *predictors


def _legacy_predictor(params: dict) -> dict:
    gru = params["GRUCell_0"]
    values = legacy_dense(params, "net", "Dense_0", "Dense_1", "Dense_2")
    zeros = np.zeros_like(gru["hn"]["bias"])
    values.update(
        {
            "history.weight_ih_l0": np.concatenate(
                [gru[f"i{gate}"]["kernel"].T for gate in "rzn"]
            ),
            "history.weight_hh_l0": np.concatenate(
                [gru[f"h{gate}"]["kernel"].T for gate in "rzn"]
            ),
            "history.bias_ih_l0": np.concatenate(
                [gru[f"i{gate}"]["bias"] for gate in "rzn"]
            ),
            "history.bias_hh_l0": np.concatenate([zeros, zeros, gru["hn"]["bias"]]),
        }
    )
    return values


@dataclass(frozen=True, slots=True)
class FrozenCompletion:
    spec: ModelSpec
    config: TrainingConfig
    source: dict
    codec: ObservationCodec
    predictor: LatentPredictor

    @classmethod
    def load(cls, directory: Path) -> "FrozenCompletion":
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema") not in (CHECKPOINT_SCHEMA, LEGACY_SCHEMA):
            raise ValueError("Expected a latent completion checkpoint")
        spec = ModelSpec(
            **{**manifest["model"], "goals": tuple(manifest["model"]["goals"])}
        )
        config = TrainingConfig(**manifest["config"])
        values = serialization.msgpack_restore(
            (directory / "weights.msgpack").read_bytes()
        )
        codec_weights = values["codec"]
        predictor_weights = values["conditioned"]
        if manifest["schema"] == LEGACY_SCHEMA:
            codec_weights = {
                name: value
                for section, params in codec_weights["params"].items()
                for name, value in legacy_dense(
                    params, section, "layers_0", "layers_2"
                ).items()
            }
            predictor_weights = _legacy_predictor(predictor_weights["params"])
        codec = ObservationCodec(
            config.width, config.latent_size, spec.observation_size
        )
        predictor = LatentPredictor(spec, config.width, config.latent_size)
        restore_weights(codec, codec_weights)
        restore_weights(predictor, predictor_weights)
        return cls(spec, config, manifest["source"], codec, predictor)

    @torch.no_grad()
    def forecast(self, inputs: PackedInput):
        return self.predictor(*predictor_inputs(self.codec, inputs))

    @torch.no_grad()
    def encode(self, observations):
        return self.codec.encode(tensor(observations))

    @torch.no_grad()
    def decode(self, latents):
        return self.codec.decode(tensor(latents))
