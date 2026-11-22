"""Train a matched latent prediction comparison and score its held-out windows."""

from collections.abc import Sequence
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from flax import serialization

from worth_reaching.data.artifacts import code_revision, write_json
from worth_reaching.data.completion import CompletionExample, load_completion_corpus
from worth_reaching.envs import grid_routes
from worth_reaching.models.completion import (
    CHECKPOINT_SCHEMA,
    LatentPredictor,
    ModelSpec,
    ObservationCodec,
    TrainingConfig,
    fit,
    pack_windows,
    predict,
)
from worth_reaching.models.training import describe, tensor, validate_resume, weights


def score_paired_routes(
    predicted: np.ndarray,
    targets: np.ndarray,
    valid: np.ndarray,
    layouts: Sequence[str],
) -> dict[str, object]:
    """Compare each prediction with its own and the alternate goal's route."""
    step_count = targets.shape[1]
    paired_targets = targets.reshape(-1, 2, step_count, targets.shape[-1])
    actual = paired_targets[..., :2]
    estimated = predicted[..., :2].reshape(actual.shape)
    destinations = np.stack(
        (paired_targets[:, 0, 0, 4:6], paired_targets[:, 1, 0, 6:8]), axis=1
    )
    at_destination = np.all(actual == destinations[:, :, None, :], axis=-1)
    before_arrival = ~np.maximum.accumulate(at_destination.any(axis=1), axis=1)
    different_cell = np.any(actual[:, 0] != actual[:, 1], axis=-1)
    scored = (
        valid.reshape(-1, 2, step_count).all(axis=1) & before_arrival & different_cell
    )
    eligible = scored.any(axis=1)
    if not eligible.any():
        return {"worlds": 0, "layouts": 0, "divergent_steps": 0}
    weights = scored[eligible]
    actual, estimated = actual[eligible], estimated[eligible]
    cells = grid_routes.SIZE - 1

    def by_world(distance: np.ndarray) -> np.ndarray:
        return np.sum(np.where(weights[:, None, :], distance, 0.0), axis=(1, 2)) / (
            2 * weights.sum(axis=1)
        )

    matched = by_world(cells * np.abs(estimated - actual).sum(axis=-1))
    swapped = by_world(cells * np.abs(estimated - actual[:, ::-1]).sum(axis=-1))
    exact = np.all(np.rint(estimated * cells) == np.rint(actual * cells), axis=-1)
    margin = swapped - matched
    wins = (margin > 1e-6) + 0.5 * (np.abs(margin) <= 1e-6)
    maps, map_index = np.unique(np.asarray(layouts)[eligible], return_inverse=True)
    map_counts = np.bincount(map_index)
    map_margins = np.bincount(map_index, weights=margin)
    draws = np.random.default_rng(20).integers(len(maps), size=(2000, len(maps)))
    bootstrapped = map_margins[draws].sum(axis=1) / map_counts[draws].sum(axis=1)
    return {
        "worlds": int(eligible.sum()),
        "layouts": len(maps),
        "divergent_steps": int(scored.sum()),
        "matched_route_error_cells": float(matched.mean()),
        "swapped_route_error_cells": float(swapped.mean()),
        "matched_route_win_rate": float(wins.mean()),
        "route_error_improvement_layout_bootstrap_95": np.quantile(
            bootstrapped, [0.025, 0.975]
        ).tolist(),
        "exact_cell_rate": float(by_world(exact).mean()),
    }


@torch.no_grad()
def _evaluate(
    examples: Sequence[CompletionExample],
    spec: ModelSpec,
    config: TrainingConfig,
    codec: ObservationCodec,
    conditioned: LatentPredictor,
    goal_blind: LatentPredictor,
    *,
    paired_grid: bool,
) -> dict[str, object]:
    latent_targets, observations, masks = [], [], []
    outputs = {"conditioned": [], "goal_blind": []}
    for start in range(0, len(examples), config.batch_size):
        inputs, targets = pack_windows(
            examples[start : start + config.batch_size], spec, config
        )
        mask = np.asarray(inputs.queries) > 0
        masks.append(mask)
        observations.append(targets)
        latent_targets.append(codec.encode(tensor(targets)).numpy())
        for arm, predictor, public in (
            ("conditioned", conditioned, inputs),
            ("goal_blind", goal_blind, inputs.without_goal()),
        ):
            latent, decoded = predict(codec, predictor, public)
            outputs[arm].append((np.asarray(latent), np.asarray(decoded)))
    mask = np.concatenate(masks)
    targets = np.concatenate(observations)
    target_latents = np.concatenate(latent_targets)[mask]
    reconstruction = codec.decode(tensor(target_latents))
    scores = {
        "windows": len(examples),
        "target_steps": int(mask.sum()),
        "target_variance": float(np.var(target_latents, axis=0).mean()),
        "codec_reconstruction_mse": float(
            np.square(np.asarray(reconstruction) - targets[mask]).mean()
        ),
    }
    paired_layouts = (
        [grid_routes.layout_id(example.world_id) for example in examples[::2]]
        if paired_grid
        else []
    )
    for arm, predictions in outputs.items():
        latent, decoded = (
            np.concatenate(chunks) for chunks in zip(*predictions, strict=True)
        )
        scores[arm] = {
            "latent_mse": float(np.square(latent[mask] - target_latents).mean()),
            "observation_mse": float(np.square(decoded[mask] - targets[mask]).mean()),
            **(
                score_paired_routes(decoded, targets, mask, paired_layouts)
                if paired_grid
                else {}
            ),
        }
    return scores


def run_completion_training(
    out: Path,
    config: TrainingConfig,
    *,
    source: Path | None,
    evaluate_test: bool,
    resume: Path | None = None,
) -> dict[str, object]:
    if source is None:
        datasets = grid_routes.grid_demonstrations(include_test=evaluate_test)
        spec = ModelSpec(
            8, grid_routes.ACTION_COUNT, grid_routes.GOALS, grid_routes.BUDGET
        )
        origin = {
            "kind": "two_passage_grid_v2",
            "split_seed": grid_routes.SPLIT_SEED,
            "group_by": "passages_and_goal_locations",
        }
    else:
        from worth_reaching.envs.craftax import ACTION_NAMES

        if evaluate_test:
            raise ValueError("--evaluate-test is available for the grid diagnostic")
        corpus = load_completion_corpus(source)
        datasets = {
            split: list(corpus.iter_examples(split)) for split in ("train", "dev")
        }
        if not datasets["train"] or not datasets["dev"]:
            raise ValueError("Training requires both train and development windows")
        spec = ModelSpec(
            datasets["train"][0].inputs.observations.shape[-1],
            len(ACTION_NAMES),
            (corpus.goal,),
            corpus.budget,
        )
        origin = {
            "kind": "craftax_windows",
            "path": str(source.resolve()),
            "manifest_sha256": sha256(
                (source / "manifest.json").read_bytes()
            ).hexdigest(),
        }

    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "manifest.json",
        {
            "schema": CHECKPOINT_SCHEMA,
            "config": asdict(config),
            "model": asdict(spec),
            "source": origin,
            "code_revision": code_revision(),
            "library_versions": {
                name: version(name) for name in ("torch", "lightning", "numpy")
            },
            "compute": describe(),
            "new_simulator_steps": 0,
        },
    )
    worlds = {
        split: sorted({example.world_id for example in rows})
        for split, rows in datasets.items()
    }
    write_json(out / "splits.json", worlds)
    codec, conditioned, goal_blind = fit(
        datasets["train"],
        spec,
        config,
        out / "checkpoints",
        validate_resume(out, resume),
    )
    summary = {
        "training_windows": len(datasets["train"]),
        "world_counts": {split: len(ids) for split, ids in worlds.items()},
        "evaluation": {
            split: _evaluate(
                rows,
                spec,
                config,
                codec,
                conditioned,
                goal_blind,
                paired_grid=source is None,
            )
            for split, rows in datasets.items()
            if split != "train"
        },
    }
    (out / "weights.msgpack").write_bytes(
        serialization.to_bytes(
            {
                "codec": weights(codec),
                "conditioned": weights(conditioned),
                "goal_blind": weights(goal_blind),
            }
        )
    )
    write_json(out / "summary.json", summary)
    return summary
