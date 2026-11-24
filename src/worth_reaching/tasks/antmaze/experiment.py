"""Fit masked completion and try decoded goals in matched AntMaze rollouts."""

import gzip
import json
from dataclasses import asdict
from hashlib import file_digest
from pathlib import Path

import numpy as np
from flax import serialization

from worth_reaching.data.artifacts import code_revision, write_json
from worth_reaching.models.continuous_completion import CompletionConfig, fit
from worth_reaching.models.training import describe, stream_seed, validate_resume
from worth_reaching.planning.receding_goals import RecedingGoals
from worth_reaching.tasks.antmaze import environment as antmaze
from worth_reaching.tasks.antmaze.baseline import (
    EvalConfig,
    evaluation_starts,
    paired_metrics,
    rollout,
    summarize_episodes,
)
from worth_reaching.tasks.antmaze.checkpoint import load_policy
from worth_reaching.tasks.antmaze.dataset import (
    ACTION_SIZE,
    DATASETS,
    load_antmaze_archive,
)


def _fingerprint(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _summarize_probe(candidate_episodes: list[list[dict]], direct: list[dict]) -> dict:
    succeeded = np.array(
        [[row["success"] for row in arm] for arm in candidate_episodes], dtype=bool
    )
    any_success = succeeded.any(axis=0)
    return {
        "starts": len(direct),
        "starts_with_any_success": int(any_success.sum()),
        "starts_with_mixed_outcomes": int((any_success & ~succeeded.all(axis=0)).sum()),
        "starts_where_alternative_succeeds_and_first_fails": int(
            (succeeded[1:].any(axis=0) & ~succeeded[0]).sum()
        ),
        "starts_where_candidate_succeeds_and_direct_fails": int(
            (any_success & ~np.array([row["success"] for row in direct])).sum()
        ),
        "per_first_route": [
            {
                "first_route": route,
                "outcomes": summarize_episodes(episodes),
                "paired_vs_direct": paired_metrics(direct, episodes),
            }
            for route, episodes in enumerate(candidate_episodes)
        ],
    }


def run(
    dataset: str,
    data_dir: Path,
    checkpoint: Path,
    out: Path,
    config: CompletionConfig,
    evaluation: EvalConfig,
    start_bank: Path | None,
    *,
    probe_first_candidates: bool = False,
    resume: Path | None = None,
) -> dict:
    if dataset not in DATASETS or evaluation.action_limit <= config.queries:
        raise ValueError(
            "Use an AntMaze dataset and an evaluation budget longer than the query count"
        )
    if probe_first_candidates and config.candidates < 2:
        raise ValueError("The first-candidate probe needs at least two sampled routes")
    archive = load_antmaze_archive(data_dir / f"{dataset}.npz")
    validation = load_antmaze_archive(data_dir / f"{dataset}-val.npz")
    if archive.fingerprint == validation.fingerprint:
        raise ValueError("Training and validation must use distinct AntMaze archives")
    actor = load_policy(checkpoint)
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "manifest.json",
        {
            "schema": "antmaze_latent_completion_v2",
            "environment": antmaze.ENVIRONMENT,
            "environment_versions": dict(
                zip(("ogbench", "gymnasium", "mujoco"), antmaze.VERSIONS, strict=True)
            ),
            "dataset": dataset,
            "data_dir": str(data_dir.resolve()),
            "train_sha256": archive.fingerprint,
            "validation_sha256": validation.fingerprint,
            "executor": str(checkpoint.resolve()),
            "executor_weights_sha256": _fingerprint(checkpoint / "weights.msgpack"),
            "executor_normalization_sha256": _fingerprint(
                checkpoint / "normalization.npz"
            ),
            "training": asdict(config),
            "evaluation": asdict(evaluation),
            "compute": describe(),
            "code_revision": code_revision(),
            "start_source": str(start_bank.resolve()) if start_bank else None,
            "selection": "first sampled route, earliest predicted state",
            "probe_first_candidates": probe_first_candidates,
        },
    )
    settings = {
        "history": config.history,
        "queries": config.queries,
        "max_horizon": config.max_horizon,
    }
    windows = archive.sample_windows(
        np.random.default_rng([config.seed, 800]), config.windows, **settings
    )
    mean = archive.observations.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(archive.observations.std(axis=0, dtype=np.float64), 1e-3).astype(
        np.float32
    )
    eval_windows = validation.sample_windows(
        np.random.default_rng([config.seed, 801]), 512, **settings
    )
    model = fit(
        windows,
        eval_windows,
        mean,
        scale,
        config,
        out / "checkpoints",
        validate_resume(
            out, resume, legacy_defaults={"training": {"learning_rate": 0.001}}
        ),
    )
    metrics = model.validate(eval_windows)
    (out / "weights.msgpack").write_bytes(
        serialization.to_bytes(
            {
                **model.weights,
                "mean": mean,
                "scale": scale,
                "flow_scale": model.flow_scale.numpy(),
            }
        )
    )
    write_json(out / "validation.json", metrics)

    env = antmaze.AntMaze()
    direct = []
    candidate_episodes: list[list[dict]] = [
        [] for _ in range(config.candidates if probe_first_candidates else 1)
    ]
    guided = candidate_episodes[0]
    try:
        starts = evaluation_starts(env, evaluation, start_bank)
        antmaze.save_starts(out / "starts.npz", starts)
        with gzip.open(out / "plans.jsonl.gz", "wt", encoding="utf-8") as stream:
            for start in starts:
                direct.append(rollout(env, start, actor, evaluation.action_limit))
                key = stream_seed(config.seed, start.task, start.seed)
                controller = RecedingGoals(
                    actor, model, key, evaluation.action_limit, action_size=ACTION_SIZE
                )
                episode = rollout(env, start, controller, evaluation.action_limit)
                guided.append({**episode, "plans": len(controller.plans)})
                for plan in controller.plans:
                    stream.write(
                        json.dumps(
                            {
                                "task": start.task,
                                "seed": start.seed,
                                **plan,
                                "executed_actions": min(
                                    plan["deadline"], episode["steps"]
                                )
                                - plan["step"],
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                for route, episodes in enumerate(candidate_episodes[1:], start=1):
                    alternative = RecedingGoals(
                        actor,
                        model,
                        key,
                        evaluation.action_limit,
                        action_size=ACTION_SIZE,
                        first_route=route,
                    )
                    episodes.append(
                        rollout(env, start, alternative, evaluation.action_limit)
                    )
    finally:
        env.close()
    write_json(out / "episodes.json", {"direct": direct, "latent": guided})
    summary = {
        "validation": metrics,
        "direct": summarize_episodes(direct),
        "latent": summarize_episodes(guided),
        "paired": paired_metrics(direct, guided),
        "plans": sum(row["plans"] for row in guided),
        "training_environment_steps": 0,
        "evaluation_environment_steps": sum(row["steps"] for row in direct)
        + sum(row["steps"] for arm in candidate_episodes for row in arm),
    }
    if probe_first_candidates:
        write_json(out / "first_candidates.json", candidate_episodes)
        summary["first_candidate_probe"] = _summarize_probe(candidate_episodes, direct)
    write_json(out / "summary.json", summary)
    return summary
