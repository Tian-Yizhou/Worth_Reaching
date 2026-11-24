"""Run learned latent planning on the maintained Maze2D benchmarks."""

import gzip
import json
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
from flax import serialization

from worth_reaching.data.artifacts import code_revision, write_json
from worth_reaching.data.maze_windows import MaskedBatch
from worth_reaching.data.pointmaze import (
    ACTION_SIZE,
    load_dataset,
    split_trajectories,
)
from worth_reaching.envs.pointmaze import SEED_BASE, PointMaze
from worth_reaching.envs.pointmaze_tasks import TASKS
from worth_reaching.evaluation.maze_planning import Policy, RecedingGoals
from worth_reaching.models.maze_completion import (
    BudgetedCompletionConfig,
    LearnedCompletion,
    fit,
)
from worth_reaching.models.training import describe, stream_seed, validate_resume

ALIGNED_SCHEMA = "pointmaze_latent_completion_v3"
BUDGET_SCHEMA = "pointmaze_latent_completion_v4"


def track(observation: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Use the position and velocity feedback from Diffuser's Maze2D runner."""
    position_error = target[:2] - observation[:2]
    velocity_error = target[2:] - observation[2:]
    return np.clip(position_error + velocity_error, -1.0, 1.0)


def rollout(
    env: PointMaze, seed: int, limit: int, policy: Policy
) -> tuple[dict, np.ndarray]:
    state, goal = env.reset(seed)
    start = state.tolist()
    closest = float(np.linalg.norm(state[:2] - goal[:2]))
    trajectory = [state]
    first_success = None
    total_reward = 0.0
    for step in range(1, limit + 1):
        outcome = env.step(policy(state, goal))
        state = outcome.observation
        trajectory.append(state)
        total_reward += outcome.reward
        closest = min(closest, float(np.linalg.norm(state[:2] - goal[:2])))
        if outcome.success and first_success is None:
            first_success = step
        if outcome.done:
            break
    return {
        "seed": seed,
        "start": start,
        "goal": goal.tolist(),
        "steps": step,
        "success": first_success is not None,
        "first_success_step": first_success,
        "sparse_return": total_reward,
        "closest_distance": closest,
    }, np.stack(trajectory)


def _summarize(episodes: list[dict]) -> dict:
    arrival_steps = [
        episode["first_success_step"] for episode in episodes if episode["success"]
    ]
    successes = len(arrival_steps)
    return {
        "episodes": len(episodes),
        "successes": successes,
        "success_rate": successes / len(episodes),
        "mean_first_success_step": float(np.mean(arrival_steps))
        if arrival_steps
        else None,
        "mean_sparse_return": float(np.mean([e["sparse_return"] for e in episodes])),
        "mean_closest_distance": float(
            np.mean([e["closest_distance"] for e in episodes])
        ),
    }


def _window_coverage(batch: MaskedBatch, limit: int, supporting_episodes: int) -> dict:
    budgets, present = batch.context.remaining, batch.context.present
    spans = np.minimum(budgets, limit)
    return {
        "windows": len(spans),
        "median_span": float(np.median(spans)),
        "max_span": int(spans.max()),
        "max_remaining_budget": int(budgets.max()),
        "windows_with_budget_above_span": int(np.count_nonzero(budgets > spans)),
        "full_span_reset_windows": int(
            np.count_nonzero((spans == limit) & (present.sum(axis=1) == 1))
        ),
        "episodes_long_enough": supporting_episodes,
    }


def _evaluate_budget(
    env: PointMaze,
    out: Path,
    model: LearnedCompletion,
    limit: int,
    seeds: list[int],
    resets: dict[int, tuple[list, list]],
) -> dict:
    direct, latent, recorded = [], [], {}
    with gzip.open(out / "plans.jsonl.gz", "wt", encoding="utf-8") as plans:
        for seed in seeds:
            control, control_path = rollout(env, seed, limit, track)
            planner = RecedingGoals(
                track,
                model,
                stream_seed(model.config.seed, seed),
                limit,
                action_size=ACTION_SIZE,
            )
            guided, guided_path = rollout(env, seed, limit, planner)
            reset = (control["start"], control["goal"])
            if reset != (guided["start"], guided["goal"]) or reset != resets.setdefault(
                seed, reset
            ):
                raise ValueError(
                    "PointMaze seeded reset failed to reproduce the public start and goal"
                )
            direct.append(control)
            latent.append(guided)
            recorded[f"direct_{seed}"] = control_path
            recorded[f"latent_{seed}"] = guided_path
            for plan in planner.plans:
                remaining = limit - plan["step"]
                plans.write(
                    json.dumps(
                        {
                            "seed": seed,
                            "remaining_budget": remaining,
                            "prediction_span": min(remaining, model.config.max_horizon),
                            **plan,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )

    np.savez_compressed(out / "trajectories.npz", **recorded)
    write_json(out / "episodes.json", {"direct": direct, "latent": latent})
    return {
        "action_limit": limit,
        "direct": _summarize(direct),
        "latent": _summarize(latent),
        "only_direct": sum(
            a["success"] and not b["success"]
            for a, b in zip(direct, latent, strict=True)
        ),
        "only_latent": sum(
            b["success"] and not a["success"]
            for a, b in zip(direct, latent, strict=True)
        ),
        "training_environment_steps": 0,
        "evaluation_environment_steps": sum(row["steps"] for row in (*direct, *latent)),
    }


def run(
    data_dir: Path,
    out: Path,
    config: BudgetedCompletionConfig,
    *,
    task_name: str,
    split: str,
    repeats: int,
    budgets: list[int] | None,
    prediction_span: int | None,
    resume: Path | None = None,
) -> dict:
    task = TASKS[task_name]
    budgets = [task.action_limit] if budgets is None else sorted(budgets)
    if (
        not budgets
        or len(set(budgets)) != len(budgets)
        or not all(config.queries < limit <= task.action_limit for limit in budgets)
        or not 0 < repeats <= SEED_BASE["test"] - SEED_BASE["dev"]
    ):
        raise ValueError(
            f"Choose distinct action budgets in {config.queries + 1}..{task.action_limit} "
            "and disjoint reset seeds"
        )
    dataset, file_hashes = load_dataset(data_dir, task.dataset_id)
    train, validation, episode_split = split_trajectories(dataset)
    train_lengths, validation_lengths = (
        train.episode_lengths,
        validation.episode_lengths,
    )
    eligible = train_lengths[train_lengths > config.queries]
    if not len(eligible):
        raise ValueError(
            "PointMaze training episodes are too short for the completion queries"
        )
    horizon_source = (
        "requested" if prediction_span is not None else "training_episode_p90"
    )
    if prediction_span is None:
        prediction_span = min(
            task.action_limit,
            int(np.quantile(eligible, 0.9, method="lower")),
            int(validation_lengths.max()),
        )
    train_support = int(np.count_nonzero(train_lengths >= prediction_span))
    validation_support = int(np.count_nonzero(validation_lengths >= prediction_span))
    if (
        not config.queries < prediction_span <= task.action_limit
        or not train_support
        or not validation_support
    ):
        raise ValueError(
            f"The {prediction_span}-step prediction span is unsupported by the PointMaze protocol "
            f"(longest train/validation episodes: {train_lengths.max()}/{validation_lengths.max()}); "
            "choose a covered --prediction-span value within the official limit or supply longer trajectories"
        )
    config = replace(
        config, max_horizon=prediction_span, max_action_budget=task.action_limit
    )
    settings = dict(
        history=config.history,
        queries=config.queries,
        max_horizon=prediction_span,
        max_action_budget=task.action_limit,
    )
    training = train.sample_windows(
        np.random.default_rng([config.seed, 800]),
        config.windows,
        full_span_windows=max(1, config.windows // 4),
        **settings,
    )
    validation_windows = validation.sample_windows(
        np.random.default_rng(801),
        512,
        full_span_windows=128,
        **settings,
    )
    seeds = list(range(SEED_BASE[split], SEED_BASE[split] + repeats))
    native = dataset.recover_environment(eval_env=True)
    try:
        env = PointMaze(native, task)
        out.mkdir(parents=True, exist_ok=False)
        evaluation = {"split": split, "seeds": seeds}
        multiple = len(budgets) > 1
        manifest = {
            "schema": BUDGET_SCHEMA,
            "task": task_name,
            "environment": native.spec.id,
            "versions": {
                name: version(name)
                for name in ("minari", "gymnasium-robotics", "gymnasium", "mujoco")
            },
            "dataset": task.dataset_id,
            "dataset_files_sha256": file_hashes,
            "training": asdict(config),
            "window_sampling": {
                "span_source": horizon_source,
                "train": _window_coverage(training, prediction_span, train_support),
                "validation": _window_coverage(
                    validation_windows, prediction_span, validation_support
                ),
            },
            "evaluation": {
                **evaluation,
                **(
                    {"action_budgets": budgets}
                    if multiple
                    else {"action_limit": budgets[0]}
                ),
            },
            "selection": "first sampled route, earliest queried state",
            "compute": describe(),
            "code_revision": code_revision(),
        }
        write_json(out / "manifest.json", manifest)
        write_json(out / "split.json", episode_split)
        mean = train.observations.mean(axis=0)
        scale = np.maximum(train.observations.std(axis=0), 1e-3)
        model = fit(
            training,
            validation_windows,
            mean,
            scale,
            config,
            out / "checkpoints",
            validate_resume(out, resume),
        )
        metrics = model.validate(validation_windows)
        write_json(out / "validation.json", metrics)
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

        by_budget, resets = {}, {}
        for limit in budgets:
            arm = out / f"budget-{limit}" if multiple else out
            if multiple:
                arm.mkdir()
                (arm / "weights.msgpack").hardlink_to(out / "weights.msgpack")
                write_json(
                    arm / "manifest.json",
                    {**manifest, "evaluation": {**evaluation, "action_limit": limit}},
                )
            result = _evaluate_budget(env, arm, model, limit, seeds, resets)
            by_budget[limit] = result
            if multiple:
                write_json(arm / "summary.json", result)
    finally:
        native.close()

    if multiple:
        summary = {
            "validation": metrics,
            "budgets": by_budget,
            "training_environment_steps": 0,
            "evaluation_environment_steps": sum(
                row["evaluation_environment_steps"] for row in by_budget.values()
            ),
        }
    else:
        summary = {"validation": metrics, **by_budget[budgets[0]]}
    write_json(out / "summary.json", summary)
    return summary
