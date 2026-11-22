"""Run learned latent planning on the maintained Maze2D benchmarks."""

import gzip
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import jax
import numpy as np
from flax import serialization

from worth_reaching.data.artifacts import code_revision, write_json
from worth_reaching.data.pointmaze import (
    ACTION_SIZE,
    load_dataset,
    split_trajectories,
)
from worth_reaching.envs.pointmaze import SEED_BASE, PointMaze
from worth_reaching.envs.pointmaze_tasks import TASKS
from worth_reaching.evaluation.maze_planning import Policy, RecedingGoals
from worth_reaching.models.maze_completion import CompletionConfig, fit
from worth_reaching.models.parallel import describe


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
    successes = sum(episode["success"] for episode in episodes)
    return {
        "episodes": len(episodes),
        "successes": successes,
        "success_rate": successes / len(episodes),
        "mean_sparse_return": float(np.mean([e["sparse_return"] for e in episodes])),
        "mean_closest_distance": float(
            np.mean([e["closest_distance"] for e in episodes])
        ),
    }


def run(
    data_dir: Path,
    out: Path,
    config: CompletionConfig,
    *,
    task_name: str,
    split: str,
    repeats: int,
    limit: int | None,
) -> dict:
    task = TASKS[task_name]
    if limit is None:
        limit = task.action_limit
    if not config.queries < limit <= task.action_limit or not 0 < repeats <= (
        SEED_BASE["test"] - SEED_BASE["dev"]
    ):
        raise ValueError(
            f"Choose an action budget in {config.queries + 1}..{task.action_limit} "
            "and disjoint reset seeds"
        )
    dataset, file_hashes = load_dataset(data_dir, task.dataset_id)
    train, validation, episode_split = split_trajectories(dataset)
    seeds = list(range(SEED_BASE[split], SEED_BASE[split] + repeats))
    native = dataset.recover_environment(eval_env=True)
    try:
        env = PointMaze(native, task)
        out.mkdir(parents=True, exist_ok=False)
        write_json(
            out / "manifest.json",
            {
                "schema": "pointmaze_latent_completion_v1",
                "task": task_name,
                "environment": native.spec.id,
                "versions": {
                    name: version(name)
                    for name in ("minari", "gymnasium-robotics", "gymnasium", "mujoco")
                },
                "dataset": task.dataset_id,
                "dataset_files_sha256": file_hashes,
                "training": asdict(config),
                "evaluation": {"split": split, "seeds": seeds, "action_limit": limit},
                "selection": "first sampled route, earliest queried state",
                "compute": describe(),
                "code_revision": code_revision(),
            },
        )
        write_json(out / "split.json", episode_split)
        settings = dict(
            history=config.history,
            queries=config.queries,
            max_horizon=config.max_horizon,
        )
        training = train.sample_windows(
            np.random.default_rng([config.seed, 800]), config.windows, **settings
        )
        mean = train.observations.mean(axis=0)
        scale = np.maximum(train.observations.std(axis=0), 1e-3)
        model = fit(training, mean, scale, config)
        validation_windows = validation.sample_windows(
            np.random.default_rng(801), 512, **settings
        )
        metrics = model.validate(validation_windows)
        write_json(out / "validation.json", metrics)
        (out / "weights.msgpack").write_bytes(
            serialization.to_bytes(
                {
                    **model.weights,
                    "mean": mean,
                    "scale": scale,
                    "flow_scale": model.flow_scale,
                }
            )
        )

        direct, latent = [], []
        recorded = {}
        with gzip.open(out / "plans.jsonl.gz", "wt", encoding="utf-8") as plans:
            for seed in seeds:
                control, control_path = rollout(env, seed, limit, track)
                planner = RecedingGoals(
                    track,
                    model,
                    jax.random.fold_in(jax.random.key(config.seed), seed),
                    limit,
                    action_size=ACTION_SIZE,
                )
                guided, guided_path = rollout(env, seed, limit, planner)
                if (control["start"], control["goal"]) != (
                    guided["start"],
                    guided["goal"],
                ):
                    raise ValueError(
                        "PointMaze seeded reset failed to reproduce the public start and goal"
                    )
                direct.append(control)
                latent.append(guided)
                recorded[f"direct_{seed}"] = control_path
                recorded[f"latent_{seed}"] = guided_path
                for plan in planner.plans:
                    plans.write(
                        json.dumps({"seed": seed, **plan}, separators=(",", ":")) + "\n"
                    )
    finally:
        native.close()

    np.savez_compressed(out / "trajectories.npz", **recorded)
    write_json(out / "episodes.json", {"direct": direct, "latent": latent})
    summary = {
        "validation": metrics,
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
    write_json(out / "summary.json", summary)
    return summary
