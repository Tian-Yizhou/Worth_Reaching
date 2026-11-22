"""Train a shared Scene executor and test decoded plans on official tasks."""

import gzip
import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import jax
import numpy as np
from flax import serialization

from worth_reaching.data.artifacts import code_revision, write_json
from worth_reaching.data.ogbench import StateArchive, load_state_archive
from worth_reaching.envs import scene
from worth_reaching.evaluation.maze_planning import Policy, RecedingGoals
from worth_reaching.models import gcbc
from worth_reaching.models.maze_completion import CompletionConfig, fit
from worth_reaching.models.parallel import (
    describe,
    replicate,
    shard_batch,
    single_device,
)


@dataclass(frozen=True, slots=True)
class ExecutorConfig:
    updates: int = 2500
    width: int = 256
    batch_size: int = 256
    learning_rate: float = 0.0003
    short_goal_steps: int = 64

    def __post_init__(self) -> None:
        if self.updates < 1:
            raise ValueError("Use at least one Scene executor update")


def _fit_executor(
    train: StateArchive,
    normalization: tuple[np.ndarray, np.ndarray],
    seed: int,
    config: ExecutorConfig,
) -> tuple[Policy, dict]:
    mean, scale = normalization
    model, state = gcbc.initialize(
        config.width,
        seed,
        config.learning_rate,
        observation_size=scene.OBSERVATION_SIZE,
        action_size=scene.ACTION_SIZE,
        relative_coordinates=0,
    )
    state = replicate(state)
    rng = np.random.default_rng([seed, 600])
    for index in range(config.updates):
        cap = config.short_goal_steps if index % 2 == 0 else None
        batch = train.sample(rng, config.batch_size, max_goal_steps=cap)
        inputs, valid = shard_batch(
            (
                (batch.observations - mean) / scale,
                (batch.goals - mean) / scale,
                batch.actions,
            )
        )
        state = gcbc.update(state, *inputs, valid)
    weights = single_device(state.params)
    actor = gcbc.policy_from_weights(model, weights, mean, scale)
    return actor, weights


def rollout(
    env: scene.Scene, initial: np.ndarray, goal: np.ndarray, policy: Policy, limit: int
) -> tuple[dict, np.ndarray, np.ndarray]:
    observation = initial
    states, actions = [observation], []
    for step in range(1, limit + 1):
        action = policy(observation, goal)
        observation, success, done = env.step(action)
        states.append(observation)
        actions.append(action)
        if success or done:
            break
    return {"success": success, "steps": step}, np.stack(states), np.stack(actions)


def _summarize(episodes: list[dict], tasks: tuple[int, ...]) -> dict:
    return {
        "episodes": len(episodes),
        "successes": sum(row["success"] for row in episodes),
        "success_rate": float(np.mean([row["success"] for row in episodes])),
        "per_task": {
            str(task): {
                "episodes": sum(row["task"] == task for row in episodes),
                "successes": sum(
                    row["task"] == task and row["success"] for row in episodes
                ),
            }
            for task in tasks
        },
    }


def run(
    data_dir: Path,
    out: Path,
    config: CompletionConfig,
    executor: ExecutorConfig,
    *,
    split: str,
    task: int | None,
    repeats: int,
    limit: int,
) -> dict:
    if not config.queries < limit <= scene.ACTION_LIMIT or not 0 < repeats <= (
        scene.SEED_BASE["test"] - scene.SEED_BASE["dev"]
    ):
        raise ValueError("Choose a valid Scene action budget and disjoint reset seeds")
    train, validation = [
        load_state_archive(
            data_dir / f"{scene.DATASET}{suffix}.npz",
            observation_size=scene.OBSERVATION_SIZE,
            action_size=scene.ACTION_SIZE,
        )
        for suffix in ("", "-val")
    ]
    if train.fingerprint == validation.fingerprint:
        raise ValueError("Training and validation must use distinct Scene archives")
    tasks = scene.TASKS if task is None else (task,)
    seeds = range(scene.SEED_BASE[split], scene.SEED_BASE[split] + repeats)
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "manifest.json",
        {
            "schema": "ogbench_scene_latent_v1",
            "environment": scene.ENVIRONMENT,
            "observation": "official_state",
            "versions": {
                name: version(name) for name in ("ogbench", "gymnasium", "mujoco")
            },
            "dataset": scene.DATASET,
            "train_sha256": train.fingerprint,
            "validation_sha256": validation.fingerprint,
            "training": asdict(config),
            "executor": asdict(executor),
            "evaluation": {
                "split": split,
                "tasks": tasks,
                "seeds": list(seeds),
                "action_limit": limit,
            },
            "selection": "first sampled route, earliest queried state",
            "compute": describe(),
            "code_revision": code_revision(),
        },
    )
    mean = train.observations.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(train.observations.std(axis=0, dtype=np.float64), 1e-3).astype(
        np.float32
    )
    actor, actor_weights = _fit_executor(train, (mean, scale), config.seed, executor)
    settings = dict(
        history=config.history, queries=config.queries, max_horizon=config.max_horizon
    )
    windows = train.sample_windows(
        np.random.default_rng([config.seed, 800]), config.windows, **settings
    )
    model = fit(windows, mean, scale, config)
    heldout = validation.sample_windows(np.random.default_rng(801), 512, **settings)
    metrics = model.validate(heldout, compare_xy=False)
    for label, cap in (("short", executor.short_goal_steps), ("all", None)):
        batch = validation.sample(np.random.default_rng(803), 1024, max_goal_steps=cap)
        errors = actor(batch.observations, batch.goals) - batch.actions
        metrics[f"executor_{label}_action_mse"] = float(np.square(errors).mean())
    write_json(out / "validation.json", metrics)
    (out / "weights.msgpack").write_bytes(
        serialization.to_bytes(
            {
                **model.weights,
                "executor": actor_weights,
                "mean": mean,
                "scale": scale,
                "flow_scale": model.flow_scale,
            }
        )
    )

    episodes = {"direct": [], "latent": []}
    starts, trajectories = [], {}
    env = scene.Scene()
    try:
        with gzip.open(out / "plans.jsonl.gz", "wt", encoding="utf-8") as stream:
            for task_id in tasks:
                for seed in seeds:
                    start = env.capture(task_id, seed)
                    starts.append(start)
                    key = jax.random.fold_in(jax.random.key(config.seed), task_id)
                    planner = RecedingGoals(
                        actor,
                        model,
                        jax.random.fold_in(key, seed),
                        limit,
                        action_size=scene.ACTION_SIZE,
                    )
                    for arm, policy in (("direct", actor), ("latent", planner)):
                        initial = (
                            start.observation if arm == "direct" else env.restore(start)
                        )
                        result, states, actions = rollout(
                            env, initial, start.goal, policy, limit
                        )
                        episodes[arm].append({"task": task_id, "seed": seed, **result})
                        trajectories[f"{arm}_{task_id}_{seed}_observations"] = states
                        trajectories[f"{arm}_{task_id}_{seed}_actions"] = actions
                    for plan in planner.plans:
                        stream.write(
                            json.dumps(
                                {"task": task_id, "seed": seed, **plan},
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
    finally:
        env.close()
    np.savez_compressed(
        out / "starts.npz",
        ids=[(s.task, s.seed) for s in starts],
        observation=np.stack([s.observation for s in starts]),
        goal=np.stack([s.goal for s in starts]),
        physics=np.stack([s.physics for s in starts]),
        state_spec=scene.STATE_SPEC,
    )
    np.savez_compressed(out / "trajectories.npz", **trajectories)
    write_json(out / "episodes.json", episodes)
    paired = list(zip(episodes["direct"], episodes["latent"], strict=True))
    summary = {
        "validation": metrics,
        **{arm: _summarize(rows, tasks) for arm, rows in episodes.items()},
        "only_direct": sum(a["success"] and not b["success"] for a, b in paired),
        "only_latent": sum(b["success"] and not a["success"] for a, b in paired),
        "training_environment_steps": 0,
        "evaluation_environment_steps": sum(a["steps"] + b["steps"] for a, b in paired),
    }
    write_json(out / "summary.json", summary)
    return summary
