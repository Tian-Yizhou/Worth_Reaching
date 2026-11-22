"""Train a shared OGBench manipulation executor and test decoded plans."""

import gzip
import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from flax import serialization

from worth_reaching.data.artifacts import code_revision, write_json
from worth_reaching.data.ogbench import StateArchive, load_state_archive
from worth_reaching.envs.ogbench_manipulation import STATE_SPEC, Manipulation
from worth_reaching.envs.ogbench_tasks import (
    ACTION_SIZE,
    SEED_OFFSETS,
    TASKS,
    ManipulationTask,
)
from worth_reaching.evaluation.maze_planning import Policy, RecedingGoals
from worth_reaching.models import gcbc
from worth_reaching.models.maze_completion import CompletionConfig, fit
from worth_reaching.models.training import (
    describe,
    regression,
    stage_resume,
    stream_seed,
    train_stage,
    validate_resume,
    weights,
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
            raise ValueError("Use at least one executor update")


def _fit_executor(
    train: StateArchive,
    normalization: tuple[np.ndarray, np.ndarray],
    seed: int,
    config: ExecutorConfig,
    out: Path,
    resume: Path | None,
) -> tuple[Policy, dict]:
    mean, scale = normalization
    torch.manual_seed(seed)
    model = train_stage(
        gcbc.GCBC(config.width, len(mean), ACTION_SIZE, relative_coordinates=0),
        gcbc.BehaviorBatches(
            train,
            mean,
            scale,
            (seed, 600),
            config.batch_size,
            config.short_goal_steps,
            alternate_horizons=True,
        ),
        regression,
        steps=config.updates,
        learning_rate=config.learning_rate,
        out=out,
        resume=resume,
    )
    return gcbc.policy_from_model(model, mean, scale), weights(model)


def rollout(
    env: Manipulation, initial: np.ndarray, goal: np.ndarray, policy: Policy, limit: int
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
    benchmark: ManipulationTask,
    data_dir: Path,
    out: Path,
    config: CompletionConfig,
    executor: ExecutorConfig,
    *,
    split: str,
    task: int | None,
    repeats: int,
    limit: int | None,
    resume: Path | None = None,
) -> dict:
    if limit is None:
        limit = benchmark.action_limit
    if (
        not config.queries < limit <= benchmark.action_limit
        or not 0 < repeats <= SEED_OFFSETS["test"]
    ):
        raise ValueError(
            f"Choose an action budget in {config.queries + 1}..{benchmark.action_limit} "
            f"for {benchmark.environment} and disjoint reset seeds"
        )
    train, validation = [
        load_state_archive(
            data_dir / f"{benchmark.dataset}{suffix}.npz",
            observation_size=benchmark.observation_size,
            action_size=ACTION_SIZE,
        )
        for suffix in ("", "-val")
    ]
    if train.fingerprint == validation.fingerprint:
        raise ValueError("Training and validation must use distinct OGBench archives")
    tasks = TASKS if task is None else (task,)
    start_seed = benchmark.seed_base + SEED_OFFSETS[split]
    seeds = range(start_seed, start_seed + repeats)
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "manifest.json",
        {
            "schema": f"ogbench_{benchmark.family}_latent_v2",
            "environment": benchmark.environment,
            "observation": "official_state",
            "versions": {
                name: version(name) for name in ("ogbench", "gymnasium", "mujoco")
            },
            "dataset": benchmark.dataset,
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
    previous = validate_resume(out, resume)
    actor, actor_weights = _fit_executor(
        train,
        (mean, scale),
        config.seed,
        executor,
        out / "checkpoints" / "executor",
        stage_resume(previous, "executor"),
    )
    settings = dict(
        history=config.history, queries=config.queries, max_horizon=config.max_horizon
    )
    windows = train.sample_windows(
        np.random.default_rng([config.seed, 800]), config.windows, **settings
    )
    model = fit(windows, mean, scale, config, out / "checkpoints", previous)
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
                "flow_scale": model.flow_scale.numpy(),
            }
        )
    )

    episodes = {"direct": [], "latent": []}
    starts, trajectories = [], {}
    env = Manipulation(benchmark)
    try:
        with gzip.open(out / "plans.jsonl.gz", "wt", encoding="utf-8") as stream:
            for task_id in tasks:
                for seed in seeds:
                    start = env.capture(task_id, seed)
                    starts.append(start)
                    planner = RecedingGoals(
                        actor,
                        model,
                        stream_seed(config.seed, task_id, seed),
                        limit,
                        action_size=ACTION_SIZE,
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
        state_spec=STATE_SPEC,
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
