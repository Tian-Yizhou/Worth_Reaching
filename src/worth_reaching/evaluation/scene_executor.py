"""Qualify a goal-conditioned Scene executor on the official tasks.

The executor is trained on the Scene play archive and evaluated by tracking the
official goal observation from seeded resets until the native success event or the
action limit. Direct success that stays above zero is a precondition for any
comparison of intermediate goals in Scene.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import partial
from hashlib import sha256
from importlib.metadata import version
from itertools import product
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

from worth_reaching.data.artifacts import code_revision, write_json
from worth_reaching.data.ogbench import load_state_archive
from worth_reaching.models.gcivl import GCIVL, GCIVLConfig, fit_gcivl
from worth_reaching.models.training import (
    WEIGHTS_FILE,
    describe,
    load_weights,
    restore_weights,
    save_weights,
    weights,
)
from worth_reaching.tasks.ogbench_manipulation.environment import Manipulation
from worth_reaching.tasks.ogbench_manipulation.spec import (
    ACTION_SIZE,
    SCENE,
    SEED_OFFSETS,
)

SCHEMA = "scene_executor_v1"


def _restore(weights_path: Path, config: GCIVLConfig) -> GCIVL:
    model = GCIVL(SCENE.observation_size, SCENE.observation_size, ACTION_SIZE, config)
    return restore_weights(model, load_weights(weights_path)["executor"])


def _rollouts(cases, *, weights_path: Path, config: GCIVLConfig, limit: int):
    torch.set_num_threads(1)
    model = _restore(weights_path, config)
    env = Manipulation(SCENE)
    results = []
    try:
        for task, seed in cases:
            start = env.capture(task, seed)
            observation = start.observation
            success, step = False, 0
            for step in range(1, limit + 1):
                observation, success, done = env.step(
                    model.act(observation, start.goal)
                )
                if success or done:
                    break
            results.append(
                {"task": task, "seed": seed, "success": success, "steps": step}
            )
    finally:
        env.close()
    return results


def _offline_metrics(model: GCIVL, archive, rng: np.random.Generator) -> dict:
    """Value of goals at increasing distances in held-out trajectories."""
    rows = rng.choice(archive.rows, size=2048)
    final = archive.ends[np.searchsorted(archive.ends, rows)]
    metrics = {}
    with torch.no_grad():
        for steps in (1, 10, 50, 200):
            goals = np.minimum(rows + steps, final)
            first, second = model.value(
                torch.as_tensor(archive.observations[rows]),
                torch.as_tensor(archive.observations[goals]),
            )
            metrics[f"value_at_{steps}_steps"] = float(((first + second) / 2).mean())
        actions = model(
            torch.as_tensor(archive.observations[rows]),
            torch.as_tensor(archive.observations[np.minimum(rows + 10, final)]),
        ).clamp(-1, 1)
        metrics["action_mse_goal_10_steps"] = float(
            (actions - torch.as_tensor(archive.actions[rows])).square().mean()
        )
    return metrics


def run(
    data_dir: Path,
    out: Path,
    config: GCIVLConfig,
    *,
    seed: int,
    split: str,
    tasks: tuple[int, ...],
    repeats: int,
    eval_workers: int,
    resume: Path | None = None,
) -> dict:
    if not 0 < repeats <= SEED_OFFSETS["test"] or eval_workers < 1:
        raise ValueError(
            "Use positive repeats within the split and at least one worker"
        )
    train, validation = (
        load_state_archive(
            data_dir / f"{SCENE.dataset}{suffix}.npz",
            observation_size=SCENE.observation_size,
            action_size=ACTION_SIZE,
        )
        for suffix in ("", "-val")
    )
    out.mkdir(parents=True, exist_ok=False)
    start_seed = SCENE.seed_base + SEED_OFFSETS[split]
    seeds = list(range(start_seed, start_seed + repeats))
    model = fit_gcivl(
        train,
        ACTION_SIZE,
        config,
        seed=seed,
        out=out / "checkpoints" / "executor",
        resume=resume,
    )
    weights_path = out / WEIGHTS_FILE
    save_weights(weights_path, {"executor": weights(model)})
    metrics = _offline_metrics(model, validation, np.random.default_rng(803))
    write_json(
        out / "manifest.json",
        {
            "schema": SCHEMA,
            "environment": SCENE.environment,
            "executor": {"kind": "gcivl", **asdict(config)},
            "seed": seed,
            "train_sha256": train.fingerprint,
            "validation_sha256": validation.fingerprint,
            "evaluation": {"split": split, "tasks": tasks, "seeds": seeds},
            "versions": {
                name: version(name) for name in ("ogbench", "mujoco", "torch")
            },
            "compute": {**describe(), "eval_workers": eval_workers},
            "code_revision": code_revision(),
            "weights_sha256": sha256(weights_path.read_bytes()).hexdigest(),
        },
    )
    cases = list(product(tasks, seeds))
    shards = [cases[index::eval_workers] for index in range(eval_workers)]
    evaluate = partial(
        _rollouts, weights_path=weights_path, config=config, limit=SCENE.action_limit
    )
    with ProcessPoolExecutor(len(shards), mp_context=get_context("spawn")) as pool:
        episodes = sorted(
            (row for shard in pool.map(evaluate, shards) for row in shard),
            key=lambda row: (row["task"], row["seed"]),
        )
    write_json(out / "episodes.json", episodes)
    summary = {
        "offline": metrics,
        "success_rate": float(np.mean([row["success"] for row in episodes])),
        "per_task": {
            str(task): float(
                np.mean([row["success"] for row in episodes if row["task"] == task])
            )
            for task in tasks
        },
        "evaluation_environment_steps": sum(row["steps"] for row in episodes),
    }
    write_json(out / "summary.json", summary)
    return summary
